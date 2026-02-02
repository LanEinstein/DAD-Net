# -*- coding: utf-8 -*-
"""
MicroFlowNeXt: Motion-Sensitive Backbone for Micro-Expression Recognition

This module implements the MicroFlowNeXt architecture, a lightweight ConvNeXt-style
backbone specifically designed for optical flow-based micro-expression recognition.

Key Components:
    - MicroSELayer: Peak-Enhanced Channel Attention using dual pooling (avg + max)
    - MicroSelfAttention: Self-attention with micro-change enhancement branch
    - MicroBlock: Gradient-aware block for capturing subtle motion patterns
    - Spatial Transformer Network (STN): Geometric normalization with flow vector correction

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth


class MicroSELayer(nn.Module):
    """
    Peak-Enhanced Channel Attention Module.

    Combines global average pooling and max pooling to capture both
    average statistics and peak activations, which is crucial for
    detecting subtle micro-expression movements.

    Args:
        channel (int): Number of input channels.
        reduction (int): Channel reduction ratio for the bottleneck. Default: 32.
    """

    def __init__(self, channel: int, reduction: int = 32):
        super(MicroSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(channel // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Channel-reweighted tensor of shape (B, C, H, W).
        """
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = torch.cat([y_avg, y_max], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MicroSelfAttention(nn.Module):
    """
    Lightweight Self-Attention Module with Micro-Change Enhancement.

    Computes self-attention over the feature dimension with an additional
    enhancement branch to amplify subtle variations in the input.

    Args:
        input_dim (int): Dimension of input features.
        enhancement_weight (float): Weight for the micro-enhancement branch. Default: 0.1.
    """

    def __init__(self, input_dim: int, enhancement_weight: float = 0.1):
        super(MicroSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.enhancement_weight = enhancement_weight

        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, input_dim)
        self.micro_enhance = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (B, D) or (B, N, D).

        Returns:
            Tuple of (attended_output, attention_weights).
        """
        x = self.norm(x)
        proj = self.projection(x)

        scores = torch.matmul(proj, proj.transpose(-2, -1))
        scores = scores / math.sqrt(self.input_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, x)

        enhanced = self.micro_enhance(x)
        output = output + self.enhancement_weight * enhanced

        return output, attention_weights


class LayerNorm(nn.Module):
    """
    Layer Normalization supporting both channels-first and channels-last formats.

    Args:
        normalized_shape (int): Input shape from an expected input.
        eps (float): Epsilon for numerical stability. Default: 1e-6.
        data_format (str): Either "channels_last" or "channels_first". Default: "channels_last".
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(
                f"data_format must be 'channels_last' or 'channels_first', got {data_format}"
            )
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MicroBlock(nn.Module):
    """
    Gradient-Aware Block for Micro-Expression Recognition.

    Combines depthwise convolution with a gradient enhancement pathway
    to capture subtle motion patterns and fine-grained texture changes.

    Args:
        dim (int): Number of input/output channels.
        drop_path (float): Stochastic depth rate. Default: 0.0.
        gradient_weight (float): Weight for the gradient enhancement pathway. Default: 0.1.
    """

    def __init__(self, dim: int, drop_path: float = 0.0, gradient_weight: float = 0.1):
        super().__init__()
        self.gradient_weight = gradient_weight

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.grad_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )
        self.stochastic_depth = StochasticDepth(drop_path, "row")

        self.register_buffer('sobel_x',
            torch.tensor([[[[-1., 0., 1.]]]], dtype=torch.float32))
        self.register_buffer('sobel_y',
            torch.tensor([[[[-1.], [0.], [1.]]]], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        residual = x

        x_main = self.dwconv(x)

        sobel_x = self.sobel_x.repeat(x.shape[1], 1, 1, 1)
        sobel_y = self.sobel_y.repeat(x.shape[1], 1, 1, 1)

        dx = F.conv2d(x, sobel_x, padding=(0, 1), groups=x.shape[1])
        dy = F.conv2d(x, sobel_y, padding=(1, 0), groups=x.shape[1])
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        x_grad = self.grad_conv(grad_mag)

        x = x_main + self.gradient_weight * x_grad

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv(x)
        x = x.permute(0, 3, 1, 2)

        x = residual + self.stochastic_depth(x)
        return x


class MicroFlowNeXt(nn.Module):
    """
    MicroFlowNeXt: Motion-Sensitive Backbone for Optical Flow Analysis.

    A lightweight ConvNeXt-style architecture with:
    - Spatial Transformer Network (STN) for geometric normalization
    - Gradient-aware blocks for capturing subtle motion patterns
    - Peak-enhanced channel attention for micro-expression detection
    - Micro-change enhanced self-attention for final classification

    Args:
        in_chans (int): Number of input channels. Default: 2 (optical flow u, v).
        num_classes (int): Number of output classes. Default: 3.
        depths (list): Number of blocks in each stage. Default: [1, 1, 1, 1].
        dims (list): Channel dimensions for each stage. Default: [48, 96, 192, 384].
        drop_path_rate (float): Stochastic depth rate. Default: 0.0.
        input_size (int): Expected input spatial size. Default: 224.
    """

    def __init__(
        self,
        in_chans: int = 2,
        num_classes: int = 3,
        depths: list = None,
        dims: list = None,
        drop_path_rate: float = 0.0,
        input_size: int = 224,
    ):
        super().__init__()

        if dims is None:
            dims = [48, 96, 192, 384]
        if depths is None:
            depths = [1, 1, 1, 1]

        if len(depths) < 2 or len(depths) > 4:
            raise ValueError(f"Number of stages must be between 2 and 4, got {len(depths)}")
        if len(dims) != len(depths):
            raise ValueError(f"dims length ({len(dims)}) must match depths length ({len(depths)})")

        self.dims = dims
        self.depths = depths
        self.num_stages = len(depths)
        self.num_classes = num_classes

        self._build_stn(in_chans, input_size)
        self._build_stages(in_chans, dims, depths, drop_path_rate)
        self._build_head(dims[-1], num_classes)
        self._init_weights()

    def _build_stn(self, in_chans: int, input_size: int):
        """Construct the Spatial Transformer Network."""
        self.localization = nn.Sequential(
            nn.Conv2d(in_chans, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        localization_output_size = 16 * (input_size // 4) * (input_size // 4)
        self.fc_loc = nn.Sequential(
            nn.Linear(localization_output_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

    def _build_stages(self, in_chans: int, dims: list, depths: list, drop_path_rate: float):
        """Construct the feature extraction stages."""
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(self.num_stages - 1):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                MicroSELayer(dims[i + 1], reduction=16),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[
                    MicroBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

    def _build_head(self, final_dim: int, num_classes: int):
        """Construct the classification head."""
        self.norm = nn.LayerNorm(final_dim, eps=1e-6)
        self.attention = MicroSelfAttention(final_dim)
        self.head = nn.Linear(final_dim, num_classes)

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def _flow_vector_correction(self, flow_data: torch.Tensor,
                                 theta: torch.Tensor) -> torch.Tensor:
        """
        Apply geometric correction to optical flow vectors after spatial transformation.

        Ensures physical consistency by transforming the flow vectors according to
        the same affine transformation applied to the spatial domain.

        Args:
            flow_data: Resampled optical flow data of shape (B, 2, H, W).
            theta: Affine transformation matrix of shape (B, 2, 3).

        Returns:
            Corrected optical flow data of shape (B, 2, H, W).
        """
        B, C, H, W = flow_data.shape

        A = theta[:, :2, :2]
        flow_reshaped = flow_data.view(B, 2, H * W)
        flow_corrected = torch.bmm(A, flow_reshaped)

        det_A = torch.det(A)
        scale_factor = torch.sqrt(torch.abs(det_A) + 1e-8)
        scale_factor = scale_factor.view(B, 1, 1)
        flow_corrected = flow_corrected / scale_factor

        return flow_corrected.view(B, 2, H, W)

    def stn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Spatial Transformer Network with flow vector correction.

        Args:
            x: Input optical flow tensor of shape (B, 2, H, W).

        Returns:
            Geometrically normalized optical flow of shape (B, 2, H, W).
        """
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_resampled = F.grid_sample(x, grid, align_corners=True)
        x_corrected = self._flow_vector_correction(x_resampled, theta)

        return x_corrected

    def forward_features_staged(self, x: torch.Tensor) -> list:
        """
        Extract hierarchical features from each stage.

        Args:
            x: Input tensor of shape (B, 2, H, W).

        Returns:
            List of feature tensors from each stage.
        """
        features_at_stages = []
        x = self.stn(x)

        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features_at_stages.append(x)

        return features_at_stages

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract final feature representation.

        Args:
            x: Input tensor of shape (B, 2, H, W).

        Returns:
            Global feature vector of shape (B, D).
        """
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward_head(self, x: torch.Tensor) -> tuple:
        """
        Compute classification logits from final stage features.

        Args:
            x: Last stage feature map of shape (B, C, H, W).

        Returns:
            Tuple of (logits, attention_weights).
        """
        x_pooled = self.norm(x.mean([-2, -1]))
        x_attended, attention_weights = self.attention(x_pooled)
        logits = self.head(x_attended)
        return logits, attention_weights

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> tuple:
        """
        Forward pass with optional loss computation.

        Args:
            x: Input optical flow tensor of shape (B, 2, H, W).
            labels: Optional ground truth labels of shape (B,).

        Returns:
            If labels provided: (predictions, logits, loss)
            Otherwise: (predictions, logits)
        """
        x = self.stn(x)
        x = self.forward_features(x)
        x, weights = self.attention(x)
        logits = self.head(x)

        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            return torch.argmax(logits, dim=1), logits, loss

        return torch.argmax(logits, dim=1), logits


def get_microflownext(
    num_classes: int = 3,
    model_size: str = "micro",
    drop_path_rate: float = 0.0,
    input_size: int = 224,
) -> MicroFlowNeXt:
    """
    Factory function to create MicroFlowNeXt models with predefined configurations.

    Args:
        num_classes: Number of output classes.
        model_size: Model size variant - "ultralight", "nano", "micro", "tiny", or "small".
        drop_path_rate: Stochastic depth rate.
        input_size: Expected input spatial size.

    Returns:
        Configured MicroFlowNeXt model.
    """
    configs = {
        "ultralight": {"depths": [1, 1, 1, 1], "dims": [24, 48, 64, 128]},
        "nano": {"depths": [1, 1, 1, 1], "dims": [32, 64, 128, 256]},
        "micro": {"depths": [1, 1, 1, 1], "dims": [48, 96, 192, 384]},
        "tiny": {"depths": [1, 1, 2, 1], "dims": [64, 128, 256, 512]},
        "small": {"depths": [1, 2, 3, 1], "dims": [96, 192, 384, 768]},
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]

    return MicroFlowNeXt(
        in_chans=2,
        num_classes=num_classes,
        depths=config["depths"],
        dims=config["dims"],
        drop_path_rate=drop_path_rate,
        input_size=input_size,
    )


def get_microflownext_custom(
    num_classes: int = 3,
    depths: list = None,
    dims: list = None,
    drop_path_rate: float = 0.0,
    input_size: int = 224,
) -> MicroFlowNeXt:
    """
    Factory function to create MicroFlowNeXt models with custom configurations.

    Args:
        num_classes: Number of output classes.
        depths: Number of blocks in each stage.
        dims: Channel dimensions for each stage.
        drop_path_rate: Stochastic depth rate.
        input_size: Expected input spatial size.

    Returns:
        Configured MicroFlowNeXt model.
    """
    if depths is None:
        depths = [1, 1, 1, 1]
    if dims is None:
        dims = [48, 96, 192, 384]

    return MicroFlowNeXt(
        in_chans=2,
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        input_size=input_size,
    )
