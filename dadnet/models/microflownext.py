"""MicroFlowNeXt: a motion-sensitive backbone for optical-flow micro-expression
recognition.

The backbone separates two kinds of motion carried by an optical-flow field:
nuisance global motion, which is suppressed by a flow-consistent spatial
transformer, and weak local motion, which is preserved and amplified by
gradient-aware blocks and peak-enhanced channel attention. A restrained
self-attention head refines the pooled feature before classification.

Components:
    FlowConsistentSTN  - affine normalization with inverse-Jacobian flow reorientation
    GradientAwareBlock - depthwise convolution fused with a gradient-magnitude branch
    PeakEnhancedChannelAttention - dual avg/max pooled channel reweighting
    MicroSelfAttention - shared query/key token attention with an enhancement branch
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth


class PeakEnhancedChannelAttention(nn.Module):
    """Channel attention that combines global average and global max pooling.

    Max pooling preserves sparse but discriminative responses that average
    pooling would dilute (paper Eq. 6-7).

    Args:
        channels: Number of input channels.
        reduction: Bottleneck reduction ratio for the shared MLP.
    """

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        descriptor = torch.cat(
            [self.avg_pool(x).view(b, c), self.max_pool(x).view(b, c)], dim=1
        )
        weight = self.mlp(descriptor).view(b, c, 1, 1)
        return x * weight.expand_as(x)


class MicroSelfAttention(nn.Module):
    """Token self-attention with a shared query/key projection (paper Eq. 9).

    Sharing the query and key projection removes one value projection and
    halves the query-key parameters, limiting token mixing that would dilute
    sparse motion cues. An additive enhancement branch retains fine variation.

    Args:
        dim: Token embedding dimension.
        enhancement_weight: Weight on the enhancement branch (beta).
    """

    def __init__(self, dim: int, enhancement_weight: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.enhancement_weight = enhancement_weight
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, dim)
        self.enhance = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(x)
        projected = self.projection(x)
        scores = torch.matmul(projected, projected.transpose(-2, -1)) / math.sqrt(self.dim)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, x) + self.enhancement_weight * self.enhance(x)
        return output, weights


class _IdentityAttention(nn.Module):
    """Head replacement that bypasses self-attention for the ablation study."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return x, None


class LayerNorm2d(nn.Module):
    """Layer normalization over the channel dimension of an NCHW tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GradientAwareBlock(nn.Module):
    """ConvNeXt-style block with an additive gradient-magnitude branch (paper Eq. 4-5).

    A main depthwise branch captures context while a gradient branch enhances
    weak local variation through channel-wise central differences.

    Args:
        dim: Number of channels.
        drop_path: Stochastic depth rate.
        gradient_weight: Weight on the gradient branch (alpha).
    """

    def __init__(self, dim: int, drop_path: float = 0.0, gradient_weight: float = 0.1) -> None:
        super().__init__()
        self.gradient_weight = gradient_weight
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.grad_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.stochastic_depth = StochasticDepth(drop_path, "row")
        self.register_buffer(
            "central_x", torch.tensor([[[[-1.0, 0.0, 1.0]]]], dtype=torch.float32)
        )
        self.register_buffer(
            "central_y", torch.tensor([[[[-1.0], [0.0], [1.0]]]], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        main = self.dwconv(x)

        kernel_x = self.central_x.repeat(x.shape[1], 1, 1, 1)
        kernel_y = self.central_y.repeat(x.shape[1], 1, 1, 1)
        grad_x = F.conv2d(x, kernel_x, padding=(0, 1), groups=x.shape[1])
        grad_y = F.conv2d(x, kernel_y, padding=(1, 0), groups=x.shape[1])
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
        grad = self.grad_conv(grad_mag)

        x = main + self.gradient_weight * grad
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv(self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return residual + self.stochastic_depth(x)


class MicroFlowNeXt(nn.Module):
    """Optical-flow backbone for micro-expression recognition.

    Args:
        in_chans: Number of input channels (2 for optical-flow u, v).
        num_classes: Number of output classes.
        depths: Number of blocks per stage.
        dims: Channel width per stage.
        drop_path_rate: Maximum stochastic depth rate.
        input_size: Expected square input spatial size.
        skip_stn: Disable the spatial transformer (ablation).
        skip_channel_attention: Disable peak-enhanced channel attention (ablation).
        skip_head_attention: Replace the self-attention head with identity (ablation).
    """

    det_low: float = 0.5
    det_high: float = 2.0
    inv_solve_eps: float = 1e-6

    def __init__(
        self,
        in_chans: int = 2,
        num_classes: int = 3,
        depths: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        drop_path_rate: float = 0.0,
        input_size: int = 224,
        skip_stn: bool = False,
        skip_channel_attention: bool = False,
        skip_head_attention: bool = False,
    ) -> None:
        super().__init__()
        dims = [48, 96, 192, 384] if dims is None else list(dims)
        depths = [1, 1, 1, 1] if depths is None else list(depths)
        if not 2 <= len(depths) <= 4:
            raise ValueError(f"number of stages must be in [2, 4], got {len(depths)}")
        if len(dims) != len(depths):
            raise ValueError(f"len(dims)={len(dims)} must equal len(depths)={len(depths)}")

        self.dims = dims
        self.depths = depths
        self.num_stages = len(depths)
        self.num_classes = num_classes
        self.skip_stn = skip_stn
        self.skip_channel_attention = skip_channel_attention
        self.skip_head_attention = skip_head_attention
        self._last_theta: Optional[torch.Tensor] = None

        self._build_stn(in_chans, input_size)
        self._build_stages(in_chans, dims, depths, drop_path_rate)
        self._build_head(dims[-1], num_classes)
        self._init_weights()

    def _build_stn(self, in_chans: int, input_size: int) -> None:
        self.localization = nn.Sequential(
            nn.Conv2d(in_chans, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        loc_dim = 16 * (input_size // 4) * (input_size // 4)
        self.fc_loc = nn.Sequential(
            nn.Linear(loc_dim, 32), nn.ReLU(True), nn.Linear(32, 6)
        )

    def _build_stages(
        self, in_chans: int, dims: List[int], depths: List[int], drop_path_rate: float
    ) -> None:
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stages - 1):
            attention = (
                nn.Identity()
                if self.skip_channel_attention
                else PeakEnhancedChannelAttention(dims[i + 1], reduction=16)
            )
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2), attention
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = [r.item() for r in torch.linspace(0, drop_path_rate, sum(depths))]
        cursor = 0
        for i in range(self.num_stages):
            blocks = [
                GradientAwareBlock(dim=dims[i], drop_path=dp_rates[cursor + j])
                for j in range(depths[i])
            ]
            self.stages.append(nn.Sequential(*blocks))
            cursor += depths[i]

    def _build_head(self, final_dim: int, num_classes: int) -> None:
        self.norm = nn.LayerNorm(final_dim, eps=1e-6)
        self.attention = (
            _IdentityAttention() if self.skip_head_attention else MicroSelfAttention(final_dim)
        )
        self.head = nn.Linear(final_dim, num_classes)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def _reorient_flow(self, flow: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Reorient sampled flow vectors by the inverse affine linear part.

        ``affine_grid`` defines a target-to-source map, so a displacement
        sampled at the source must be mapped back by ``A^{-1}`` to stay
        consistent in target coordinates (paper Eq. 3).
        """
        b, _, h, w = flow.shape
        linear = theta[:, :2, :2]
        eye = torch.eye(2, dtype=linear.dtype, device=linear.device).expand_as(linear)
        linear_reg = linear + self.inv_solve_eps * eye
        flow_flat = flow.reshape(b, 2, h * w)
        corrected = torch.linalg.solve(linear_reg, flow_flat)
        return corrected.reshape(b, 2, h, w)

    def spatial_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the flow-consistent spatial transformer to the input field."""
        if self.skip_stn:
            self._last_theta = None
            return x
        features = self.localization(x).view(x.size(0), -1)
        theta = self.fc_loc(features).view(-1, 2, 3)
        self._last_theta = theta
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        resampled = F.grid_sample(x, grid, align_corners=True)
        return self._reorient_flow(resampled, theta)

    def forward_features_staged(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return the feature map produced by each backbone stage."""
        x = self.spatial_transform(x)
        features = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple:
        features = self.forward_features_staged(x)
        pooled = self.norm(features[-1].mean([-2, -1]))
        attended, _ = self.attention(pooled)
        logits = self.head(attended)
        predictions = torch.argmax(logits, dim=1)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            return predictions, logits, loss
        return predictions, logits


MODEL_SIZE_CONFIGS = {
    "ultralight": {"depths": [1, 1, 1, 1], "dims": [24, 48, 64, 128]},
    "nano": {"depths": [1, 1, 1, 1], "dims": [32, 64, 128, 256]},
    "micro": {"depths": [1, 1, 1, 1], "dims": [48, 96, 192, 384]},
    "tiny": {"depths": [1, 1, 2, 1], "dims": [64, 128, 256, 512]},
    "small": {"depths": [1, 2, 3, 1], "dims": [96, 192, 384, 768]},
    "base": {"depths": [1, 1, 1, 1], "dims": [128, 256, 512, 1024]},
}


def get_microflownext(
    num_classes: int = 3,
    model_size: str = "micro",
    drop_path_rate: float = 0.0,
    input_size: int = 224,
    **ablation_flags: bool,
) -> MicroFlowNeXt:
    """Build a MicroFlowNeXt from a named width configuration.

    Args:
        num_classes: Number of output classes.
        model_size: One of ``MODEL_SIZE_CONFIGS``.
        drop_path_rate: Maximum stochastic depth rate.
        input_size: Expected square input spatial size.
        ablation_flags: Optional ``skip_stn`` / ``skip_channel_attention`` /
            ``skip_head_attention`` switches forwarded to the model.
    """
    if model_size not in MODEL_SIZE_CONFIGS:
        raise ValueError(
            f"unknown model_size {model_size!r}; choose from {list(MODEL_SIZE_CONFIGS)}"
        )
    config = MODEL_SIZE_CONFIGS[model_size]
    return MicroFlowNeXt(
        in_chans=2,
        num_classes=num_classes,
        depths=config["depths"],
        dims=config["dims"],
        drop_path_rate=drop_path_rate,
        input_size=input_size,
        **ablation_flags,
    )


__all__ = [
    "MicroFlowNeXt",
    "GradientAwareBlock",
    "PeakEnhancedChannelAttention",
    "MicroSelfAttention",
    "LayerNorm2d",
    "MODEL_SIZE_CONFIGS",
    "get_microflownext",
]
