# -*- coding: utf-8 -*-
"""
DAD-Net: Distribution-Aligned Dual-Stream Framework

This module implements the DAD-Net architecture for cross-domain micro-expression
recognition. It combines a frozen macro-expression teacher branch with a trainable
micro-expression student branch through multi-level feature distribution alignment.

Key Components:
    - Dual-stream architecture (frozen teacher + trainable student)
    - Feature adapters for cross-domain dimension matching
    - Multi-level GSWD (Gaussian Sliced-Wasserstein Distance) alignment
    - Configurable stage-wise alignment

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .microflownext import MicroFlowNeXt


class FeatureAdapter(nn.Module):
    """
    Feature Adapter for Cross-Domain Dimension Matching.

    Transforms feature maps from the macro-expression branch to match
    the dimensions of the micro-expression branch for distribution alignment.

    Args:
        in_channels (int): Number of input channels (macro-expression features).
        out_channels (int): Number of output channels (micro-expression features).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(FeatureAdapter, self).__init__()
        self.adapter_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape (B, C_in, H, W).

        Returns:
            Adapted feature map of shape (B, C_out, H, W).
        """
        x = self.adapter_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


def gaussian_sliced_wasserstein_distance(
    target: torch.Tensor,
    source: torch.Tensor,
    num_projections: int = 100,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Gaussian Sliced-Wasserstein Distance (GSWD) between two distributions.

    Uses Monte Carlo approximation with random projections to estimate the
    Wasserstein-2 distance under Gaussian assumptions.

    Args:
        target: Target distribution features of shape (B, ...).
        source: Source distribution features of shape (B, ...).
        num_projections: Number of random projection directions (L). Default: 100.
        eps: Small constant for numerical stability. Default: 1e-8.

    Returns:
        GSWD loss tensor of shape (B,) for DataParallel compatibility.
    """
    device = source.device
    source_flat = source.view(source.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    batch_size, feature_dim = source_flat.shape

    if feature_dim == 0:
        return torch.zeros(batch_size, device=device, requires_grad=True)

    directions = torch.randn(num_projections, feature_dim, device=device)
    directions = F.normalize(directions, dim=1)

    proj_source = source_flat @ directions.T
    proj_target = target_flat @ directions.T

    mean_source = proj_source.mean(dim=0)
    mean_target = proj_target.mean(dim=0)
    std_source = proj_source.std(dim=0, unbiased=False).clamp(min=eps)
    std_target = proj_target.std(dim=0, unbiased=False).clamp(min=eps)

    wasserstein_sq = (mean_source - mean_target) ** 2 + (std_source - std_target) ** 2
    loss_scalar = wasserstein_sq.mean()
    loss_tensor = loss_scalar.expand(batch_size)

    return loss_tensor


class DADNet(nn.Module):
    """
    DAD-Net: Distribution-Aligned Dual-Stream Network.

    A dual-branch framework that combines cross-domain knowledge transfer
    from macro-expressions with motion-aware learning for micro-expressions.

    The macro-expression teacher branch provides high-level semantic priors,
    while the micro-expression student branch is trained with multi-level
    feature distribution alignment using GSWD.

    Args:
        num_classes (int): Number of output classes.
        micro_config (tuple): Configuration for micro-expression branch (depths, dims).
        macro_config (tuple): Configuration for macro-expression branch (depths, dims).
        alignment_stages (dict): Stage-wise alignment configuration {stage_idx: bool}.
        macro_weights_path (str): Path to pre-trained macro-expression weights.
        alignment_weight (float): Weight for alignment loss (lambda). Default: 0.5.
        num_projections (int): Number of GSWD random projections. Default: 100.
        in_chans (int): Number of input channels. Default: 2.
        input_size (int): Expected input spatial size. Default: 224.
    """

    def __init__(
        self,
        num_classes: int,
        micro_config: Tuple[List[int], List[int]],
        macro_config: Tuple[List[int], List[int]],
        alignment_stages: Dict[int, bool],
        macro_weights_path: Optional[str] = None,
        alignment_weight: float = 0.5,
        num_projections: int = 100,
        in_chans: int = 2,
        input_size: int = 224,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.alignment_stages = alignment_stages
        self.alignment_weight = alignment_weight
        self.num_projections = num_projections
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        micro_depths, micro_dims = micro_config
        macro_depths, macro_dims = macro_config

        self.micro_dims = micro_dims
        self.macro_dims = macro_dims
        self.micro_num_stages = len(micro_depths)
        self.macro_num_stages = len(macro_depths)

        self.micro_branch = MicroFlowNeXt(
            in_chans=in_chans,
            num_classes=num_classes,
            depths=micro_depths,
            dims=micro_dims,
            drop_path_rate=0.0,
            input_size=input_size,
        )

        self.macro_branch = MicroFlowNeXt(
            in_chans=in_chans,
            num_classes=num_classes,
            depths=macro_depths,
            dims=macro_dims,
            drop_path_rate=0.0,
            input_size=input_size,
        )

        self._load_and_freeze_macro_branch(macro_weights_path)
        self._build_adapters()

    def _load_and_freeze_macro_branch(self, weights_path: Optional[str]):
        """Load pre-trained weights and freeze the macro-expression branch."""
        if weights_path is not None:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)

                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

                macro_state = self.macro_branch.state_dict()
                filtered_state = {}

                for k, v in state_dict.items():
                    clean_key = k
                    for prefix in ['module.', 'model.', 'backbone.']:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]

                    if clean_key in macro_state:
                        if not (clean_key.startswith("head.") or clean_key.startswith("attention.")):
                            if v.shape == macro_state[clean_key].shape:
                                filtered_state[clean_key] = v

                self.macro_branch.load_state_dict(filtered_state, strict=False)

            except Exception as e:
                print(f"Warning: Failed to load macro branch weights: {e}")

        for param in self.macro_branch.parameters():
            param.requires_grad = False
        self.macro_branch.eval()

    def _build_adapters(self):
        """Construct feature adapters for dimension-mismatched alignment stages."""
        self.adapters = nn.ModuleList()
        self.adapter_map = {}
        adapter_idx = 0

        max_stages = max(self.micro_num_stages, self.macro_num_stages)

        for stage in range(max_stages):
            if self.alignment_stages.get(stage, False):
                if stage < self.macro_num_stages and stage < self.micro_num_stages:
                    macro_dim = self.macro_dims[stage]
                    micro_dim = self.micro_dims[stage]

                    if macro_dim != micro_dim:
                        adapter = FeatureAdapter(macro_dim, micro_dim)
                        self.adapters.append(adapter)
                        self.adapter_map[stage] = adapter_idx
                        adapter_idx += 1

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        Forward pass with optional loss computation.

        Args:
            x: Input optical flow tensor of shape (B, 2, H, W).
            labels: Optional ground truth labels of shape (B,).

        Returns:
            If labels provided:
                (predictions, logits, attention_weights, total_loss, alignment_loss)
            Otherwise:
                (predictions, logits, attention_weights)
        """
        batch_size = x.size(0)
        total_alignment_loss = torch.zeros(batch_size, device=x.device)

        with torch.no_grad():
            macro_features = self.macro_branch.forward_features_staged(x)

        x_micro = self.micro_branch.stn(x)

        feat_micro = x_micro
        for stage in range(self.micro_num_stages):
            feat_micro = self.micro_branch.downsample_layers[stage](feat_micro)
            feat_micro = self.micro_branch.stages[stage](feat_micro)

            if self.alignment_stages.get(stage, False) and stage < self.macro_num_stages:
                feat_macro = macro_features[stage]

                if stage in self.adapter_map:
                    feat_macro = self.adapters[self.adapter_map[stage]](feat_macro)

                align_loss = gaussian_sliced_wasserstein_distance(
                    feat_micro, feat_macro,
                    num_projections=self.num_projections
                )
                total_alignment_loss = total_alignment_loss + align_loss

        feat_pooled = self.micro_branch.norm(feat_micro.mean([-2, -1]))
        feat_attended, attention_weights = self.micro_branch.attention(feat_pooled)
        logits = self.micro_branch.head(feat_attended)
        predictions = torch.argmax(logits, dim=1)

        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            alignment_loss_mean = total_alignment_loss.mean()
            total_loss = cls_loss + self.alignment_weight * alignment_loss_mean
            return predictions, logits, attention_weights, total_loss, alignment_loss_mean

        return predictions, logits, attention_weights


def get_dad_net(
    num_classes: int = 3,
    micro_model_size: str = "micro",
    macro_model_size: str = "nano",
    alignment_stages: Optional[Dict[int, bool]] = None,
    macro_weights_path: Optional[str] = None,
    alignment_weight: float = 0.5,
    num_projections: int = 100,
    input_size: int = 224,
) -> DADNet:
    """
    Factory function to create DAD-Net models with predefined configurations.

    Args:
        num_classes: Number of output classes.
        micro_model_size: Size variant for micro-expression branch.
        macro_model_size: Size variant for macro-expression branch.
        alignment_stages: Stage-wise alignment configuration.
        macro_weights_path: Path to pre-trained macro-expression weights.
        alignment_weight: Weight for alignment loss (lambda).
        num_projections: Number of GSWD random projections.
        input_size: Expected input spatial size.

    Returns:
        Configured DADNet model.
    """
    configs = {
        "nano": {"depths": [1, 1, 1, 1], "dims": [32, 64, 128, 256]},
        "micro": {"depths": [1, 1, 1, 1], "dims": [48, 96, 192, 384]},
        "tiny": {"depths": [1, 1, 2, 1], "dims": [64, 128, 256, 512]},
        "small": {"depths": [1, 2, 3, 1], "dims": [96, 192, 384, 768]},
        "ultralight": {"depths": [1, 1, 1, 1], "dims": [24, 48, 64, 128]},
    }

    if micro_model_size not in configs:
        raise ValueError(f"Unknown micro_model_size: {micro_model_size}")
    if macro_model_size not in configs:
        raise ValueError(f"Unknown macro_model_size: {macro_model_size}")

    micro_cfg = configs[micro_model_size]
    macro_cfg = configs[macro_model_size]

    micro_config = (micro_cfg["depths"], micro_cfg["dims"])
    macro_config = (macro_cfg["depths"], macro_cfg["dims"])

    if alignment_stages is None:
        alignment_stages = {0: False, 1: False, 2: True, 3: True}

    return DADNet(
        num_classes=num_classes,
        micro_config=micro_config,
        macro_config=macro_config,
        alignment_stages=alignment_stages,
        macro_weights_path=macro_weights_path,
        alignment_weight=alignment_weight,
        num_projections=num_projections,
        in_chans=2,
        input_size=input_size,
    )
