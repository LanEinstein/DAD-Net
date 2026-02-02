# -*- coding: utf-8 -*-
"""
Gaussian Sliced-Wasserstein Distance (GSWD) Loss

This module implements the GSWD loss function for multi-level feature distribution
alignment between macro-expression and micro-expression feature spaces.

The GSWD computes the Wasserstein-2 distance under Gaussian assumptions using
Monte Carlo approximation with random projections. This approach is more effective
than L2, MMD, or KL divergence for aligning feature distributions across domains.

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def gaussian_sliced_wasserstein_distance(
    target: torch.Tensor,
    source: torch.Tensor,
    num_projections: int = 100,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Gaussian Sliced-Wasserstein Distance between two feature distributions.

    The GSWD approximates the 2-Wasserstein distance by:
    1. Randomly sampling L unit direction vectors
    2. Projecting both distributions onto these directions
    3. Computing 1D Gaussian Wasserstein distance for each projection
    4. Averaging over all projections

    For 1D Gaussian distributions N(mu1, sigma1^2) and N(mu2, sigma2^2):
        W_2^2 = (mu1 - mu2)^2 + (sigma1 - sigma2)^2

    Args:
        target: Target distribution features of shape (B, C, H, W) or (B, D).
        source: Source distribution features of shape (B, C, H, W) or (B, D).
        num_projections: Number of random projection directions (L). Default: 100.
        eps: Small constant for numerical stability. Default: 1e-8.

    Returns:
        GSWD loss tensor. Returns scalar if both inputs have same batch size,
        otherwise returns tensor of shape (B,) for DataParallel compatibility.
    """
    device = source.device

    source_flat = source.view(source.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    batch_size, feature_dim = source_flat.shape

    if feature_dim == 0:
        return torch.zeros(1, device=device, requires_grad=True)

    directions = torch.randn(num_projections, feature_dim, device=device)
    directions = F.normalize(directions, dim=1)

    proj_source = source_flat @ directions.T
    proj_target = target_flat @ directions.T

    mean_source = proj_source.mean(dim=0)
    mean_target = proj_target.mean(dim=0)
    std_source = proj_source.std(dim=0, unbiased=False).clamp(min=eps)
    std_target = proj_target.std(dim=0, unbiased=False).clamp(min=eps)

    wasserstein_sq = (mean_source - mean_target) ** 2 + (std_source - std_target) ** 2

    gswd = wasserstein_sq.mean()

    return gswd


class GaussianSlicedWassersteinLoss(nn.Module):
    """
    Gaussian Sliced-Wasserstein Distance Loss Module.

    A PyTorch module wrapper for the GSWD loss function with configurable
    number of random projections.

    Args:
        num_projections (int): Number of random projection directions. Default: 100.
        eps (float): Numerical stability constant. Default: 1e-8.
    """

    def __init__(self, num_projections: int = 100, eps: float = 1e-8):
        super(GaussianSlicedWassersteinLoss, self).__init__()
        self.num_projections = num_projections
        self.eps = eps

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GSWD loss between target and source distributions.

        Args:
            target: Target feature distribution.
            source: Source feature distribution.

        Returns:
            GSWD loss value.
        """
        return gaussian_sliced_wasserstein_distance(
            target, source,
            num_projections=self.num_projections,
            eps=self.eps
        )


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.

    Applies label smoothing to prevent overconfident predictions and
    improve generalization, which is particularly beneficial for
    micro-expression recognition with limited training data.

    Args:
        smoothing (float): Label smoothing factor. Default: 0.05.
        reduction (str): Reduction method - "mean", "sum", or "none". Default: "mean".
    """

    def __init__(self, smoothing: float = 0.05, reduction: str = "mean"):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Predicted logits of shape (B, C).
            targets: Ground truth labels of shape (B,).

        Returns:
            Loss value.
        """
        return F.cross_entropy(
            logits, targets,
            label_smoothing=self.smoothing,
            reduction=self.reduction
        )


class DADNetLoss(nn.Module):
    """
    Combined Loss Function for DAD-Net Training.

    Combines classification loss with multi-level feature alignment loss:
        L_total = L_cls + lambda * L_align

    Args:
        alignment_weight (float): Weight for alignment loss (lambda). Default: 0.5.
        num_projections (int): Number of GSWD projections. Default: 100.
        label_smoothing (float): Label smoothing factor. Default: 0.05.
    """

    def __init__(
        self,
        alignment_weight: float = 0.5,
        num_projections: int = 100,
        label_smoothing: float = 0.05,
    ):
        super(DADNetLoss, self).__init__()
        self.alignment_weight = alignment_weight
        self.cls_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.align_loss = GaussianSlicedWassersteinLoss(num_projections=num_projections)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        micro_features: Optional[list] = None,
        macro_features: Optional[list] = None,
        alignment_stages: Optional[dict] = None,
    ) -> tuple:
        """
        Compute combined DAD-Net loss.

        Args:
            logits: Predicted logits of shape (B, C).
            targets: Ground truth labels of shape (B,).
            micro_features: List of micro-expression branch features per stage.
            macro_features: List of macro-expression branch features per stage.
            alignment_stages: Dictionary indicating which stages to align.

        Returns:
            Tuple of (total_loss, classification_loss, alignment_loss).
        """
        cls_loss = self.cls_loss(logits, targets)

        align_loss = torch.tensor(0.0, device=logits.device)

        if micro_features is not None and macro_features is not None:
            if alignment_stages is None:
                alignment_stages = {i: True for i in range(len(micro_features))}

            num_aligned = 0
            for stage, should_align in alignment_stages.items():
                if should_align and stage < len(micro_features) and stage < len(macro_features):
                    stage_loss = self.align_loss(
                        micro_features[stage],
                        macro_features[stage]
                    )
                    align_loss = align_loss + stage_loss
                    num_aligned += 1

            if num_aligned > 0:
                align_loss = align_loss / num_aligned

        total_loss = cls_loss + self.alignment_weight * align_loss

        return total_loss, cls_loss, align_loss
