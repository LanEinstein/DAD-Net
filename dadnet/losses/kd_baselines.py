"""Knowledge-distillation baselines for the alignment-objective comparison.

Each loss is a drop-in replacement for GSWD at an aligned stage. Inputs are a
student feature map and a teacher feature map that the dual-stream adapter has
already matched in channel count. Four-dimensional inputs are global-average
pooled to channel vectors where a loss operates on pooled descriptors.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pool(feature: torch.Tensor) -> torch.Tensor:
    """Global-average pool a feature map to a channel vector; pass vectors through."""
    return feature.mean([-2, -1]) if feature.dim() == 4 else feature


class FitNetLoss(nn.Module):
    """Mean-squared-error hint regression on flattened features (Romero et al., 2015)."""

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = student.reshape(student.size(0), -1)
        t = teacher.reshape(teacher.size(0), -1)
        return F.mse_loss(s, t)


class L2FeatureLoss(nn.Module):
    """Plain squared L2 distance between flattened features (no projector)."""

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = student.reshape(student.size(0), -1)
        t = teacher.reshape(teacher.size(0), -1)
        return (s - t).pow(2).mean()


class MMDLoss(nn.Module):
    """Maximum mean discrepancy with a multi-kernel Gaussian estimator.

    Args:
        kernel_mul: Geometric spacing between kernel bandwidths.
        kernel_num: Number of Gaussian kernels in the bandwidth ladder.
    """

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5) -> None:
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        diff = total.unsqueeze(0) - total.unsqueeze(1)
        l2 = diff.pow(2).sum(dim=-1)
        bandwidth = l2.detach().sum() / (n * n - n + 1e-8)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        return sum(
            torch.exp(-l2 / (bandwidth * (self.kernel_mul ** i) + 1e-8))
            for i in range(self.kernel_num)
        )

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s, t = _pool(student), _pool(teacher)
        n = s.size(0)
        kernels = self._gaussian_kernel(s, t)
        xx = kernels[:n, :n].mean()
        yy = kernels[n:, n:].mean()
        xy = kernels[:n, n:].mean()
        yx = kernels[n:, :n].mean()
        return xx + yy - xy - yx


class KLFeatureLoss(nn.Module):
    """Temperature-scaled KL divergence between pooled channel distributions."""

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = F.log_softmax(_pool(student) / self.temperature, dim=-1)
        t = F.softmax(_pool(teacher) / self.temperature, dim=-1)
        return F.kl_div(s, t, reduction="batchmean") * (self.temperature ** 2)


class PKTLoss(nn.Module):
    """Probabilistic knowledge transfer on similarity distributions (Passalis & Tefas, 2018)."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = F.normalize(_pool(student), dim=-1)
        t = F.normalize(_pool(teacher), dim=-1)
        s_prob = F.softmax(torch.mm(s, s.T), dim=-1).clamp(min=self.eps)
        t_prob = F.softmax(torch.mm(t, t.T), dim=-1).clamp(min=self.eps)
        return (t_prob * (t_prob.log() - s_prob.log())).sum(dim=-1).mean()


class CRDLoss(nn.Module):
    """Contrastive representation distillation with in-batch negatives (Tian et al., 2020).

    Args:
        student_dim: Channel count of the pooled student feature.
        teacher_dim: Channel count of the pooled teacher feature.
        embed_dim: Shared contrastive embedding dimension.
        temperature: Softmax temperature for the InfoNCE objective.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        embed_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.student_proj = nn.Sequential(
            nn.Linear(student_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, embed_dim)
        )
        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = F.normalize(self.student_proj(_pool(student)), dim=-1)
        t = F.normalize(self.teacher_proj(_pool(teacher).detach()), dim=-1)
        logits = torch.mm(s, t.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


def get_kd_loss(
    method: str,
    student_dim: Optional[int] = None,
    teacher_dim: Optional[int] = None,
) -> nn.Module:
    """Build a knowledge-distillation baseline by name.

    Args:
        method: One of ``fitnet``, ``l2``, ``mmd``, ``kl``, ``pkt``, ``crd``.
        student_dim: Student channel count (required for ``crd``).
        teacher_dim: Teacher channel count (required for ``crd``).
    """
    method = method.lower()
    if method == "fitnet":
        return FitNetLoss()
    if method == "l2":
        return L2FeatureLoss()
    if method == "mmd":
        return MMDLoss()
    if method == "kl":
        return KLFeatureLoss()
    if method == "pkt":
        return PKTLoss()
    if method == "crd":
        if student_dim is None or teacher_dim is None:
            raise ValueError("crd requires student_dim and teacher_dim")
        return CRDLoss(student_dim, teacher_dim)
    raise ValueError(
        f"unknown method {method!r}; choose from fitnet, l2, mmd, kl, pkt, crd"
    )


__all__ = [
    "FitNetLoss",
    "L2FeatureLoss",
    "MMDLoss",
    "KLFeatureLoss",
    "PKTLoss",
    "CRDLoss",
    "get_kd_loss",
]
