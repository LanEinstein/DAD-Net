"""Gaussian Sliced-Wasserstein Distance (GSWD) for feature-distribution alignment.

GSWD aligns teacher and student feature distributions without pointwise
correspondence. A stage feature map of shape ``(B, C, H, W)`` is read as
``B * H * W`` token vectors in ``R^C`` (paper Eq. 8). Each Gaussian is projected
onto ``L`` random unit directions of ``R^C``; the squared Wasserstein-2 distance
between two one-dimensional Gaussians has the closed form
``(m1 - m2)^2 + (s1 - s2)^2`` (paper Eq. 10), and the loss averages it over the
``L`` projections (paper Eq. 11).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
    """Reshape a feature map into token vectors over the channel dimension.

    ``(B, C, H, W)`` becomes ``(B * H * W, C)``; a ``(B, C)`` vector is returned
    unchanged. Each token is one point in the ``C``-dimensional feature space.
    """
    if feature.dim() == 4:
        b, c, h, w = feature.shape
        return feature.permute(0, 2, 3, 1).reshape(b * h * w, c)
    if feature.dim() == 2:
        return feature
    raise ValueError(f"expected a 2D or 4D tensor, got shape {tuple(feature.shape)}")


def gaussian_sliced_wasserstein_distance(
    student: torch.Tensor,
    teacher: torch.Tensor,
    num_projections: int = 100,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the GSWD between two feature distributions.

    Args:
        student: Student feature map ``(B, C, H, W)`` or vector ``(B, C)``.
        teacher: Teacher feature map ``(B, C, H, W)`` or vector ``(B, C)``;
            its channel count must match the student.
        num_projections: Number of random projection directions ``L``.
        eps: Lower bound on the projected standard deviations.

    Returns:
        Scalar GSWD loss.
    """
    student_tokens = _to_tokens(student)
    teacher_tokens = _to_tokens(teacher)

    channels = student_tokens.shape[1]
    if channels != teacher_tokens.shape[1]:
        raise ValueError(
            f"channel mismatch: student {channels} vs teacher {teacher_tokens.shape[1]}"
        )
    if channels == 0:
        return torch.zeros(1, device=student.device, requires_grad=True)

    directions = torch.randn(num_projections, channels, device=student.device)
    directions = F.normalize(directions, dim=1)

    student_proj = student_tokens @ directions.T
    teacher_proj = teacher_tokens @ directions.T

    mean_diff = student_proj.mean(dim=0) - teacher_proj.mean(dim=0)
    std_diff = (
        student_proj.std(dim=0, unbiased=False).clamp(min=eps)
        - teacher_proj.std(dim=0, unbiased=False).clamp(min=eps)
    )
    return (mean_diff.pow(2) + std_diff.pow(2)).mean()


class GaussianSlicedWassersteinLoss(nn.Module):
    """Module wrapper around :func:`gaussian_sliced_wasserstein_distance`."""

    def __init__(self, num_projections: int = 100, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_projections = num_projections
        self.eps = eps

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return gaussian_sliced_wasserstein_distance(
            student, teacher, num_projections=self.num_projections, eps=self.eps
        )


__all__ = ["gaussian_sliced_wasserstein_distance", "GaussianSlicedWassersteinLoss"]
