"""Inference-time corruptions for out-of-distribution robustness evaluation.

The student consumes a two-channel optical-flow field, so image-style
corruptions are mapped to flow analogs (paper Sec. IV-G):

    gaussian_noise - additive noise with standard deviation a fraction of the
                     per-sample flow standard deviation
    blur           - per-channel spatial Gaussian blur (motion smoothing)
    intensity      - multiplicative motion attenuation (weaker captured motion)
    downsample     - bilinear down-then-up sampling (lossy-compression analog)

Severity index 0 is the clean identity, so every retention curve shares a
baseline point. Each corruption is deterministic given ``(clip_index, level)``.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

SEVERITY: Dict[str, List[float]] = {
    "gaussian_noise": [0.1, 0.2, 0.3, 0.4, 0.5],
    "blur": [0.5, 1.0, 1.5, 2.0, 2.5],
    "intensity": [0.8, 0.6, 0.4, 0.2, 0.1],
    "downsample": [0.75, 0.5, 0.375, 0.25, 0.125],
}
PERTURBATIONS = tuple(SEVERITY)


def _odd_kernel(sigma: float) -> int:
    """Return an odd Gaussian kernel size covering three standard deviations."""
    size = int(2 * round(3 * sigma) + 1)
    return max(3, size | 1)


def apply(flow: torch.Tensor, kind: str, level: float, clip_index: int) -> torch.Tensor:
    """Return a perturbed copy of a ``(2, H, W)`` flow field.

    Args:
        flow: Clean optical-flow field; never modified in place.
        kind: One of :data:`PERTURBATIONS`.
        level: Severity value from :data:`SEVERITY`.
        clip_index: Index seeding the deterministic noise draw.
    """
    x = flow.clone()
    if kind == "gaussian_noise":
        std = float(x.std()) or 1.0
        generator = torch.Generator().manual_seed((clip_index * 131 + int(level * 1000)) & 0x7FFFFFFF)
        return x + torch.randn(x.shape, generator=generator) * (level * std)
    if kind == "blur":
        size = _odd_kernel(level)
        return gaussian_blur(x, kernel_size=[size, size], sigma=[level, level])
    if kind == "intensity":
        return x * level
    if kind == "downsample":
        _, h, w = x.shape
        sh, sw = max(1, round(h * level)), max(1, round(w * level))
        down = F.interpolate(x.unsqueeze(0), size=(sh, sw), mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        return up.squeeze(0)
    raise ValueError(f"unknown perturbation {kind!r}; choose from {PERTURBATIONS}")


__all__ = ["SEVERITY", "PERTURBATIONS", "apply"]
