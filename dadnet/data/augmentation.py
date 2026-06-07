"""Flow-consistent augmentation for two-channel optical-flow fields.

Geometric transforms reorient the flow vectors so the augmented field stays
physically consistent. Three modes match the paper:

    none     - identity
    hflip    - horizontal flip with probability 0.5; the horizontal component
               is negated when the spatial axis is mirrored
    standard - with probability 0.5, apply light rotation (uniform in
               [-5, 5] degrees) and/or isotropic scaling (uniform in
               [0.95, 1.05]), each gated by an independent coin flip
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF


def horizontal_flip_flow(flow: torch.Tensor) -> torch.Tensor:
    """Flip a ``(2, H, W)`` field horizontally and negate its u component."""
    flipped = torch.flip(flow, dims=(-1,)).clone()
    flipped[0] = -flipped[0]
    return flipped


def rotate_flow(flow: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate a ``(2, H, W)`` field and rotate its flow vectors by the same angle."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rotated = TF.rotate(flow.unsqueeze(0), angle_deg).squeeze(0)
    u, v = rotated[0].clone(), rotated[1].clone()
    out = rotated.clone()
    out[0] = cos_a * u - sin_a * v
    out[1] = sin_a * u + cos_a * v
    return out


def scale_flow(flow: torch.Tensor, scale: float) -> torch.Tensor:
    """Resize a ``(2, H, W)`` field by ``scale`` and inversely rescale magnitude."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    h, w = flow.shape[1:]
    resized = TF.resize(flow.unsqueeze(0), [max(1, int(h * scale)), max(1, int(w * scale))], antialias=True)
    resized = TF.resize(resized, [h, w], antialias=True).squeeze(0)
    return resized * (1.0 / scale)


class FlowAugmentation:
    """Callable flow-consistent augmentation transform.

    Args:
        mode: One of ``none``, ``hflip``, ``standard``.
        apply_prob: Probability of entering the geometric branch in ``standard``.
        rotate_deg: Maximum absolute rotation in degrees for ``standard``.
        scale_range: Lower and upper isotropic scale bounds for ``standard``.
        hflip_prob: Flip probability for ``hflip``.
    """

    def __init__(
        self,
        mode: str = "standard",
        apply_prob: float = 0.5,
        rotate_deg: float = 5.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        hflip_prob: float = 0.5,
    ) -> None:
        if mode not in ("none", "hflip", "standard"):
            raise ValueError(f"unknown mode {mode!r}; choose none, hflip, or standard")
        self.mode = mode
        self.apply_prob = apply_prob
        self.rotate_deg = rotate_deg
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.dim() != 3 or flow.shape[0] != 2:
            raise ValueError(f"expected a (2, H, W) field, got {tuple(flow.shape)}")
        if self.mode == "none":
            return flow
        if self.mode == "hflip":
            if random.random() < self.hflip_prob:
                return horizontal_flip_flow(flow)
            return flow
        if random.random() > self.apply_prob:
            return flow
        if random.random() > 0.5:
            flow = rotate_flow(flow, random.uniform(-self.rotate_deg, self.rotate_deg))
        if random.random() > 0.5:
            flow = scale_flow(flow, random.uniform(*self.scale_range))
        return flow


def build_augmentation(mode: str) -> "FlowAugmentation | None":
    """Return a :class:`FlowAugmentation` for ``mode``, or ``None`` for ``none``."""
    return None if mode == "none" else FlowAugmentation(mode=mode)


__all__ = [
    "FlowAugmentation",
    "build_augmentation",
    "horizontal_flip_flow",
    "rotate_flow",
    "scale_flow",
]
