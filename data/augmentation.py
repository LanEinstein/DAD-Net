# -*- coding: utf-8 -*-
"""
Flow-Consistent Data Augmentation for Optical Flow

This module provides data augmentation techniques specifically designed for
optical flow data, ensuring physical consistency of flow vectors after
geometric transformations.

Key Augmentations:
    - Horizontal flip with flow vector inversion
    - Rotation with flow vector rotation
    - Scale with flow magnitude adjustment
    - Additive noise
    - Contrast adjustment

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import math
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Optional, Tuple


def horizontal_flip_optical_flow(flow: torch.Tensor) -> torch.Tensor:
    """
    Apply horizontal flip to optical flow with vector correction.

    Flips the spatial dimensions and negates the horizontal (u) component
    to maintain physical consistency.

    Args:
        flow: Optical flow tensor of shape (2, H, W) or (B, 2, H, W).

    Returns:
        Horizontally flipped optical flow.
    """
    if flow.dim() == 3:
        flipped = torch.flip(flow, dims=[2])
        flipped[0] = -flipped[0]
    else:
        flipped = torch.flip(flow, dims=[3])
        flipped[:, 0] = -flipped[:, 0]

    return flipped


def rotate_optical_flow(
    flow: torch.Tensor,
    angle: float,
) -> torch.Tensor:
    """
    Rotate optical flow with vector direction correction.

    Rotates both the spatial layout and the flow vectors to maintain
    physical consistency.

    Args:
        flow: Optical flow tensor of shape (2, H, W).
        angle: Rotation angle in degrees.

    Returns:
        Rotated optical flow.
    """
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    flow_batch = flow.unsqueeze(0)
    rotated = TF.rotate(flow_batch, angle)
    rotated = rotated.squeeze(0)

    u = rotated[0].clone()
    v = rotated[1].clone()

    rotated[0] = cos_theta * u - sin_theta * v
    rotated[1] = sin_theta * u + cos_theta * v

    return rotated


def scale_optical_flow(
    flow: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """
    Scale optical flow with magnitude adjustment.

    Scales the spatial layout and adjusts flow magnitudes to maintain
    physical consistency.

    Args:
        flow: Optical flow tensor of shape (2, H, W).
        scale_factor: Scaling factor (>1 for zoom in, <1 for zoom out).

    Returns:
        Scaled optical flow.
    """
    _, h, w = flow.shape

    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    flow_batch = flow.unsqueeze(0)
    scaled = F.interpolate(flow_batch, size=(new_h, new_w), mode='bilinear', align_corners=True)
    scaled = F.interpolate(scaled, size=(h, w), mode='bilinear', align_corners=True)
    scaled = scaled.squeeze(0)

    scaled = scaled / scale_factor

    return scaled


def add_gaussian_noise(
    flow: torch.Tensor,
    std: float = 0.05,
) -> torch.Tensor:
    """
    Add Gaussian noise to optical flow.

    Args:
        flow: Optical flow tensor.
        std: Standard deviation of Gaussian noise. Default: 0.05.

    Returns:
        Noisy optical flow.
    """
    noise = torch.randn_like(flow) * std
    return flow + noise


def adjust_contrast(
    flow: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """
    Adjust contrast of optical flow magnitude.

    Args:
        flow: Optical flow tensor.
        factor: Contrast factor (>1 increases contrast, <1 decreases).

    Returns:
        Contrast-adjusted optical flow.
    """
    mean = flow.mean()
    return (flow - mean) * factor + mean


class OpticalFlowAugmentation:
    """
    Flow-Consistent Data Augmentation Module.

    Applies random augmentations to optical flow data while maintaining
    physical consistency of flow vectors.

    Args:
        horizontal_flip_prob: Probability of horizontal flip. Default: 0.5.
        rotation_range: Maximum rotation angle in degrees. Default: 10.
        scale_range: Scale factor range as (min, max). Default: (0.9, 1.1).
        noise_std: Standard deviation for Gaussian noise. Default: 0.02.
        contrast_range: Contrast factor range as (min, max). Default: (0.9, 1.1).
        enable_flip: Enable horizontal flip augmentation. Default: True.
        enable_rotation: Enable rotation augmentation. Default: True.
        enable_scale: Enable scale augmentation. Default: True.
        enable_noise: Enable noise augmentation. Default: False.
        enable_contrast: Enable contrast augmentation. Default: False.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        rotation_range: float = 10.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.02,
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        enable_flip: bool = True,
        enable_rotation: bool = True,
        enable_scale: bool = True,
        enable_noise: bool = False,
        enable_contrast: bool = False,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.contrast_range = contrast_range
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.enable_scale = enable_scale
        self.enable_noise = enable_noise
        self.enable_contrast = enable_contrast

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to optical flow.

        Args:
            flow: Optical flow tensor of shape (2, H, W).

        Returns:
            Augmented optical flow tensor.
        """
        if self.enable_flip and random.random() < self.horizontal_flip_prob:
            flow = horizontal_flip_optical_flow(flow)

        if self.enable_rotation and random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            flow = rotate_optical_flow(flow, angle)

        if self.enable_scale and random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            flow = scale_optical_flow(flow, scale)

        if self.enable_noise and random.random() < 0.5:
            flow = add_gaussian_noise(flow, self.noise_std)

        if self.enable_contrast and random.random() < 0.5:
            factor = random.uniform(*self.contrast_range)
            flow = adjust_contrast(flow, factor)

        return flow

    def __repr__(self) -> str:
        return (
            f"OpticalFlowAugmentation("
            f"flip={self.enable_flip}, "
            f"rotation={self.enable_rotation}({self.rotation_range}°), "
            f"scale={self.enable_scale}{self.scale_range}, "
            f"noise={self.enable_noise}, "
            f"contrast={self.enable_contrast})"
        )
