# -*- coding: utf-8 -*-
"""
DAD-Net Loss Functions

This module provides loss functions for training the DAD-Net framework,
including the Gaussian Sliced-Wasserstein Distance for feature alignment.
"""

from .gswd import (
    GaussianSlicedWassersteinLoss,
    gaussian_sliced_wasserstein_distance,
    LabelSmoothingCrossEntropy,
    DADNetLoss,
)

__all__ = [
    'GaussianSlicedWassersteinLoss',
    'gaussian_sliced_wasserstein_distance',
    'LabelSmoothingCrossEntropy',
    'DADNetLoss',
]
