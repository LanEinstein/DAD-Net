# -*- coding: utf-8 -*-
"""
DAD-Net Data Loading and Preprocessing

This module provides data loading utilities, dataset classes, and preprocessing
functions for optical flow-based micro-expression recognition.
"""

from .dataset import (
    MicroExpressionDataset,
    create_loso_splits,
    create_train_test_splits,
)

from .preprocessing import (
    compute_optical_flow,
    align_and_crop_face,
    normalize_optical_flow,
    process_video_clip,
)

from .augmentation import (
    OpticalFlowAugmentation,
    rotate_optical_flow,
    scale_optical_flow,
    horizontal_flip_optical_flow,
)

__all__ = [
    'MicroExpressionDataset',
    'create_loso_splits',
    'create_train_test_splits',
    'compute_optical_flow',
    'align_and_crop_face',
    'normalize_optical_flow',
    'process_video_clip',
    'OpticalFlowAugmentation',
    'rotate_optical_flow',
    'scale_optical_flow',
    'horizontal_flip_optical_flow',
]
