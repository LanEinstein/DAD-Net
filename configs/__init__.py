# -*- coding: utf-8 -*-
"""
DAD-Net Configuration Management

This module provides default configuration values and configuration utilities.
"""

from .default import (
    DEFAULT_CONFIG,
    get_dataset_config,
    get_model_config,
    get_training_config,
)

__all__ = [
    'DEFAULT_CONFIG',
    'get_dataset_config',
    'get_model_config',
    'get_training_config',
]
