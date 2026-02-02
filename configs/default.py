# -*- coding: utf-8 -*-
"""
Default Configuration for DAD-Net

This module provides default configuration values for model architecture,
training hyperparameters, and dataset settings.

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
    # Model Configuration
    'model': {
        'micro_model_size': 'micro',
        'macro_model_size': 'ultralight',
        'num_classes': 3,
        'input_size': 224,
        'in_channels': 2,
        'drop_path_rate': 0.0,
    },

    # Dual-Stream Configuration
    'dual_stream': {
        'alignment_stages': {0: False, 1: False, 2: True, 3: True},
        'alignment_weight': 0.5,
        'num_projections': 100,
        'freeze_macro_branch': True,
    },

    # Training Configuration
    'training': {
        'batch_size': 8,
        'epochs': 205,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.05,
        'warmup_epochs': 5,
        'min_lr_ratio': 0.01,
    },

    # Optimizer Configuration
    'optimizer': {
        'name': 'adabelief',
        'betas': (0.9, 0.999),
        'eps': 1e-16,
        'weight_decouple': True,
        'rectify': False,
    },

    # Early Stopping Configuration
    'early_stopping': {
        'patience': 100,
        'min_delta': 0.0,
        'mode': 'max',
        'restore_best_weights': True,
    },

    # Data Augmentation Configuration
    'augmentation': {
        'enable': True,
        'horizontal_flip_prob': 0.5,
        'rotation_range': 10.0,
        'scale_range': (0.9, 1.1),
        'enable_noise': False,
        'enable_contrast': False,
    },

    # Dataset Configuration
    'dataset': {
        'emotion_map': {'negative': 0, 'positive': 1, 'surprise': 2},
        'normalize': True,
        'target_size': (224, 224),
    },

    # Logging Configuration
    'logging': {
        'log_interval': 10,
        'save_checkpoints': True,
        'checkpoint_interval': 10,
    },
}


DATASET_CONFIGS = {
    '4dme': {
        'num_classes': 3,
        'emotion_map': {'negative': 0, 'positive': 1, 'surprise': 2},
        'alignment_weight': 0.5,
        'augmentation_enable': True,
        'eval_protocol': 'loso',
    },
    'casme3': {
        'num_classes': 3,
        'emotion_map': {'negative': 0, 'positive': 1, 'surprise': 2},
        'alignment_weight': 0.5,
        'augmentation_enable': True,
        'eval_protocol': 'loso',
    },
    'dfme': {
        'num_classes': 3,
        'emotion_map': {'negative': 0, 'positive': 1, 'surprise': 2},
        'alignment_weight': 0.1,
        'augmentation_enable': False,
        'eval_protocol': 'official_split',
    },
    'casme2': {
        'num_classes': 3,
        'emotion_map': {'negative': 0, 'positive': 1, 'surprise': 2},
        'alignment_weight': 0.5,
        'augmentation_enable': True,
        'eval_protocol': 'loso',
    },
}


MODEL_SIZE_CONFIGS = {
    'nano': {'depths': [1, 1, 1, 1], 'dims': [32, 64, 128, 256]},
    'micro': {'depths': [1, 1, 1, 1], 'dims': [48, 96, 192, 384]},
    'tiny': {'depths': [1, 1, 2, 1], 'dims': [64, 128, 256, 512]},
    'small': {'depths': [1, 2, 3, 1], 'dims': [96, 192, 384, 768]},
    'ultralight': {'depths': [1, 1, 1, 1], 'dims': [24, 48, 64, 128]},
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset-specific configuration.

    Args:
        dataset_name: Name of the dataset ('4dme', 'casme3', 'dfme', 'casme2').

    Returns:
        Dictionary containing dataset-specific settings.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name].copy()


def get_model_config(model_size: str) -> Dict[str, Any]:
    """
    Get model size configuration.

    Args:
        model_size: Size variant ('nano', 'micro', 'tiny', 'small', 'ultralight').

    Returns:
        Dictionary containing depths and dims for the model.
    """
    model_size = model_size.lower()
    if model_size not in MODEL_SIZE_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available: {list(MODEL_SIZE_CONFIGS.keys())}"
        )
    return MODEL_SIZE_CONFIGS[model_size].copy()


def get_training_config(
    dataset_name: Optional[str] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Get complete training configuration with optional dataset-specific defaults.

    Args:
        dataset_name: Optional dataset name for dataset-specific defaults.
        **overrides: Key-value pairs to override default values.

    Returns:
        Complete training configuration dictionary.
    """
    config = {k: v.copy() if isinstance(v, dict) else v for k, v in DEFAULT_CONFIG.items()}

    if dataset_name is not None:
        dataset_config = get_dataset_config(dataset_name)
        config['model']['num_classes'] = dataset_config['num_classes']
        config['dataset']['emotion_map'] = dataset_config['emotion_map']
        config['dual_stream']['alignment_weight'] = dataset_config['alignment_weight']
        config['augmentation']['enable'] = dataset_config['augmentation_enable']

    for key, value in overrides.items():
        if '.' in key:
            parts = key.split('.')
            target = config
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        elif key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    return config
