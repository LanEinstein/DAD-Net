# -*- coding: utf-8 -*-
"""
Training Utilities for DAD-Net

This module provides training utilities including learning rate schedulers,
early stopping, and random seed management.

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: Count only trainable parameters. Default: True.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.01,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a cosine learning rate schedule with linear warmup.

    The learning rate linearly increases during warmup, then follows a
    cosine decay schedule.

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles. Default: 0.5 (half cycle).
        min_lr_ratio: Minimum learning rate as ratio of initial. Default: 0.01.
        last_epoch: Index of last epoch. Default: -1.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps)
        total = float(max(1, num_training_steps - num_warmup_steps))
        progress = min(progress / total, 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.

    Monitors a metric and stops training when it stops improving for
    a specified number of epochs.

    Args:
        patience: Number of epochs to wait for improvement. Default: 10.
        min_delta: Minimum change to qualify as improvement. Default: 0.0.
        mode: "min" for loss metrics, "max" for accuracy metrics. Default: "min".
        restore_best_weights: Restore model to best state on stop. Default: True.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None

        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(
        self,
        score: float,
        model: Optional[nn.Module] = None,
    ) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value.
            model: Model to save best state from. Default: None.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_state_dict = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_state_dict = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_model(self, model: nn.Module):
        """
        Restore model to best state.

        Args:
            model: Model to restore weights to.
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking training/validation metrics across batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.

        Args:
            val: New value to add.
            n: Weight/count for this value. Default: 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
