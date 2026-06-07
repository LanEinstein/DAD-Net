"""Reproducibility and bookkeeping helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and enable deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters, optionally restricted to trainable ones."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


class AverageMeter:
    """Track a running average of a scalar value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.sum / self.count if self.count else 0.0


__all__ = ["set_seed", "count_parameters", "AverageMeter"]
