"""Training loop, metrics, and reproducibility utilities."""

from .metrics import (
    compute_accuracy,
    compute_metrics,
    compute_uar,
    compute_uf1,
    pooled_metrics,
)
from .trainer import (
    build_optimizer,
    build_scheduler,
    evaluate,
    fit_trajectory,
    train_one_epoch,
)
from .utils import AverageMeter, count_parameters, set_seed

__all__ = [
    "compute_uf1",
    "compute_uar",
    "compute_accuracy",
    "compute_metrics",
    "pooled_metrics",
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
    "fit_trajectory",
    "set_seed",
    "count_parameters",
    "AverageMeter",
]
