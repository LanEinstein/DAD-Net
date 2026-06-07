"""Shared training and evaluation loop for the student and teacher streams.

A single trajectory trains one model. When a validation loader is supplied, the
best epoch is selected by validation UF1 (early stopping on patience); otherwise
the model trains for the full cosine schedule. Held-out test loaders are scored
once with the selected weights.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..configs.default import TrainingConfig
from .metrics import compute_metrics


def _is_dual(model: nn.Module) -> bool:
    """Return True for a dual-stream model that emits an alignment loss."""
    return getattr(model, "macro_branch", None) is not None


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Construct the optimizer over the model's trainable parameters."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adamw":
        return torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construct the per-epoch learning-rate scheduler."""
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs, eta_min=config.min_lr
        )
    raise ValueError(f"unknown scheduler {config.scheduler!r}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one optimization epoch and return the mean batch loss."""
    model.train()
    dual = _is_dual(model)
    total_loss, total_count = 0.0, 0
    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        loss = outputs[3] if dual else outputs[2]
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_count += inputs.size(0)
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """Score a loader and return metrics with the true and predicted labels."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for inputs, labels, _ in loader:
        predictions = model(inputs.to(device))[0]
        y_pred.extend(predictions.cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())
    return compute_metrics(y_true, y_pred), y_true, y_pred


def _selection_key(metrics: Dict[str, float], selection_metric: str) -> Tuple[float, float]:
    """Return a comparable key so ties on the primary metric break on the other."""
    if selection_metric == "uf1":
        return metrics["uf1"], metrics["uar"]
    if selection_metric == "uar":
        return metrics["uar"], metrics["uf1"]
    raise ValueError(f"unknown selection metric {selection_metric!r}")


def fit_trajectory(
    model: nn.Module,
    fit_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    test_loaders: Optional[Dict[str, DataLoader]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """Train one trajectory and evaluate the held-out test loaders.

    Args:
        model: Model to train (single- or dual-stream).
        fit_loader: Training data loader.
        config: Optimization recipe.
        device: Compute device.
        val_loader: Optional validation loader driving best-epoch selection.
        test_loaders: Optional named loaders scored once with the final weights.
        logger: Optional logger for per-epoch progress.

    Returns:
        A dictionary with the best validation metrics (when a validation loader
        is given) and, for each test loader, its metrics and predictions.
    """
    model = model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    best_key = (-1.0, -1.0)
    best_state = None
    best_val: Dict[str, float] = {}
    epochs_since_best = 0

    for epoch in range(config.max_epochs):
        loss = train_one_epoch(model, fit_loader, optimizer, device)
        scheduler.step()

        if val_loader is None:
            if logger:
                logger.info("epoch %d/%d  loss=%.4f", epoch + 1, config.max_epochs, loss)
            continue

        val_metrics, _, _ = evaluate(model, val_loader, device)
        key = _selection_key(val_metrics, config.selection_metric)
        if key > best_key:
            best_key, best_val = key, val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if logger:
            logger.info(
                "epoch %d/%d  loss=%.4f  val_uf1=%.4f  val_uar=%.4f  best_uf1=%.4f",
                epoch + 1, config.max_epochs, loss,
                val_metrics["uf1"], val_metrics["uar"], best_val.get("uf1", 0.0),
            )
        if epochs_since_best >= config.patience:
            if logger:
                logger.info("early stopping at epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    results: Dict[str, object] = {"best_val": best_val}
    for name, loader in (test_loaders or {}).items():
        metrics, y_true, y_pred = evaluate(model, loader, device)
        results[name] = {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}
    return results


__all__ = [
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
    "fit_trajectory",
]
