"""Evaluation metrics for micro-expression recognition.

UF1 (unweighted F1) and UAR (unweighted average recall) are macro-averaged and,
under LOSO, computed once on the predictions pooled across all folds rather than
averaged per fold.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_uf1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Macro-averaged F1 over all classes."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_uar(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Macro-averaged recall over all classes."""
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Overall classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Return UF1, UAR, and accuracy as a dictionary."""
    return {
        "uf1": compute_uf1(y_true, y_pred),
        "uar": compute_uar(y_true, y_pred),
        "accuracy": compute_accuracy(y_true, y_pred),
    }


def pooled_metrics(
    fold_true: Sequence[Sequence[int]], fold_pred: Sequence[Sequence[int]]
) -> Dict[str, float]:
    """Compute metrics on predictions concatenated across folds."""
    y_true = np.concatenate([np.asarray(f) for f in fold_true]) if fold_true else np.array([])
    y_pred = np.concatenate([np.asarray(f) for f in fold_pred]) if fold_pred else np.array([])
    return compute_metrics(y_true, y_pred)


__all__ = [
    "compute_uf1",
    "compute_uar",
    "compute_accuracy",
    "compute_metrics",
    "pooled_metrics",
]
