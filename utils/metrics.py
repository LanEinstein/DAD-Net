# -*- coding: utf-8 -*-
"""
Evaluation Metrics for Micro-Expression Recognition

This module provides evaluation metrics commonly used in micro-expression
recognition research, including UF1 (Unweighted F1), UAR (Unweighted Average
Recall), and confusion matrix logging.

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from typing import Dict, List, Optional, Tuple
import os


def compute_uf1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Unweighted F1-Score (UF1).

    UF1 is the macro-averaged F1-score, treating all classes equally
    regardless of their sample sizes.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        UF1 score in range [0, 1].
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_uar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Unweighted Average Recall (UAR).

    UAR is the macro-averaged recall, treating all classes equally
    regardless of their sample sizes.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        UAR score in range [0, 1].
    """
    return recall_score(y_true, y_pred, average='macro', zero_division=0)


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy in range [0, 1].
    """
    return accuracy_score(y_true, y_pred)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all standard evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing 'uf1', 'uar', and 'accuracy'.
    """
    return {
        'uf1': compute_uf1(y_true, y_pred),
        'uar': compute_uar(y_true, y_pred),
        'accuracy': compute_accuracy(y_true, y_pred),
    }


class ConfusionMatrixLogger:
    """
    Utility class for logging and visualizing confusion matrices.

    Args:
        class_names: List of class names for axis labels.
        save_dir: Directory for saving confusion matrix plots.
    """

    def __init__(
        self,
        class_names: List[str],
        save_dir: Optional[str] = None,
    ):
        self.class_names = class_names
        self.save_dir = save_dir
        self.accumulated_y_true = []
        self.accumulated_y_pred = []

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        Update accumulated predictions.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
        """
        self.accumulated_y_true.extend(y_true.tolist())
        self.accumulated_y_pred.extend(y_pred.tolist())

    def compute(self, normalize: bool = True) -> np.ndarray:
        """
        Compute confusion matrix from accumulated predictions.

        Args:
            normalize: Whether to normalize rows to sum to 1. Default: True.

        Returns:
            Confusion matrix as numpy array.
        """
        cm = confusion_matrix(
            self.accumulated_y_true,
            self.accumulated_y_pred,
            labels=list(range(len(self.class_names)))
        )

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm = cm.astype('float') / row_sums

        return cm

    def plot(
        self,
        save_name: str = "confusion_matrix.png",
        normalize: bool = True,
        figsize: Tuple[int, int] = (8, 6),
        cmap: str = "Blues",
    ):
        """
        Plot and save confusion matrix.

        Args:
            save_name: Filename for saved plot.
            normalize: Whether to normalize the matrix. Default: True.
            figsize: Figure size as (width, height). Default: (8, 6).
            cmap: Colormap name. Default: "Blues".
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Warning: matplotlib/seaborn not available for plotting")
            return

        cm = self.compute(normalize=normalize)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if self.save_dir is not None:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=150)

        plt.close()

    def reset(self):
        """Reset accumulated predictions."""
        self.accumulated_y_true = []
        self.accumulated_y_pred = []

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute metrics from accumulated predictions.

        Returns:
            Dictionary containing 'uf1', 'uar', and 'accuracy'.
        """
        y_true = np.array(self.accumulated_y_true)
        y_pred = np.array(self.accumulated_y_pred)
        return compute_metrics(y_true, y_pred)
