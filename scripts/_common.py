"""Helpers shared by the command-line training and evaluation scripts."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

# Make the package importable when scripts are run from a source checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dadnet.data import FlowSample, MicroFlowDataset, build_augmentation  # noqa: E402


def get_device(requested: str = "auto") -> torch.device:
    """Resolve a torch device string, defaulting to CUDA when available."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def setup_logger(name: str) -> logging.Logger:
    """Return a stream logger that prints a timestamped single line per record."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def make_loader(
    samples: Sequence[FlowSample],
    batch_size: int,
    train: bool,
    augmentation: str = "none",
    num_workers: int = 4,
) -> DataLoader:
    """Build a data loader, applying augmentation only on training partitions."""
    transform = build_augmentation(augmentation) if train else None
    dataset = MicroFlowDataset(samples, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def write_json(path: str, payload: dict) -> None:
    """Write a JSON file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def warn_if_teacher_missing(
    alignment_weight: float, macro_weights_path: Optional[str], logger: logging.Logger
) -> None:
    """Warn when alignment is active but no teacher checkpoint was supplied."""
    if alignment_weight > 0 and not macro_weights_path:
        logger.warning(
            "alignment_weight=%.3g but no --macro-weights given; the teacher is "
            "randomly initialized. Train a teacher first or set lambda to 0.",
            alignment_weight,
        )


__all__ = [
    "get_device",
    "setup_logger",
    "make_loader",
    "write_json",
    "warn_if_teacher_missing",
]
