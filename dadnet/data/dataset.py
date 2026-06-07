"""Optical-flow dataset and subject-aware splitting utilities.

A dataset sample is a pre-computed TV-L1 optical-flow field stored as a NumPy
array, paired with an integer class label and a subject identifier. Index files
are CSVs with the columns ``flow_path``, ``label``, and ``subject``. Splits are
built by subject so that no subject appears in more than one partition.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

INDEX_COLUMNS = ("flow_path", "label", "subject")


@dataclass(frozen=True)
class FlowSample:
    """One indexed optical-flow clip."""

    flow_path: str
    label: int
    subject: str


def read_index(csv_path: str) -> List[FlowSample]:
    """Read an index CSV into a list of :class:`FlowSample`."""
    frame = pd.read_csv(csv_path)
    missing = [c for c in INDEX_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    return [
        FlowSample(str(row.flow_path), int(row.label), str(row.subject))
        for row in frame.itertuples(index=False)
    ]


class MicroFlowDataset(Dataset):
    """Dataset of pre-computed optical-flow fields.

    Args:
        samples: Indexed clips to serve.
        target_size: Output spatial size; fields are center cropped or padded.
        transform: Optional flow-consistent augmentation applied to each field.
        normalize: Apply per-channel z-score normalization when True.
    """

    def __init__(
        self,
        samples: Sequence[FlowSample],
        target_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.target_size = target_size
        self.transform = transform
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def _load_field(self, path: str) -> np.ndarray:
        """Load a flow array and return it in ``(H, W, 2)`` layout."""
        data = np.load(path)
        if data.ndim == 4 and data.shape[0] == 1 and data.shape[-1] == 2:
            data = data.squeeze(0)
        elif data.ndim == 4 and data.shape[-1] == 2:
            data = data[data.shape[0] // 2]
        elif data.ndim == 3 and data.shape[0] == 2:
            data = data.transpose(1, 2, 0)
        elif data.ndim == 3 and data.shape[-1] == 2:
            pass
        else:
            raise ValueError(f"unexpected flow shape {data.shape} in {path}")
        return data

    def _resize(self, data: np.ndarray) -> np.ndarray:
        """Center crop or symmetric pad each channel to the target size."""
        h, w = data.shape[:2]
        th, tw = self.target_size
        if (h, w) == (th, tw):
            return data
        channels = []
        for index in range(data.shape[2]):
            channel = data[:, :, index]
            if h > th:
                start = (h - th) // 2
                channel = channel[start:start + th, :]
            elif h < th:
                pad = (th - h) // 2
                channel = np.pad(channel, ((pad, th - h - pad), (0, 0)), mode="constant")
            if w > tw:
                start = (w - tw) // 2
                channel = channel[:, start:start + tw]
            elif w < tw:
                pad = (tw - w) // 2
                channel = np.pad(channel, ((0, 0), (pad, tw - w - pad)), mode="constant")
            channels.append(channel[:th, :tw])
        return np.stack(channels, axis=-1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        field = self._resize(self._load_field(sample.flow_path))
        flow = torch.from_numpy(field.transpose(2, 0, 1).copy()).float()
        if self.normalize:
            mean = flow.mean(dim=(1, 2), keepdim=True)
            std = flow.std(dim=(1, 2), keepdim=True)
            flow = (flow - mean) / (std + 1e-6)
        if self.transform is not None:
            flow = self.transform(flow)
        return flow, sample.label, idx


def filter_by_subjects(
    samples: Sequence[FlowSample], subjects: Sequence[str]
) -> List[FlowSample]:
    """Return the clips whose subject is in ``subjects``."""
    wanted = set(subjects)
    return [s for s in samples if s.subject in wanted]


def build_loso_folds(
    samples: Sequence[FlowSample],
) -> List[Tuple[List[FlowSample], List[FlowSample], str]]:
    """Build leave-one-subject-out folds.

    Returns one ``(train, test, subject)`` tuple per subject, where the test
    partition holds exactly that subject's clips.
    """
    by_subject = defaultdict(list)
    for sample in samples:
        by_subject[sample.subject].append(sample)
    folds = []
    for test_subject in sorted(by_subject):
        test = by_subject[test_subject]
        train = [s for subj, clips in by_subject.items() if subj != test_subject for s in clips]
        folds.append((train, test, test_subject))
    return folds


def carve_internal_validation(
    samples: Sequence[FlowSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[FlowSample], List[FlowSample]]:
    """Split clips into fit and validation partitions disjoint by subject.

    Subjects are shuffled with the given seed and assigned to validation until
    the clip share reaches ``val_ratio``; the rest form the fit partition.

    Returns:
        A ``(fit, validation)`` pair with no shared subject.
    """
    by_subject = defaultdict(list)
    for sample in samples:
        by_subject[sample.subject].append(sample)
    subjects = sorted(by_subject)
    random.Random(seed).shuffle(subjects)

    target = int(round(len(samples) * val_ratio))
    val_subjects, count = set(), 0
    for subject in subjects:
        if count >= target:
            break
        val_subjects.add(subject)
        count += len(by_subject[subject])

    fit = [s for s in samples if s.subject not in val_subjects]
    validation = [s for s in samples if s.subject in val_subjects]
    return fit, validation


__all__ = [
    "FlowSample",
    "MicroFlowDataset",
    "INDEX_COLUMNS",
    "read_index",
    "filter_by_subjects",
    "build_loso_folds",
    "carve_internal_validation",
]
