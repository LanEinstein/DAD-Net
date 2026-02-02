# -*- coding: utf-8 -*-
"""
Micro-Expression Dataset Classes

This module provides PyTorch Dataset implementations for loading and processing
optical flow data for micro-expression recognition tasks.

Supported datasets:
    - CASME II
    - CASME III
    - 4DME
    - DFME

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict


class MicroExpressionDataset(Dataset):
    """
    PyTorch Dataset for Micro-Expression Recognition.

    Loads pre-computed optical flow data and applies optional transformations.
    Supports both file-path-based and DataFrame-based data organization.

    Args:
        data_source: Either a list of file paths or a pandas DataFrame with columns
                    ['File_Path', 'Emotion', 'Subject'].
        emotion_map: Dictionary mapping emotion strings to class indices.
        target_size: Target spatial size for optical flow data. Default: (224, 224).
        transform: Optional transform to apply to optical flow data.
        normalize: Whether to apply z-score normalization. Default: True.
    """

    EMOTION_MAPS = {
        '3class': {'negative': 0, 'positive': 1, 'surprise': 2},
        '4class': {'negative': 0, 'positive': 1, 'surprise': 2, 'others': 3},
        '7class': {
            'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'contempt': 6
        },
    }

    def __init__(
        self,
        data_source,
        emotion_map: Optional[Dict[str, int]] = None,
        target_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ):
        if emotion_map is None:
            emotion_map = self.EMOTION_MAPS['3class']

        self.emotion_map = emotion_map
        self.target_size = target_size
        self.transform = transform
        self.normalize = normalize

        if isinstance(data_source, pd.DataFrame):
            self.dataframe = data_source.copy()
            self.use_dataframe = True
        else:
            self.file_paths = list(data_source)
            self.use_dataframe = False

    def __len__(self) -> int:
        if self.use_dataframe:
            return len(self.dataframe)
        return len(self.file_paths)

    def _load_optical_flow(self, file_path: str) -> np.ndarray:
        """Load and preprocess optical flow data from file."""
        data = np.load(file_path)

        if data.ndim == 4 and data.shape[0] == 1 and data.shape[-1] == 2:
            data = data.squeeze(0)
        elif data.ndim == 4 and data.shape[-1] == 2:
            data = data[data.shape[0] // 2]
        elif data.ndim == 3 and data.shape[0] == 2:
            data = data.transpose(1, 2, 0)
        elif data.ndim == 3 and data.shape[-1] == 2:
            pass
        else:
            raise ValueError(f"Unexpected optical flow shape: {data.shape}")

        return data

    def _resize_optical_flow(self, data: np.ndarray) -> np.ndarray:
        """Resize optical flow to target size using center crop/pad."""
        h, w = data.shape[:2]
        th, tw = self.target_size

        if h == th and w == tw:
            return data

        resized_channels = []
        for ch_idx in range(data.shape[2]):
            channel = data[:, :, ch_idx]

            if h > th:
                h_start = (h - th) // 2
                channel = channel[h_start:h_start + th, :]
            elif h < th:
                pad_h = (th - h) // 2
                channel = np.pad(channel, ((pad_h, th - h - pad_h), (0, 0)), mode='constant')

            if w > tw:
                w_start = (w - tw) // 2
                channel = channel[:, w_start:w_start + tw]
            elif w < tw:
                pad_w = (tw - w) // 2
                channel = np.pad(channel, ((0, 0), (pad_w, tw - w - pad_w)), mode='constant')

            resized_channels.append(channel[:th, :tw])

        return np.stack(resized_channels, axis=-1)

    def _get_label_from_path(self, file_path: str) -> int:
        """Extract emotion label from file path structure."""
        emotion = os.path.basename(os.path.dirname(file_path))
        return self.emotion_map.get(emotion.lower(), 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (optical_flow_tensor, label, sample_index).
        """
        if self.use_dataframe:
            row = self.dataframe.iloc[idx]
            file_path = row['File_Path']
            label_str = row['Emotion']
            label = self.emotion_map.get(label_str.lower(), 0)
            sample_idx = row.name
        else:
            file_path = self.file_paths[idx]
            label = self._get_label_from_path(file_path)
            sample_idx = idx

        try:
            data = self._load_optical_flow(file_path)
            data = self._resize_optical_flow(data)

            data = data.transpose(2, 0, 1)
            data = torch.from_numpy(data.copy()).float()

            if self.normalize:
                mean = data.mean(dim=(1, 2), keepdim=True)
                std = data.std(dim=(1, 2), keepdim=True)
                data = (data - mean) / (std + 1e-6)

            if self.transform is not None:
                data = self.transform(data)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            data = torch.zeros((2, self.target_size[0], self.target_size[1]), dtype=torch.float)

        return data, label, sample_idx


def create_loso_splits(
    data_dir: str,
    emotion_map: Optional[Dict[str, int]] = None,
) -> List[Tuple[List[str], List[str], str]]:
    """
    Create Leave-One-Subject-Out (LOSO) cross-validation splits.

    Args:
        data_dir: Root directory containing emotion subdirectories.
        emotion_map: Dictionary mapping emotion strings to class indices.

    Returns:
        List of tuples (train_paths, test_paths, subject_id) for each fold.
    """
    if emotion_map is None:
        emotion_map = MicroExpressionDataset.EMOTION_MAPS['3class']

    subject_files = defaultdict(list)

    for emotion in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        if emotion.lower() not in emotion_map:
            continue

        for file_name in os.listdir(emotion_dir):
            if not file_name.endswith('.npy'):
                continue

            file_path = os.path.join(emotion_dir, file_name)
            parts = file_name.split('_')
            if len(parts) >= 3:
                subject_id = '_'.join(parts[1:-1])
                subject_files[subject_id].append(file_path)

    splits = []
    subjects = sorted(subject_files.keys())

    for test_subject in subjects:
        test_paths = subject_files[test_subject]
        train_paths = []
        for subject, paths in subject_files.items():
            if subject != test_subject:
                train_paths.extend(paths)

        splits.append((train_paths, test_paths, test_subject))

    return splits


def create_train_test_splits(
    data_dir: str,
    test_ratio: float = 0.1,
    emotion_map: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create stratified train/test splits by subject ID.

    Ensures that all samples from the same subject are in either train or test set,
    preventing data leakage.

    Args:
        data_dir: Root directory containing emotion subdirectories.
        test_ratio: Proportion of subjects to use for testing. Default: 0.1.
        emotion_map: Dictionary mapping emotion strings to class indices.
        seed: Random seed for reproducibility. Default: 42.

    Returns:
        Tuple of (train_paths, test_paths).
    """
    import random
    random.seed(seed)

    if emotion_map is None:
        emotion_map = MicroExpressionDataset.EMOTION_MAPS['3class']

    emotion_subject_files = defaultdict(lambda: defaultdict(list))

    for emotion in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        emotion_lower = emotion.lower()
        if emotion_lower not in emotion_map:
            continue

        for file_name in os.listdir(emotion_dir):
            if not file_name.endswith('.npy'):
                continue

            file_path = os.path.join(emotion_dir, file_name)
            parts = file_name.split('_')
            if len(parts) >= 3:
                subject_id = '_'.join(parts[1:-1])
                emotion_subject_files[emotion_lower][subject_id].append(file_path)

    train_paths = []
    test_paths = []

    for emotion, subject_dict in emotion_subject_files.items():
        subjects = list(subject_dict.keys())
        subjects.sort()

        test_count = max(1, int(len(subjects) * test_ratio))
        random.shuffle(subjects)

        test_subjects = subjects[:test_count]
        train_subjects = subjects[test_count:]

        for subject in train_subjects:
            train_paths.extend(subject_dict[subject])

        for subject in test_subjects:
            test_paths.extend(subject_dict[subject])

    return train_paths, test_paths
