"""Per-benchmark configurations and the shared training recipe.

``BENCHMARKS`` encodes the configurations reported in the paper: student width,
the four-stage alignment mask, the alignment weight lambda, the augmentation
mode, and the protocol. LOSO benchmarks use the full student; DFME benchmarks
use the capacity-matched student selected on internal validation, whose pruned
modules are recorded by the ``skip_*`` flags.

``TRAINING`` holds the optimization recipe shared by every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization recipe shared across benchmarks."""

    optimizer: str = "adamw"
    learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-4
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 30
    scheduler: str = "cosine"
    min_lr: float = 1.0e-7
    label_smoothing: float = 0.05
    num_projections: int = 100
    selection_metric: str = "uf1"
    internal_val_ratio: float = 0.2
    dfme_seeds: Tuple[int, ...] = (4, 5, 7, 31, 42)
    loso_seed: int = 42


TRAINING = TrainingConfig()


@dataclass(frozen=True)
class BenchmarkConfig:
    """Single-benchmark configuration of the DAD-Net family.

    Attributes:
        dataset: Source dataset name.
        num_classes: Number of classes for this setting.
        protocol: Evaluation protocol, ``loso`` or ``dfme``.
        student_size: Student width key in ``MODEL_SIZE_CONFIGS``.
        teacher_size: Teacher width key (matched to the student by default).
        stage_mask: Four-character ``T``/``F`` mask of active alignment stages.
        alignment_weight: Alignment weight lambda (0 disables the teacher term).
        augmentation: ``standard``, ``hflip``, or ``none``.
        skip_stn: Prune the student spatial transformer.
        skip_channel_attention: Prune student channel attention.
        skip_head_attention: Prune the student self-attention head.
    """

    dataset: str
    num_classes: int
    protocol: str
    student_size: str
    teacher_size: str
    stage_mask: str
    alignment_weight: float
    augmentation: str
    skip_stn: bool = False
    skip_channel_attention: bool = False
    skip_head_attention: bool = False

    def alignment_stages(self) -> Dict[int, bool]:
        """Return the alignment mask as a ``{stage_index: active}`` map."""
        return parse_stage_mask(self.stage_mask)


def parse_stage_mask(mask: str) -> Dict[int, bool]:
    """Convert a ``T``/``F`` stage mask such as ``TFTT`` into a stage map."""
    mask = mask.upper()
    if any(ch not in "TF" for ch in mask):
        raise ValueError(f"stage mask must contain only T/F, got {mask!r}")
    return {i: ch == "T" for i, ch in enumerate(mask)}


BENCHMARKS: Dict[str, BenchmarkConfig] = {
    "4dme": BenchmarkConfig(
        dataset="4dme", num_classes=3, protocol="loso",
        student_size="base", teacher_size="base",
        stage_mask="TFTT", alignment_weight=1.5, augmentation="standard",
    ),
    "casme3": BenchmarkConfig(
        dataset="casme3", num_classes=3, protocol="loso",
        student_size="base", teacher_size="base",
        stage_mask="TFFT", alignment_weight=1.0, augmentation="hflip",
    ),
    "dfme_3class": BenchmarkConfig(
        dataset="dfme", num_classes=3, protocol="dfme",
        student_size="small", teacher_size="small",
        stage_mask="FFTT", alignment_weight=0.0, augmentation="standard",
        skip_channel_attention=True, skip_head_attention=True,
    ),
    "dfme_4class": BenchmarkConfig(
        dataset="dfme", num_classes=4, protocol="dfme",
        student_size="base", teacher_size="base",
        stage_mask="FFTT", alignment_weight=0.0, augmentation="standard",
        skip_channel_attention=True, skip_head_attention=True,
    ),
    "dfme_7class": BenchmarkConfig(
        dataset="dfme", num_classes=7, protocol="dfme",
        student_size="micro", teacher_size="micro",
        stage_mask="FFTT", alignment_weight=1.0, augmentation="standard",
        skip_stn=True, skip_channel_attention=True, skip_head_attention=True,
    ),
}


def get_benchmark(name: str) -> BenchmarkConfig:
    """Return the configuration registered under ``name``."""
    key = name.lower()
    if key not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {list(BENCHMARKS)}")
    return BENCHMARKS[key]


# Macro teacher pretraining on the posed CK+ dataset (paper Sec. IV-A).
@dataclass(frozen=True)
class MacroTeacherConfig:
    """Recipe for pretraining the frozen macro-expression teacher on CK+."""

    dataset: str = "ckplus"
    num_classes: int = 3
    learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-4
    batch_size: int = 32
    max_epochs: int = 100
    scheduler: str = "cosine"
    min_lr: float = 1.0e-7
    weighted_sampler: bool = True
    label_map: Dict[str, int] = field(
        default_factory=lambda: {"negative": 0, "positive": 1, "surprise": 2}
    )


MACRO_TEACHER = MacroTeacherConfig()


__all__ = [
    "TrainingConfig",
    "TRAINING",
    "BenchmarkConfig",
    "BENCHMARKS",
    "get_benchmark",
    "parse_stage_mask",
    "MacroTeacherConfig",
    "MACRO_TEACHER",
]
