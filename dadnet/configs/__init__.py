"""Benchmark configurations and the shared training recipe."""

from .default import (
    BENCHMARKS,
    MACRO_TEACHER,
    TRAINING,
    BenchmarkConfig,
    MacroTeacherConfig,
    TrainingConfig,
    get_benchmark,
    parse_stage_mask,
)

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
