"""Construct models directly from a benchmark configuration."""

from __future__ import annotations

from typing import Optional

from .configs.default import BenchmarkConfig
from .models import DADNet, MicroFlowNeXt, get_dad_net, get_microflownext


def build_dadnet(
    config: BenchmarkConfig,
    macro_weights_path: Optional[str] = None,
    num_projections: int = 100,
    label_smoothing: float = 0.05,
    input_size: int = 224,
) -> DADNet:
    """Build the dual-stream DAD-Net described by a benchmark configuration."""
    return get_dad_net(
        num_classes=config.num_classes,
        micro_model_size=config.student_size,
        macro_model_size=config.teacher_size,
        alignment_stages=config.alignment_stages(),
        macro_weights_path=macro_weights_path,
        alignment_weight=config.alignment_weight,
        num_projections=num_projections,
        input_size=input_size,
        label_smoothing=label_smoothing,
        skip_stn=config.skip_stn,
        skip_channel_attention=config.skip_channel_attention,
        skip_head_attention=config.skip_head_attention,
    )


def build_student(config: BenchmarkConfig, input_size: int = 224) -> MicroFlowNeXt:
    """Build the standalone student backbone described by a configuration."""
    return get_microflownext(
        num_classes=config.num_classes,
        model_size=config.student_size,
        input_size=input_size,
        skip_stn=config.skip_stn,
        skip_channel_attention=config.skip_channel_attention,
        skip_head_attention=config.skip_head_attention,
    )


__all__ = ["build_dadnet", "build_student"]
