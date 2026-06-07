"""DAD-Net: a distribution-aligned dual-stream network.

A frozen macro-expression teacher and a trainable micro-expression student are
both instantiated as MicroFlowNeXt backbones. The student is regularized to
match the teacher feature distribution at several backbone stages through GSWD,
a closed-form, correspondence-free objective. The teacher is frozen so that it
supplies a stable macro-motion prior rather than drifting toward the scarce
micro-expression data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.gswd import gaussian_sliced_wasserstein_distance
from ..losses.kd_baselines import get_kd_loss
from .microflownext import MODEL_SIZE_CONFIGS, MicroFlowNeXt


class FeatureAdapter(nn.Module):
    """Project teacher features to the student channel width (paper Eq. 1).

    A 1x1 convolution followed by batch normalization makes stage-wise
    alignment well defined when the two streams differ in channel width.

    Args:
        in_channels: Teacher channel count.
        out_channels: Student channel count.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class DADNet(nn.Module):
    """Distribution-aligned dual-stream network.

    Args:
        num_classes: Number of output classes.
        micro_config: ``(depths, dims)`` for the trainable student stream.
        macro_config: ``(depths, dims)`` for the frozen teacher stream.
        alignment_stages: Map ``{stage_index: active}`` selecting aligned stages.
        macro_weights_path: Checkpoint with pre-trained teacher weights.
        alignment_weight: Weight on the active-stage alignment term (lambda).
        num_projections: Number of GSWD random projections.
        in_chans: Number of input channels.
        input_size: Expected square input spatial size.
        kd_method: Alignment objective at active stages
            (``gswd`` or a baseline from :func:`get_kd_loss`).
        label_smoothing: Label-smoothing factor for the classification loss.
        skip_stn: Disable the student spatial transformer (ablation).
        skip_channel_attention: Disable student channel attention (ablation).
        skip_head_attention: Disable the student self-attention head (ablation).
    """

    def __init__(
        self,
        num_classes: int,
        micro_config: Tuple[List[int], List[int]],
        macro_config: Tuple[List[int], List[int]],
        alignment_stages: Dict[int, bool],
        macro_weights_path: Optional[str] = None,
        alignment_weight: float = 0.5,
        num_projections: int = 100,
        in_chans: int = 2,
        input_size: int = 224,
        kd_method: str = "gswd",
        label_smoothing: float = 0.05,
        skip_stn: bool = False,
        skip_channel_attention: bool = False,
        skip_head_attention: bool = False,
    ) -> None:
        super().__init__()
        micro_depths, micro_dims = micro_config
        macro_depths, macro_dims = macro_config

        self.num_classes = num_classes
        self.alignment_stages = dict(alignment_stages)
        self.alignment_weight = alignment_weight
        self.num_projections = num_projections
        self.kd_method = kd_method.lower()
        self.label_smoothing = label_smoothing
        self.micro_dims = list(micro_dims)
        self.macro_dims = list(macro_dims)
        self.micro_num_stages = len(micro_depths)
        self.macro_num_stages = len(macro_depths)

        self.micro_branch = MicroFlowNeXt(
            in_chans=in_chans,
            num_classes=num_classes,
            depths=micro_depths,
            dims=micro_dims,
            drop_path_rate=0.0,
            input_size=input_size,
            skip_stn=skip_stn,
            skip_channel_attention=skip_channel_attention,
            skip_head_attention=skip_head_attention,
        )
        self.macro_branch = MicroFlowNeXt(
            in_chans=in_chans,
            num_classes=num_classes,
            depths=macro_depths,
            dims=macro_dims,
            drop_path_rate=0.0,
            input_size=input_size,
        )

        self._build_adapters()
        self._build_kd_losses()
        self._load_and_freeze_teacher(macro_weights_path)

    def _build_adapters(self) -> None:
        """Create a teacher-to-student adapter for each active mismatched stage."""
        self.adapters = nn.ModuleList()
        self.adapter_map: Dict[int, int] = {}
        for stage in range(max(self.micro_num_stages, self.macro_num_stages)):
            if not self.alignment_stages.get(stage, False):
                continue
            if stage >= self.micro_num_stages or stage >= self.macro_num_stages:
                continue
            macro_dim, micro_dim = self.macro_dims[stage], self.micro_dims[stage]
            if macro_dim != micro_dim:
                self.adapter_map[stage] = len(self.adapters)
                self.adapters.append(FeatureAdapter(macro_dim, micro_dim))

    def _build_kd_losses(self) -> None:
        """Instantiate per-stage baseline losses when ``kd_method`` is not GSWD."""
        self.kd_losses = nn.ModuleDict()
        if self.kd_method == "gswd":
            return
        for stage, active in self.alignment_stages.items():
            if not active or stage >= min(self.micro_num_stages, self.macro_num_stages):
                continue
            dim = self.micro_dims[stage]
            self.kd_losses[str(stage)] = get_kd_loss(
                self.kd_method, student_dim=dim, teacher_dim=dim
            )

    def _load_and_freeze_teacher(self, weights_path: Optional[str]) -> None:
        """Load teacher weights when provided, then freeze the teacher stream."""
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(state, dict) and key in state:
                    state = state[key]
                    break
            target = self.macro_branch.state_dict()
            filtered = {}
            for key, value in state.items():
                clean = key
                for prefix in ("module.", "model.", "backbone."):
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                if clean.startswith(("head.", "attention.")):
                    continue
                if clean in target and value.shape == target[clean].shape:
                    filtered[clean] = value
            self.macro_branch.load_state_dict(filtered, strict=False)
        for param in self.macro_branch.parameters():
            param.requires_grad = False
        self.macro_branch.eval()

    def train(self, mode: bool = True) -> "DADNet":
        """Keep the frozen teacher in evaluation mode at all times."""
        super().train(mode)
        self.macro_branch.eval()
        return self

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple:
        with torch.no_grad():
            teacher_features = self.macro_branch.forward_features_staged(x)

        feature = self.micro_branch.spatial_transform(x)
        total_alignment = torch.zeros(1, device=x.device)
        num_aligned = 0

        for stage in range(self.micro_num_stages):
            feature = self.micro_branch.downsample_layers[stage](feature)
            feature = self.micro_branch.stages[stage](feature)

            if not self.alignment_stages.get(stage, False) or stage >= self.macro_num_stages:
                continue
            teacher_feature = teacher_features[stage]
            if stage in self.adapter_map:
                teacher_feature = self.adapters[self.adapter_map[stage]](teacher_feature)
            if self.kd_method == "gswd":
                stage_loss = gaussian_sliced_wasserstein_distance(
                    feature, teacher_feature, num_projections=self.num_projections
                )
            else:
                stage_loss = self.kd_losses[str(stage)](feature, teacher_feature)
            total_alignment = total_alignment + stage_loss
            num_aligned += 1

        # Average over active stages so alignment strength is independent of
        # the number of aligned stages (paper Eq. 13).
        alignment = total_alignment / num_aligned if num_aligned > 0 else total_alignment

        pooled = self.micro_branch.norm(feature.mean([-2, -1]))
        attended, attention_weights = self.micro_branch.attention(pooled)
        logits = self.micro_branch.head(attended)
        predictions = torch.argmax(logits, dim=1)

        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            total_loss = cls_loss + self.alignment_weight * alignment
            return predictions, logits, attention_weights, total_loss, alignment
        return predictions, logits, attention_weights


def get_dad_net(
    num_classes: int = 3,
    micro_model_size: str = "micro",
    macro_model_size: str = "nano",
    alignment_stages: Optional[Dict[int, bool]] = None,
    macro_weights_path: Optional[str] = None,
    alignment_weight: float = 0.5,
    num_projections: int = 100,
    input_size: int = 224,
    kd_method: str = "gswd",
    label_smoothing: float = 0.05,
    skip_stn: bool = False,
    skip_channel_attention: bool = False,
    skip_head_attention: bool = False,
) -> DADNet:
    """Build a DAD-Net from named width configurations.

    Args:
        num_classes: Number of output classes.
        micro_model_size: Student width key in ``MODEL_SIZE_CONFIGS``.
        macro_model_size: Teacher width key in ``MODEL_SIZE_CONFIGS``.
        alignment_stages: Aligned-stage map; defaults to the last two stages.
        macro_weights_path: Checkpoint with pre-trained teacher weights.
        alignment_weight: Weight on the active-stage alignment term (lambda).
        num_projections: Number of GSWD random projections.
        input_size: Expected square input spatial size.
        kd_method: Alignment objective at active stages.
        label_smoothing: Label-smoothing factor for the classification loss.
        skip_stn: Disable the student spatial transformer (ablation).
        skip_channel_attention: Disable student channel attention (ablation).
        skip_head_attention: Disable the student self-attention head (ablation).
    """
    for name, size in (("micro_model_size", micro_model_size), ("macro_model_size", macro_model_size)):
        if size not in MODEL_SIZE_CONFIGS:
            raise ValueError(f"unknown {name} {size!r}; choose from {list(MODEL_SIZE_CONFIGS)}")
    micro = MODEL_SIZE_CONFIGS[micro_model_size]
    macro = MODEL_SIZE_CONFIGS[macro_model_size]
    if alignment_stages is None:
        alignment_stages = {0: False, 1: False, 2: True, 3: True}
    return DADNet(
        num_classes=num_classes,
        micro_config=(micro["depths"], micro["dims"]),
        macro_config=(macro["depths"], macro["dims"]),
        alignment_stages=alignment_stages,
        macro_weights_path=macro_weights_path,
        alignment_weight=alignment_weight,
        num_projections=num_projections,
        in_chans=2,
        input_size=input_size,
        kd_method=kd_method,
        label_smoothing=label_smoothing,
        skip_stn=skip_stn,
        skip_channel_attention=skip_channel_attention,
        skip_head_attention=skip_head_attention,
    )


__all__ = ["DADNet", "FeatureAdapter", "get_dad_net"]
