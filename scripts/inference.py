"""Run a trained DAD-Net student on a single optical-flow clip.

The student backbone is rebuilt from a benchmark configuration and loaded from a
checkpoint (either a standalone student or a full dual-stream model, whose
``micro_branch`` weights are extracted automatically).

Example:
    python scripts/inference.py --benchmark dfme_3class \
        --checkpoint outputs/dfme_3class_student.pth \
        --input sample_flow.npy
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from _common import get_device

from dadnet.builder import build_student
from dadnet.configs import get_benchmark
from dadnet.data import MicroFlowDataset, FlowSample

LABELS_3CLASS = {0: "negative", 1: "positive", 2: "surprise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAD-Net student inference")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="optical-flow .npy file")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _extract_student_state(state: dict) -> dict:
    """Return student weights, stripping the ``micro_branch.`` prefix when present."""
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    prefixed = {k[len("micro_branch."):]: v for k, v in state.items() if k.startswith("micro_branch.")}
    return prefixed or state


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    config = get_benchmark(args.benchmark)
    model = build_student(config).to(device)
    model.load_state_dict(
        _extract_student_state(torch.load(args.checkpoint, map_location=device)), strict=False
    )
    model.eval()

    # Reuse the dataset loader so preprocessing matches training exactly.
    dataset = MicroFlowDataset([FlowSample(args.input, 0, "infer")], transform=None)
    flow, _, _ = dataset[0]

    with torch.no_grad():
        prediction, logits = model(flow.unsqueeze(0).to(device))
    probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    predicted = int(prediction.item())

    print(f"predicted class: {predicted} ({LABELS_3CLASS.get(predicted, 'class ' + str(predicted))})")
    for index, prob in enumerate(probabilities):
        print(f"  class {index}: {prob:.4f}")


if __name__ == "__main__":
    main()
