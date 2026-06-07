"""Evaluate out-of-distribution robustness of a trained student on DFME test B.

Each corruption is applied to the flow field at inference only, with no
retraining. The retention at each severity is the corrupted UF1 divided by the
clean UF1 (severity index 0). Results are written per corruption and severity.

Example:
    python scripts/evaluate_robustness.py --benchmark dfme_3class \
        --checkpoint outputs/dfme_3class_student.pth \
        --testb-index data/dfme/testB_3class.csv \
        --output outputs/robustness_3class.json
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch

from _common import get_device, setup_logger, write_json

from dadnet.builder import build_student
from dadnet.configs import get_benchmark
from dadnet.data import MicroFlowDataset, read_index
from dadnet.data.perturbations import PERTURBATIONS, SEVERITY, apply
from dadnet.engine import compute_metrics, set_seed


def _extract_student_state(state: dict) -> dict:
    """Return student weights, stripping the ``micro_branch.`` prefix when present."""
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    prefixed = {k[len("micro_branch."):]: v for k, v in state.items() if k.startswith("micro_branch.")}
    return prefixed or state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness evaluation on DFME test B")
    parser.add_argument("--benchmark", required=True,
                        choices=["dfme_3class", "dfme_4class", "dfme_7class"])
    parser.add_argument("--checkpoint", required=True, help="trained student checkpoint")
    parser.add_argument("--testb-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def _score(model, dataset, device, kind: str, level: float) -> Dict[str, float]:
    """Score the dataset with one corruption applied per clip (level 0 = clean)."""
    y_true: List[int] = []
    y_pred: List[int] = []
    for index in range(len(dataset)):
        flow, label, _ = dataset[index]
        if level != 0.0:
            flow = apply(flow, kind, level, clip_index=index)
        prediction = model(flow.unsqueeze(0).to(device))[0]
        y_pred.append(int(prediction.item()))
        y_true.append(int(label))
    return compute_metrics(y_true, y_pred)


def main() -> None:
    args = parse_args()
    logger = setup_logger("evaluate_robustness")
    device = get_device(args.device)
    set_seed(args.seed)

    config = get_benchmark(args.benchmark)
    model = build_student(config).to(device)
    model.load_state_dict(
        _extract_student_state(torch.load(args.checkpoint, map_location=device)), strict=False
    )
    model.eval()

    dataset = MicroFlowDataset(read_index(args.testb_index), transform=None)
    clean = _score(model, dataset, device, PERTURBATIONS[0], 0.0)["uf1"]
    logger.info("clean UF1=%.4f over %d clips", clean, len(dataset))

    curves: Dict[str, Dict] = {}
    for kind in PERTURBATIONS:
        retention = [1.0]
        uf1_values = [clean]
        for level in SEVERITY[kind]:
            uf1 = _score(model, dataset, device, kind, level)["uf1"]
            uf1_values.append(uf1)
            retention.append(uf1 / clean if clean > 0 else 0.0)
            logger.info("%s level=%.3g uf1=%.4f retention=%.3f", kind, level, uf1, retention[-1])
        curves[kind] = {
            "severity": [0.0] + list(SEVERITY[kind]),
            "uf1": uf1_values,
            "retention": retention,
        }

    write_json(args.output, {
        "benchmark": args.benchmark,
        "clean_uf1": clean,
        "curves": curves,
    })
    logger.info("results written to %s", args.output)


if __name__ == "__main__":
    main()
