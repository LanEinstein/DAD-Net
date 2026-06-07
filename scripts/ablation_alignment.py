"""Compare alignment objectives (GSWD against distillation baselines).

The same student and teacher are trained with each alignment objective at the
active stages, under the DFME internal-validation protocol. Test A and test B
UF1 are reported per objective. Distribution-level objectives (GSWD, KL) are
expected to lead the pointwise and contrastive baselines (paper Table VI).

Example:
    python scripts/ablation_alignment.py --benchmark dfme_7class \
        --train-index data/dfme/train_7class.csv \
        --testa-index data/dfme/testA_7class.csv \
        --testb-index data/dfme/testB_7class.csv \
        --macro-weights weights/teacher_micro.pth \
        --alignment-weight 0.3 \
        --output outputs/alignment_ablation.json
"""

from __future__ import annotations

import argparse
from typing import List

from _common import get_device, make_loader, setup_logger, warn_if_teacher_missing, write_json

from dadnet.configs import TRAINING, get_benchmark
from dadnet.data import carve_internal_validation, read_index
from dadnet.engine import fit_trajectory, set_seed
from dadnet.models import get_dad_net

DEFAULT_METHODS = ["gswd", "kl", "mmd", "l2", "fitnet", "crd"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alignment-objective ablation")
    parser.add_argument("--benchmark", required=True,
                        choices=["dfme_3class", "dfme_4class", "dfme_7class"])
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--testa-index", required=True)
    parser.add_argument("--testb-index", required=True)
    parser.add_argument("--macro-weights", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--alignment-weight", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=TRAINING.max_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--val-ratio", type=float, default=TRAINING.internal_val_ratio)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("ablation_alignment")
    device = get_device(args.device)

    config = get_benchmark(args.benchmark)
    training = TRAINING.__class__(max_epochs=args.epochs, batch_size=args.batch_size)
    warn_if_teacher_missing(args.alignment_weight, args.macro_weights, logger)

    train_samples = read_index(args.train_index)
    testa_samples = read_index(args.testa_index)
    testb_samples = read_index(args.testb_index)

    results: List[dict] = []
    for method in args.methods:
        set_seed(args.seed)
        fit_samples, val_samples = carve_internal_validation(
            train_samples, args.val_ratio, args.seed
        )
        # Canonical full student so the alignment objective is the only variable.
        model = get_dad_net(
            num_classes=config.num_classes,
            micro_model_size=config.student_size,
            macro_model_size=config.teacher_size,
            alignment_stages=config.alignment_stages(),
            macro_weights_path=args.macro_weights,
            alignment_weight=args.alignment_weight,
            num_projections=training.num_projections,
            kd_method=method,
        )
        fit_loader = make_loader(fit_samples, training.batch_size, train=True,
                                 augmentation=config.augmentation, num_workers=args.num_workers)
        val_loader = make_loader(val_samples, training.batch_size, train=False,
                                 num_workers=args.num_workers)
        test_loaders = {
            "testA": make_loader(testa_samples, training.batch_size, train=False,
                                 num_workers=args.num_workers),
            "testB": make_loader(testb_samples, training.batch_size, train=False,
                                 num_workers=args.num_workers),
        }
        outcome = fit_trajectory(model, fit_loader, training, device,
                                 val_loader=val_loader, test_loaders=test_loaders)
        record = {
            "method": method,
            "val_uf1": outcome["best_val"].get("uf1", 0.0),
            "testA_uf1": outcome["testA"]["metrics"]["uf1"],
            "testB_uf1": outcome["testB"]["metrics"]["uf1"],
        }
        results.append(record)
        logger.info("method=%s testA_uf1=%.4f testB_uf1=%.4f",
                    method, record["testA_uf1"], record["testB_uf1"])

    write_json(args.output, {
        "benchmark": args.benchmark,
        "alignment_weight": args.alignment_weight,
        "seed": args.seed,
        "results": results,
    })
    logger.info("results written to %s", args.output)


if __name__ == "__main__":
    main()
