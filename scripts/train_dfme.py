"""Train DAD-Net on the DFME official split with internal-validation selection.

For each seed, a subject-disjoint validation partition is carved from the
official training set; the best epoch is selected on that validation partition,
and test A and test B are each evaluated once. Metrics are reported as the
mean and standard deviation across seeds.

Example:
    python scripts/train_dfme.py --benchmark dfme_3class \
        --train-index data/dfme/train_3class.csv \
        --testa-index data/dfme/testA_3class.csv \
        --testb-index data/dfme/testB_3class.csv \
        --output outputs/dfme_3class.json
"""

from __future__ import annotations

import argparse
from statistics import mean, pstdev
from typing import Dict, List

from _common import get_device, make_loader, setup_logger, warn_if_teacher_missing, write_json

from dadnet.builder import build_dadnet
from dadnet.configs import TRAINING, get_benchmark
from dadnet.data import carve_internal_validation, read_index
from dadnet.engine import count_parameters, fit_trajectory, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DFME official-split training for DAD-Net")
    parser.add_argument("--benchmark", required=True,
                        choices=["dfme_3class", "dfme_4class", "dfme_7class"])
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--testa-index", required=True)
    parser.add_argument("--testb-index", required=True)
    parser.add_argument("--macro-weights", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=TRAINING.max_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--val-ratio", type=float, default=TRAINING.internal_val_ratio)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAINING.dfme_seeds))
    parser.add_argument("--save-student", default=None,
                        help="save the first seed's student backbone to this path")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _summarize(values: List[float]) -> Dict[str, float]:
    """Return the mean and population standard deviation of a list of values."""
    return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    args = parse_args()
    logger = setup_logger("train_dfme")
    device = get_device(args.device)

    config = get_benchmark(args.benchmark)
    training = TRAINING.__class__(max_epochs=args.epochs, batch_size=args.batch_size)
    warn_if_teacher_missing(config.alignment_weight, args.macro_weights, logger)

    train_samples = read_index(args.train_index)
    testa_samples = read_index(args.testa_index)
    testb_samples = read_index(args.testb_index)
    logger.info(
        "%s: train=%d testA=%d testB=%d student=%s stage=%s lambda=%.3g seeds=%s",
        args.benchmark, len(train_samples), len(testa_samples), len(testb_samples),
        config.student_size, config.stage_mask, config.alignment_weight, args.seeds,
    )

    per_seed = []
    for seed in args.seeds:
        set_seed(seed)
        fit_samples, val_samples = carve_internal_validation(
            train_samples, args.val_ratio, seed
        )
        model = build_dadnet(config, macro_weights_path=args.macro_weights,
                             num_projections=training.num_projections)
        if not per_seed:
            logger.info("trainable parameters: %d", count_parameters(model))

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

        result = fit_trajectory(model, fit_loader, training, device,
                                val_loader=val_loader, test_loaders=test_loaders)
        if args.save_student and not per_seed:
            import torch
            torch.save({"model": model.micro_branch.state_dict(),
                        "benchmark": args.benchmark}, args.save_student)
            logger.info("saved student backbone to %s", args.save_student)
        record = {
            "seed": seed,
            "val": result["best_val"],
            "testA": result["testA"]["metrics"],
            "testB": result["testB"]["metrics"],
        }
        per_seed.append(record)
        logger.info(
            "seed %d  val_uf1=%.4f  testA_uf1=%.4f  testB_uf1=%.4f",
            seed, record["val"].get("uf1", 0.0),
            record["testA"]["uf1"], record["testB"]["uf1"],
        )

    summary = {
        partition: {
            metric: _summarize([r[partition][metric] for r in per_seed])
            for metric in ("uf1", "uar")
        }
        for partition in ("testA", "testB")
    }
    logger.info(
        "testA UF1 %.4f+/-%.4f  testB UF1 %.4f+/-%.4f",
        summary["testA"]["uf1"]["mean"], summary["testA"]["uf1"]["std"],
        summary["testB"]["uf1"]["mean"], summary["testB"]["uf1"]["std"],
    )

    write_json(args.output, {
        "benchmark": args.benchmark,
        "protocol": "dfme_official_split",
        "seeds": args.seeds,
        "config": vars(config),
        "summary": summary,
        "per_seed": per_seed,
    })
    logger.info("results written to %s", args.output)


if __name__ == "__main__":
    main()
