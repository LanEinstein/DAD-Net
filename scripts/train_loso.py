"""Train DAD-Net under the MEGC leave-one-subject-out protocol (4DME, CASME3).

Each subject is the test fold exactly once. The student trains on all other
subjects for the full cosine schedule, and UF1/UAR are computed on the
predictions pooled across folds.

Example:
    python scripts/train_loso.py --benchmark 4dme \
        --index /path/to/4dme_3class.csv \
        --macro-weights weights/teacher_base.pth \
        --output outputs/4dme_loso.json
"""

from __future__ import annotations

import argparse

from _common import get_device, make_loader, setup_logger, warn_if_teacher_missing, write_json

from dadnet.builder import build_dadnet
from dadnet.configs import TRAINING, get_benchmark
from dadnet.data import build_loso_folds, read_index
from dadnet.engine import count_parameters, fit_trajectory, pooled_metrics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOSO training for DAD-Net")
    parser.add_argument("--benchmark", required=True, choices=["4dme", "casme3"])
    parser.add_argument("--index", required=True, help="index CSV: flow_path,label,subject")
    parser.add_argument("--macro-weights", default=None, help="pre-trained teacher checkpoint")
    parser.add_argument("--output", required=True, help="results JSON path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=TRAINING.max_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--seed", type=int, default=TRAINING.loso_seed)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("train_loso")
    device = get_device(args.device)
    set_seed(args.seed)

    config = get_benchmark(args.benchmark)
    training = TRAINING.__class__(max_epochs=args.epochs, batch_size=args.batch_size)
    warn_if_teacher_missing(config.alignment_weight, args.macro_weights, logger)

    samples = read_index(args.index)
    folds = build_loso_folds(samples)
    logger.info(
        "%s: %d folds, %d clips, student=%s teacher=%s stage=%s lambda=%.3g aug=%s",
        args.benchmark, len(folds), len(samples), config.student_size,
        config.teacher_size, config.stage_mask, config.alignment_weight, config.augmentation,
    )

    fold_true, fold_pred, fold_records = [], [], []
    for index, (train_samples, test_samples, subject) in enumerate(folds):
        model = build_dadnet(config, macro_weights_path=args.macro_weights,
                             num_projections=training.num_projections)
        if index == 0:
            logger.info("trainable parameters: %d", count_parameters(model))

        train_loader = make_loader(train_samples, training.batch_size, train=True,
                                   augmentation=config.augmentation, num_workers=args.num_workers)
        test_loader = make_loader(test_samples, training.batch_size, train=False,
                                  num_workers=args.num_workers)

        result = fit_trajectory(model, train_loader, training, device,
                                test_loaders={"test": test_loader})
        test = result["test"]
        fold_true.append(test["y_true"])
        fold_pred.append(test["y_pred"])
        fold_records.append({"fold": index + 1, "subject": subject, **test["metrics"]})
        logger.info("fold %d/%d subject=%s uf1=%.4f uar=%.4f",
                    index + 1, len(folds), subject, test["metrics"]["uf1"], test["metrics"]["uar"])

    pooled = pooled_metrics(fold_true, fold_pred)
    logger.info("pooled UF1=%.4f UAR=%.4f ACC=%.4f", pooled["uf1"], pooled["uar"], pooled["accuracy"])

    write_json(args.output, {
        "benchmark": args.benchmark,
        "protocol": "loso",
        "seed": args.seed,
        "config": vars(config),
        "pooled": pooled,
        "folds": fold_records,
    })
    logger.info("results written to %s", args.output)


if __name__ == "__main__":
    main()
