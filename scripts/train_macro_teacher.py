"""Pretrain the frozen macro-expression teacher on the posed CK+ dataset.

The teacher is a MicroFlowNeXt backbone trained on CK+ optical flow with the
three micro-expression categories (negative, positive, surprise). A class-
balanced sampler counters CK+ label imbalance. The resulting checkpoint is
passed to the student training scripts via ``--macro-weights``.

Example:
    python scripts/train_macro_teacher.py \
        --index data/ckplus/index_3class.csv \
        --model-size base \
        --output weights/teacher_base.pth
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from _common import get_device, setup_logger

from dadnet.configs import MACRO_TEACHER, TRAINING
from dadnet.data import FlowAugmentation, MicroFlowDataset, read_index
from dadnet.engine import build_scheduler, count_parameters, set_seed, train_one_epoch
from dadnet.models import get_microflownext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Macro-expression teacher pretraining")
    parser.add_argument("--index", required=True, help="CK+ index CSV: flow_path,label,subject")
    parser.add_argument("--output", required=True, help="teacher checkpoint path")
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--num-classes", type=int, default=MACRO_TEACHER.num_classes)
    parser.add_argument("--epochs", type=int, default=MACRO_TEACHER.max_epochs)
    parser.add_argument("--lr", type=float, default=MACRO_TEACHER.learning_rate)
    parser.add_argument("--batch-size", type=int, default=MACRO_TEACHER.batch_size)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    return parser.parse_args()


def _build_sampler(labels) -> WeightedRandomSampler:
    """Build a sampler whose draw probability is inverse to class frequency."""
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def main() -> None:
    args = parse_args()
    logger = setup_logger("train_macro_teacher")
    device = get_device(args.device)
    set_seed(args.seed)

    samples = read_index(args.index)
    labels = [s.label for s in samples]
    logger.info("CK+ teacher: %d clips, classes=%s", len(samples), sorted(set(labels)))

    dataset = MicroFlowDataset(samples, transform=FlowAugmentation(mode="standard"))
    if args.no_weighted_sampler:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            sampler=_build_sampler(labels),
                            num_workers=args.num_workers, pin_memory=True)

    model = get_microflownext(num_classes=args.num_classes, model_size=args.model_size).to(device)
    logger.info("teacher %s, trainable parameters: %d", args.model_size, count_parameters(model))

    training = TRAINING.__class__(learning_rate=args.lr, max_epochs=args.epochs,
                                  min_lr=MACRO_TEACHER.min_lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=MACRO_TEACHER.weight_decay)
    scheduler = build_scheduler(optimizer, training)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("epoch %d/%d  loss=%.4f", epoch + 1, args.epochs, loss)

    torch.save({"model": model.state_dict(), "model_size": args.model_size,
                "num_classes": args.num_classes}, args.output)
    logger.info("teacher checkpoint written to %s", args.output)


if __name__ == "__main__":
    main()
