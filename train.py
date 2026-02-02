# -*- coding: utf-8 -*-
"""
DAD-Net Training Script

This script provides training functionality for the DAD-Net framework,
supporting both single-stream (baseline) and dual-stream (DAD-Net) models.

Usage:
    python train.py --dataset 4dme --data_dir /path/to/data --output_dir ./results

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Handle imports for both package and script execution
try:
    from models import MicroFlowNeXt, DADNet, get_microflownext, get_dad_net
    from data import MicroExpressionDataset, OpticalFlowAugmentation, create_loso_splits
    from losses import DADNetLoss
    from utils import (
        compute_metrics,
        get_cosine_schedule_with_warmup,
        EarlyStopping,
        set_seed,
        count_parameters,
        ConfusionMatrixLogger,
    )
    from configs import get_training_config, get_dataset_config
except ImportError:
    # Add parent directory to path for script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import MicroFlowNeXt, DADNet, get_microflownext, get_dad_net
    from data import MicroExpressionDataset, OpticalFlowAugmentation, create_loso_splits
    from losses import DADNetLoss
    from utils import (
        compute_metrics,
        get_cosine_schedule_with_warmup,
        EarlyStopping,
        set_seed,
        count_parameters,
        ConfusionMatrixLogger,
    )
    from configs import get_training_config, get_dataset_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DAD-Net for Micro-Expression Recognition'
    )

    parser.add_argument('--dataset', type=str, default='4dme',
                        choices=['4dme', 'casme3', 'dfme', 'casme2'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--macro_weights', type=str, default=None,
                        help='Path to pre-trained macro-expression weights')

    parser.add_argument('--model_type', type=str, default='dad_net',
                        choices=['baseline', 'dad_net'],
                        help='Model type to train')
    parser.add_argument('--micro_size', type=str, default='micro',
                        help='Micro-expression branch model size')
    parser.add_argument('--macro_size', type=str, default='ultralight',
                        help='Macro-expression branch model size')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=205,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    parser.add_argument('--alignment_weight', type=float, default=None,
                        help='GSWD alignment loss weight (lambda)')
    parser.add_argument('--num_projections', type=int, default=100,
                        help='Number of GSWD random projections')
    parser.add_argument('--alignment_stages', type=str, default='2,3',
                        help='Comma-separated stage indices for alignment')

    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    is_dual_model: bool = False,
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        device: Device to use.
        scheduler: Optional learning rate scheduler.
        is_dual_model: Whether the model is DAD-Net dual-stream.

    Returns:
        Dictionary containing training metrics.
    """
    model.train()

    total_loss = 0.0
    total_align_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        if is_dual_model:
            preds, logits, _, loss, align_loss = model(inputs, labels)
            total_align_loss += align_loss.item() * inputs.size(0)
        else:
            preds, logits, loss = model(inputs, labels)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    num_samples = len(all_labels)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / num_samples

    if is_dual_model:
        metrics['align_loss'] = total_align_loss / num_samples

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_dual_model: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation data loader.
        device: Device to use.
        is_dual_model: Whether the model is DAD-Net dual-stream.

    Returns:
        Dictionary containing evaluation metrics.
    """
    model.eval()

    total_loss = 0.0
    total_align_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        if is_dual_model:
            preds, logits, _, loss, align_loss = model(inputs, labels)
            total_align_loss += align_loss.item() * inputs.size(0)
        else:
            preds, logits, loss = model(inputs, labels)

        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    num_samples = len(all_labels)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / num_samples

    if is_dual_model:
        metrics['align_loss'] = total_align_loss / num_samples

    return metrics


def train_loso(
    args,
    config: Dict,
    device: torch.device,
):
    """
    Train with Leave-One-Subject-Out cross-validation.

    Args:
        args: Command line arguments.
        config: Training configuration.
        device: Device to use.
    """
    splits = create_loso_splits(args.data_dir, config['dataset']['emotion_map'])
    print(f"Created {len(splits)} LOSO folds")

    all_fold_results = []

    for fold_idx, (train_paths, test_paths, test_subject) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(splits)} - Test Subject: {test_subject}")
        print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")
        print('='*60)

        transform = None
        if config['augmentation']['enable']:
            transform = OpticalFlowAugmentation(
                horizontal_flip_prob=config['augmentation']['horizontal_flip_prob'],
                rotation_range=config['augmentation']['rotation_range'],
                scale_range=config['augmentation']['scale_range'],
            )

        train_dataset = MicroExpressionDataset(
            train_paths,
            emotion_map=config['dataset']['emotion_map'],
            transform=transform,
        )
        test_dataset = MicroExpressionDataset(
            test_paths,
            emotion_map=config['dataset']['emotion_map'],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        is_dual_model = args.model_type == 'dad_net'

        if is_dual_model:
            alignment_stages = {
                int(s): True for s in args.alignment_stages.split(',')
            }
            for i in range(4):
                if i not in alignment_stages:
                    alignment_stages[i] = False

            model = get_dad_net(
                num_classes=config['model']['num_classes'],
                micro_model_size=args.micro_size,
                macro_model_size=args.macro_size,
                alignment_stages=alignment_stages,
                macro_weights_path=args.macro_weights,
                alignment_weight=config['dual_stream']['alignment_weight'],
                num_projections=args.num_projections,
            )
        else:
            model = get_microflownext(
                num_classes=config['model']['num_classes'],
                model_size=args.micro_size,
            )

        model = model.to(device)

        if fold_idx == 0:
            print(f"Model parameters: {count_parameters(model):,}")

        try:
            from adabelief_pytorch import AdaBelief
            optimizer = AdaBelief(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                betas=config['optimizer']['betas'],
                eps=config['optimizer']['eps'],
                weight_decouple=config['optimizer']['weight_decouple'],
                rectify=config['optimizer']['rectify'],
                print_change_log=False,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
            )

        num_training_steps = len(train_loader) * config['training']['epochs']
        num_warmup_steps = len(train_loader) * config['training']['warmup_epochs']

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=config['training']['min_lr_ratio'],
        )

        early_stopping = EarlyStopping(
            patience=config['early_stopping']['patience'],
            mode=config['early_stopping']['mode'],
            restore_best_weights=config['early_stopping']['restore_best_weights'],
        )

        best_uf1 = 0.0
        best_epoch = 0

        for epoch in range(config['training']['epochs']):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device, scheduler, is_dual_model
            )
            test_metrics = evaluate(model, test_loader, device, is_dual_model)

            if test_metrics['uf1'] > best_uf1:
                best_uf1 = test_metrics['uf1']
                best_epoch = epoch + 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}: "
                      f"Train UF1={train_metrics['uf1']:.4f}, "
                      f"Test UF1={test_metrics['uf1']:.4f}, "
                      f"Best={best_uf1:.4f}")

            if early_stopping(test_metrics['uf1'], model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        early_stopping.restore_best_model(model)
        final_metrics = evaluate(model, test_loader, device, is_dual_model)

        fold_result = {
            'fold': fold_idx + 1,
            'subject': test_subject,
            'best_epoch': best_epoch,
            **{f'test_{k}': v for k, v in final_metrics.items()},
        }
        all_fold_results.append(fold_result)

        print(f"Fold {fold_idx + 1} Final: UF1={final_metrics['uf1']:.4f}, "
              f"UAR={final_metrics['uar']:.4f}, ACC={final_metrics['accuracy']:.4f}")

    avg_uf1 = np.mean([r['test_uf1'] for r in all_fold_results])
    avg_uar = np.mean([r['test_uar'] for r in all_fold_results])
    avg_acc = np.mean([r['test_accuracy'] for r in all_fold_results])

    std_uf1 = np.std([r['test_uf1'] for r in all_fold_results])
    std_uar = np.std([r['test_uar'] for r in all_fold_results])

    print(f"\n{'='*60}")
    print("LOSO Cross-Validation Results")
    print('='*60)
    print(f"UF1: {avg_uf1:.4f} +/- {std_uf1:.4f}")
    print(f"UAR: {avg_uar:.4f} +/- {std_uar:.4f}")
    print(f"ACC: {avg_acc:.4f}")

    results_path = os.path.join(args.output_dir, 'loso_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'folds': all_fold_results,
            'summary': {
                'uf1_mean': avg_uf1, 'uf1_std': std_uf1,
                'uar_mean': avg_uar, 'uar_std': std_uar,
                'accuracy_mean': avg_acc,
            },
            'config': config,
        }, f, indent=2)

    print(f"Results saved to {results_path}")


def main():
    """Main training entry point."""
    args = parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = get_training_config(
        dataset_name=args.dataset,
        **{
            'training.batch_size': args.batch_size,
            'training.epochs': args.epochs,
            'training.learning_rate': args.lr,
            'training.weight_decay': args.weight_decay,
        }
    )

    if args.alignment_weight is not None:
        config['dual_stream']['alignment_weight'] = args.alignment_weight

    if args.no_augmentation:
        config['augmentation']['enable'] = False

    print(f"\nTraining Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model_type}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")

    if args.model_type == 'dad_net':
        print(f"  Alignment Weight: {config['dual_stream']['alignment_weight']}")
        print(f"  Alignment Stages: {args.alignment_stages}")

    train_loso(args, config, device)


if __name__ == '__main__':
    main()
