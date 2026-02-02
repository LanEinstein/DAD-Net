# -*- coding: utf-8 -*-
"""
DAD-Net Inference Script

This script provides inference functionality for trained DAD-Net models,
supporting both single sample and batch inference.

Usage:
    python inference.py --checkpoint model.pth --input sample.npy

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Optional, List, Tuple

# Handle imports for both package and script execution
try:
    from models import MicroFlowNeXt, DADNet, get_microflownext, get_dad_net
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import MicroFlowNeXt, DADNet, get_microflownext, get_dad_net


EMOTION_LABELS = {
    0: 'negative',
    1: 'positive',
    2: 'surprise',
}


def load_model(
    checkpoint_path: str,
    model_type: str = 'dad_net',
    num_classes: int = 3,
    micro_size: str = 'micro',
    macro_size: str = 'ultralight',
    device: torch.device = None,
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        model_type: Model type ('baseline' or 'dad_net').
        num_classes: Number of output classes.
        micro_size: Micro-expression branch model size.
        macro_size: Macro-expression branch model size.
        device: Device to load model to.

    Returns:
        Loaded model in evaluation mode.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'dad_net':
        model = get_dad_net(
            num_classes=num_classes,
            micro_model_size=micro_size,
            macro_model_size=macro_size,
            macro_weights_path=None,
        )
    else:
        model = get_microflownext(
            num_classes=num_classes,
            model_size=micro_size,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def preprocess_optical_flow(
    flow: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess optical flow for model input.

    Args:
        flow: Optical flow array of shape (H, W, 2) or (2, H, W).
        target_size: Target spatial size.
        normalize: Whether to apply z-score normalization.

    Returns:
        Preprocessed tensor of shape (1, 2, H, W).
    """
    if flow.ndim == 3 and flow.shape[-1] == 2:
        flow = flow.transpose(2, 0, 1)
    elif flow.ndim == 3 and flow.shape[0] == 2:
        pass
    else:
        raise ValueError(f"Unexpected flow shape: {flow.shape}")

    c, h, w = flow.shape
    th, tw = target_size

    if h != th or w != tw:
        resized = np.zeros((c, th, tw), dtype=flow.dtype)
        for i in range(c):
            if h > th:
                start = (h - th) // 2
                channel = flow[i, start:start+th, :]
            else:
                channel = np.pad(flow[i], ((0, th-h), (0, 0)), mode='constant')[:th, :]

            if w > tw:
                start = (w - tw) // 2
                channel = channel[:, start:start+tw]
            else:
                channel = np.pad(channel, ((0, 0), (0, tw-w)), mode='constant')[:, :tw]

            resized[i] = channel
        flow = resized

    tensor = torch.from_numpy(flow.copy()).float()

    if normalize:
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True)
        tensor = (tensor - mean) / (std + 1e-6)

    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    flow: np.ndarray,
    device: torch.device = None,
    return_probs: bool = False,
) -> Tuple:
    """
    Predict emotion for a single optical flow sample.

    Args:
        model: Trained model.
        flow: Optical flow array.
        device: Device for inference.
        return_probs: Whether to return probability distribution.

    Returns:
        Tuple of (predicted_label, confidence) or
        (predicted_label, confidence, probabilities) if return_probs=True.
    """
    if device is None:
        device = next(model.parameters()).device

    input_tensor = preprocess_optical_flow(flow).to(device)

    outputs = model(input_tensor)
    logits = outputs[1]

    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()

    pred_label = EMOTION_LABELS.get(pred_idx, f'class_{pred_idx}')

    if return_probs:
        return pred_label, confidence, probs[0].cpu().numpy()
    return pred_label, confidence


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    flows: List[np.ndarray],
    device: torch.device = None,
    batch_size: int = 32,
) -> List[Tuple[str, float]]:
    """
    Predict emotions for a batch of optical flow samples.

    Args:
        model: Trained model.
        flows: List of optical flow arrays.
        device: Device for inference.
        batch_size: Batch size for inference.

    Returns:
        List of (predicted_label, confidence) tuples.
    """
    if device is None:
        device = next(model.parameters()).device

    results = []

    for i in range(0, len(flows), batch_size):
        batch_flows = flows[i:i + batch_size]
        batch_tensors = torch.cat([
            preprocess_optical_flow(f) for f in batch_flows
        ], dim=0).to(device)

        outputs = model(batch_tensors)
        logits = outputs[1]

        probs = torch.softmax(logits, dim=1)
        pred_indices = torch.argmax(probs, dim=1)
        confidences = probs.gather(1, pred_indices.unsqueeze(1)).squeeze(1)

        for pred_idx, conf in zip(pred_indices.cpu().numpy(), confidences.cpu().numpy()):
            pred_label = EMOTION_LABELS.get(int(pred_idx), f'class_{pred_idx}')
            results.append((pred_label, float(conf)))

    return results


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description='DAD-Net Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input optical flow file (.npy)')
    parser.add_argument('--model_type', type=str, default='dad_net',
                        choices=['baseline', 'dad_net'],
                        help='Model type')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--micro_size', type=str, default='micro',
                        help='Micro-expression branch model size')
    parser.add_argument('--show_probs', action='store_true',
                        help='Show probability distribution')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint,
        model_type=args.model_type,
        num_classes=args.num_classes,
        micro_size=args.micro_size,
        device=device,
    )

    print(f"Loading input from {args.input}...")
    flow = np.load(args.input)

    if args.show_probs:
        pred_label, confidence, probs = predict_single(
            model, flow, device, return_probs=True
        )
        print(f"\nPrediction: {pred_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"\nProbability Distribution:")
        for idx, prob in enumerate(probs):
            label = EMOTION_LABELS.get(idx, f'class_{idx}')
            print(f"  {label}: {prob:.4f}")
    else:
        pred_label, confidence = predict_single(model, flow, device)
        print(f"\nPrediction: {pred_label}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == '__main__':
    main()
