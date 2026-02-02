# -*- coding: utf-8 -*-
"""
Optical Flow Preprocessing for Micro-Expression Recognition

This module provides functions for computing TV-L1 optical flow between
onset and apex frames, along with face detection and alignment utilities.

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple


def align_and_crop_face(
    image: np.ndarray,
    face_detector,
    shape_predictor,
    target_size: Tuple[int, int] = (224, 224),
    forehead_extension: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Detect, align, and crop face from an image.

    Uses dlib face detector and 68-point landmark predictor for alignment.
    Extends the crop region to include the forehead for complete facial coverage.

    Args:
        image: Input BGR image.
        face_detector: dlib face detector instance.
        shape_predictor: dlib shape predictor instance.
        target_size: Output size for cropped face. Default: (224, 224).
        forehead_extension: Extension ratio for forehead coverage. Default: 0.3.

    Returns:
        Cropped and resized face image, or None if no face detected.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    faces = face_detector(gray, 1)
    if len(faces) == 0:
        return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())

    shape = shape_predictor(gray, face)
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    forehead_ext = int(forehead_extension * (landmarks[8, 1] - landmarks[19, 1]))
    x_min = max(0, face.left())
    y_min = max(0, face.top() - forehead_ext)
    x_max = min(image.shape[1], face.right())
    y_max = min(image.shape[0], face.bottom())

    face_img = image[y_min:y_max, x_min:x_max]

    if face_img.size == 0:
        return None

    face_img = cv2.resize(face_img, target_size)

    return face_img


def compute_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    method: str = "tvl1",
) -> np.ndarray:
    """
    Compute optical flow between two frames.

    Args:
        frame1: First frame (onset frame).
        frame2: Second frame (apex frame).
        method: Optical flow algorithm - "tvl1" or "farneback". Default: "tvl1".

    Returns:
        Optical flow array of shape (H, W, 2) containing horizontal (u) and
        vertical (v) flow components.
    """
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
        gray2 = frame2

    if method == "tvl1":
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(gray1, gray2, None)
    elif method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
    else:
        raise ValueError(f"Unknown optical flow method: {method}")

    return flow


def normalize_optical_flow(
    flow: np.ndarray,
    method: str = "zscore",
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """
    Normalize optical flow values.

    Args:
        flow: Optical flow array of shape (H, W, 2) or (2, H, W).
        method: Normalization method - "zscore", "minmax", or "clip". Default: "zscore".
        clip_percentile: Percentile for clipping extreme values. Default: 99.0.

    Returns:
        Normalized optical flow array.
    """
    flow = flow.astype(np.float32)

    if method == "zscore":
        mean = flow.mean()
        std = flow.std()
        flow = (flow - mean) / (std + 1e-6)

    elif method == "minmax":
        flow_min = flow.min()
        flow_max = flow.max()
        flow = (flow - flow_min) / (flow_max - flow_min + 1e-6)

    elif method == "clip":
        clip_val = np.percentile(np.abs(flow), clip_percentile)
        flow = np.clip(flow, -clip_val, clip_val)
        flow = flow / (clip_val + 1e-6)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return flow


def process_video_clip(
    onset_path: str,
    apex_path: str,
    face_detector,
    shape_predictor,
    target_size: Tuple[int, int] = (224, 224),
    flow_method: str = "tvl1",
) -> Optional[np.ndarray]:
    """
    Process a micro-expression video clip to extract optical flow.

    Performs face detection, alignment, and optical flow computation
    between onset and apex frames.

    Args:
        onset_path: Path to onset frame image.
        apex_path: Path to apex frame image.
        face_detector: dlib face detector instance.
        shape_predictor: dlib shape predictor instance.
        target_size: Target size for face crops. Default: (224, 224).
        flow_method: Optical flow algorithm. Default: "tvl1".

    Returns:
        Optical flow array of shape (H, W, 2), or None if processing fails.
    """
    onset_img = cv2.imread(onset_path)
    apex_img = cv2.imread(apex_path)

    if onset_img is None or apex_img is None:
        return None

    onset_face = align_and_crop_face(
        onset_img, face_detector, shape_predictor, target_size
    )
    apex_face = align_and_crop_face(
        apex_img, face_detector, shape_predictor, target_size
    )

    if onset_face is None or apex_face is None:
        return None

    flow = compute_optical_flow(onset_face, apex_face, method=flow_method)

    return flow


def visualize_optical_flow(
    flow: np.ndarray,
    method: str = "hsv",
) -> np.ndarray:
    """
    Convert optical flow to a color visualization.

    Args:
        flow: Optical flow array of shape (H, W, 2).
        method: Visualization method - "hsv" or "arrows". Default: "hsv".

    Returns:
        BGR visualization image.
    """
    if method == "hsv":
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    else:
        raise ValueError(f"Unknown visualization method: {method}")
