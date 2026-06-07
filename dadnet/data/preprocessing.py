"""Face alignment and TV-L1 optical-flow extraction from onset/apex frames.

The pipeline detects and crops the face on the onset and apex frames with a
dlib 68-point landmark predictor, then computes a dense TV-L1 optical-flow
field between them. OpenCV is required at runtime; the dlib model file is a
third-party asset downloaded separately (see the README).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional preprocessing dependency
    cv2 = None


def _require_cv2() -> None:
    """Raise a clear error when OpenCV is unavailable."""
    if cv2 is None:
        raise ImportError("opencv-python (and opencv-contrib-python for TV-L1) is required")


def align_and_crop_face(
    image: np.ndarray,
    face_detector,
    shape_predictor,
    target_size: Tuple[int, int] = (224, 224),
    forehead_extension: float = 0.3,
) -> Optional[np.ndarray]:
    """Detect the largest face, extend the crop upward, and resize it.

    Args:
        image: BGR image.
        face_detector: dlib frontal face detector.
        shape_predictor: dlib 68-point landmark predictor.
        target_size: Output crop size.
        forehead_extension: Upward extension as a fraction of face height.

    Returns:
        The cropped face, or ``None`` when no face is found.
    """
    _require_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    faces = face_detector(gray, 1)
    if not faces:
        return None
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    shape = shape_predictor(gray, face)
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    extension = int(forehead_extension * (landmarks[8, 1] - landmarks[19, 1]))
    x_min, y_min = max(0, face.left()), max(0, face.top() - extension)
    x_max, y_max = min(image.shape[1], face.right()), min(image.shape[0], face.bottom())
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return cv2.resize(crop, target_size)


def compute_optical_flow(onset: np.ndarray, apex: np.ndarray, method: str = "tvl1") -> np.ndarray:
    """Compute optical flow between two frames.

    Args:
        onset: Onset frame.
        apex: Apex frame.
        method: ``tvl1`` or ``farneback``.

    Returns:
        Flow field of shape ``(H, W, 2)`` with horizontal and vertical channels.
    """
    _require_cv2()
    gray1 = cv2.cvtColor(onset, cv2.COLOR_BGR2GRAY) if onset.ndim == 3 else onset
    gray2 = cv2.cvtColor(apex, cv2.COLOR_BGR2GRAY) if apex.ndim == 3 else apex
    if method == "tvl1":
        return cv2.optflow.DualTVL1OpticalFlow_create().calc(gray1, gray2, None)
    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
    raise ValueError(f"unknown optical flow method {method!r}")


def process_clip(
    onset_path: str,
    apex_path: str,
    face_detector,
    shape_predictor,
    target_size: Tuple[int, int] = (224, 224),
    method: str = "tvl1",
) -> Optional[np.ndarray]:
    """Align both frames of a clip and return their optical-flow field.

    Returns:
        Flow field of shape ``(H, W, 2)``, or ``None`` when a frame is missing
        or no face is detected.
    """
    _require_cv2()
    onset_img, apex_img = cv2.imread(onset_path), cv2.imread(apex_path)
    if onset_img is None or apex_img is None:
        return None
    onset_face = align_and_crop_face(onset_img, face_detector, shape_predictor, target_size)
    apex_face = align_and_crop_face(apex_img, face_detector, shape_predictor, target_size)
    if onset_face is None or apex_face is None:
        return None
    return compute_optical_flow(onset_face, apex_face, method=method)


__all__ = ["align_and_crop_face", "compute_optical_flow", "process_clip"]
