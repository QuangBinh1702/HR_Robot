"""Shared liveness tensor contract for ONNX and RKNN backends."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LivenessResult:
    fake_score: float
    real_score: float
    is_real: bool


def _ensure_bgr(face_crop: np.ndarray) -> np.ndarray:
    if len(face_crop.shape) == 2:
        return cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
    return face_crop


def crop_liveness_face(
    frame: np.ndarray,
    bbox: list[float] | tuple[float, float, float, float],
    scale: float,
) -> np.ndarray:
    """Crop a square face region with reflection padding, matching the reference anti-spoof repo."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    max_dim = max(width, height)
    crop_size = max(1, int(round(max_dim * max(scale, 1.0))))
    center_x = x1 + (width / 2.0)
    center_y = y1 + (height / 2.0)

    crop_x1 = int(round(center_x - (crop_size / 2.0)))
    crop_y1 = int(round(center_y - (crop_size / 2.0)))
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(frame_w, crop_x2)
    src_y2 = min(frame_h, crop_y2)

    top = max(0, -crop_y1)
    left = max(0, -crop_x1)
    bottom = max(0, crop_y2 - frame_h)
    right = max(0, crop_x2 - frame_w)

    crop = frame[src_y1:src_y2, src_x1:src_x2]
    if crop.size == 0:
        raise ValueError("Invalid liveness crop")

    return cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def _resize_with_reflect_padding(rgb_face: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = input_size[1], input_size[0]
    src_h, src_w = rgb_face.shape[:2]
    ratio = min(target_w / max(1, src_w), target_h / max(1, src_h))
    scaled_w = max(1, int(round(src_w * ratio)))
    scaled_h = max(1, int(round(src_h * ratio)))
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(rgb_face, (scaled_w, scaled_h), interpolation=interpolation)

    delta_w = target_w - scaled_w
    delta_h = target_h - scaled_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def _prepare_rgb_face(face_crop: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    bgr = _ensure_bgr(face_crop)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _resize_with_reflect_padding(rgb, input_size)


def _probability_threshold_to_logit(threshold: float) -> float:
    clipped = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    return float(np.log(clipped / (1.0 - clipped)))


def prepare_onnx_liveness_input(
    face_crop: np.ndarray,
    input_size: tuple[int, int],
) -> np.ndarray:
    rgb = _prepare_rgb_face(face_crop, input_size).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def prepare_rknn_liveness_input(
    face_crop: np.ndarray,
    input_size: tuple[int, int],
) -> np.ndarray:
    rgb = _prepare_rgb_face(face_crop, input_size).astype(np.uint8)
    return np.expand_dims(rgb, axis=0)


def decode_binary_liveness_scores(
    output: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float, bool]:
    scores = np.asarray(output, dtype=np.float32).reshape(-1)
    if scores.size != 2:
        raise ValueError(f"Liveness model output must contain exactly 2 scores, got {scores.size}")

    if np.all(scores >= 0.0) and np.isclose(float(scores.sum()), 1.0, atol=1e-4):
        real_score, fake_score = scores.tolist()
        logit_diff = float(np.log(max(real_score, 1e-6)) - np.log(max(fake_score, 1e-6)))
    else:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores)
        real_score, fake_score = probs.tolist()
        logit_diff = float(scores[0] - scores[1])

    return float(fake_score), float(real_score), logit_diff >= _probability_threshold_to_logit(threshold)


def validate_liveness_output_shape(output_shape: list | tuple) -> None:
    last_dim = output_shape[-1]
    if last_dim != 2:
        raise RuntimeError(
            "This anti-spoof model must output exactly 2 classes (real/spoof). "
            f"Got output shape ending with {last_dim!r}."
        )


__all__ = [
    "LivenessResult",
    "crop_liveness_face",
    "decode_binary_liveness_scores",
    "prepare_onnx_liveness_input",
    "prepare_rknn_liveness_input",
    "validate_liveness_output_shape",
]
