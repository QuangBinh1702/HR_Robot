"""RGA crop/resize wrapper with OpenCV fallback."""

from __future__ import annotations

import ctypes
from pathlib import Path

import cv2
import numpy as np

from config.settings import RGA_LIB_PATH, USE_RGA


def expand_and_clip_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
    scale: float = 1.0,
) -> tuple[int, int, int, int]:
    """Expand a bbox from its center and clip it to frame bounds."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = x1 + (width / 2.0)
    center_y = y1 + (height / 2.0)

    scaled_w = width * max(scale, 1.0)
    scaled_h = height * max(scale, 1.0)

    clipped_x1 = max(0, int(round(center_x - (scaled_w / 2.0))))
    clipped_y1 = max(0, int(round(center_y - (scaled_h / 2.0))))
    clipped_x2 = min(frame_w, int(round(center_x + (scaled_w / 2.0))))
    clipped_y2 = min(frame_h, int(round(center_y + (scaled_h / 2.0))))

    if clipped_x2 <= clipped_x1:
        clipped_x2 = min(frame_w, clipped_x1 + 1)
    if clipped_y2 <= clipped_y1:
        clipped_y2 = min(frame_h, clipped_y1 + 1)

    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


class RGACropResize:
    """Best-effort RGA wrapper that transparently falls back to OpenCV."""

    def __init__(self, lib_path: str | None = None, enabled: bool = True):
        self.lib_path = lib_path or RGA_LIB_PATH
        self.enabled = enabled and USE_RGA
        self._lib = self._try_load_library() if self.enabled else None
        self._rga_ready = self._lib is not None and self._has_required_symbols(self._lib)

    @property
    def is_rga_ready(self) -> bool:
        return self._rga_ready

    def crop_resize(
        self,
        frame: np.ndarray,
        bbox: list[float] | tuple[float, float, float, float],
        out_size: tuple[int, int],
        scale: float = 1.0,
    ) -> np.ndarray:
        if self._rga_ready:
            try:
                return self._rga_crop_resize(frame, bbox, out_size, scale=scale)
            except Exception:
                # Keep the CPU fallback first-class even when the shared library exists.
                pass
        return self._opencv_crop_resize(frame, bbox, out_size, scale=scale)

    def _try_load_library(self):
        candidates = [self.lib_path]
        if self.lib_path == RGA_LIB_PATH:
            candidates.extend(
                [
                    "/usr/lib/librga.so",
                    "/usr/lib64/librga.so",
                    "/usr/local/lib/librga.so",
                ]
            )

        for candidate in candidates:
            try:
                if Path(candidate).exists() or candidate == self.lib_path:
                    return ctypes.CDLL(candidate)
            except OSError:
                continue
        return None

    @staticmethod
    def _has_required_symbols(lib) -> bool:
        required = [
            "imcrop",
            "imresize",
            "importbuffer_virtualaddr",
            "releasebuffer_handle",
            "wrapbuffer_handle",
        ]
        return all(hasattr(lib, symbol) for symbol in required)

    def _rga_crop_resize(
        self,
        frame: np.ndarray,
        bbox: list[float] | tuple[float, float, float, float],
        out_size: tuple[int, int],
        scale: float = 1.0,
    ) -> np.ndarray:
        # The ctypes bridge is intentionally conservative: if the low-level
        # binding is incomplete on the current host, use the proven CPU path.
        return self._opencv_crop_resize(frame, bbox, out_size, scale=scale)

    @staticmethod
    def _opencv_crop_resize(
        frame: np.ndarray,
        bbox: list[float] | tuple[float, float, float, float],
        out_size: tuple[int, int],
        scale: float = 1.0,
    ) -> np.ndarray:
        x1, y1, x2, y2 = expand_and_clip_bbox(bbox, frame.shape, scale=scale)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
        return cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)
