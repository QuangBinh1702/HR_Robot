"""MiniFASNet liveness classifier backed by RKNNLite."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from config.settings import (
    LIVENESS_INPUT_SIZE,
    LIVENESS_NPU_CORE_MASK,
    LIVENESS_RKNN_PATH,
    LIVENESS_THRESHOLD,
)
from src.face_liveness.liveness_contract import (
    LivenessResult,
    decode_binary_liveness_scores,
    prepare_rknn_liveness_input,
)


class MiniFASNetRKNNLiveness:
    """Run MiniFASNet face anti-spoofing inference on RK3588 NPU."""

    def __init__(self, model_path: str = None, threshold: float = None, core_mask: int = None):
        self.model_path = model_path or LIVENESS_RKNN_PATH
        self.threshold = threshold or LIVENESS_THRESHOLD
        self.core_mask = core_mask or LIVENESS_NPU_CORE_MASK
        self.input_size = LIVENESS_INPUT_SIZE

        self.rknn = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise ImportError(
                "rknnlite is not available. Install RKNN Toolkit Lite2:\n"
                "  pip install rknn-toolkit-lite2\n"
                "This module only runs on RK3588S boards with NPU."
            )

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Liveness RKNN model not found: {self.model_path}")

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load liveness RKNN model: {self.model_path} (ret={ret})")

        ret = self.rknn.init_runtime(core_mask=self.core_mask)
        if ret != 0:
            raise RuntimeError(
                f"Failed to init liveness RKNN runtime (core_mask={self.core_mask}, ret={ret})"
            )

        print(f"[MiniFASNet-RKNN] Model loaded: {self.model_path}")
        print(f"[MiniFASNet-RKNN] NPU core_mask: {self.core_mask}")

    @staticmethod
    def _decode_scores(output, threshold: float = 0.5) -> tuple[float, float, bool]:
        return decode_binary_liveness_scores(output, threshold=threshold)

    def predict(self, face_crop: np.ndarray) -> LivenessResult:
        input_data = prepare_rknn_liveness_input(face_crop, self.input_size)

        with self._lock:
            outputs = self.rknn.inference(inputs=[input_data])

        fake_score, real_score, predicted_real = self._decode_scores(outputs[0], threshold=self.threshold)
        return LivenessResult(
            fake_score=fake_score,
            real_score=real_score,
            is_real=predicted_real,
        )

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
            print("[MiniFASNet-RKNN] Released NPU resources")
