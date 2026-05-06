"""MiniFASNet liveness classifier backed by ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config.settings import LIVENESS_INPUT_SIZE, LIVENESS_ONNX_PATH, LIVENESS_THRESHOLD
from src.face_liveness.liveness_contract import (
    LivenessResult,
    decode_binary_liveness_scores,
    prepare_onnx_liveness_input,
    validate_liveness_output_shape,
)
from src.onnxruntime_cuda import get_onnxruntime_providers


class MiniFASNetONNXLiveness:
    """Run MiniFASNet face anti-spoofing inference with ONNX Runtime."""

    def __init__(self, model_path: str = None, threshold: float = None):
        self.model_path = model_path or LIVENESS_ONNX_PATH
        self.threshold = threshold or LIVENESS_THRESHOLD
        self.input_size = LIVENESS_INPUT_SIZE

        self.session = None
        self.input_name = None
        self._load_model()

    def _load_model(self):
        import onnxruntime as ort

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Liveness ONNX model not found: {self.model_path}")

        providers, provider_label = get_onnxruntime_providers()
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._validate_model_signature()

        print(f"[MiniFASNet-ONNX] Model loaded: {self.model_path}")
        print(f"[MiniFASNet-ONNX] Provider: {provider_label}")

    def _validate_model_signature(self) -> None:
        validate_liveness_output_shape(self.session.get_outputs()[0].shape)

    @staticmethod
    def _decode_scores(output, threshold: float = 0.5) -> tuple[float, float, bool]:
        return decode_binary_liveness_scores(output, threshold=threshold)

    def predict(self, face_crop: np.ndarray) -> LivenessResult:
        input_data = prepare_onnx_liveness_input(face_crop, self.input_size)
        outputs = self.session.run(None, {self.input_name: input_data})
        fake_score, real_score, predicted_real = self._decode_scores(outputs[0], threshold=self.threshold)
        return LivenessResult(
            fake_score=fake_score,
            real_score=real_score,
            is_real=predicted_real,
        )
