"""GhostFaceNet RKNN recognizer."""

from __future__ import annotations

from config.settings import GHOSTFACENET_NPU_CORE_MASK, GHOSTFACENET_RKNN_PATH
from src.face_recognition.recognize_mobilefacenet_rknn import MobileFaceNetRKNNRecognizer


class GhostFaceNetRKNNRecognizer(MobileFaceNetRKNNRecognizer):
    """GhostFaceNet shares the same alignment and embedding flow as MobileFaceNet."""

    def __init__(self, model_path: str = None, threshold: float = None, core_mask: int = None):
        super().__init__(
            model_path=model_path or GHOSTFACENET_RKNN_PATH,
            threshold=threshold,
            core_mask=core_mask or GHOSTFACENET_NPU_CORE_MASK,
        )
