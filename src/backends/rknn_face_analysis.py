"""
RKNN Face Analysis - NPU-accelerated face detection + recognition.
Drop-in replacement for insightface.app.FaceAnalysis on RK3588S.
"""

import sys
from pathlib import Path

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    SCRFD_RKNN_PATH, ARCFACE_RKNN_PATH,
    SCRFD_NPU_CORE_MASK, ARCFACE_NPU_CORE_MASK,
    DETECTION_INPUT_SIZE,
)


@dataclass
class RKNNFace:
    """Face result compatible with InsightFace Face object."""
    bbox: np.ndarray
    det_score: float
    kps: Optional[np.ndarray]
    normed_embedding: Optional[np.ndarray]


class RKNNFaceAnalysis:
    """
    NPU-accelerated face analysis for RK3588S.
    API-compatible with insightface.app.FaceAnalysis.
    """

    def __init__(self, det_model_path=None, rec_model_path=None,
                 det_size=None, det_core_mask=None, rec_core_mask=None):
        from src.face_detection.detect_scrfd_rknn import SCRFDRKNNDetector
        from src.face_recognition.recognize_mobilefacenet_rknn import MobileFaceNetRKNNRecognizer

        self.detector = SCRFDRKNNDetector(
            model_path=det_model_path or SCRFD_RKNN_PATH,
            input_size=det_size or DETECTION_INPUT_SIZE,
            core_mask=det_core_mask or SCRFD_NPU_CORE_MASK,
        )
        self.recognizer = MobileFaceNetRKNNRecognizer(
            model_path=rec_model_path or ARCFACE_RKNN_PATH,
            core_mask=rec_core_mask or ARCFACE_NPU_CORE_MASK,
        )

    def get(self, image: np.ndarray) -> list[RKNNFace]:
        """
        Detect faces and extract embeddings.
        Compatible with insightface FaceAnalysis.get().
        """
        detections = self.detector.detect(image)
        faces = []

        for det in detections:
            kps = det.get('keypoints')
            embedding = None

            if kps is not None and len(kps) >= 5:
                aligned = self.recognizer.align_face(image, kps)
                if aligned is not None:
                    embedding = self.recognizer.extract_embedding(aligned)

            kps_array = np.array(kps, dtype=np.float32) if kps else None

            faces.append(RKNNFace(
                bbox=np.array(det['bbox'], dtype=np.float32),
                det_score=float(det['score']),
                kps=kps_array,
                normed_embedding=embedding,
            ))

        return faces

    def detect(self, image: np.ndarray) -> list[RKNNFace]:
        """Detect faces only (no recognition). Faster for headcount/preview."""
        detections = self.detector.detect(image)
        return [
            RKNNFace(
                bbox=np.array(det['bbox'], dtype=np.float32),
                det_score=float(det['score']),
                kps=np.array(det['keypoints'], dtype=np.float32) if det.get('keypoints') else None,
                normed_embedding=None,
            )
            for det in detections
        ]

    def release(self):
        """Release NPU resources."""
        self.detector.release()
        self.recognizer.release()
