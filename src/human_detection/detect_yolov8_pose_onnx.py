"""
YOLOv8 pose detector (ONNX Runtime) for local development on CPU/GPU.

Returns person boxes with optional 17 body keypoints (COCO format).
This module is API-compatible with detect_yolov8_pose_rknn.YOLOv8PoseRKNNDetector
via detect_persons(image) -> list[dict].
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
 
from config.settings import (
    YOLOV8N_POSE_ONNX_PATH,
    YOLO_PERSON_CONF_THRESHOLD,
    YOLO_PERSON_NMS_THRESHOLD,
    YOLO_INPUT_SIZE,
    YOLO_POSE_KEYPOINT_THRESHOLD,
)
from src.onnxruntime_cuda import get_onnxruntime_providers


class YOLOv8PoseONNXDetector:
    """YOLOv8 pose ONNX detector optimized for local validation."""

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float | None = None,
        nms_threshold: float | None = None,
        input_size: tuple[int, int] | None = None,
        kpt_threshold: float | None = None,
    ):
        self.model_path = model_path or YOLOV8N_POSE_ONNX_PATH
        self.conf_threshold = conf_threshold or YOLO_PERSON_CONF_THRESHOLD
        self.nms_threshold = nms_threshold or YOLO_PERSON_NMS_THRESHOLD
        self.input_size = input_size or YOLO_INPUT_SIZE
        self.kpt_threshold = kpt_threshold or YOLO_POSE_KEYPOINT_THRESHOLD
        self.session = None
        self.input_name = None
        self._load_model()

    def _load_model(self) -> None:
        import onnxruntime as ort

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLOv8 pose ONNX model not found: {self.model_path}\n"
                "Set YOLOV8N_POSE_ONNX_PATH or copy model to models/yolov8n-pose.onnx"
            )

        providers, backend = get_onnxruntime_providers()

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[YOLOv8-Pose-ONNX] Model loaded: {self.model_path}")
        print(f"[YOLOv8-Pose-ONNX] Backend: {backend}")
        print(f"[YOLOv8-Pose-ONNX] Input size: {self.input_size}")

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        input_w, input_h = self.input_size

        scale = min(input_w / w, input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_data = np.transpose(padded, (2, 0, 1))[None, ...]
        return input_data, scale, h, w

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x - (w / 2.0)
        y1 = y - (h / 2.0)
        x2 = x + (w / 2.0)
        y2 = y + (h / 2.0)
        return np.stack([x1, y1, x2, y2], axis=1)

    def _decode_pose_output(self, output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(output)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected YOLO pose output shape: {arr.shape}")

        if arr.shape[0] <= 80 and arr.shape[1] > arr.shape[0]:
            pred = arr.T
        else:
            pred = arr

        dims = pred.shape[1]
        if dims < 56:
            raise RuntimeError(f"Unsupported YOLO pose output layout: {pred.shape}")

        boxes = pred[:, :4]

        if dims == 56:
            scores = pred[:, 4]
            kpt_flat = pred[:, 5:56]
        else:
            kpt_start = dims - 51
            obj = pred[:, 4]
            cls_scores = pred[:, 5:kpt_start]
            if cls_scores.size > 0:
                scores = obj * np.max(cls_scores, axis=1)
            else:
                scores = obj
            kpt_flat = pred[:, kpt_start:kpt_start + 51]

        return boxes, scores, kpt_flat

    def detect_persons(self, image: np.ndarray) -> list[dict]:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        input_data, scale, orig_h, orig_w = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_data})
        if not outputs:
            return []

        boxes_xywh, scores, kpt_flat = self._decode_pose_output(outputs[0])
        keep_mask = scores > self.conf_threshold
        if not keep_mask.any():
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        scores = scores[keep_mask]
        kpt_flat = kpt_flat[keep_mask]

        if scores.shape[0] > 100:
            top_idx = np.argpartition(scores, -100)[-100:]
            boxes_xywh = boxes_xywh[top_idx]
            scores = scores[top_idx]
            kpt_flat = kpt_flat[top_idx]

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        boxes_xyxy /= scale
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        kpts = kpt_flat.reshape(-1, 17, 3).copy()
        kpts[:, :, 0] /= scale
        kpts[:, :, 1] /= scale
        kpts[:, :, 0] = np.clip(kpts[:, :, 0], 0, orig_w)
        kpts[:, :, 1] = np.clip(kpts[:, :, 1], 0, orig_h)

        boxes_xywh_nms = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = b.tolist()
            boxes_xywh_nms.append([x1, y1, x2 - x1, y2 - y1])

        if len(boxes_xywh_nms) == 1:
            keep = [0]
        else:
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh_nms,
                scores.tolist(),
                self.conf_threshold,
                self.nms_threshold,
            )
            if len(indices) == 0:
                return []
            keep = indices.flatten().tolist()

        persons = []
        for idx in keep:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            keypoints = []
            for xk, yk, ck in kpts[idx].tolist():
                keypoints.append([float(xk), float(yk), float(ck)])

            persons.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(scores[idx]),
                    "class_id": 0,
                    "class_name": "person",
                    "keypoints": keypoints,
                }
            )

        return persons

    def release(self) -> None:
        self.session = None
