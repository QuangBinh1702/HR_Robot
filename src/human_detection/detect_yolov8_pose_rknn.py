"""
YOLOv8n-pose person detector (RKNNLite) for RK3588S.

Returns person boxes with optional 17 body keypoints (COCO format).
This module is API-compatible with detect_yolov8_rknn.YOLOv8PersonRKNNDetector
via detect_persons(image) -> list[dict].
"""

import threading
from pathlib import Path

import cv2
import numpy as np

from config.settings import (
    YOLOV8N_POSE_RKNN_PATH,
    YOLO_PERSON_CONF_THRESHOLD,
    YOLO_PERSON_NMS_THRESHOLD,
    YOLO_INPUT_SIZE,
    YOLO_NPU_CORE_MASK,
    YOLO_POSE_KEYPOINT_THRESHOLD,
)


class YOLOv8PoseRKNNDetector:
    """YOLOv8 pose RKNN detector optimized for person skeleton inference."""

    # COCO-17 skeleton connections
    SKELETON_EDGES = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
    ]

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = None,
        nms_threshold: float = None,
        input_size: tuple[int, int] = None,
        core_mask: int = None,
        kpt_threshold: float = None,
    ):
        self.model_path = model_path or YOLOV8N_POSE_RKNN_PATH
        self.conf_threshold = conf_threshold or YOLO_PERSON_CONF_THRESHOLD
        self.nms_threshold = nms_threshold or YOLO_PERSON_NMS_THRESHOLD
        self.input_size = input_size or YOLO_INPUT_SIZE
        self.core_mask = core_mask or YOLO_NPU_CORE_MASK
        self.kpt_threshold = kpt_threshold or YOLO_POSE_KEYPOINT_THRESHOLD

        self.rknn = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        try:
            from rknnlite.api import RKNNLite  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "rknnlite is not available. Install RKNN Toolkit Lite2:\n"
                "  pip install rknn-toolkit-lite2\n"
                "This module only runs on RK3588S boards with NPU."
            )

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLOv8 pose RKNN model not found: {self.model_path}\n"
                "Set YOLOV8N_POSE_RKNN_PATH or copy model to models/yolov8n-pose.rknn"
            )

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load YOLOv8 pose RKNN model: {self.model_path} (ret={ret})")

        ret = self.rknn.init_runtime(core_mask=self.core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init YOLOv8 pose RKNN runtime (core_mask={self.core_mask}, ret={ret})")

        print(f"[YOLOv8-Pose-RKNN] Model loaded: {self.model_path}")
        print(f"[YOLOv8-Pose-RKNN] NPU core_mask: {self.core_mask}")
        print(f"[YOLOv8-Pose-RKNN] Input size: {self.input_size}")

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        input_w, input_h = self.input_size

        scale = min(input_w / w, input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(padded, axis=0)
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

        # Accept both (C, N) and (N, C)
        if arr.shape[0] <= 80 and arr.shape[1] > arr.shape[0]:
            pred = arr.T
        else:
            pred = arr

        dims = pred.shape[1]
        if dims < 56:
            raise RuntimeError(f"Unsupported YOLO pose output layout: {pred.shape}")

        boxes = pred[:, :4]

        # Common layouts:
        # 56 dims: [x,y,w,h,score,kpt(51)]
        # >56 dims: [x,y,w,h,obj,cls...,kpt(51)]
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

        with self._lock:
            outputs = self.rknn.inference(inputs=[input_data])

        if not outputs:
            return []

        boxes_xywh, scores, kpt_flat = self._decode_pose_output(outputs[0])

        keep_mask = scores > self.conf_threshold
        if not keep_mask.any():
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        scores = scores[keep_mask]
        kpt_flat = kpt_flat[keep_mask]

        # Cap candidates to limit CPU in NMS and keypoint transform.
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

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
            print("[YOLOv8-Pose-RKNN] Released NPU resources")
