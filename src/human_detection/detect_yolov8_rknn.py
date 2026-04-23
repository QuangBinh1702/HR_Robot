"""
YOLOv8n person detector (RKNNLite) for RK3588S.

Detects only class 0 (person) and returns filtered person boxes.
"""

import threading
from pathlib import Path

import cv2
import numpy as np

from config.settings import (
    YOLOV8N_RKNN_PATH,
    YOLO_PERSON_CONF_THRESHOLD,
    YOLO_PERSON_NMS_THRESHOLD,
    YOLO_INPUT_SIZE,
    YOLO_NPU_CORE_MASK,
)


class YOLOv8PersonRKNNDetector:
    """YOLOv8n RKNN detector optimized for person-only inference."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = None,
        nms_threshold: float = None,
        input_size: tuple[int, int] = None,
        core_mask: int = None,
    ):
        self.model_path = model_path or YOLOV8N_RKNN_PATH
        self.conf_threshold = conf_threshold or YOLO_PERSON_CONF_THRESHOLD
        self.nms_threshold = nms_threshold or YOLO_PERSON_NMS_THRESHOLD
        self.input_size = input_size or YOLO_INPUT_SIZE
        self.core_mask = core_mask or YOLO_NPU_CORE_MASK

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
            raise FileNotFoundError(
                f"YOLOv8 RKNN model not found: {self.model_path}\n"
                "Convert and copy YOLOv8n model to models/yolov8n.rknn"
            )

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load YOLOv8 RKNN model: {self.model_path} (ret={ret})")

        ret = self.rknn.init_runtime(core_mask=self.core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init YOLOv8 RKNN runtime (core_mask={self.core_mask}, ret={ret})")

        print(f"[YOLOv8-RKNN] Model loaded: {self.model_path}")
        print(f"[YOLOv8-RKNN] NPU core_mask: {self.core_mask}")
        print(f"[YOLOv8-RKNN] Input size: {self.input_size}")

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

    def _decode_yolov8_output(self, output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(output)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        # Expected YOLOv8 export shape: (84, N) or (N, 84)
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected YOLO output shape: {arr.shape}")

        if arr.shape[0] == 84:
            pred = arr.T
        elif arr.shape[1] == 84:
            pred = arr
        else:
            raise RuntimeError(f"Unsupported YOLO output layout: {arr.shape}")

        boxes = pred[:, :4]
        class_scores = pred[:, 4:]

        person_scores = class_scores[:, self.PERSON_CLASS_ID]
        cls_ids = np.full(person_scores.shape, self.PERSON_CLASS_ID, dtype=np.int32)
        return boxes, person_scores, cls_ids

    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        # boxes are normalized in input-space pixels after RKNN export for YOLOv8
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = x - (w / 2.0)
        y1 = y - (h / 2.0)
        x2 = x + (w / 2.0)
        y2 = y + (h / 2.0)
        return np.stack([x1, y1, x2, y2], axis=1)

    def detect_persons(self, image: np.ndarray) -> list[dict]:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        input_data, scale, orig_h, orig_w = self._preprocess(image)

        with self._lock:
            outputs = self.rknn.inference(inputs=[input_data])

        if not outputs:
            return []

        boxes_xywh, person_scores, cls_ids = self._decode_yolov8_output(outputs[0])

        keep_mask = person_scores > self.conf_threshold
        if not keep_mask.any():
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        person_scores = person_scores[keep_mask]
        cls_ids = cls_ids[keep_mask]

        # Keep highest-confidence candidates before NMS to cut CPU overhead.
        if person_scores.shape[0] > 200:
            top_idx = np.argpartition(person_scores, -200)[-200:]
            boxes_xywh = boxes_xywh[top_idx]
            person_scores = person_scores[top_idx]
            cls_ids = cls_ids[top_idx]

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        boxes_xyxy /= scale

        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        boxes_xywh_nms = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = b.tolist()
            boxes_xywh_nms.append([x1, y1, x2 - x1, y2 - y1])

        if len(boxes_xywh_nms) == 1:
            x1, y1, x2, y2 = boxes_xyxy[0]
            return [
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(person_scores[0]),
                    "class_id": int(cls_ids[0]),
                    "class_name": "person",
                }
            ]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh_nms,
            person_scores.tolist(),
            self.conf_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        persons = []
        for idx in indices.flatten().tolist():
            x1, y1, x2, y2 = boxes_xyxy[idx]
            persons.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(person_scores[idx]),
                    "class_id": int(cls_ids[idx]),
                    "class_name": "person",
                }
            )

        return persons

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
            print("[YOLOv8-RKNN] Released NPU resources")
