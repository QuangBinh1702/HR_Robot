"""Asynchronous person-gated face pipeline for RK3588S."""

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from config.settings import (
    PIPELINE_QUEUE_SIZE,
    PERSON_DETECT_BACKEND,
    USE_PERSON_GATE,
    USE_PERSON_POSE,
    YOLOV8N_RKNN_PATH,
    YOLOV8N_POSE_ONNX_PATH,
    YOLOV8N_POSE_RKNN_PATH,
    YOLO_PERSON_MIN_BBOX_HEIGHT,
    PERSON_TRIGGER_CONSECUTIVE_FRAMES,
    PERSON_TRIGGER_MISS_FRAMES,
    PERSON_DETECT_INTERVAL,
    FACE_DETECT_INTERVAL,
)
from src.camera_utils import open_camera
from src.human_detection.detect_yolov8_rknn import YOLOv8PersonRKNNDetector
from src.human_detection.detect_yolov8_pose_onnx import YOLOv8PoseONNXDetector
from src.human_detection.detect_yolov8_pose_rknn import YOLOv8PoseRKNNDetector
from src.human_detection.person_gate import PersonTemporalGate


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass
class PersonPacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray
    persons: list[dict]
    trigger: bool
    gate_hits: int
    gate_misses: int


@dataclass
class PipelineSnapshot:
    frame_id: int = -1
    frame: np.ndarray = None
    person_boxes: list[dict] = None
    trigger: bool = False
    gate_hits: int = 0
    gate_misses: int = 0
    face_results: list[dict] = None


class AsyncPersonGatedPipeline:
    """Queue-based async pipeline: camera -> YOLO person gate -> face analysis."""

    def __init__(self, face_pipeline, attendance_manager=None, on_status_update=None):
        self.face_pipeline = face_pipeline
        self.attendance = attendance_manager
        self.on_status_update = on_status_update

        self.person_detector = None
        self.person_gate = PersonTemporalGate(
            min_bbox_height=YOLO_PERSON_MIN_BBOX_HEIGHT,
            required_consecutive_hits=PERSON_TRIGGER_CONSECUTIVE_FRAMES,
            miss_frames_to_reset=PERSON_TRIGGER_MISS_FRAMES,
        )

        self.frame_q: queue.Queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        self.person_q: queue.Queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)

        self.stop_event = threading.Event()
        self.snapshot_lock = threading.Lock()
        self.snapshot = PipelineSnapshot(person_boxes=[], face_results=[])

        self._threads = []

    def _put_latest(self, q: queue.Queue, item):
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put_nowait(item)

    def _camera_worker(self):
        cap = open_camera()
        if cap is None:
            print("ERROR: Cannot open camera")
            self.stop_event.set()
            return

        frame_id = 0
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                packet = FramePacket(frame_id=frame_id, timestamp=time.time(), frame=frame)
                frame_id += 1
                try:
                    self._put_latest(self.frame_q, packet)
                except queue.Full:
                    pass
        finally:
            cap.release()

    def _person_worker(self):
        detect_interval = max(1, PERSON_DETECT_INTERVAL)
        last_persons = []
        last_trigger = True
        last_gate_hits = 0
        last_gate_misses = 0

        while not self.stop_event.is_set():
            try:
                frame_packet = self.frame_q.get(timeout=0.2)
            except queue.Empty:
                continue

            persons = []
            trigger = True
            gate_hits = 0
            gate_misses = 0
            if USE_PERSON_GATE and self.person_detector is not None:
                should_detect = (frame_packet.frame_id % detect_interval) == 0
                if should_detect:
                    persons = self.person_detector.detect_persons(frame_packet.frame)
                    filtered, gate_state = self.person_gate.update(persons)
                    persons = filtered
                    trigger = gate_state.trigger_active
                    gate_hits = gate_state.consecutive_hits
                    gate_misses = gate_state.consecutive_misses

                    last_persons = persons
                    last_trigger = trigger
                    last_gate_hits = gate_hits
                    last_gate_misses = gate_misses
                else:
                    persons = last_persons
                    trigger = last_trigger
                    gate_hits = last_gate_hits
                    gate_misses = last_gate_misses

            packet = PersonPacket(
                frame_id=frame_packet.frame_id,
                timestamp=frame_packet.timestamp,
                frame=frame_packet.frame,
                persons=persons,
                trigger=trigger,
                gate_hits=gate_hits,
                gate_misses=gate_misses,
            )
            try:
                self._put_latest(self.person_q, packet)
            except queue.Full:
                pass

    def _face_worker(self):
        detect_interval = max(1, FACE_DETECT_INTERVAL)
        last_face_results = []

        while not self.stop_event.is_set():
            try:
                person_packet = self.person_q.get(timeout=0.2)
            except queue.Empty:
                continue

            face_results = []
            summary = None
            if person_packet.trigger:
                should_detect = (person_packet.frame_id % detect_interval) == 0
                if should_detect:
                    face_results = self.face_pipeline.detect_faces(person_packet.frame)
                    last_face_results = face_results
                else:
                    face_results = last_face_results

            if self.attendance is not None:
                summary = self.attendance.process_results(face_results)

            if self.on_status_update and summary is not None:
                summary = dict(summary)
                summary["person_trigger"] = person_packet.trigger
                summary["person_count"] = len(person_packet.persons)
                summary["person_boxes"] = person_packet.persons
                self.on_status_update(summary)

            with self.snapshot_lock:
                self.snapshot = PipelineSnapshot(
                    frame_id=person_packet.frame_id,
                    frame=person_packet.frame,
                    person_boxes=person_packet.persons,
                    trigger=person_packet.trigger,
                    gate_hits=person_packet.gate_hits,
                    gate_misses=person_packet.gate_misses,
                    face_results=face_results,
                )

    def start(self):
        if USE_PERSON_GATE:
            try:
                if USE_PERSON_POSE:
                    self.person_detector = self._create_pose_detector()
                    print(f"[PersonGate] Mode: pose skeleton ({self._person_pose_backend_name()})")
                else:
                    self.person_detector = YOLOv8PersonRKNNDetector(model_path=YOLOV8N_RKNN_PATH)
                    print("[PersonGate] Mode: person bbox")
            except Exception as e:
                print(f"[PersonGate] Failed to init YOLOv8 RKNN detector: {e}")
                print("[PersonGate] Fallback: run face stages without person gating")
                self.person_detector = None

        self.stop_event.clear()
        self._threads = [
            threading.Thread(target=self._camera_worker, daemon=True, name="camera-worker"),
            threading.Thread(target=self._person_worker, daemon=True, name="person-worker"),
            threading.Thread(target=self._face_worker, daemon=True, name="face-worker"),
        ]
        for t in self._threads:
            t.start()

    def _person_pose_backend_name(self) -> str:
        if PERSON_DETECT_BACKEND in {"onnx", "rknn"}:
            return PERSON_DETECT_BACKEND
        return "auto"

    def _create_pose_detector(self):
        backend = PERSON_DETECT_BACKEND

        if backend == "onnx":
            return YOLOv8PoseONNXDetector(model_path=YOLOV8N_POSE_ONNX_PATH)

        if backend == "rknn":
            return YOLOv8PoseRKNNDetector(model_path=YOLOV8N_POSE_RKNN_PATH)

        try:
            return YOLOv8PoseRKNNDetector(model_path=YOLOV8N_POSE_RKNN_PATH)
        except Exception as rknn_error:
            print(f"[PersonGate] RKNN pose unavailable: {rknn_error}")
            return YOLOv8PoseONNXDetector(model_path=YOLOV8N_POSE_ONNX_PATH)

    def get_latest_snapshot(self) -> PipelineSnapshot:
        with self.snapshot_lock:
            return self.snapshot

    def stop(self):
        self.stop_event.set()
        for t in self._threads:
            t.join(timeout=1.0)

        if self.person_detector is not None:
            self.person_detector.release()


def draw_person_boxes(frame, persons: list[dict], trigger: bool):
    vis = frame.copy()
    confidence_threshold = 0.35
    skeleton_color = (0, 230, 255) if trigger else (150, 150, 150)
    joint_color = (0, 255, 170) if trigger else (165, 165, 165)
    skeleton_edges = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
        (2, 4), (0, 5), (0, 6),
    ]

    for p in persons:
        keypoints = p.get("keypoints") or []
        if len(keypoints) < 17:
            continue

        overlay = vis.copy()
        for edge in skeleton_edges:
            i, j = edge
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci > confidence_threshold and cj > confidence_threshold:
                cv2.line(
                    overlay,
                    (int(xi), int(yi)),
                    (int(xj), int(yj)),
                    skeleton_color,
                    3,
                    lineType=cv2.LINE_AA,
                )

        vis = cv2.addWeighted(overlay, 0.88, vis, 0.12, 0)

        for xk, yk, ck in keypoints:
            if ck > confidence_threshold:
                radius = 3 if ck < 0.65 else 4
                cv2.circle(
                    vis,
                    (int(xk), int(yk)),
                    radius,
                    joint_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    vis,
                    (int(xk), int(yk)),
                    radius + 1,
                    (24, 24, 24),
                    1,
                    lineType=cv2.LINE_AA,
                )
    return vis
