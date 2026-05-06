"""Parallel liveness-aware pipeline for RK3588S."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from config.settings import (
    FACE_DETECT_INTERVAL,
    LIVENESS_CROP_SCALE,
    LIVENESS_INPUT_SIZE,
    MAX_FACES_PER_FRAME,
    PERSON_DETECT_BACKEND,
    PERSON_DETECT_INTERVAL,
    PERSON_TRIGGER_CONSECUTIVE_FRAMES,
    PERSON_TRIGGER_MISS_FRAMES,
    PIPELINE_QUEUE_SIZE,
    USE_LIVENESS,
    USE_NPU,
    USE_PERSON_GATE,
    USE_PERSON_POSE,
    YOLOV8N_POSE_ONNX_PATH,
    YOLOV8N_POSE_RKNN_PATH,
    YOLOV8N_RKNN_PATH,
    YOLO_PERSON_MIN_BBOX_HEIGHT,
)
from src.camera_utils import open_camera
from src.face_detection.detect_scrfd import SCRFDDetector
from src.face_detection.detect_scrfd_rknn import SCRFDRKNNDetector
from src.face_liveness.liveness_contract import crop_liveness_face
from src.face_liveness.liveness_minifasnet_rknn import LivenessResult, MiniFASNetRKNNLiveness
from src.face_recognition.recognize_ghostfacenet_rknn import GhostFaceNetRKNNRecognizer
from src.human_detection.detect_yolov8_pose_rknn import YOLOv8PoseRKNNDetector
from src.human_detection.detect_yolov8_rknn import YOLOv8PersonRKNNDetector
from src.human_detection.person_gate import PersonTemporalGate
from src.pipeline_async import draw_person_boxes
from src.rga.rga_ops import RGACropResize

try:
    from src.face_liveness.liveness_minifasnet_onnx import MiniFASNetONNXLiveness
except ModuleNotFoundError:
    MiniFASNetONNXLiveness = None

try:
    from src.face_recognition.recognize_arcface import ArcFaceRecognizer
except ModuleNotFoundError:
    ArcFaceRecognizer = None

try:
    from src.human_detection.detect_yolov8_pose_onnx import YOLOv8PoseONNXDetector
except ModuleNotFoundError:
    YOLOv8PoseONNXDetector = None


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass
class DispatchPacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray
    persons: list[dict]
    face_detections: list[dict]
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


def _bbox_area(detection: dict) -> float:
    x1, y1, x2, y2 = detection["bbox"]
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def select_top_faces(detections: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    return sorted(detections, key=_bbox_area, reverse=True)[:limit]


def run_liveness_gate(
    frame: np.ndarray,
    detections: list[dict],
    cropper,
    liveness_model,
    recognize_face,
    max_faces: int,
    liveness_input_size: tuple[int, int],
    liveness_crop_scale: float = 1.0,
) -> list[dict]:
    results = []
    for detection in select_top_faces(detections, limit=max_faces):
        liveness_result = LivenessResult(fake_score=0.0, real_score=1.0, is_real=True)
        if liveness_model is not None:
            face_crop = crop_liveness_face(frame, detection["bbox"], scale=liveness_crop_scale)
            liveness_result = liveness_model.predict(face_crop)

        face_result = {
            "bbox": [int(v) for v in detection["bbox"]],
            "score": float(detection.get("score", 0.0)),
            "keypoints": detection.get("keypoints"),
            "status": "REAL" if liveness_result.is_real else "FAKE",
            "liveness_score": liveness_result.real_score,
            "fake_score": liveness_result.fake_score,
            "name": "Unknown",
            "confidence": 0.0,
            "member_id": None,
        }
        if not liveness_result.is_real:
            results.append(face_result)
            continue

        recognition = recognize_face(frame, detection)
        face_result.update(recognition)
        results.append(face_result)
    return results


class AsyncPipelineV2:
    """Queue-based camera pipeline with parallel detection and serial Core 1 gating."""

    def __init__(self, face_pipeline, attendance_manager=None, on_status_update=None):
        self.face_pipeline = face_pipeline
        self.attendance = attendance_manager
        self.on_status_update = on_status_update

        self.face_detector = self._build_face_detector()
        self.person_detector = None
        self.person_gate = PersonTemporalGate(
            min_bbox_height=YOLO_PERSON_MIN_BBOX_HEIGHT,
            required_consecutive_hits=PERSON_TRIGGER_CONSECUTIVE_FRAMES,
            miss_frames_to_reset=PERSON_TRIGGER_MISS_FRAMES,
        )
        self.liveness_model = self._build_liveness_model()
        self.recognizer = self._build_recognizer()
        self.cropper = RGACropResize()

        self.frame_q: queue.Queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        self.dispatch_q: queue.Queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)

        self.stop_event = threading.Event()
        self.snapshot_lock = threading.Lock()
        self.snapshot = PipelineSnapshot(person_boxes=[], face_results=[])
        self._threads = []
        self._dispatch_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dispatch")

    def _build_face_detector(self):
        detector = getattr(self.face_pipeline.app, "detector", None)
        if detector is not None:
            return detector
        if not USE_NPU:
            return SCRFDDetector()
        return SCRFDRKNNDetector()

    def _build_liveness_model(self):
        if not USE_LIVENESS:
            return None
        if not USE_NPU:
            try:
                if MiniFASNetONNXLiveness is None:
                    raise ModuleNotFoundError(
                        "src.face_liveness.liveness_minifasnet_onnx is missing"
                    )
                return MiniFASNetONNXLiveness()
            except Exception as exc:
                print(f"[PipelineV2] ONNX liveness unavailable: {exc}")
                return None
        try:
            return MiniFASNetRKNNLiveness()
        except Exception as exc:
            print(f"[PipelineV2] Liveness unavailable: {exc}")
            return None

    def _build_recognizer(self):
        if not USE_NPU:
            try:
                if ArcFaceRecognizer is None:
                    raise ModuleNotFoundError(
                        "src.face_recognition.recognize_arcface is missing"
                    )
                return ArcFaceRecognizer(threshold=self.face_pipeline.threshold)
            except Exception as exc:
                print(f"[PipelineV2] ArcFace ONNX unavailable: {exc}")
                return None
        try:
            return GhostFaceNetRKNNRecognizer(threshold=self.face_pipeline.threshold)
        except Exception as ghost_exc:
            print(f"[PipelineV2] GhostFaceNet unavailable: {ghost_exc}")
            recognizer = getattr(self.face_pipeline.app, "recognizer", None)
            if recognizer is not None:
                return recognizer
            print("[PipelineV2] No NPU recognizer available, face matching disabled")
            return None

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
                self._put_latest(self.frame_q, packet)
        finally:
            cap.release()

    def _dispatch_worker(self):
        person_interval = max(1, PERSON_DETECT_INTERVAL)
        face_interval = max(1, FACE_DETECT_INTERVAL)

        last_persons = []
        last_faces = []
        last_trigger = True
        last_gate_hits = 0
        last_gate_misses = 0

        while not self.stop_event.is_set():
            try:
                frame_packet = self.frame_q.get(timeout=0.2)
            except queue.Empty:
                continue

            should_person_detect = USE_PERSON_GATE and self.person_detector is not None
            should_person_detect = should_person_detect and (frame_packet.frame_id % person_interval) == 0
            should_face_detect = (frame_packet.frame_id % face_interval) == 0

            future_persons = None
            future_faces = None
            if should_person_detect:
                future_persons = self._dispatch_pool.submit(
                    self.person_detector.detect_persons,
                    frame_packet.frame,
                )
            if should_face_detect:
                future_faces = self._dispatch_pool.submit(
                    self.face_detector.detect,
                    frame_packet.frame,
                )

            persons = last_persons
            trigger = last_trigger
            gate_hits = last_gate_hits
            gate_misses = last_gate_misses
            if future_persons is not None:
                persons = future_persons.result()
                filtered, gate_state = self.person_gate.update(persons)
                persons = filtered
                trigger = gate_state.trigger_active
                gate_hits = gate_state.consecutive_hits
                gate_misses = gate_state.consecutive_misses
                last_persons = persons
                last_trigger = trigger
                last_gate_hits = gate_hits
                last_gate_misses = gate_misses

            if future_faces is not None:
                last_faces = future_faces.result()
            face_detections = last_faces

            packet = DispatchPacket(
                frame_id=frame_packet.frame_id,
                timestamp=frame_packet.timestamp,
                frame=frame_packet.frame,
                persons=persons,
                face_detections=face_detections,
                trigger=trigger,
                gate_hits=gate_hits,
                gate_misses=gate_misses,
            )
            self._put_latest(self.dispatch_q, packet)

    def _recognize_face(self, frame: np.ndarray, detection: dict) -> dict:
        if self.recognizer is None:
            return {"name": "Unknown", "confidence": 0.0, "member_id": None}

        keypoints = detection.get("keypoints")
        aligned = None
        if keypoints and hasattr(self.recognizer, "align_face"):
            aligned = self.recognizer.align_face(frame, keypoints)
        if aligned is None:
            aligned = self.cropper.crop_resize(
                frame,
                detection["bbox"],
                getattr(self.recognizer, "ARCFACE_INPUT_SIZE", (112, 112)),
            )

        embedding = self.recognizer.extract_embedding(aligned)
        name, confidence, member_id = self.face_pipeline.cache.match(
            embedding,
            self.face_pipeline.threshold,
        )
        return {
            "name": name,
            "confidence": confidence,
            "member_id": member_id,
        }

    def _core1_worker(self):
        while not self.stop_event.is_set():
            try:
                packet = self.dispatch_q.get(timeout=0.2)
            except queue.Empty:
                continue

            face_results = []
            if packet.trigger and packet.face_detections:
                face_results = run_liveness_gate(
                    frame=packet.frame,
                    detections=packet.face_detections,
                    cropper=self.cropper,
                    liveness_model=self.liveness_model,
                    recognize_face=self._recognize_face,
                    max_faces=MAX_FACES_PER_FRAME,
                    liveness_input_size=LIVENESS_INPUT_SIZE,
                    liveness_crop_scale=LIVENESS_CROP_SCALE,
                )

            summary = None
            if self.attendance is not None:
                summary = self.attendance.process_results(face_results)

            if self.on_status_update and summary is not None:
                summary = dict(summary)
                summary["person_trigger"] = packet.trigger
                summary["person_count"] = len(packet.persons)
                summary["person_boxes"] = packet.persons
                self.on_status_update(summary)

            with self.snapshot_lock:
                self.snapshot = PipelineSnapshot(
                    frame_id=packet.frame_id,
                    frame=packet.frame,
                    person_boxes=packet.persons,
                    trigger=packet.trigger,
                    gate_hits=packet.gate_hits,
                    gate_misses=packet.gate_misses,
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
            except Exception as exc:
                print(f"[PersonGate] Failed to init YOLO detector: {exc}")
                print("[PersonGate] Fallback: run face stages without person gating")
                self.person_detector = None

        self.stop_event.clear()
        self._threads = [
            threading.Thread(target=self._camera_worker, daemon=True, name="camera-worker"),
            threading.Thread(target=self._dispatch_worker, daemon=True, name="dispatch-worker"),
            threading.Thread(target=self._core1_worker, daemon=True, name="core1-worker"),
        ]
        for thread in self._threads:
            thread.start()

    def _person_pose_backend_name(self) -> str:
        if PERSON_DETECT_BACKEND in {"onnx", "rknn"}:
            return PERSON_DETECT_BACKEND
        return "auto"

    def _create_pose_detector(self):
        if PERSON_DETECT_BACKEND == "onnx":
            if YOLOv8PoseONNXDetector is None:
                raise ModuleNotFoundError(
                    "src.human_detection.detect_yolov8_pose_onnx is missing"
                )
            return YOLOv8PoseONNXDetector(model_path=YOLOV8N_POSE_ONNX_PATH)
        if PERSON_DETECT_BACKEND == "rknn":
            return YOLOv8PoseRKNNDetector(model_path=YOLOV8N_POSE_RKNN_PATH)
        try:
            return YOLOv8PoseRKNNDetector(model_path=YOLOV8N_POSE_RKNN_PATH)
        except Exception as rknn_error:
            print(f"[PersonGate] RKNN pose unavailable: {rknn_error}")
            if YOLOv8PoseONNXDetector is None:
                raise ModuleNotFoundError(
                    "src.human_detection.detect_yolov8_pose_onnx is missing"
                )
            return YOLOv8PoseONNXDetector(model_path=YOLOV8N_POSE_ONNX_PATH)

    def get_latest_snapshot(self) -> PipelineSnapshot:
        with self.snapshot_lock:
            return self.snapshot

    def stop(self):
        self.stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)

        self._dispatch_pool.shutdown(wait=False, cancel_futures=True)

        for component in [
            self.person_detector,
            self.liveness_model,
            self.recognizer,
        ]:
            if component is not None and hasattr(component, "release"):
                component.release()


__all__ = [
    "AsyncPipelineV2",
    "DispatchPacket",
    "FramePacket",
    "PipelineSnapshot",
    "draw_person_boxes",
    "run_liveness_gate",
    "select_top_faces",
]
