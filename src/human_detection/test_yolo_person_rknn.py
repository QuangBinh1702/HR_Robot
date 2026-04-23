"""
Test YOLOv8n RKNN person detection on image/camera.

Usage:
    python src/human_detection/test_yolo_person_rknn.py
    python src/human_detection/test_yolo_person_rknn.py --image path/to/image.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    PERSON_TRIGGER_CONSECUTIVE_FRAMES,
    PERSON_TRIGGER_MISS_FRAMES,
    YOLO_PERSON_MIN_BBOX_HEIGHT,
)
from src.camera_utils import open_camera
from src.human_detection.detect_yolov8_rknn import YOLOv8PersonRKNNDetector
from src.human_detection.person_gate import PersonTemporalGate


def draw_persons(frame, persons: list[dict], trigger: bool, gate_hits: int, gate_misses: int):
    vis = frame.copy()
    color = (0, 255, 255) if trigger else (120, 120, 120)

    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        conf = p.get("score", 0.0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"person {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    gate_text = f"Gate: {'ON' if trigger else 'OFF'} | hit={gate_hits} miss={gate_misses}"
    cv2.putText(vis, gate_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return vis


def run_image(detector: YOLOv8PersonRKNNDetector, gate: PersonTemporalGate, image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image: {image_path}")
        return

    t0 = time.time()
    persons = detector.detect_persons(image)
    dt_ms = (time.time() - t0) * 1000.0

    persons, gate_state = gate.update(persons)

    print(f"Detected persons: {len(persons)} | trigger={gate_state.trigger_active} | {dt_ms:.1f} ms")
    for idx, p in enumerate(persons, start=1):
        print(f"  {idx}. score={p['score']:.3f}, bbox={p['bbox']}")

    vis = draw_persons(image, persons, gate_state.trigger_active, gate_state.consecutive_hits, gate_state.consecutive_misses)
    cv2.imshow("YOLOv8n Person Detection (RKNN)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(detector: YOLOv8PersonRKNNDetector, gate: PersonTemporalGate):
    cap = open_camera()
    if cap is None:
        print("Error: Cannot open camera")
        return

    print(f"[Camera] Started (index={CAMERA_INDEX}, {CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    print("Press 'q' to quit")

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            t0 = time.time()
            raw_persons = detector.detect_persons(frame)
            infer_ms = (time.time() - t0) * 1000.0

            persons, gate_state = gate.update(raw_persons)
            vis = draw_persons(frame, persons, gate_state.trigger_active, gate_state.consecutive_hits, gate_state.consecutive_misses)

            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()

            info = f"FPS: {fps_display} | persons: {len(persons)} | infer: {infer_ms:.1f} ms"
            cv2.putText(vis, info, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            cv2.imshow("YOLOv8n Person Detection (RKNN)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8n RKNN person detector")
    parser.add_argument("--image", type=str, default=None, help="Path to image (omit for camera mode)")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOv8 RKNN model")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--core-mask", type=int, default=None, help="NPU core mask")
    parser.add_argument("--min-height", type=int, default=YOLO_PERSON_MIN_BBOX_HEIGHT, help="Min person bbox height")
    parser.add_argument("--hits", type=int, default=PERSON_TRIGGER_CONSECUTIVE_FRAMES, help="Consecutive hit frames")
    parser.add_argument("--miss", type=int, default=PERSON_TRIGGER_MISS_FRAMES, help="Consecutive miss frames to reset")
    args = parser.parse_args()

    detector = YOLOv8PersonRKNNDetector(
        model_path=args.model,
        conf_threshold=args.threshold,
        core_mask=args.core_mask,
    )
    gate = PersonTemporalGate(
        min_bbox_height=args.min_height,
        required_consecutive_hits=args.hits,
        miss_frames_to_reset=args.miss,
    )

    try:
        if args.image:
            run_image(detector, gate, args.image)
        else:
            run_camera(detector, gate)
    finally:
        detector.release()


if __name__ == "__main__":
    main()
