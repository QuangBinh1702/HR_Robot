"""
Capture MiniFASNet calibration face crops directly from camera.

Example:
    python scripts/capture_minifasnet_calibration.py --count 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LIVENESS_CROP_SCALE, LIVENESS_INPUT_SIZE  # noqa: E402
from src.camera_utils import open_camera  # noqa: E402
from src.face_detection.detect_scrfd import SCRFDDetector  # noqa: E402
from src.rga.rga_ops import expand_and_clip_bbox  # noqa: E402


def release_if_supported(obj) -> None:
    release = getattr(obj, "release", None)
    if callable(release):
        release()


def next_face_index(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0

    max_index = -1
    for path in output_dir.glob("face_*.jpg"):
        try:
            max_index = max(max_index, int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max_index + 1


def save_face_crop(
    frame,
    bbox: list[float] | tuple[float, float, float, float],
    output_dir: Path,
    index: int,
    scale: float,
) -> Path:
    x1, y1, x2, y2 = expand_and_clip_bbox(bbox, frame.shape, scale=scale)
    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, LIVENESS_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"face_{index:04d}.jpg"
    if not cv2.imwrite(str(output_path), resized):
        raise RuntimeError(f"Failed to save crop to {output_path}")
    return output_path


def pick_largest_face(detections: list[dict]) -> dict | None:
    if not detections:
        return None
    return max(
        detections,
        key=lambda det: max(0.0, det["bbox"][2] - det["bbox"][0]) * max(0.0, det["bbox"][3] - det["bbox"][1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture MiniFASNet calibration crops from camera")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "calibration" / f"faces_{LIVENESS_INPUT_SIZE[0]}"),
        help=f"Directory to store {LIVENESS_INPUT_SIZE[0]}x{LIVENESS_INPUT_SIZE[1]} face crops",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of face crops to capture",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.7,
        help="Minimum seconds between saved crops",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=LIVENESS_CROP_SCALE,
        help="Face crop expansion scale before resizing",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    detector = SCRFDDetector()
    cap = open_camera()
    if cap is None:
        release_if_supported(detector)
        raise RuntimeError("Cannot open camera")

    saved_count = 0
    next_index = next_face_index(output_dir)
    last_saved_at = 0.0

    print("=" * 60)
    print("MiniFASNet Calibration Capture")
    print("=" * 60)
    print(f"Output dir : {output_dir}")
    print(f"Target     : {args.count} crops")
    print(f"Interval   : {args.interval:.2f}s")
    print("Controls   : q=quit, s=save now")
    print("=" * 60)

    try:
        while saved_count < args.count:
            ret, frame = cap.read()
            if not ret:
                continue

            detections = detector.detect(frame)
            face = pick_largest_face(detections)
            vis = frame.copy()
            now = time.time()

            if face is not None:
                x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    "Face detected",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if now - last_saved_at >= args.interval:
                    save_face_crop(
                        frame=frame,
                        bbox=face["bbox"],
                        output_dir=output_dir,
                        index=next_index,
                        scale=args.scale,
                    )
                    saved_count += 1
                    next_index += 1
                    last_saved_at = now

            status = f"Saved: {saved_count}/{args.count}"
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("MiniFASNet Calibration Capture", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and face is not None:
                save_face_crop(
                    frame=frame,
                    bbox=face["bbox"],
                    output_dir=output_dir,
                    index=next_index,
                    scale=args.scale,
                )
                saved_count += 1
                next_index += 1
                last_saved_at = now

    finally:
        cap.release()
        cv2.destroyAllWindows()
        release_if_supported(detector)

    print(f"[DONE] Saved {saved_count} crops to {output_dir}")


if __name__ == "__main__":
    main()
