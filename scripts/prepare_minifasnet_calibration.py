"""
Prepare MiniFASNet calibration crops from a directory of source images.

Example:
    python scripts/prepare_minifasnet_calibration.py \
        --input-dir data/calibration/source_images \
        --output-dir data/calibration/faces_80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LIVENESS_CROP_SCALE  # noqa: E402
from config.settings import LIVENESS_INPUT_SIZE  # noqa: E402
from src.face_detection.detect_scrfd import SCRFDDetector  # noqa: E402
from src.rga.rga_ops import expand_and_clip_bbox  # noqa: E402


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def pick_largest_face(detections: list[dict]) -> dict | None:
    if not detections:
        return None
    return max(
        detections,
        key=lambda det: max(0.0, det["bbox"][2] - det["bbox"][0]) * max(0.0, det["bbox"][3] - det["bbox"][1]),
    )


def process_image(
    image_path: Path,
    detector,
    output_dir: Path,
    index: int,
    scale: float,
) -> Path | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    face = pick_largest_face(detector.detect(image))
    if face is None:
        return None

    x1, y1, x2, y2 = expand_and_clip_bbox(face["bbox"], image.shape, scale=scale)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, LIVENESS_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"face_{index:04d}.jpg"
    if not cv2.imwrite(str(output_path), resized):
        return None
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MiniFASNet calibration face crops")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "calibration" / f"faces_{LIVENESS_INPUT_SIZE[0]}"),
        help=f"Directory to write {LIVENESS_INPUT_SIZE[0]}x{LIVENESS_INPUT_SIZE[1]} face crops",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of crops to export",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=LIVENESS_CROP_SCALE,
        help="Face crop expansion scale before resizing",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")

    detector = SCRFDDetector()
    exported = 0
    skipped = 0

    try:
        for image_path in image_paths:
            if exported >= args.limit:
                break
            output_path = process_image(
                image_path=image_path,
                detector=detector,
                output_dir=output_dir,
                index=exported,
                scale=args.scale,
            )
            if output_path is None:
                skipped += 1
                continue
            exported += 1
            print(f"[OK] {image_path.name} -> {output_path.name}")
    finally:
        if hasattr(detector, "release"):
            detector.release()

    print(f"[DONE] Exported {exported} crops to {output_dir}")
    print(f"[DONE] Skipped {skipped} images")


if __name__ == "__main__":
    main()
