"""Run anti-spoof detection on an image with the shared liveness contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LIVENESS_CROP_SCALE, USE_NPU  # noqa: E402
from src.face_detection.detect_scrfd import SCRFDDetector  # noqa: E402
from src.face_liveness.liveness_contract import crop_liveness_face  # noqa: E402
from src.face_liveness.liveness_minifasnet_onnx import MiniFASNetONNXLiveness  # noqa: E402
from src.face_liveness.liveness_minifasnet_rknn import MiniFASNetRKNNLiveness  # noqa: E402


def build_liveness_model(backend: str):
    if backend == "onnx":
        return MiniFASNetONNXLiveness()
    if backend == "rknn":
        return MiniFASNetRKNNLiveness()
    raise ValueError(f"Unsupported backend: {backend}")


def annotate(image, bbox, status: str, real_score: float, fake_score: float) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 255, 0) if status == "REAL" else (0, 0, 255)
    label = f"{status} real={real_score:.3f} fake={fake_score:.3f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test anti-spoof on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the source image")
    parser.add_argument(
        "--backend",
        choices=["onnx", "rknn"],
        default="rknn" if USE_NPU else "onnx",
        help="Liveness backend to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save annotated output image",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = SCRFDDetector()
    liveness_model = build_liveness_model(args.backend)

    try:
        detections = detector.detect(image)
        results = []
        for index, detection in enumerate(detections):
            face_crop = crop_liveness_face(image, detection["bbox"], scale=LIVENESS_CROP_SCALE)
            prediction = liveness_model.predict(face_crop)
            status = "REAL" if prediction.is_real else "FAKE"
            results.append(
                {
                    "index": index,
                    "bbox": [round(float(v), 2) for v in detection["bbox"]],
                    "status": status,
                    "real_score": round(float(prediction.real_score), 6),
                    "fake_score": round(float(prediction.fake_score), 6),
                }
            )
            annotate(image, detection["bbox"], status, prediction.real_score, prediction.fake_score)
    finally:
        if hasattr(detector, "release"):
            detector.release()
        if hasattr(liveness_model, "release"):
            liveness_model.release()

    print(json.dumps({"backend": args.backend, "results": results}, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        print(f"[OK] Saved annotated output: {output_path}")


if __name__ == "__main__":
    main()
