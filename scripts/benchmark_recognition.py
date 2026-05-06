"""
Benchmark recognition backends on a directory of aligned face crops.

This script is intended to run on the RK3588S board for realistic latency
numbers, but it can still be used on Windows/Linux dev machines to validate
that available backends initialize and produce embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    ARCFACE_MODEL_PATH,
    ARCFACE_RKNN_PATH,
    GHOSTFACENET_RKNN_PATH,
)
from src.face_recognition.recognize_arcface import ArcFaceRecognizer  # noqa: E402
from src.face_recognition.recognize_ghostfacenet_rknn import GhostFaceNetRKNNRecognizer  # noqa: E402
from src.face_recognition.recognize_mobilefacenet_rknn import MobileFaceNetRKNNRecognizer  # noqa: E402


def load_images(image_dir: Path, limit: int) -> list[np.ndarray]:
    images = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        images.append(image)
        if len(images) >= limit:
            break
    return images


def benchmark_model(name: str, recognizer, images: list[np.ndarray], iterations: int) -> dict:
    if not images:
        raise ValueError("No images available for benchmarking")

    latencies_ms = []
    embedding_norms = []

    for idx in range(iterations):
        image = images[idx % len(images)]
        start = time.perf_counter()
        embedding = recognizer.extract_embedding(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        embedding_norms.append(float(np.linalg.norm(embedding)))

    return {
        "model": name,
        "iterations": iterations,
        "avg_ms": round(float(np.mean(latencies_ms)), 3),
        "min_ms": round(float(np.min(latencies_ms)), 3),
        "max_ms": round(float(np.max(latencies_ms)), 3),
        "avg_embedding_norm": round(float(np.mean(embedding_norms)), 5),
    }


def build_recognizers():
    return [
        ("arcface_onnx", lambda: ArcFaceRecognizer(model_path=ARCFACE_MODEL_PATH)),
        ("mobilefacenet_rknn", lambda: MobileFaceNetRKNNRecognizer(model_path=ARCFACE_RKNN_PATH)),
        ("ghostfacenet_rknn", lambda: GhostFaceNetRKNNRecognizer(model_path=GHOSTFACENET_RKNN_PATH)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark recognition backends")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "calibration" / "faces"),
        help="Directory of aligned 112x112 face crops",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=16,
        help="Maximum number of images to load",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of embedding extraction iterations per model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "benchmark_recognition.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Benchmark image dir not found: {image_dir}")

    images = load_images(image_dir, limit=args.limit_images)
    if not images:
        raise RuntimeError(f"No readable images found in {image_dir}")

    results = []
    for name, factory in build_recognizers():
        recognizer = None
        try:
            recognizer = factory()
            results.append(benchmark_model(name, recognizer, images, iterations=args.iterations))
            print(f"[OK] Benchmarked {name}")
        except Exception as exc:
            results.append({"model": name, "error": str(exc)})
            print(f"[WARN] Skipped {name}: {exc}")
        finally:
            if recognizer is not None and hasattr(recognizer, "release"):
                recognizer.release()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] Saved benchmark report: {output_path}")


if __name__ == "__main__":
    main()
