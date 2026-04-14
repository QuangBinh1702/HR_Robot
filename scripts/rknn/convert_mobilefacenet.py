"""
Convert MobileFaceNet (ArcFace) ONNX → RKNN for RK3588S NPU deployment.
Run on PC (x86_64) with rknn-toolkit2 installed.

Usage:
    python scripts/rknn/convert_mobilefacenet.py
    python scripts/rknn/convert_mobilefacenet.py --quantize fp16
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_calibration_dataset(calib_dir: Path, output_file: Path, max_images: int = 200):
    """Generate dataset.txt from calibration face images (112x112 aligned faces)."""
    if not calib_dir.exists():
        print(f"⚠ Calibration dir not found: {calib_dir}")
        print(f"  Please add 100-200 aligned face images (112x112) to: {calib_dir}")
        return False

    images = sorted(
        [f for f in calib_dir.iterdir()
         if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    )[:max_images]

    if len(images) < 20:
        print(f"⚠ Only {len(images)} calibration images found. Recommend 100-200.")
        if len(images) == 0:
            return False

    with open(output_file, 'w') as f:
        for img in images:
            f.write(f"{img}\n")

    print(f"✓ Dataset file: {output_file} ({len(images)} images)")
    return True


def convert(quantize: str = "int8"):
    """Convert MobileFaceNet ONNX to RKNN."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("✗ rknn-toolkit2 not installed!")
        print("  Install on PC: pip install rknn-toolkit2")
        sys.exit(1)

    onnx_path = PROJECT_ROOT / "models" / "arcface_rec.onnx"
    rknn_path = PROJECT_ROOT / "models" / "arcface_rec.rknn"
    calib_dir = PROJECT_ROOT / "data" / "calibration" / "faces"
    dataset_file = PROJECT_ROOT / "data" / "calibration" / "arcface_dataset.txt"

    if not onnx_path.exists():
        print(f"✗ ONNX model not found: {onnx_path}")
        print(f"  Run: python scripts/download_models.py")
        sys.exit(1)

    do_quantization = quantize == "int8"

    if do_quantization:
        if not prepare_calibration_dataset(calib_dir, dataset_file):
            print("✗ Cannot quantize without calibration data.")
            print("  Use --quantize fp16 for FP16 mode (no calibration needed)")
            sys.exit(1)

    rknn = RKNN(verbose=True)

    # MobileFaceNet/ArcFace preprocessing: normalize to [-1, 1]
    # mean=127.5, std=127.5 → (pixel - 127.5) / 127.5
    print("--> Configuring MobileFaceNet model")
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
        target_platform="rk3588",
        optimization_level=3,
    )

    print(f"--> Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(
        model=str(onnx_path),
        inputs=['input.1'],
        input_size_list=[[1, 3, 112, 112]],
    )
    if ret != 0:
        print("✗ Failed to load ONNX model")
        sys.exit(1)

    print(f"--> Building RKNN ({'INT8' if do_quantization else 'FP16'})")
    if do_quantization:
        ret = rknn.build(do_quantization=True, dataset=str(dataset_file))
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("✗ Build failed")
        sys.exit(1)

    print(f"--> Exporting: {rknn_path}")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        print("✗ Export failed")
        sys.exit(1)

    rknn.release()

    size_mb = rknn_path.stat().st_size / 1024 / 1024
    print(f"\n✅ MobileFaceNet RKNN model saved: {rknn_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert MobileFaceNet ONNX → RKNN")
    parser.add_argument("--quantize", choices=["int8", "fp16"], default="int8",
                        help="Quantization mode (default: int8)")
    args = parser.parse_args()
    convert(quantize=args.quantize)


if __name__ == "__main__":
    main()
