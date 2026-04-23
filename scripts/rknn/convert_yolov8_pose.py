"""
Convert YOLOv8 pose ONNX -> RKNN for RK3588S deployment.

Recommended flow:
1) Export ONNX from ultralytics with input 640x640.
2) Convert to RKNN on x86 host.
3) Deploy models/yolov8n-pose.rknn to RK3588S board.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_calibration_dataset(calib_dir: Path, output_file: Path, max_images: int = 300) -> bool:
    """Generate dataset list for INT8 quantization."""
    if not calib_dir.exists():
        print(f"[WARN] Calibration dir not found: {calib_dir}")
        return False

    images = sorted(
        [f for f in calib_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    )[:max_images]

    if len(images) < 50:
        print(f"[WARN] Only {len(images)} images found (recommended 100-300)")
        if len(images) == 0:
            return False

    with open(output_file, "w", encoding="utf-8") as f:
        for img in images:
            f.write(f"{img}\n")

    print(f"[OK] Dataset file created: {output_file} ({len(images)} images)")
    return True


def convert(onnx_path: Path, output_path: Path, quantize: str = "int8") -> None:
    """Convert YOLOv8 pose ONNX to RKNN."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERR] rknn-toolkit2 is not installed.")
        print("      Install on host PC: pip install rknn-toolkit2")
        sys.exit(1)

    if not onnx_path.exists():
        print(f"[ERR] ONNX model not found: {onnx_path}")
        sys.exit(1)

    do_quantization = quantize == "int8"
    dataset_file = PROJECT_ROOT / "data" / "calibration" / "yolov8_pose_dataset.txt"
    calib_dir = PROJECT_ROOT / "data" / "calibration"

    if do_quantization:
        if not prepare_calibration_dataset(calib_dir, dataset_file):
            print("[ERR] Cannot build INT8 without calibration data.")
            print("      Use --quantize fp16 to skip calibration.")
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)

    print("--> Configuring YOLOv8 pose model")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
        optimization_level=3,
    )

    print(f"--> Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path), input_size_list=[[1, 3, 640, 640]])
    if ret != 0:
        print("[ERR] Failed to load ONNX model.")
        rknn.release()
        sys.exit(1)

    print(f"--> Building RKNN ({'INT8' if do_quantization else 'FP16'})")
    if do_quantization:
        ret = rknn.build(do_quantization=True, dataset=str(dataset_file))
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("[ERR] RKNN build failed.")
        rknn.release()
        sys.exit(1)

    print(f"--> Exporting RKNN: {output_path}")
    ret = rknn.export_rknn(str(output_path))
    rknn.release()
    if ret != 0:
        print("[ERR] RKNN export failed.")
        sys.exit(1)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[OK] Saved RKNN model: {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLOv8 pose ONNX -> RKNN")
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(PROJECT_ROOT / "models" / "yolov8n-pose.onnx"),
        help="Path to YOLOv8 pose ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "yolov8n-pose.rknn"),
        help="Output RKNN path",
    )
    parser.add_argument(
        "--quantize",
        choices=["int8", "fp16"],
        default="int8",
        help="Quantization mode (default: int8)",
    )
    args = parser.parse_args()

    convert(Path(args.onnx), Path(args.output), quantize=args.quantize)


if __name__ == "__main__":
    main()
