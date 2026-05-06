"""
Convert MiniFASNet ONNX -> RKNN for RK3588S deployment.

Recommended flow:
1) Export the anti-spoof model to ONNX with static RGB input.
2) Convert on x86_64 Linux with rknn-toolkit2 installed.
3) Deploy the resulting `.rknn` file to the RK3588S board.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_calibration_dataset(calib_dir: Path, output_file: Path, max_images: int = 200) -> bool:
    """Create dataset list file for INT8 quantization."""
    if not calib_dir.exists():
        print(f"[WARN] Calibration dir not found: {calib_dir}")
        print("       Add 100-200 face crops matching the model input size for reliable INT8 calibration.")
        return False

    images = sorted(
        [f for f in calib_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    )[:max_images]

    if len(images) < 20:
        print(f"[WARN] Only {len(images)} calibration images found (recommended 100-200).")
        if len(images) == 0:
            return False

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for img in images:
            f.write(f"{img}\n")

    print(f"[OK] Dataset file created: {output_file} ({len(images)} images)")
    return True


def get_onnx_input_meta(onnx_path: Path) -> tuple[str, list, str]:
    """Return (input_name, input_shape, input_layout)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    if not inputs:
        raise ValueError("ONNX model has no inputs")

    input_name = inputs[0].name
    input_shape = list(inputs[0].shape)
    if len(input_shape) != 4:
        raise ValueError(f"Invalid ONNX input shape: {input_shape}. Expected 4D image tensor.")

    if input_shape[1] == 3:
        input_layout = "nchw"
    elif input_shape[-1] == 3:
        input_layout = "nhwc"
    else:
        raise ValueError(
            f"Could not detect channel dimension from input shape: {input_shape}. "
            "Expected channel=3 in axis 1 or axis -1."
        )

    return input_name, input_shape, input_layout


def _sanitize_input_shape_for_rknn(input_shape: list) -> list[int] | None:
    """Convert ONNX input shape to static positive ints for RKNN, if possible."""
    if len(input_shape) != 4:
        return None

    sanitized = []
    for idx, dim in enumerate(input_shape):
        if isinstance(dim, int):
            value = dim
        else:
            dim_str = str(dim).strip().lower()
            if dim is None or dim_str in ("none", "?", "-1", "", "batch_size"):
                value = 1 if idx == 0 else -1
            else:
                try:
                    value = int(dim)
                except (TypeError, ValueError):
                    value = -1

        sanitized.append(value)

    if any(v <= 0 for v in sanitized):
        return None

    return sanitized


def _shape_needs_fix(input_shape: list) -> bool:
    """Return True if shape has dynamic/non-positive/non-int dims."""
    if len(input_shape) != 4:
        return True
    for dim in input_shape:
        if not isinstance(dim, int) or dim <= 0:
            return True
    return False


def _calibration_subdir_name(input_shape: list) -> str:
    """Infer calibration crop folder name from the ONNX input size."""
    if len(input_shape) != 4:
        return "faces_128"

    spatial_dims = [dim for dim in input_shape if isinstance(dim, int) and dim not in (1, 3) and dim > 0]
    if not spatial_dims:
        return "faces_128"

    return f"faces_{min(spatial_dims)}"


def convert(onnx_path: Path, output_path: Path, quantize: str = "int8") -> None:
    """Convert MiniFASNet ONNX model to RKNN."""
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
    dataset_file = PROJECT_ROOT / "data" / "calibration" / "anti_spoof_dataset.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)

    try:
        input_name, input_shape, input_layout = get_onnx_input_meta(onnx_path)
        calib_dir = PROJECT_ROOT / "data" / "calibration" / _calibration_subdir_name(input_shape)
        print(f"[INFO] ONNX input: name={input_name}, shape={input_shape}, layout={input_layout}")

        if do_quantization:
            if not prepare_calibration_dataset(calib_dir, dataset_file):
                print("[ERR] Cannot build INT8 without calibration data.")
                print("      Use --quantize fp16 to skip calibration.")
                sys.exit(1)

        print("--> Configuring MiniFASNet")
        if input_layout == "nchw":
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                quant_img_RGB2BGR=False,
                target_platform="rk3588",
                optimization_level=3,
            )
        else:
            rknn.config(
                target_platform="rk3588",
                optimization_level=0,
            )
            print("[WARN] NHWC input detected: skip mean/std in config.")
            print("[WARN] Runtime preprocess should keep RGB uint8 input.")

        print(f"--> Loading ONNX: {onnx_path}")
        static_shape = _sanitize_input_shape_for_rknn(input_shape)
        needs_fix = _shape_needs_fix(input_shape)

        if needs_fix:
            if static_shape is None:
                print(f"[ERR] Input shape not supported by RKNN and cannot be sanitized: {input_shape}")
                sys.exit(1)

            print(f"[INFO] Dynamic/invalid ONNX shape detected: {input_shape}")
            print(f"[INFO] Forcing static input_size_list={static_shape}")
            try:
                ret = rknn.load_onnx(
                    model=str(onnx_path),
                    inputs=[input_name],
                    input_size_list=[static_shape],
                )
            except Exception as exc:
                print(f"[ERR] load_onnx with static shape failed: {exc}")
                sys.exit(1)
        else:
            try:
                ret = rknn.load_onnx(model=str(onnx_path))
            except Exception as exc:
                print(f"[WARN] load_onnx(model=...) failed: {exc}")
                if static_shape is None:
                    print("[ERR] Could not derive static input shape for fallback.")
                    sys.exit(1)
                print(f"[WARN] Retry with static input_size_list={static_shape}")
                ret = rknn.load_onnx(
                    model=str(onnx_path),
                    inputs=[input_name],
                    input_size_list=[static_shape],
                )
        if ret != 0:
            print("[ERR] Failed to load ONNX into RKNN toolkit.")
            sys.exit(1)

        print(f"--> Building RKNN ({'INT8' if do_quantization else 'FP16'})")
        if do_quantization:
            ret = rknn.build(do_quantization=True, dataset=str(dataset_file))
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            print("[ERR] RKNN build failed.")
            sys.exit(1)

        print(f"--> Exporting RKNN: {output_path}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            print("[ERR] RKNN export failed.")
            sys.exit(1)

        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"[OK] Saved RKNN model: {output_path} ({size_mb:.1f} MB)")
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MiniFASNet ONNX -> RKNN")
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(PROJECT_ROOT / "models" / "anti_spoof.onnx"),
        help="Path to anti-spoof ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "anti_spoof.rknn"),
        help="Output RKNN path",
    )
    parser.add_argument(
        "--quantize",
        choices=["int8", "fp16"],
        default="int8",
        help="Quantization mode (default: int8)",
    )
    args = parser.parse_args()

    convert(
        onnx_path=Path(args.onnx),
        output_path=Path(args.output),
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
