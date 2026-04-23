"""
Convert GhostFaceNet ONNX -> RKNN for RK3588S deployment.

This script supports two practical flows:
1) Use an existing ONNX model (recommended):
   python scripts/rknn/convert_ghostfacenet.py --onnx models/ghostfacenet/ghostface_rec.onnx

2) Auto-download a known GhostFaceNet-compatible ONNX and convert:
   python scripts/rknn/convert_ghostfacenet.py --download-onnx

Notes:
- Run conversion on x86_64 Linux with rknn-toolkit2 installed.
- For recognition models, FP16 is usually safer for accuracy than INT8.
"""

import argparse
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_GHOST_ONNX_URL = (
    "https://raw.githubusercontent.com/andestech/ModelZoo/master/"
    "GhostFaceNet/Model/ghostface_fp32.onnx"
)


def prepare_calibration_dataset(calib_dir: Path, output_file: Path, max_images: int = 200) -> bool:
    """Generate dataset list for INT8 quantization."""
    if not calib_dir.exists():
        print(f"[WARN] Calibration dir not found: {calib_dir}")
        print("       Add 100-200 aligned face images (112x112) for robust INT8 quantization.")
        return False

    images = sorted(
        [f for f in calib_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    )[:max_images]

    if len(images) < 20:
        print(f"[WARN] Only {len(images)} calibration images found (recommended 100-200).")
        if len(images) == 0:
            return False

    with open(output_file, "w", encoding="utf-8") as f:
        for img in images:
            f.write(f"{img}\n")

    print(f"[OK] Dataset file created: {output_file} ({len(images)} images)")
    return True


def download_onnx(url: str, dst: Path, overwrite: bool = False) -> None:
    """Download ONNX file if it does not exist."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1024 and not overwrite:
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"[OK] ONNX already exists: {dst} ({size_mb:.1f} MB)")
        return

    if dst.exists() and overwrite:
        dst.unlink(missing_ok=True)

    print(f"--> Downloading ONNX from: {url}")
    urllib.request.urlretrieve(url, str(dst))

    if not dst.exists() or dst.stat().st_size <= 1024:
        raise RuntimeError(f"Downloaded ONNX is invalid: {dst}")

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"[OK] Downloaded ONNX: {dst} ({size_mb:.1f} MB)")


def get_onnx_input_meta(onnx_path: Path) -> tuple[str, list, str]:
    """Return (input_name, input_shape, input_layout)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    if not inputs:
        raise ValueError("ONNX has no inputs")

    input_name = inputs[0].name
    input_shape = list(inputs[0].shape)
    if len(input_shape) != 4:
        raise ValueError(f"Invalid ONNX input shape: {input_shape}")

    # Heuristic: channels is usually 3 for RGB/BGR inputs
    if input_shape[1] == 3:
        input_layout = "nchw"
    elif input_shape[-1] == 3:
        input_layout = "nhwc"
    else:
        # fallback: some exports may leave dynamic dims; assume NHWC for GhostFace from Keras
        input_layout = "nhwc"

    return input_name, input_shape, input_layout


def validate_onnx_input_shape(onnx_path: Path) -> bool:
    """Validate ONNX input is image tensor, not embedding vector."""
    try:
        input_name, shape, layout = get_onnx_input_meta(onnx_path)
    except ImportError:
        print("[WARN] onnxruntime not installed, skip ONNX shape validation.")
        return True
    except Exception as exc:
        print(f"[ERR] {exc}")
        print("      This file is likely NOT a face image embedding model.")
        print("      Expected image-like input shape, e.g. [1,112,112,3] or [1,3,112,112].")
        return False

    print(f"[OK] ONNX input detected: name={input_name}, shape={shape}, layout={layout}")

    return True


def patch_minmax_constants_for_rknn(onnx_path: Path) -> Path:
    """
    Patch ONNX Min/Max constants from [1,1,1,1] -> scalar.

    Root cause:
    - Some GhostFace ONNX exports encode clip bounds as rank-4 tensors [1,1,1,1].
    - RKNN optimizer rewrites Min/Max -> Clip and then expects scalar min/max.
    - This can crash with: "min should be a scalar".
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("[WARN] onnx package not installed, skip Min/Max scalar patch.")
        return onnx_path

    model = onnx.load(str(onnx_path))
    graph = model.graph

    init_index = {tensor.name: idx for idx, tensor in enumerate(graph.initializer)}
    to_patch = set()

    for node in graph.node:
        if node.op_type not in ("Min", "Max"):
            continue
        for const_name in node.input[1:]:
            if const_name in init_index:
                tensor = graph.initializer[init_index[const_name]]
                dims = list(tensor.dims)
                elem_count = 1
                for d in dims:
                    elem_count *= int(d)
                if elem_count == 1 and len(dims) > 0:
                    to_patch.add(const_name)

    if not to_patch:
        return onnx_path

    patched = 0
    for name in to_patch:
        idx = init_index[name]
        old_tensor = graph.initializer[idx]
        arr = numpy_helper.to_array(old_tensor)
        scalar_arr = arr.reshape(())
        new_tensor = numpy_helper.from_array(scalar_arr, name=name)
        graph.initializer[idx].CopyFrom(new_tensor)
        patched += 1

    patched_path = onnx_path.with_name(onnx_path.stem + ".rknnfix.onnx")
    onnx.save(model, str(patched_path))
    print(f"[OK] Patched {patched} Min/Max constants to scalar for RKNN: {patched_path}")
    return patched_path


def convert(onnx_path: Path, output_path: Path, quantize: str = "fp16") -> None:
    """Convert GhostFaceNet ONNX to RKNN."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERR] rknn-toolkit2 is not installed.")
        print("      Install on host PC: pip install rknn-toolkit2")
        sys.exit(1)

    if not onnx_path.exists():
        print(f"[ERR] ONNX not found: {onnx_path}")
        sys.exit(1)

    if not validate_onnx_input_shape(onnx_path):
        print("[ERR] Wrong ONNX file selected for recognizer conversion.")
        print("      Use a GhostFace recognizer ONNX with image input (112x112x3).")
        print("      Quick fix: rerun with --download-onnx.")
        sys.exit(1)

    onnx_path_for_build = patch_minmax_constants_for_rknn(onnx_path)

    do_quantization = quantize == "int8"
    dataset_file = PROJECT_ROOT / "data" / "calibration" / "ghostface_dataset.txt"
    calib_dir = PROJECT_ROOT / "data" / "calibration" / "faces"

    if do_quantization:
        if not prepare_calibration_dataset(calib_dir, dataset_file):
            print("[ERR] Cannot build INT8 without calibration data.")
            print("      Use --quantize fp16 to skip calibration.")
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)

    input_name, input_shape, input_layout = get_onnx_input_meta(onnx_path_for_build)

    print("--> Configuring GhostFaceNet model")
    # NOTE:
    # - For some NHWC ONNX exports, rknn-toolkit2 validates mean/std against dim-1
    #   (112 here) instead of channel=3 and fails with:
    #   "expect 112".
    # - To avoid this toolkit quirk, skip mean/std for NHWC inputs.
    # - Keep mean/std only for NCHW input where channel=3 is stable.
    if input_layout == "nchw":
        rknn.config(
            mean_values=[[127.5, 127.5, 127.5]],
            std_values=[[127.5, 127.5, 127.5]],
            quant_img_RGB2BGR=False,
            target_platform="rk3588",
            optimization_level=3,
        )
    else:
        print("[WARN] NHWC input detected, skipping mean/std in RKNN config to avoid toolkit shape bug.")
        print("[WARN] You must apply normalization in runtime preprocess for this model.")
        print("[WARN] Using optimization_level=0 to avoid RKNN clip fusion crash on this ONNX.")
        rknn.config(
            target_platform="rk3588",
            optimization_level=0,
        )

    print(f"--> Loading ONNX: {onnx_path_for_build}")
    # Avoid passing input_size_list for NHWC models to reduce load_onnx quirks.
    if input_layout == "nhwc":
        ret = rknn.load_onnx(model=str(onnx_path_for_build))
    else:
        ret = rknn.load_onnx(
            model=str(onnx_path_for_build),
            inputs=[input_name],
            input_size_list=[input_shape],
        )
    if ret != 0:
        print("[ERR] Failed to load ONNX model in RKNN toolkit.")
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
    parser = argparse.ArgumentParser(description="Convert GhostFaceNet ONNX -> RKNN")
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(PROJECT_ROOT / "models" / "ghostfacenet" / "ghostface_rec.onnx"),
        help="Path to GhostFaceNet ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "ghostface_rec.rknn"),
        help="Output RKNN path",
    )
    parser.add_argument(
        "--quantize",
        choices=["int8", "fp16"],
        default="fp16",
        help="Quantization mode (default: fp16)",
    )
    parser.add_argument(
        "--download-onnx",
        action="store_true",
        help="Auto-download a known GhostFace ONNX if --onnx does not exist",
    )
    parser.add_argument(
        "--onnx-url",
        type=str,
        default=DEFAULT_GHOST_ONNX_URL,
        help="ONNX URL used with --download-onnx",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download ONNX even if file exists",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)

    if args.download_onnx:
        download_onnx(args.onnx_url, onnx_path, overwrite=args.force_download)

    convert(onnx_path=onnx_path, output_path=output_path, quantize=args.quantize)


if __name__ == "__main__":
    main()
