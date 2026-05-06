"""
Model Download Script (Edge AI / RK3588S NPU)
Downloads lightweight SCRFD and ArcFace ONNX models using InsightFace library.
- buffalo_sc: det_500m.onnx + w600k_mbf.onnx (small & fast)
- scrfd_2.5g_kps.onnx: standalone lightweight detector with keypoints
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Direct URL for SCRFD-2.5G standalone model from InsightFace model zoo
SCRFD_2_5G_URL = (
    "https://github.com/deepinsight/insightface/releases/download/"
    "v0.7/scrfd_2.5g_kps.onnx"
)

GHOSTFACE_ONNX_URL = (
    "https://raw.githubusercontent.com/andestech/ModelZoo/master/"
    "GhostFaceNet/Model/ghostface_fp32.onnx"
)

ANTI_SPOOF_ONNX_URL = (
    "https://github.com/facenox/face-antispoof-onnx/releases/download/"
    "v1.0.0/best_model.onnx"
)

YOLOV8N_ONNX_URLS = [
    "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.onnx",
]


def download_buffalo_sc():
    """
    Use InsightFace library to download buffalo_sc models automatically.
    buffalo_sc contains: det_500m.onnx + w600k_mbf.onnx (lightweight).
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("[X] insightface not installed. Run: pip install insightface")
        return False

    print("[1/4] Downloading models via InsightFace (buffalo_sc)...")
    print("      This will download to ~/.insightface/models/buffalo_sc/\n")

    try:
        app = FaceAnalysis(name="buffalo_sc", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("  [OK] InsightFace buffalo_sc models downloaded successfully!\n")
    except Exception as e:
        print(f"  [X] Failed to download buffalo_sc: {e}")
        return False

    return True


def download_scrfd_2_5g_standalone():
    """
    Try to download the standalone SCRFD-2.5G-KPS model.
    This is a better lightweight detector than det_500m from buffalo_sc.
    """
    print("[2/4] Downloading standalone SCRFD-2.5G-KPS model...")

    dst = MODELS_DIR / "scrfd_2.5g_kps.onnx"
    if dst.exists():
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  [OK] Already exists: {dst.name} ({size_mb:.1f} MB)\n")
        return True

    # Try via insightface model zoo API first
    try:
        from insightface.utils import storage
        model_path = storage.download("models", "scrfd_2.5g_kps.onnx")
        if model_path and Path(model_path).exists():
            shutil.copy2(model_path, dst)
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  [OK] Downloaded via insightface zoo: {dst.name} ({size_mb:.1f} MB)\n")
            return True
    except Exception:
        pass

    # Fallback: direct URL download
    try:
        print(f"  Downloading from: {SCRFD_2_5G_URL}")
        urllib.request.urlretrieve(SCRFD_2_5G_URL, str(dst))
        if dst.exists() and dst.stat().st_size > 1024:
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  [OK] Downloaded: {dst.name} ({size_mb:.1f} MB)\n")
            return True
        else:
            dst.unlink(missing_ok=True)
            print("  [X] Downloaded file too small, removed.\n")
            return False
    except Exception as e:
        dst.unlink(missing_ok=True)
        print(f"  [!] Could not download standalone SCRFD-2.5G: {e}")
        print("  Will fall back to det_500m.onnx from buffalo_sc.\n")
        return False


def copy_models_to_project():
    """
    Copy lightweight models to the project models/ directory.
    - Recognition: w600k_mbf.onnx -> models/arcface_rec.onnx
    - Detection: prefer scrfd_2.5g_kps.onnx, fallback det_500m.onnx -> models/scrfd_det.onnx
    """
    print("[3/4] Copying models to", MODELS_DIR, "...")

    home = Path.home()
    insightface_dir = home / ".insightface" / "models"
    buffalo_sc_dir = insightface_dir / "buffalo_sc"

    if not buffalo_sc_dir.exists():
        print(f"  [X] buffalo_sc directory not found: {buffalo_sc_dir}")
        return False

    # List all files in buffalo_sc
    onnx_files = list(buffalo_sc_dir.glob("*.onnx"))
    print(f"  Found {len(onnx_files)} ONNX files in buffalo_sc:")
    for f in onnx_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    - {f.name} ({size_mb:.1f} MB)")

    # Copy recognition model: w600k_mbf.onnx -> arcface_rec.onnx
    rec_copied = False
    rec_src = buffalo_sc_dir / "w600k_mbf.onnx"
    if rec_src.exists():
        dst = MODELS_DIR / "arcface_rec.onnx"
        shutil.copy2(rec_src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  [OK] Recognition: w600k_mbf.onnx -> {dst.name} ({size_mb:.1f} MB)")
        rec_copied = True
    else:
        print("  [X] w600k_mbf.onnx not found in buffalo_sc")

    # Copy detection model: prefer scrfd_2.5g_kps.onnx, fallback to det_500m.onnx
    det_copied = False
    scrfd_standalone = MODELS_DIR / "scrfd_2.5g_kps.onnx"
    det_dst = MODELS_DIR / "scrfd_det.onnx"

    if scrfd_standalone.exists():
        # Use standalone SCRFD-2.5G (better than det_500m)
        shutil.copy2(scrfd_standalone, det_dst)
        size_mb = det_dst.stat().st_size / 1024 / 1024
        print(f"  [OK] Detection: scrfd_2.5g_kps.onnx -> {det_dst.name} ({size_mb:.1f} MB)")
        det_copied = True
    else:
        # Fallback to det_500m from buffalo_sc
        det_src = buffalo_sc_dir / "det_500m.onnx"
        if det_src.exists():
            shutil.copy2(det_src, det_dst)
            size_mb = det_dst.stat().st_size / 1024 / 1024
            print(f"  [OK] Detection: det_500m.onnx -> {det_dst.name} ({size_mb:.1f} MB)")
            det_copied = True
        else:
            print("  [X] No detection model found (neither scrfd_2.5g_kps nor det_500m)")

    return det_copied and rec_copied


def verify_models():
    """Verify that models exist, are lightweight, and can be loaded."""
    print("\n" + "=" * 60)
    print("Verifying models...")
    print("=" * 60)

    # Check models directory
    onnx_files = list(MODELS_DIR.glob("*.onnx"))

    if not onnx_files:
        print("[X] No .onnx files found in models/")
        return False

    print(f"\nModels in {MODELS_DIR}:")
    total_size = 0
    for f in onnx_files:
        size_mb = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  [OK] {f.name} ({size_mb:.1f} MB)")

    print(f"\n  Total model size: {total_size:.1f} MB")
    if total_size < 100:
        print("  [OK] Lightweight models confirmed (suitable for edge AI / RK3588S NPU)")
    else:
        print("  [!] Models may be too large for edge deployment")

    # Quick test with InsightFace using buffalo_sc
    print("\nQuick inference test (buffalo_sc)...")
    try:
        import numpy as np
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_sc", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Create dummy image
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = app.get(dummy)
        print(f"  [OK] Inference OK (detected {len(faces)} faces on blank image)")
    except Exception as e:
        print(f"  [!] Inference test note: {e}")

    return True


def download_ghostface_onnx():
    """Download a GhostFace-compatible ONNX for later RKNN conversion."""
    print("[4/5] Downloading GhostFace ONNX (optional test model)...")

    ghost_dir = MODELS_DIR / "ghostfacenet"
    ghost_dir.mkdir(parents=True, exist_ok=True)
    dst = ghost_dir / "ghostface_rec.onnx"

    if dst.exists() and dst.stat().st_size > 1024:
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  [OK] Already exists: {dst.name} ({size_mb:.1f} MB)\n")
        return True

    try:
        print(f"  Downloading from: {GHOSTFACE_ONNX_URL}")
        urllib.request.urlretrieve(GHOSTFACE_ONNX_URL, str(dst))
        if dst.exists() and dst.stat().st_size > 1024:
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  [OK] Downloaded: {dst.name} ({size_mb:.1f} MB)\n")
            return True
        dst.unlink(missing_ok=True)
        print("  [X] Downloaded file too small, removed.\n")
        return False
    except Exception as e:
        dst.unlink(missing_ok=True)
        print(f"  [!] Could not download GhostFace ONNX: {e}\n")
        return False


def download_anti_spoof_onnx():
    """Download the reference 2-class anti-spoof ONNX model."""
    print("[5/6] Downloading anti-spoof ONNX (Windows/RK3588 parity model)...")

    dst = MODELS_DIR / "anti_spoof.onnx"
    if dst.exists() and dst.stat().st_size > 1024:
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  [OK] Already exists: {dst.name} ({size_mb:.1f} MB)\n")
        return True

    try:
        print(f"  Downloading from: {ANTI_SPOOF_ONNX_URL}")
        urllib.request.urlretrieve(ANTI_SPOOF_ONNX_URL, str(dst))
        if dst.exists() and dst.stat().st_size > 1024:
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  [OK] Downloaded: {dst.name} ({size_mb:.1f} MB)\n")
            return True
        dst.unlink(missing_ok=True)
        print("  [X] Downloaded file too small, removed.\n")
        return False
    except Exception as e:
        dst.unlink(missing_ok=True)
        print(f"  [!] Could not download anti-spoof ONNX: {e}\n")
        return False


def download_yolov8n_onnx():
    """Download YOLOv8n ONNX for RKNN conversion."""
    print("[6/6] Downloading YOLOv8n ONNX (for person detection)...")

    dst = MODELS_DIR / "yolov8n.onnx"
    if dst.exists() and dst.stat().st_size > 1024:
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  [OK] Already exists: {dst.name} ({size_mb:.1f} MB)\n")
        return True

    last_error = None
    for url in YOLOV8N_ONNX_URLS:
        try:
            print(f"  Downloading from: {url}")
            urllib.request.urlretrieve(url, str(dst))
            if dst.exists() and dst.stat().st_size > 1024:
                size_mb = dst.stat().st_size / 1024 / 1024
                print(f"  [OK] Downloaded: {dst.name} ({size_mb:.1f} MB)\n")
                return True
            dst.unlink(missing_ok=True)
            last_error = RuntimeError("Downloaded file too small")
        except Exception as e:
            dst.unlink(missing_ok=True)
            last_error = e

    print(f"  [!] Could not download YOLOv8n ONNX from all sources: {last_error}")

    # Fallback: export ONNX locally with ultralytics if available
    print("  Trying local export via ultralytics...")
    export_cmd = [
        sys.executable,
        "-c",
        (
            "from ultralytics import YOLO; "
            "m=YOLO('yolov8n.pt'); "
            "m.export(format='onnx', imgsz=640, opset=12, simplify=True, dynamic=False)"
        ),
    ]
    try:
        subprocess.run(export_cmd, check=True)
        local_onnx = PROJECT_ROOT / "yolov8n.onnx"
        if local_onnx.exists() and local_onnx.stat().st_size > 1024:
            shutil.move(str(local_onnx), str(dst))
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  [OK] Exported locally: {dst.name} ({size_mb:.1f} MB)\n")
            return True
    except Exception as e:
        print(f"  [!] Local export failed: {e}")

    print("  Fallback manual steps:")
    print("    pip install ultralytics")
    print("    yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12")
    print("    move yolov8n.onnx models/yolov8n.onnx\n")
    return False


def main():
    print("=" * 60)
    print("HR Robot - Edge AI Model Download Script")
    print("Target: RK3588S NPU (lightweight models)")
    print("=" * 60)
    print(f"Model directory: {MODELS_DIR}\n")

    # Check if models already exist
    existing = list(MODELS_DIR.glob("*.onnx"))
    if existing:
        print("Existing models found:")
        for f in existing:
            print(f"  [OK] {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")
        print()

    # Step 1: Download buffalo_sc via InsightFace
    buffalo_ok = download_buffalo_sc()

    # Step 2: Try to download standalone SCRFD-2.5G
    scrfd_ok = download_scrfd_2_5g_standalone()

    # Step 3: Copy models to project directory
    if buffalo_ok:
        copy_ok = copy_models_to_project()
    else:
        copy_ok = False
        print("[3/4] Skipped (buffalo_sc download failed).")

    # Step 4: Optional GhostFace ONNX download
    ghost_ok = download_ghostface_onnx()

    anti_spoof_ok = download_anti_spoof_onnx()

    # Step 6: YOLOv8n ONNX for person detector conversion
    yolo_ok = download_yolov8n_onnx()

    # Step 5: Verify
    if copy_ok:
        print("\n[OK] Lightweight models downloaded and copied successfully!")
    else:
        print("\n[!] Auto-download had issues.")
        print("But InsightFace models should be in ~/.insightface/models/buffalo_sc/")
        print("You can use InsightFace wrapper directly without manual model files.")

    if ghost_ok:
        print("[OK] GhostFace ONNX ready at models/ghostfacenet/ghostface_rec.onnx")
        print("     Convert to RKNN with: python scripts/rknn/convert_ghostfacenet.py")

    if anti_spoof_ok:
        print("[OK] Anti-spoof ONNX ready at models/anti_spoof.onnx")
        print("     Convert to RKNN with: python scripts/rknn/convert_minifasnet.py")

    if yolo_ok:
        print("[OK] YOLOv8n ONNX ready at models/yolov8n.onnx")
        print("     Convert to RKNN with: python scripts/rknn/convert_yolov8n.py")

    verify_models()
    print("\nDone!")


if __name__ == "__main__":
    main()
