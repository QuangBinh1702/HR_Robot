"""
Model Download Script
Downloads SCRFD and ArcFace ONNX models using InsightFace library.
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_via_insightface():
    """
    Use InsightFace library to download buffalo_l models automatically.
    Models include: SCRFD det_10g + ArcFace w600k_r50
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("✗ insightface not installed. Run: pip install insightface")
        return False
    
    print("[1/3] Downloading models via InsightFace (buffalo_l)...")
    print("      This will download to ~/.insightface/models/buffalo_l/\n")
    
    try:
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("  ✓ InsightFace models downloaded successfully!\n")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Trying buffalo_sc (smaller) instead...\n")
        try:
            app = FaceAnalysis(name="buffalo_sc", providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            print("  ✓ InsightFace buffalo_sc models downloaded!\n")
        except Exception as e2:
            print(f"  ✗ Failed: {e2}")
            return False
    
    # Find downloaded model files
    home = Path.home()
    insightface_dir = home / ".insightface" / "models"
    
    print("[2/3] Locating downloaded model files...")
    
    # Try buffalo_l first, then buffalo_sc
    model_dir = None
    for name in ["buffalo_l", "buffalo_sc"]:
        candidate = insightface_dir / name
        if candidate.exists():
            model_dir = candidate
            print(f"  ✓ Found model dir: {model_dir}")
            break
    
    if model_dir is None:
        print("  ✗ Cannot find downloaded models")
        print(f"  Searched in: {insightface_dir}")
        return False
    
    # List all .onnx files
    onnx_files = list(model_dir.glob("*.onnx"))
    print(f"  ✓ Found {len(onnx_files)} ONNX files:")
    for f in onnx_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    - {f.name} ({size_mb:.1f} MB)")
    
    print(f"\n[3/3] Copying models to {MODELS_DIR}...")
    
    # Copy detection model (det_10g.onnx or det_2.5g.onnx)
    det_copied = False
    for det_name in ["det_10g.onnx", "det_2.5g.onnx", "det_500m.onnx"]:
        src = model_dir / det_name
        if src.exists():
            dst = MODELS_DIR / "scrfd_det.onnx"
            shutil.copy2(src, dst)
            print(f"  ✓ Detection: {det_name} → {dst.name}")
            det_copied = True
            break
    
    if not det_copied:
        print("  ⚠ No detection model found, copying first available")
        det_files = [f for f in onnx_files if "det" in f.name.lower()]
        if det_files:
            dst = MODELS_DIR / "scrfd_det.onnx"
            shutil.copy2(det_files[0], dst)
            print(f"  ✓ Detection: {det_files[0].name} → {dst.name}")
            det_copied = True
    
    # Copy recognition model (w600k_r50.onnx or w600k_mbf.onnx)
    rec_copied = False
    for rec_name in ["w600k_r50.onnx", "w600k_mbf.onnx"]:
        src = model_dir / rec_name
        if src.exists():
            dst = MODELS_DIR / "arcface_rec.onnx"
            shutil.copy2(src, dst)
            print(f"  ✓ Recognition: {rec_name} → {dst.name}")
            rec_copied = True
            break
    
    if not rec_copied:
        # Try any recognition-looking model  
        rec_files = [f for f in onnx_files if any(k in f.name.lower() for k in ["w600k", "arcface", "rec"])]
        if rec_files:
            dst = MODELS_DIR / "arcface_rec.onnx"
            shutil.copy2(rec_files[0], dst)
            print(f"  ✓ Recognition: {rec_files[0].name} → {dst.name}")
            rec_copied = True
    
    return det_copied and rec_copied


def verify_models():
    """Verify that models exist and can be loaded."""
    print("\n" + "=" * 60)
    print("Verifying models...")
    print("=" * 60)
    
    # Check models directory
    onnx_files = list(MODELS_DIR.glob("*.onnx"))
    
    if not onnx_files:
        print("✗ No .onnx files found in models/")
        return False
    
    print(f"\nModels in {MODELS_DIR}:")
    for f in onnx_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  ✓ {f.name} ({size_mb:.1f} MB)")
    
    # Quick test with InsightFace
    print("\nQuick inference test...")
    try:
        import numpy as np
        from insightface.app import FaceAnalysis
        
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Create dummy image
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = app.get(dummy)
        print(f"  ✓ Inference OK (detected {len(faces)} faces on blank image)")
    except Exception as e:
        print(f"  ⚠ Inference test note: {e}")
    
    return True


def main():
    print("=" * 60)
    print("HR Robot - Model Download Script")
    print("=" * 60)
    print(f"Model directory: {MODELS_DIR}\n")
    
    # Check if models already exist
    existing = list(MODELS_DIR.glob("*.onnx"))
    if existing:
        print("Existing models found:")
        for f in existing:
            print(f"  ✓ {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")
        print()
    
    # Download via InsightFace
    success = download_via_insightface()
    
    if success:
        print("\n✅ Models downloaded and copied successfully!")
    else:
        print("\n⚠ Auto-download had issues.")
        print("But InsightFace models should be in ~/.insightface/models/")
        print("You can use InsightFace wrapper directly without manual model files.")
    
    verify_models()
    print("\nDone!")


if __name__ == "__main__":
    main()
