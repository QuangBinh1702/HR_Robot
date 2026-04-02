"""
Camera Test Script
Verify camera is working and display basic info.

Usage:
    python src/face_detection/test_camera.py
    python src/face_detection/test_camera.py --index 1
"""

import sys
import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


def test_camera(index: int = None):
    idx = index if index is not None else CAMERA_INDEX
    
    print(f"Testing camera index: {idx}")
    cap = cv2.VideoCapture(idx)
    
    if not cap.isOpened():
        print(f"✗ Cannot open camera {idx}")
        print("  Try: --index 0, --index 1, or --index 2")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Camera opened: {w}x{h} @ {fps:.0f} FPS")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to read frame")
            break
        
        cv2.putText(frame, f"Camera {idx}: {w}x{h}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    test_camera(args.index)
