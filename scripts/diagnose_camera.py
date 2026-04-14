"""Diagnose all available cameras and backends on Windows."""
import cv2
import time

BACKENDS = [
    ("DSHOW",  cv2.CAP_DSHOW),
    ("MSMF",   cv2.CAP_MSMF),
    ("ANY",    cv2.CAP_ANY),
]

print("OpenCV version:", cv2.__version__)
print("=" * 60)

for idx in range(5):
    for name, backend in BACKENDS:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            print(f"  index={idx}  {name:6s}  -> CANNOT OPEN")
            cap.release()
            continue

        # Try to read a frame with short timeout
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))

        ret, frame = cap.read()
        status = "FRAME OK" if (ret and frame is not None) else "FRAME FAIL"
        shape = f"{frame.shape}" if (ret and frame is not None) else "N/A"

        print(f"  index={idx}  {name:6s}  -> OPENED  {w}x{h} @{fps:.0f}fps  "
              f"backend_id={backend_id}  {status}  shape={shape}")
        cap.release()

    print("-" * 60)

print("\nDone. If your USB camera shows FRAME FAIL everywhere,")
print("check Device Manager and reinstall the camera driver.")
