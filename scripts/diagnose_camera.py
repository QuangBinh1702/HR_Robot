"""Diagnose all available cameras — hỗ trợ Windows và Linux (RK3588S)."""
import sys
import platform
import subprocess
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OS_NAME = platform.system()

print(f"OS: {OS_NAME}")
print(f"OpenCV version: {cv2.__version__}")
print("=" * 60)


def _fourcc_str(code: int) -> str:
    if code <= 0:
        return "N/A"
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def _test_camera(source, backend, backend_name):
    """Thử mở camera và đọc frame."""
    try:
        cap = cv2.VideoCapture(source, backend)
    except Exception as e:
        print(f"  {str(source):20s}  {backend_name:10s}  -> EXCEPTION: {e}")
        return

    if not cap.isOpened():
        print(f"  {str(source):20s}  {backend_name:10s}  -> CANNOT OPEN")
        cap.release()
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = _fourcc_str(int(cap.get(cv2.CAP_PROP_FOURCC)))

    ret, frame = cap.read()
    status = "FRAME OK" if (ret and frame is not None) else "FRAME FAIL"
    shape = f"{frame.shape}" if (ret and frame is not None) else "N/A"

    print(f"  {str(source):20s}  {backend_name:10s}  -> OPENED  {w}x{h} @{fps:.0f}fps  "
          f"fourcc={fourcc}  {status}  shape={shape}")
    cap.release()


if OS_NAME == "Windows":
    print("\n[Windows] Testing backends: DSHOW, MSMF, ANY\n")
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    ]
    for idx in range(5):
        for name, backend in backends:
            _test_camera(idx, backend, name)
        print("-" * 60)

else:
    # Linux / RK3588S
    print("\n[Linux] Listing video devices:\n")
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5
        )
        print(result.stdout if result.stdout else "(no output)")
        if result.stderr:
            print(result.stderr)
    except FileNotFoundError:
        print("  v4l2-ctl not found. Install: sudo apt install v4l-utils")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[Linux] Scanning /dev/video* with V4L2:\n")
    import glob as glob_mod
    devices = sorted(glob_mod.glob("/dev/video*"))
    if not devices:
        print("  No /dev/video* devices found!")

    for dev in devices:
        # Print device name from sysfs
        name_file = Path(f"/sys/class/video4linux/{Path(dev).name}/name")
        dev_name = name_file.read_text().strip() if name_file.exists() else "unknown"
        print(f"\n  Device: {dev} ({dev_name})")

        # Try list formats
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n")[:15]:
                    print(f"    {line}")
        except Exception:
            pass

        # Try V4L2 with different formats
        for fmt_name, fmt_code in [("YUYV", "YUYV"), ("MJPG", "MJPG")]:
            try:
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt_code))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, frame = cap.read()
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fourcc = _fourcc_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
                    status = "OK" if (ret and frame is not None) else "FAIL"
                    shape = f"{frame.shape}" if (ret and frame is not None) else "N/A"
                    print(f"    V4L2 {fmt_name}: {status}  {w}x{h}  actual_fourcc={actual_fourcc}  shape={shape}")
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"    V4L2 {fmt_name}: EXCEPTION {e}")

        print("-" * 60)

print("\nDone.")
print("Tip: On RK3588S, set CAMERA_DEVICE=/dev/videoX in .env")
print("     where X is the device that shows FRAME OK above.")
