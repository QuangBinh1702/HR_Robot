"""
HR Robot - Camera Utilities
Mở camera đa nền tảng (Windows / Linux) với tự động chọn backend.

Usage:
    from src.camera_utils import open_camera, get_camera_info

    cap = open_camera()
    if cap:
        info = get_camera_info(cap)
        print(info)
"""

import sys
import platform
import struct
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    CAMERA_DEVICE, CAMERA_BACKEND, CAMERA_PIXEL_FORMAT,
)


# ========== Helpers ==========

def _fourcc_str(code: int) -> str:
    """Chuyển mã fourcc int → chuỗi 4 ký tự (vd: 'MJPG')."""
    if code <= 0:
        return "N/A"
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def _set_fourcc(cap, fmt: str) -> bool:
    """Đặt pixel format (fourcc). Trả True nếu driver chấp nhận."""
    fourcc = cv2.VideoWriter_fourcc(*fmt)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    actual = int(cap.get(cv2.CAP_PROP_FOURCC))
    return _fourcc_str(actual) == fmt


def _configure_cap(cap, width: int, height: int, fps: int):
    """Đặt resolution và FPS cho VideoCapture."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def _try_read_frame(cap) -> bool:
    """Đọc thử 1 frame thật để xác nhận camera hoạt động."""
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True
    return False


def _try_open(source, backend, width, height, fps, pixel_format=None):
    """Thử mở camera với source/backend cụ thể. Trả về cap hoặc None."""
    try:
        cap = cv2.VideoCapture(source, backend)
    except Exception:
        return None

    if not cap.isOpened():
        cap.release()
        return None

    # Thử đặt pixel format nếu được chỉ định
    if pixel_format and pixel_format != "AUTO":
        _set_fourcc(cap, pixel_format)

    _configure_cap(cap, width, height, fps)

    if _try_read_frame(cap):
        return cap

    cap.release()
    return None


def _try_formats_linux(source, backend, width, height, fps):
    """Thử các pixel format theo thứ tự ưu tiên trên Linux."""
    formats = []
    if CAMERA_PIXEL_FORMAT not in ("AUTO", ""):
        formats.append(CAMERA_PIXEL_FORMAT)
    formats.extend(["YUYV", "MJPG"])
    # Loại trùng giữ thứ tự
    seen = set()
    unique = []
    for f in formats:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    for fmt in unique:
        cap = _try_open(source, backend, width, height, fps, pixel_format=fmt)
        if cap:
            print(f"  → Pixel format: {fmt}")
            return cap
    return None


# ========== Windows ==========

def _open_camera_windows(index, width, height, fps):
    """Thử các backend trên Windows: DSHOW → MSMF → ANY."""
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "ANY"),
    ]
    for idx in range(5):
        if index is not None and idx != index:
            continue
        for backend, name in backends:
            print(f"  Thử camera {idx} backend {name}...")
            cap = _try_open(idx, backend, width, height, fps)
            if cap:
                print(f"  ✓ Mở thành công: camera {idx} / {name}")
                return cap
    return None


# ========== Linux ==========

def _open_camera_linux(index, width, height, fps):
    """Thử mở camera trên Linux: V4L2 (device path hoặc scan) → GStreamer fallback."""

    # 1. Nếu có CAMERA_DEVICE env var → dùng trực tiếp
    if CAMERA_DEVICE:
        print(f"  Dùng CAMERA_DEVICE={CAMERA_DEVICE}")
        cap = _try_formats_linux(CAMERA_DEVICE, cv2.CAP_V4L2, width, height, fps)
        if cap:
            return cap
        print(f"  ✗ Không mở được {CAMERA_DEVICE}")

    # 2. Scan indices với V4L2
    scan_range = range(10)
    for idx in (scan_range if index is None else [index]):
        print(f"  Thử camera {idx} / V4L2...")
        cap = _try_formats_linux(idx, cv2.CAP_V4L2, width, height, fps)
        if cap:
            print(f"  ✓ Mở thành công: camera {idx} / V4L2")
            return cap

    # 3. Fallback: GStreamer pipeline
    source_idx = index if index is not None else 0
    gst_pipeline = (
        f"v4l2src device=/dev/video{source_idx} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! appsink"
    )
    print(f"  Thử GStreamer: {gst_pipeline}")
    cap = _try_open(gst_pipeline, cv2.CAP_GSTREAMER, width, height, fps)
    if cap:
        print(f"  ✓ Mở thành công qua GStreamer")
        return cap

    return None


# ========== Public API ==========

def get_camera_info(cap) -> dict:
    """Trả về thông tin thực tế đã negotiate từ camera."""
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": _fourcc_str(fourcc_int),
        "backend": cap.getBackendName() if hasattr(cap, "getBackendName") else str(backend_id),
    }


def open_camera(index=None, width=None, height=None, fps=None):
    """Mở camera với tự động chọn backend theo nền tảng.

    Returns:
        cv2.VideoCapture hoặc None nếu không mở được.
    """
    idx = index if index is not None else CAMERA_INDEX
    w = width or CAMERA_WIDTH
    h = height or CAMERA_HEIGHT
    f = fps or CAMERA_FPS

    os_name = platform.system()
    print(f"[camera_utils] OS={os_name}, index={idx}, {w}x{h}@{f}fps, "
          f"backend={CAMERA_BACKEND}, format={CAMERA_PIXEL_FORMAT}")

    cap = None

    if CAMERA_BACKEND == "auto":
        if os_name == "Windows":
            cap = _open_camera_windows(idx, w, h, f)
        else:
            cap = _open_camera_linux(idx, w, h, f)
    elif CAMERA_BACKEND == "windows":
        cap = _open_camera_windows(idx, w, h, f)
    elif CAMERA_BACKEND in ("v4l2", "gstreamer"):
        cap = _open_camera_linux(idx, w, h, f)
    else:
        print(f"  ✗ Backend không hợp lệ: {CAMERA_BACKEND}")

    if cap is None:
        print("[camera_utils] ✗ Không thể mở camera!")
        return None

    # Log thông tin thực tế
    info = get_camera_info(cap)
    print(f"[camera_utils] ✓ Camera ready: {info['width']}x{info['height']} "
          f"@ {info['fps']:.1f}fps, fourcc={info['fourcc']}, backend={info['backend']}")
    return cap


# ========== CLI test ==========

if __name__ == "__main__":
    cap = open_camera()
    if cap:
        print("\nNhấn 'q' để thoát.\n")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera Utils Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
