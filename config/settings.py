"""
HR Robot - Configuration Settings
Centralized configuration for all modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ========== Paths ==========
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FACE_DB_DIR = DATA_DIR / "face_db"
CALIBRATION_DIR = DATA_DIR / "calibration"

# Ensure directories exist
for d in [MODELS_DIR, DATA_DIR, FACE_DB_DIR, CALIBRATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== Camera ==========
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "2"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
CAMERA_DEVICE = os.getenv("CAMERA_DEVICE", "").strip()  # e.g. /dev/video2
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "auto").lower()  # auto|v4l2|gstreamer|windows
CAMERA_PIXEL_FORMAT = os.getenv("CAMERA_PIXEL_FORMAT", "auto").upper()  # auto|MJPG|YUYV

# ========== Face Detection (SCRFD) ==========
SCRFD_MODEL_PATH = str(MODELS_DIR / "scrfd_det.onnx")
SCRFD_RKNN_PATH = str(MODELS_DIR / "scrfd_det.rknn")
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.4"))
DETECTION_INPUT_SIZE = (640, 640)  # On NPU with lightweight model, (320, 320) is recommended

# ========== Face Recognition (ArcFace) ==========
ARCFACE_MODEL_PATH = str(MODELS_DIR / "arcface_rec.onnx")
ARCFACE_RKNN_PATH = str(MODELS_DIR / "arcface_rec.rknn")
EMBEDDING_SIZE = 512
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.45"))
# Cosine similarity threshold: > threshold = same person

# ========== Attendance ==========
CHECKIN_COOLDOWN_SECONDS = int(os.getenv("CHECKIN_COOLDOWN_SECONDS", "300"))  # 5 min
CHECKOUT_ABSENT_MINUTES = int(os.getenv("CHECKOUT_ABSENT_MINUTES", "10"))
# If not seen for N minutes, auto check-out
ATTENDANCE_CONFIDENCE_THRESHOLD = float(os.getenv(
    "ATTENDANCE_CONFIDENCE_THRESHOLD",
    str(RECOGNITION_THRESHOLD)
))
MIN_ATTENDANCE_HITS = int(os.getenv("MIN_ATTENDANCE_HITS", "2"))
# Minimum consecutive recognition hits before check-in
HEADCOUNT_PERSIST_INTERVAL = int(os.getenv("HEADCOUNT_PERSIST_INTERVAL", "1"))
# Seconds between SpaceStatus DB writes (heartbeat)
OVERLOAD_ALERT_COOLDOWN = int(os.getenv("OVERLOAD_ALERT_COOLDOWN", "30"))
# Seconds between repeated overload alerts

# ========== People Counter ==========
MAX_ROOM_CAPACITY = int(os.getenv("MAX_ROOM_CAPACITY", "20"))

# ========== Database ==========
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'data' / 'hr_robot.db'}"
)
# Production: postgresql://user:pass@localhost:5432/hr_robot

# ========== API Server ==========
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
SECRET_KEY = os.getenv("SECRET_KEY", "hr-robot-secret-key-change-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# ========== RKNN ==========
USE_NPU = os.getenv("USE_NPU", "false").lower() == "true"
NPU_CORE_MASK = int(os.getenv("NPU_CORE_MASK", "1"))
# 1=core0, 2=core1, 4=core2, 3=core0+1, 7=all cores
SCRFD_NPU_CORE_MASK = int(os.getenv("SCRFD_NPU_CORE_MASK", "1"))   # core0
ARCFACE_NPU_CORE_MASK = int(os.getenv("ARCFACE_NPU_CORE_MASK", "2"))  # core1
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv(
    "DEFAULT_EMBEDDING_MODEL_NAME",
    "buffalo_sc_rknn_v1" if USE_NPU else "buffalo_l"
)

# ========== Logging ==========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
