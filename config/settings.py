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
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))

# ========== Face Detection (SCRFD) ==========
SCRFD_MODEL_PATH = str(MODELS_DIR / "scrfd_det.onnx")
SCRFD_RKNN_PATH = str(MODELS_DIR / "scrfd_det.rknn")
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.4"))
DETECTION_INPUT_SIZE = (640, 640)

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

# ========== Logging ==========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
