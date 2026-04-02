# HR Robot - Edge AI Face Recognition & Attendance System

Hệ thống nhận diện khuôn mặt và điểm danh tự động chạy trên **Firefly ROC-RK3588-PC**.

## Kiến trúc

```
Camera → SCRFD (Face Detection) → ArcFace (Recognition) → Attendance Logic → FastAPI → Dashboard
                                         ↕
                                    PostgreSQL DB
                                         ↕
                                    ROS2 Topics
```

## Tech Stack

| Layer            | Technology                                |
| ---------------- | ----------------------------------------- |
| Board            | Firefly ROC-RK3588-PC, Ubuntu 22.04 ARM64 |
| Face Detection   | SCRFD-2.5GF (RKNN INT8)                   |
| Face Recognition | ArcFace-MobileFaceNet (RKNN FP16)         |
| Backend          | FastAPI + WebSocket                       |
| Database         | PostgreSQL / SQLite                       |
| Frontend         | Web Dashboard                             |
| Robot            | ROS2 Humble, rclpy                        |

## Cài đặt

```bash
# 1. Clone & setup
cd HR_Robot
python -m venv venv
source venv/bin/activate  # Linux
# hoặc: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models
python scripts/download_models.py

# 4. Test camera
python src/face_detection/test_camera.py

# 5. Run prototype
python src/face_detection/detect_scrfd.py
```

## Cấu trúc thư mục

```
HR_Robot/
├── config/                  # Cấu hình hệ thống
├── models/                  # AI models (.onnx, .rknn)
├── scripts/                 # Utility scripts
├── src/
│   ├── face_detection/      # SCRFD face detection
│   ├── face_recognition/    # ArcFace face recognition
│   ├── attendance/          # Check-in/out logic
│   ├── people_counter/      # Head count & overload
│   ├── api/                 # FastAPI backend
│   ├── dashboard/           # Web frontend
│   ├── database/            # DB models & migrations
│   └── ros2_node/           # ROS2 integration
├── data/
│   ├── face_db/             # Registered face embeddings
│   └── calibration/         # Quantization calibration images
├── tests/                   # Unit & integration tests
├── docker/                  # Docker configs
├── requirements.txt
└── README.md
```

## Thành viên

- **Nguyễn Quang Bình** (TVC) - Nhận diện & Điểm danh
- **Trương Bùi Diễn** (TV) - Nhận diện & Điểm danh
- **Lê Quang Thái** - Phần cứng & ROS2 Navigation
