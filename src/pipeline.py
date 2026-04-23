"""
HR Robot - Complete Face Recognition Pipeline
Unified pipeline: Detection → Alignment → Recognition → Registration

Supports two backends:
  - CPU: InsightFace FaceAnalysis (ONNX Runtime)
  - NPU: RKNN on RK3588S (lightweight SCRFD-2.5G + MobileFaceNet)

Set USE_NPU=true in .env to enable NPU backend.

Usage:
    python src/pipeline.py                    # Live recognition
    python src/pipeline.py --register         # Register new face
    python src/pipeline.py --register --name "Nguyen Van A"
    python src/pipeline.py --list             # List registered faces
    python src/pipeline.py --benchmark        # FPS benchmark
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    RECOGNITION_THRESHOLD, MAX_ROOM_CAPACITY,
    USE_NPU, DEFAULT_EMBEDDING_MODEL_NAME,
    SCRFD_RKNN_PATH, ARCFACE_RKNN_PATH,
    DETECTION_INPUT_SIZE, USE_PERSON_GATE, FACE_DETECT_INTERVAL,
)
from src.database.models import session_scope
from src.database.repository import FaceRepository
from src.embedding_cache import EmbeddingCache
from src.attendance import AttendanceManager
from src.app_runtime import AppRuntime
from src.api_server import start_api_server, broadcast_status
from src.camera_utils import open_camera
from src.pipeline_async import AsyncPersonGatedPipeline, draw_person_boxes


def _warn_if_not_project_venv() -> None:
    """Warn when the script is not running inside the project's virtualenv."""
    exe_path = Path(sys.executable).resolve()
    expected_venv = PROJECT_ROOT / "venv"
    if expected_venv.exists() and expected_venv not in exe_path.parents:
        print("[Warning] Dang chay bang Python ngoai venv cua project.")
        print(f"[Warning] Python hien tai : {exe_path}")
        print(f"[Warning] Nen dung       : {expected_venv / 'Scripts' / 'python.exe'}")
        print("[Warning] Lenh khuyen dung: venv\\Scripts\\python src/pipeline.py")


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline.
    Supports CPU (InsightFace) and NPU (RKNN on RK3588S) backends.
    Handles: detection, alignment, embedding, matching, registration.
    """
    
    def __init__(self, model_name: str = None, threshold: float = None):
        self.threshold = threshold or RECOGNITION_THRESHOLD
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL_NAME
        
        # Database-backed face matching
        self.repo = FaceRepository()
        self.cache = EmbeddingCache(self.repo, model_name=self.model_name)
        
        self._init_backend()
        
        # Load embeddings from SQLite into memory
        count = self.cache.rebuild()
        print(f"[Pipeline] Loaded {count} embeddings from database")
    
    def _init_backend(self):
        """Initialize face analysis backend (NPU or CPU)."""
        if USE_NPU:
            self._init_rknn()
        else:
            self._init_insightface(self.model_name)
    
    def _init_rknn(self):
        """Initialize RKNN NPU backend for RK3588S."""
        try:
            from src.backends.rknn_face_analysis import RKNNFaceAnalysis
        except ImportError as e:
            print(f"[!] RKNN backend not available: {e}")
            print("  Falling back to CPU (InsightFace)...")
            self._init_insightface(self.model_name)
            return
        
        try:
            self.app = RKNNFaceAnalysis(
                det_model_path=SCRFD_RKNN_PATH,
                rec_model_path=ARCFACE_RKNN_PATH,
                det_size=DETECTION_INPUT_SIZE,
            )
            self.backend = "RK3588S NPU (RKNN)"
            print(f"[Pipeline] NPU backend loaded [{self.backend}]")
        except Exception as e:
            print(f"[!] RKNN init failed: {e}")
            print("  Falling back to CPU (InsightFace)...")
            self._init_insightface(self.model_name)
    
    def _init_insightface(self, model_name: str):
        """Initialize InsightFace FaceAnalysis (CPU fallback)."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            print("ERROR: insightface not installed!")
            print("Run: pip install insightface onnxruntime")
            sys.exit(1)
        
        backend = "CPU (ONNX Runtime)"
        print(f"[Pipeline] Loading InsightFace model: {model_name} ({backend})")
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition'],
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.backend = backend
        print(f"[Pipeline] Model loaded successfully [{backend}]")

    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect and analyze all faces in an image.
        
        Returns list of dicts with:
            bbox, score, embedding, keypoints, name, confidence, member_id
        """
        try:
            faces = self.app.get(image)
        except Exception:
            raise
        results = []
        
        for face in faces:
            result = {
                'bbox': [int(x) for x in face.bbox],
                'score': float(face.det_score),
                'embedding': face.normed_embedding,
                'keypoints': face.kps.tolist() if face.kps is not None else None,
            }
            
            # Match against database
            name, confidence, member_id = self._match_face(face.normed_embedding)
            result['name'] = name
            result['confidence'] = confidence
            result['member_id'] = member_id
            
            results.append(result)
        
        return results
    
    def _match_face(self, embedding: np.ndarray) -> tuple:
        """Match an embedding against the database cache."""
        name, score, member_id = self.cache.match(embedding, self.threshold)
        return (name, score, member_id)
    
    def register_face(self, name: str, image: np.ndarray) -> dict:
        """
        Register a face for the given name.
        Allows adding more samples for the same person.
        Rejects if the face matches a DIFFERENT known person.
        
        Returns:
            {'success': bool, 'message': str}
        """
        faces = self.app.get(image)
        
        if not faces:
            return {'success': False, 'message': 'Không phát hiện khuôn mặt'}
        
        # Filter: keep faces that are unknown OR match the target name
        eligible_faces = []
        for f in faces:
            emb = f.normed_embedding
            matched_name, score, _ = self._match_face(emb)
            if matched_name == "Người lạ" or matched_name == name:
                eligible_faces.append(f)
            # If matched to another person → skip this face
        
        if not eligible_faces:
            return {'success': False, 'message': 'Khuôn mặt đã được đăng ký cho người khác'}
        
        # Pick the largest eligible face
        face = max(eligible_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        embedding = face.normed_embedding
        
        # Save to SQLite
        with session_scope() as session:
            member = self.repo.get_or_create_member(session, name)
            self.repo.add_embedding(session, member.id, embedding, model_name=self.model_name)
            total = self.repo.count_embeddings(session, member.id, model_name=self.model_name)
        
        self.cache.rebuild()
        
        return {
            'success': True,
            'message': f"Đã đăng ký '{name}' (mẫu #{total})"
        }
    
    def delete_face(self, name: str) -> bool:
        """Delete all embeddings for a person (keeps member row for attendance history)."""
        with session_scope() as session:
            deleted = self.repo.delete_embeddings_by_name(session, name, model_name=self.model_name)
        if deleted > 0:
            self.cache.rebuild()
            return True
        return False
    
    def list_faces(self) -> dict:
        """List all registered faces with embedding counts."""
        with session_scope() as session:
            return self.repo.list_registered_faces(session, model_name=self.model_name)
    
    def get_headcount(self, image: np.ndarray) -> int:
        """Count number of faces in image (detect-only, no recognition)."""
        if hasattr(self.app, 'detect'):
            faces = self.app.detect(image)
        else:
            faces = self.app.get(image)
        return len(faces)


# ================================================================
# CLI Modes
# ================================================================

def _load_font(size: int):
    """Load a font that supports Vietnamese characters."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

# Pre-load fonts
_FONT_LARGE = _load_font(20)
_FONT_MEDIUM = _load_font(16)
_FONT_SMALL = _load_font(14)




def draw_texts_vn(img, text_items):
    """Draw multiple Vietnamese texts with a single OpenCV<->PIL conversion."""
    if not text_items:
        return img

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for text, pos, item_font, color in text_items:
        active_font = item_font or _FONT_MEDIUM
        draw.text(pos, text, font=active_font, fill=(color[2], color[1], color[0]))

    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def get_text_size_vn(text, font=None):
    """Get (width, height) of Vietnamese text rendered with PIL font."""
    if font is None:
        font = _FONT_MEDIUM
    bbox = font.getbbox(text)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def draw_results(frame: np.ndarray, results: list, fps: int, headcount_limit: int) -> np.ndarray:
    """Draw detection/recognition results on frame."""
    vis = frame.copy()
    headcount = len(results)
    is_overloaded = headcount > headcount_limit
    text_items = []
    
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        name = r['name']
        conf = r['confidence']
        
        # Color: green=known, red=unknown
        if name == "Người lạ":
            color = (0, 0, 255)  # Red
            label = f"Người lạ ({conf:.2f})"
        else:
            color = (0, 255, 0)  # Green
            label = f"{name} ({conf:.2f})"
        
        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Label background
        tw, th = get_text_size_vn(label, _FONT_MEDIUM)
        cv2.rectangle(vis, (x1, y1 - th - 12), (x1 + tw + 4, y1), color, -1)
        text_items.append((label, (x1 + 2, y1 - th - 10), _FONT_MEDIUM, (255, 255, 255)))
    
    # Top bar info
    bar_color = (0, 0, 200) if is_overloaded else (50, 50, 50)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 45), bar_color, -1)
    
    info = f"FPS: {fps} | Số người: {headcount}/{headcount_limit}"
    if is_overloaded:
        info += " | !! QUÁ TẢI !!"

    text_items.append((info, (10, 10), _FONT_LARGE, (255, 255, 255)))
    
    # Known/Unknown count
    known = sum(1 for r in results if r['name'] != "Người lạ")
    unknown = headcount - known
    status = f"Thành viên: {known} | Người lạ: {unknown}"
    text_items.append((status, (10, vis.shape[0] - 25), _FONT_SMALL, (200, 200, 200)))

    draw_texts_vn(vis, text_items)
    
    return vis


def _build_manual_status_summary(attendance: AttendanceManager, results: list, snapshot) -> dict:
    """Build status payload without automatic attendance transitions."""
    headcount = len(results)
    known_count = sum(1 for r in results if r.get("member_id") is not None)
    unknown_count = headcount - known_count
    summary = attendance.get_status_summary()
    summary.update({
        "headcount": headcount,
        "known_count": known_count,
        "unknown_count": unknown_count,
        "is_overloaded": headcount > MAX_ROOM_CAPACITY,
        "new_checkins": [],
        "new_checkouts": [],
        "overload_alert_triggered": False,
        "person_trigger": getattr(snapshot, "trigger", False),
        "person_count": len(getattr(snapshot, "person_boxes", []) or []),
        "person_boxes": getattr(snapshot, "person_boxes", []) or [],
    })
    return summary


def _draw_manual_touch_panel(vis: np.ndarray, selected_face: dict | None, attendance: AttendanceManager, ui_state: dict):
    """Draw fixed check-in/check-out touch controls for OpenCV window."""
    h, w = vis.shape[:2]
    panel_h = 132
    panel_y = h - panel_h
    cv2.rectangle(vis, (0, panel_y), (w, h), (10, 18, 28), -1)
    cv2.rectangle(vis, (0, panel_y), (w, h), (56, 72, 92), 1)

    margin = 18
    gap = 18
    button_h = 84
    available_w = w - margin * 2 - gap
    button_w = max(220, available_w // 2)
    if button_w * 2 + gap > available_w:
        button_w = available_w // 2
    button_y = panel_y + 28

    checkin_rect = (margin, button_y, margin + button_w, button_y + button_h)
    checkout_rect = (margin + button_w + gap, button_y, margin + button_w * 2 + gap, button_y + button_h)
    ui_state["checkin_rect"] = checkin_rect
    ui_state["checkout_rect"] = checkout_rect
    ui_state["confirm_rect"] = None
    ui_state["cancel_rect"] = None

    helper_text = ui_state.get("message") or (
        f"San sang thao tac cho: {selected_face['name']}" if selected_face
        else "Canh dung 1 thanh vien trong khung hinh de bat nut thao tac"
    )
    present_count = attendance.get_status_summary().get("present_count", 0)

    text_items = [
        ("Che do diem danh thu cong", (18, panel_y + 6), _FONT_MEDIUM, (255, 255, 255)),
        (f"Dang co mat: {present_count}", (w - 180, panel_y + 6), _FONT_MEDIUM, (120, 255, 190)),
        (helper_text, (18, panel_y + 105), _FONT_SMALL, (210, 218, 232)),
    ]

    enabled = selected_face is not None
    buttons = [
        ("CHECK IN", checkin_rect, (30, 120, 80), (72, 200, 130), enabled, "Cham de xac nhan"),
        ("CHECK OUT", checkout_rect, (115, 50, 50), (228, 112, 112), enabled, "Can xac nhan 2 buoc"),
    ]

    pending_checkout = ui_state.get("pending_checkout")
    if pending_checkout:
        confirm_y = panel_y - 76
        confirm_w = 180
        confirm_gap = 14
        confirm_x2 = w - margin
        cancel_rect = (confirm_x2 - confirm_w * 2 - confirm_gap, confirm_y, confirm_x2 - confirm_w - confirm_gap, confirm_y + 58)
        confirm_rect = (confirm_x2 - confirm_w, confirm_y, confirm_x2, confirm_y + 58)
        ui_state["cancel_rect"] = cancel_rect
        ui_state["confirm_rect"] = confirm_rect
        cv2.rectangle(vis, (cancel_rect[0], cancel_rect[1]), (cancel_rect[2], cancel_rect[3]), (64, 74, 86), -1)
        cv2.rectangle(vis, (cancel_rect[0], cancel_rect[1]), (cancel_rect[2], cancel_rect[3]), (210, 216, 224), 2)
        cv2.rectangle(vis, (confirm_rect[0], confirm_rect[1]), (confirm_rect[2], confirm_rect[3]), (214, 92, 92), -1)
        cv2.rectangle(vis, (confirm_rect[0], confirm_rect[1]), (confirm_rect[2], confirm_rect[3]), (255, 240, 240), 2)
        text_items.extend([
            (f"Xac nhan check-out cho {pending_checkout['name']}?", (18, panel_y - 68), _FONT_MEDIUM, (255, 228, 228)),
            ("HUY", (cancel_rect[0] + 54, cancel_rect[1] + 15), _FONT_MEDIUM, (255, 255, 255)),
            ("XAC NHAN", (confirm_rect[0] + 24, confirm_rect[1] + 15), _FONT_MEDIUM, (255, 255, 255)),
        ])

    for label, rect, deep_color, fill_color, button_enabled, note in buttons:
        x1, y1, x2, y2 = rect
        active_fill = fill_color if button_enabled else (70, 78, 88)
        active_deep = deep_color if button_enabled else (58, 64, 72)
        border = (255, 255, 255) if button_enabled else (112, 118, 126)
        cv2.rectangle(vis, (x1, y1), (x2, y2), active_fill, -1)
        cv2.rectangle(vis, (x1, y1), (x2, y2), border, 2)
        cv2.rectangle(vis, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), active_deep, 1)
        tw, _ = get_text_size_vn(label, _FONT_LARGE)
        text_items.append((label, (x1 + (x2 - x1 - tw) // 2, y1 + 14), _FONT_LARGE, (255, 255, 255)))
        if not button_enabled:
            note = "Can dung 1 thanh vien"
        nw, _ = get_text_size_vn(note, _FONT_SMALL)
        text_items.append((note, (x1 + (x2 - x1 - nw) // 2, y1 + 52), _FONT_SMALL, (240, 246, 252)))

    draw_texts_vn(vis, text_items)
    return vis



def mode_recognition(pipeline: FaceRecognitionPipeline, attendance_manager=None, on_status_update=None):
    """Live recognition mode with manual touch-based check-in/check-out."""
    print(f"\n{'='*50}")
    print("HR Robot - Nhan dien khuon mat")
    print(f"{'='*50}")
    print(f"Camera: {CAMERA_INDEX} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    print(f"Da dang ky: {len(pipeline.list_faces())} nguoi")
    print(f"Gioi han phong: {MAX_ROOM_CAPACITY} nguoi")
    print(f"Nguong nhan dien: {pipeline.threshold}")
    print(f"Tan suat detect mat: 1/{max(1, FACE_DETECT_INTERVAL)} frame")
    print("Diem danh: thu cong bang 2 nut tren man hinh")
    print(f"\nPhim tat:")
    print(f"  R = Dang ky khuon mat moi")
    print(f"  D = Xoa khuon mat da dang ky")
    print(f"  L = Xem danh sach da dang ky")
    print(f"  Q = Thoat")
    print(f"{'='*50}\n")

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0

    # Attendance manager (use shared instance if provided)
    attendance = attendance_manager or AttendanceManager(pipeline.repo)

    runner = AsyncPersonGatedPipeline(
        face_pipeline=pipeline,
        attendance_manager=None,
        on_status_update=None,
    )
    runner.start()

    last_frame = None
    last_snapshot_id = -1
    window_name = "HR Robot - Face Recognition"
    cv2.namedWindow(window_name)
    ui_state = {
        "message": "Canh dung 1 thanh vien da nhan dien vao khung hinh de thao tac",
        "message_ttl_until": 0.0,
        "selected_face": None,
        "results": [],
        "snapshot": None,
        "pending_checkout": None,
    }

    def _set_message(text: str, ttl: float = 2.5):
        ui_state["message"] = text
        ui_state["message_ttl_until"] = time.time() + ttl

    def _inside(rect, x, y) -> bool:
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _single_known_face(face_results: list) -> dict | None:
        known_faces = [r for r in face_results if r.get("member_id") is not None]
        if len(known_faces) != 1:
            return None
        return known_faces[0]

    def _mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return

        selected_face = ui_state.get("selected_face")
        if _inside(ui_state.get("cancel_rect", (0, 0, -1, -1)), x, y):
            ui_state["pending_checkout"] = None
            _set_message("Da huy thao tac check-out")
            return

        if _inside(ui_state.get("confirm_rect", (0, 0, -1, -1)), x, y):
            pending = ui_state.get("pending_checkout")
            if pending is None:
                return
            if selected_face is None or selected_face.get("member_id") != pending["member_id"]:
                ui_state["pending_checkout"] = None
                _set_message("Nguoi trong khung da thay doi, vui long thao tac lai", ttl=3.0)
                return
            result = attendance.manual_checkout(pending["member_id"])
            ui_state["pending_checkout"] = None
            _set_message(result["message"], ttl=3.0)
            if result.get("success") and on_status_update:
                on_status_update(_build_manual_status_summary(attendance, ui_state["results"], ui_state["snapshot"]))
            return

        if _inside(ui_state.get("checkin_rect", (0, 0, -1, -1)), x, y):
            ui_state["pending_checkout"] = None
            if selected_face is None:
                _set_message("Check-in yeu cau dung 1 thanh vien hop le trong khung")
                return
            result = attendance.manual_checkin(selected_face["member_id"])
            _set_message(result["message"], ttl=3.0)
            if result.get("success") and on_status_update:
                on_status_update(_build_manual_status_summary(attendance, ui_state["results"], ui_state["snapshot"]))
            return

        if _inside(ui_state.get("checkout_rect", (0, 0, -1, -1)), x, y):
            if selected_face is None:
                _set_message("Check-out yeu cau dung 1 thanh vien hop le trong khung")
                return
            pending = ui_state.get("pending_checkout")
            if pending and pending["member_id"] == selected_face["member_id"]:
                _set_message(f"Dang cho xac nhan check-out cho {selected_face['name']}")
                return
            ui_state["pending_checkout"] = {
                "member_id": selected_face["member_id"],
                "name": selected_face["name"],
            }
            _set_message(f"Cho xac nhan check-out cho {selected_face['name']}", ttl=5.0)

    cv2.setMouseCallback(window_name, _mouse_callback)
    last_status_push = 0.0

    while True:
        snapshot = runner.get_latest_snapshot()
        if snapshot.frame is None:
            time.sleep(0.005)
            continue
        if snapshot.frame_id == last_snapshot_id:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            time.sleep(0.005)
            continue
        last_snapshot_id = snapshot.frame_id

        frame = snapshot.frame
        results = snapshot.face_results or []
        last_frame = frame
        selected_face = _single_known_face(results)
        ui_state["selected_face"] = selected_face
        ui_state["results"] = results
        ui_state["snapshot"] = snapshot
        pending = ui_state.get("pending_checkout")
        if pending and (selected_face is None or selected_face.get("member_id") != pending["member_id"]):
            ui_state["pending_checkout"] = None
            _set_message("Da huy xac nhan check-out vi khung hinh da thay doi", ttl=3.0)
        elif ui_state.get("message_ttl_until", 0.0) and time.time() > ui_state["message_ttl_until"]:
            ui_state["message"] = ""
            ui_state["message_ttl_until"] = 0.0

        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # Draw
        vis = draw_results(frame, results, fps_display, MAX_ROOM_CAPACITY)
        vis = draw_person_boxes(vis, snapshot.person_boxes or [], snapshot.trigger)
        if USE_PERSON_GATE:
            gate_text = f"Gate: {'ON' if snapshot.trigger else 'OFF'} | hit={snapshot.gate_hits} miss={snapshot.gate_misses}"
            cv2.putText(vis, gate_text, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        vis = _draw_manual_touch_panel(vis, selected_face, attendance, ui_state)

        if on_status_update and (time.time() - last_status_push) >= 0.5:
            on_status_update(_build_manual_status_summary(attendance, results, snapshot))
            last_status_push = time.time()

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Quick register from current frame
            name = input("\nNhap ten nguoi can dang ky: ").strip()
            if name and last_frame is not None:
                result = pipeline.register_face(name, last_frame)
                print(f"  → {result['message']}")
        elif key == ord('d'):
            faces = pipeline.list_faces()
            if not faces:
                print("\n  Chua co ai de xoa!")
            else:
                print(f"\nDanh sach ({len(faces)} nguoi):")
                names = list(faces.keys())
                for i, n in enumerate(names, 1):
                    print(f"  {i}. {n} ({faces[n]['num_embeddings']} mau)")
                choice = input("Nhap so thu tu can xoa (0 = huy): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(names):
                    del_name = names[int(choice) - 1]
                    confirm = input(f"Xac nhan xoa '{del_name}'? (y/n): ").strip().lower()
                    if confirm == 'y':
                        if pipeline.delete_face(del_name):
                            print(f"  ✓ Da xoa '{del_name}'")
                        else:
                            print(f"  ✗ Khong tim thay '{del_name}'")
                    else:
                        print("  → Huy xoa")
                else:
                    print("  → Huy")
        elif key == ord('l'):
            faces = pipeline.list_faces()
            print(f"\nDanh sach ({len(faces)} nguoi):")
            for n, info in faces.items():
                print(f"  - {n}: {info['num_embeddings']} mau, "
                      f"dang ky: {info['registered_at']}")
    
    runner.stop()
    cv2.destroyAllWindows()


def mode_register(pipeline: FaceRecognitionPipeline, name: str = None):
    """Interactive face registration mode."""
    cap = open_camera()
    if cap is None:
        print("ERROR: Cannot open camera")
        return
    
    if not name:
        name = input("Nhap ten nguoi can dang ky: ").strip()
    
    if not name:
        print("Ten khong duoc de trong!")
        return
    
    print(f"\n{'='*50}")
    print(f"Dang ky khuon mat: {name}")
    print(f"{'='*50}")
    print(f"  SPACE = Chup mau (nên chup 3-5 goc khac nhau)")
    print(f"  Q     = Hoan tat")
    print(f"{'='*50}\n")
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show face detection in real-time
        faces = pipeline.app.get(frame)
        vis = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2 = [int(x) for x in face.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Info bar
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 45), (50, 50, 50), -1)
        info = f"Đăng ký: {name} | Đã chụp: {capture_count} mẫu | Faces: {len(faces)}"
        draw_texts_vn(vis, [
            (info, (10, 10), _FONT_LARGE, (0, 255, 255)),
            ("SPACE=Chụp | Q=Xong", (10, vis.shape[0] - 25), _FONT_SMALL, (200, 200, 200)),
        ])
        
        cv2.imshow("HR Robot - Dang ky khuon mat", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if len(faces) == 0:
                print("  ✗ Khong thay khuon mat, thu lai")
            elif len(faces) > 1:
                print("  ✗ Nhieu khuon mat, chi de 1 nguoi truoc camera")
            else:
                result = pipeline.register_face(name, frame)
                if result['success']:
                    capture_count += 1
                    print(f"  ✓ {result['message']}")
                else:
                    print(f"  ✗ {result['message']}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nHoan tat dang ky '{name}': {capture_count} mau")
    if capture_count >= 3:
        print("✓ Du mau de nhan dien chinh xac!")
    elif capture_count > 0:
        print("⚠ Nen chup them mau (it nhat 3 goc) de tang do chinh xac")
    else:
        print("✗ Chua chup mau nao")


def mode_list(pipeline: FaceRecognitionPipeline):
    """List all registered faces."""
    faces = pipeline.list_faces()
    
    print(f"\n{'='*50}")
    print(f"Danh sach thanh vien da dang ky: {len(faces)} nguoi")
    print(f"{'='*50}")
    
    if not faces:
        print("  (Chua co ai)")
    else:
        for name, info in faces.items():
            print(f"  [{info['num_embeddings']} mau] {name}")
            print(f"         Dang ky: {info['registered_at']}")
            if 'last_updated' in info:
                print(f"         Cap nhat: {info['last_updated']}")
    
    print()


def mode_benchmark(pipeline: FaceRecognitionPipeline):
    """Benchmark FPS on camera feed."""
    cap = open_camera()
    if cap is None:
        print("ERROR: Cannot open camera")
        return
    
    print(f"\n{'='*50}")
    print("Benchmark - Do toc do xu ly")
    print(f"{'='*50}")
    print("Dang do... (30 frames)\n")
    
    times = []
    face_counts = []
    
    # Warm up
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            pipeline.detect_faces(frame)
    
    # Benchmark
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        
        t0 = time.time()
        results = pipeline.detect_faces(frame)
        dt = time.time() - t0
        
        times.append(dt)
        face_counts.append(len(results))
        print(f"  Frame {i+1:2d}: {dt*1000:6.1f}ms | {len(results)} faces")
    
    cap.release()
    
    if times:
        avg_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        max_ms = np.max(times) * 1000
        avg_fps = 1.0 / np.mean(times)
        avg_faces = np.mean(face_counts)
        
        print(f"\n{'='*50}")
        print(f"Ket qua Benchmark ({pipeline.backend})")
        print(f"{'='*50}")
        print(f"  Inference trung binh : {avg_ms:.1f}ms")
        print(f"  Inference min/max    : {min_ms:.1f}ms / {max_ms:.1f}ms")
        print(f"  FPS trung binh       : {avg_fps:.1f}")
        print(f"  So khuon mat TB      : {avg_faces:.1f}")
        print(f"  Resolution           : {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"  Backend              : {pipeline.backend}")
        print(f"  Model                : {pipeline.model_name}")
        print(f"{'='*50}")
        
        # Save benchmark result
        benchmark = {
            'timestamp': datetime.now().isoformat(),
            'backend': pipeline.backend,
            'resolution': f'{CAMERA_WIDTH}x{CAMERA_HEIGHT}',
            'avg_inference_ms': round(avg_ms, 1),
            'avg_fps': round(avg_fps, 1),
            'min_ms': round(min_ms, 1),
            'max_ms': round(max_ms, 1),
            'frames_tested': len(times),
        }
        
        benchmark_path = PROJECT_ROOT / "docs" / "benchmark.json"
        benchmarks = []
        if benchmark_path.exists():
            with open(benchmark_path, 'r') as f:
                benchmarks = json.load(f)
        benchmarks.append(benchmark)
        with open(benchmark_path, 'w') as f:
            json.dump(benchmarks, f, indent=2)
        
        print(f"\nKet qua da luu: docs/benchmark.json")


def main():
    _warn_if_not_project_venv()

    parser = argparse.ArgumentParser(description="HR Robot - Face Recognition Pipeline")
    parser.add_argument("--register", action="store_true", help="Che do dang ky khuon mat")
    parser.add_argument("--name", type=str, help="Ten nguoi dang ky (dung voi --register)")
    parser.add_argument("--delete", type=str, metavar="NAME", help="Xoa khuon mat theo ten")
    parser.add_argument("--list", action="store_true", help="Xem danh sach da dang ky")
    parser.add_argument("--benchmark", action="store_true", help="Do toc do xu ly")
    parser.add_argument("--model", type=str, default=None, 
                       help="Model/embedding version (auto-selected by backend)")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Nguong nhan dien (0.0-1.0)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("  HR Robot - He thong nhan dien khuon mat")
    print("=" * 50)
    
    pipeline = FaceRecognitionPipeline(
        model_name=args.model,
        threshold=args.threshold,
    )
    
    if args.delete:
        if pipeline.delete_face(args.delete):
            print(f"  ✓ Đã xóa '{args.delete}'")
        else:
            print(f"  ✗ Không tìm thấy '{args.delete}'")
        return
    elif args.list:
        mode_list(pipeline)
    elif args.register:
        mode_register(pipeline, name=args.name)
    elif args.benchmark:
        mode_benchmark(pipeline)
    else:
        # Create shared runtime for pipeline + API
        attendance = AttendanceManager(pipeline.repo)
        runtime = AppRuntime(repo=pipeline.repo, attendance=attendance)
        start_api_server(runtime)
        mode_recognition(pipeline, attendance_manager=attendance, on_status_update=broadcast_status)


if __name__ == "__main__":
    main()
