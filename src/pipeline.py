"""
HR Robot - Complete Face Recognition Pipeline (Phase 1)
Unified pipeline: Detection → Alignment → Recognition → Registration

Uses InsightFace's FaceAnalysis for reliable out-of-the-box performance.
This is the PRIMARY approach for Phase 1 prototype.

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
    RECOGNITION_THRESHOLD, FACE_DB_DIR, MAX_ROOM_CAPACITY
)


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline using InsightFace.
    Handles: detection, alignment, embedding, matching, registration.
    """
    
    def __init__(self, model_name: str = "buffalo_l", threshold: float = None):
        self.threshold = threshold or RECOGNITION_THRESHOLD
        self.face_db = {}  # {name: [embedding1, embedding2, ...]}
        self.face_info = {}  # {name: {registered_at, num_embeddings, ...}}
        
        self._init_insightface(model_name)
        self._load_database()
    
    def _init_insightface(self, model_name: str):
        """Initialize InsightFace FaceAnalysis."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            print("ERROR: insightface not installed!")
            print("Run: pip install insightface onnxruntime")
            sys.exit(1)
        
        print(f"[Pipeline] Loading InsightFace model: {model_name}")
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition'],
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        print(f"[Pipeline] Model loaded successfully ✓")
    
    def _load_database(self):
        """Load face database from disk."""
        db_path = FACE_DB_DIR / "face_database.json"
        emb_path = FACE_DB_DIR / "embeddings.npz"
        
        if db_path.exists() and emb_path.exists():
            # Load metadata
            with open(db_path, 'r', encoding='utf-8') as f:
                self.face_info = json.load(f)
            
            # Load embeddings
            data = np.load(emb_path, allow_pickle=True)
            for name in self.face_info:
                key = f"emb_{name}"
                if key in data:
                    self.face_db[name] = data[key]
            
            print(f"[Pipeline] Loaded {len(self.face_db)} registered faces ✓")
            for name, info in self.face_info.items():
                print(f"  - {name} ({info.get('num_embeddings', '?')} samples)")
        else:
            print("[Pipeline] No face database found (starting fresh)")
    
    def _save_database(self):
        """Save face database to disk."""
        FACE_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as JSON (human readable)
        db_path = FACE_DB_DIR / "face_database.json"
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(self.face_info, f, ensure_ascii=False, indent=2)
        
        # Save embeddings as NPZ (efficient binary)
        emb_path = FACE_DB_DIR / "embeddings.npz"
        emb_dict = {f"emb_{name}": np.array(embs) for name, embs in self.face_db.items()}
        np.savez(emb_path, **emb_dict)
        
        print(f"[Pipeline] Database saved ({len(self.face_db)} faces) ✓")
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect and analyze all faces in an image.
        
        Returns list of dicts with:
            bbox, score, embedding, keypoints, name, confidence
        """
        faces = self.app.get(image)
        results = []
        
        for face in faces:
            result = {
                'bbox': [int(x) for x in face.bbox],
                'score': float(face.det_score),
                'embedding': face.normed_embedding,
                'keypoints': face.kps.tolist() if face.kps is not None else None,
            }
            
            # Match against database
            name, confidence = self._match_face(face.normed_embedding)
            result['name'] = name
            result['confidence'] = confidence
            
            results.append(result)
        
        return results
    
    def _match_face(self, embedding: np.ndarray) -> tuple:
        """Match an embedding against the database."""
        if not self.face_db:
            return ("Người lạ", 0.0)
        
        best_name = "Người lạ"
        best_score = 0.0
        
        for name, embeddings in self.face_db.items():
            # Cosine similarity with all registered embeddings
            similarities = np.dot(embeddings, embedding)
            max_sim = float(np.max(similarities))
            
            if max_sim > best_score:
                best_score = max_sim
                best_name = name
        
        if best_score < self.threshold:
            return ("Người lạ", best_score)
        
        return (best_name, best_score)
    
    def register_face(self, name: str, image: np.ndarray) -> dict:
        """
        Register the largest UNKNOWN face in the image.
        Skips faces that are already registered.
        
        Returns:
            {'success': bool, 'message': str}
        """
        faces = self.app.get(image)
        
        if not faces:
            return {'success': False, 'message': 'Không phát hiện khuôn mặt'}
        
        # Filter out already-registered faces
        unknown_faces = []
        for f in faces:
            emb = f.normed_embedding
            recognized_name, score = self._match_face(emb)
            if recognized_name == "Người lạ":
                unknown_faces.append(f)
        
        if not unknown_faces:
            return {'success': False, 'message': 'Tất cả khuôn mặt trong frame đã được đăng ký rồi'}
        
        if len(unknown_faces) > 1:
            # Pick the largest unknown face
            face = max(unknown_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        else:
            face = unknown_faces[0]
        
        embedding = face.normed_embedding
        
        # Add to database
        if name not in self.face_db:
            self.face_db[name] = np.empty((0, embedding.shape[0]), dtype=np.float32)
            self.face_info[name] = {
                'registered_at': datetime.now().isoformat(),
                'num_embeddings': 0,
            }
        
        # Stack new embedding onto existing array
        self.face_db[name] = np.vstack([self.face_db[name], embedding.reshape(1, -1)])
        self.face_info[name]['num_embeddings'] = len(self.face_db[name])
        self.face_info[name]['last_updated'] = datetime.now().isoformat()
        
        self._save_database()
        
        return {
            'success': True, 
            'message': f"Đã đăng ký '{name}' (mẫu #{self.face_info[name]['num_embeddings']})"
        }
    
    def delete_face(self, name: str) -> bool:
        """Remove a face from the database."""
        if name in self.face_db:
            del self.face_db[name]
            del self.face_info[name]
            self._save_database()
            return True
        return False
    
    def list_faces(self) -> dict:
        """List all registered faces."""
        return self.face_info.copy()
    
    def get_headcount(self, image: np.ndarray) -> int:
        """Count number of faces in image."""
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


def put_text_vn(img, text, pos, font=None, color=(255, 255, 255)):
    """Draw Vietnamese (Unicode) text on an OpenCV image using PIL."""
    if font is None:
        font = _FONT_MEDIUM
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
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
        put_text_vn(vis, label, (x1 + 2, y1 - th - 10), _FONT_MEDIUM, (255, 255, 255))
    
    # Top bar info
    bar_color = (0, 0, 200) if is_overloaded else (50, 50, 50)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 45), bar_color, -1)
    
    info = f"FPS: {fps} | Số người: {headcount}/{headcount_limit}"
    if is_overloaded:
        info += " | !! QUÁ TẢI !!"
    
    put_text_vn(vis, info, (10, 10), _FONT_LARGE, (255, 255, 255))
    
    # Known/Unknown count
    known = sum(1 for r in results if r['name'] != "Người lạ")
    unknown = headcount - known
    status = f"Thành viên: {known} | Người lạ: {unknown}"
    put_text_vn(vis, status, (10, vis.shape[0] - 25), _FONT_SMALL, (200, 200, 200))
    
    return vis


def mode_recognition(pipeline: FaceRecognitionPipeline):
    """Live face recognition mode."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print(f"\n{'='*50}")
    print("HR Robot - Nhan dien khuon mat")
    print(f"{'='*50}")
    print(f"Camera: {CAMERA_INDEX} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    print(f"Da dang ky: {len(pipeline.face_db)} nguoi")
    print(f"Gioi han phong: {MAX_ROOM_CAPACITY} nguoi")
    print(f"Nguong nhan dien: {pipeline.threshold}")
    print(f"\nPhim tat:")
    print(f"  R = Dang ky khuon mat moi")
    print(f"  D = Xoa khuon mat da dang ky")
    print(f"  L = Xem danh sach da dang ky")
    print(f"  Q = Thoat")
    print(f"{'='*50}\n")
    
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    frame_count = 0
    detect_interval = 3  # Chỉ detect mỗi 3 frame (giảm lag)
    cached_results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect & recognize (chỉ mỗi N frame, frame còn lại dùng kết quả cũ)
        frame_count += 1
        if frame_count % detect_interval == 0:
            cached_results = pipeline.detect_faces(frame)
        results = cached_results
        
        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Draw
        vis = draw_results(frame, results, fps_display, MAX_ROOM_CAPACITY)
        cv2.imshow("HR Robot - Face Recognition", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Quick register from current frame
            name = input("\nNhap ten nguoi can dang ky: ").strip()
            if name:
                result = pipeline.register_face(name, frame)
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
    
    cap.release()
    cv2.destroyAllWindows()


def mode_register(pipeline: FaceRecognitionPipeline, name: str = None):
    """Interactive face registration mode."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
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
        put_text_vn(vis, info, (10, 10), _FONT_LARGE, (0, 255, 255))
        
        put_text_vn(vis, "SPACE=Chụp | Q=Xong", (10, vis.shape[0] - 25), _FONT_SMALL, (200, 200, 200))
        
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
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
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
        print(f"Ket qua Benchmark (CPU - ONNX Runtime)")
        print(f"{'='*50}")
        print(f"  Inference trung binh : {avg_ms:.1f}ms")
        print(f"  Inference min/max    : {min_ms:.1f}ms / {max_ms:.1f}ms")
        print(f"  FPS trung binh       : {avg_fps:.1f}")
        print(f"  So khuon mat TB      : {avg_faces:.1f}")
        print(f"  Resolution           : {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"  Backend              : CPU (ONNX Runtime)")
        print(f"{'='*50}")
        
        # Save benchmark result
        benchmark = {
            'timestamp': datetime.now().isoformat(),
            'backend': 'CPU_ONNX',
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
    parser = argparse.ArgumentParser(description="HR Robot - Face Recognition Pipeline")
    parser.add_argument("--register", action="store_true", help="Che do dang ky khuon mat")
    parser.add_argument("--name", type=str, help="Ten nguoi dang ky (dung voi --register)")
    parser.add_argument("--delete", type=str, metavar="NAME", help="Xoa khuon mat theo ten")
    parser.add_argument("--list", action="store_true", help="Xem danh sach da dang ky")
    parser.add_argument("--benchmark", action="store_true", help="Do toc do xu ly")
    parser.add_argument("--model", type=str, default="buffalo_l", 
                       help="InsightFace model (buffalo_l hoac buffalo_sc)")
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
        mode_recognition(pipeline)


if __name__ == "__main__":
    main()
