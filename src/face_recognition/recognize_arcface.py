"""
ArcFace Face Recognition - ONNX Runtime implementation
Extracts 512-d face embeddings for identity matching.

Usage:
    python src/face_recognition/recognize_arcface.py              # Live camera
    python src/face_recognition/recognize_arcface.py --register    # Register mode
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    ARCFACE_MODEL_PATH, RECOGNITION_THRESHOLD, EMBEDDING_SIZE,
    FACE_DB_DIR, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
)
from src.face_detection.detect_scrfd import SCRFDDetector


class ArcFaceRecognizer:
    """
    ArcFace MobileFaceNet - Face Recognition via embedding extraction.
    Compares face embeddings against a registered database using cosine similarity.
    """
    
    ARCFACE_INPUT_SIZE = (112, 112)  # Standard ArcFace input
    
    # Standard 5-point alignment template for 112x112
    REFERENCE_LANDMARKS = np.array([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041],   # right mouth
    ], dtype=np.float32)
    
    def __init__(self, model_path: str = None, threshold: float = None):
        self.model_path = model_path or ARCFACE_MODEL_PATH
        self.threshold = threshold or RECOGNITION_THRESHOLD
        
        self.session = None
        self.input_name = None
        
        # Face database: {name: [embedding1, embedding2, ...]}
        self.face_db = {}
        
        self._load_model()
        self._load_face_db()
    
    def _load_model(self):
        """Load ArcFace ONNX model."""
        import onnxruntime as ort
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run: python scripts/download_models.py"
            )
        
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        print(f"[ArcFace] Model loaded: {self.model_path}")
        print(f"[ArcFace] Input shape: {input_shape}")
    
    def _load_face_db(self):
        """Load registered face embeddings from disk."""
        db_path = FACE_DB_DIR / "face_database.npz"
        
        if db_path.exists():
            data = np.load(db_path, allow_pickle=True)
            self.face_db = dict(data['database'].item())
            print(f"[ArcFace] Loaded {len(self.face_db)} registered faces")
        else:
            self.face_db = {}
            print("[ArcFace] No face database found (empty)")
    
    def save_face_db(self):
        """Save face database to disk."""
        db_path = FACE_DB_DIR / "face_database.npz"
        np.savez(db_path, database=self.face_db)
        print(f"[ArcFace] Saved {len(self.face_db)} faces to {db_path}")
    
    def align_face(self, image: np.ndarray, keypoints: list) -> np.ndarray:
        """
        Align face using 5-point landmarks via similarity transform.
        This is crucial for ArcFace accuracy.
        
        Args:
            image: original BGR image
            keypoints: list of 5 (x, y) tuples from SCRFD
            
        Returns:
            Aligned face image (112x112)
        """
        if keypoints is None or len(keypoints) < 5:
            # Fallback: simple crop without alignment
            return None
        
        src_pts = np.array(keypoints[:5], dtype=np.float32)
        dst_pts = self.REFERENCE_LANDMARKS.copy()
        
        # Estimate similarity transform
        transform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if transform is None:
            return None
        
        # Warp face to standard position
        aligned = cv2.warpAffine(
            image, transform, self.ARCFACE_INPUT_SIZE,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned
    
    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract 512-d embedding from aligned face image.
        
        Args:
            aligned_face: 112x112 BGR face image
            
        Returns:
            Normalized 512-d embedding vector
        """
        # Preprocess: BGR->RGB, HWC->CHW, normalize to [-1, 1]
        face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face_float = face_rgb.astype(np.float32)
        face_float = (face_float - 127.5) / 127.5  # Normalize to [-1, 1]
        
        # HWC to CHW, add batch dimension
        face_chw = np.transpose(face_float, (2, 0, 1))
        face_batch = np.expand_dims(face_chw, axis=0)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: face_batch})
        embedding = outputs[0][0]
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def register_face(self, name: str, image: np.ndarray, 
                      keypoints: list) -> bool:
        """
        Register a face into the database.
        
        Args:
            name: person's name/ID
            image: original BGR image
            keypoints: 5-point landmarks from SCRFD
            
        Returns:
            True if registration successful
        """
        aligned = self.align_face(image, keypoints)
        if aligned is None:
            print(f"[Register] Failed to align face for {name}")
            return False
        
        embedding = self.extract_embedding(aligned)
        
        if name not in self.face_db:
            self.face_db[name] = []
        
        self.face_db[name].append(embedding)
        self.save_face_db()
        
        print(f"[Register] Registered '{name}' "
              f"(total embeddings: {len(self.face_db[name])})")
        return True
    
    def recognize(self, image: np.ndarray, 
                  keypoints: list) -> tuple:
        """
        Recognize a face against the database.
        
        Args:
            image: original BGR image
            keypoints: 5-point landmarks from SCRFD
            
        Returns:
            (name, confidence) if matched, ("Unknown", 0.0) otherwise
        """
        if not self.face_db:
            return ("Unknown", 0.0)
        
        aligned = self.align_face(image, keypoints)
        if aligned is None:
            return ("Unknown", 0.0)
        
        query_embedding = self.extract_embedding(aligned)
        
        best_name = "Unknown"
        best_score = 0.0
        
        for name, embeddings in self.face_db.items():
            # Compare against all registered embeddings for this person
            db_embeddings = np.array(embeddings)
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                db_embeddings
            )[0]
            
            max_sim = float(np.max(similarities))
            
            if max_sim > best_score:
                best_score = max_sim
                best_name = name
        
        if best_score < self.threshold:
            return ("Unknown", best_score)
        
        return (best_name, best_score)


def run_recognition(detector: SCRFDDetector, recognizer: ArcFaceRecognizer):
    """Run face recognition on live camera feed."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print(f"\n[Camera] Recognition mode started")
    print(f"[Camera] Registered faces: {list(recognizer.face_db.keys())}")
    print("Press 'q' to quit\n")
    
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 1: Detect faces
        faces = detector.detect(frame)
        
        # Step 2: Recognize each face
        vis = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face['bbox']]
            kps = face.get('keypoints')
            
            name, score = recognizer.recognize(frame, kps)
            
            # Color: green=known, red=unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({score:.2f})"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        info = f"FPS: {fps_display} | Faces: {len(faces)} | DB: {len(recognizer.face_db)} people"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("HR Robot - Face Recognition", vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def run_registration(detector: SCRFDDetector, recognizer: ArcFaceRecognizer):
    """Interactive face registration mode."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    name = input("Enter person name to register: ").strip()
    if not name:
        print("Name cannot be empty")
        return
    
    print(f"\nRegistering '{name}'")
    print("Press SPACE to capture, 'q' when done")
    print("Capture 3-5 images from different angles for best results\n")
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detector.detect(frame)
        vis = detector.draw_faces(frame, faces)
        
        info = f"Registering: {name} | Captured: {capture_count} | Faces in frame: {len(faces)}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, "SPACE=capture, Q=done", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("HR Robot - Face Registration", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to capture
            if len(faces) == 0:
                print("  No face detected, try again")
            elif len(faces) > 1:
                print("  Multiple faces detected, please show only one face")
            else:
                kps = faces[0].get('keypoints')
                if recognizer.register_face(name, frame, kps):
                    capture_count += 1
                    print(f"  ✓ Captured {capture_count}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nRegistration complete for '{name}': {capture_count} captures")


def main():
    parser = argparse.ArgumentParser(description="ArcFace Face Recognition")
    parser.add_argument("--register", action="store_true", help="Enter registration mode")
    parser.add_argument("--model", type=str, default=None, help="ArcFace model path")
    parser.add_argument("--threshold", type=float, default=None, help="Recognition threshold")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HR Robot - ArcFace Face Recognition (ONNX)")
    print("=" * 60)
    
    detector = SCRFDDetector()
    recognizer = ArcFaceRecognizer(
        model_path=args.model,
        threshold=args.threshold,
    )
    
    if args.register:
        run_registration(detector, recognizer)
    else:
        run_recognition(detector, recognizer)


if __name__ == "__main__":
    main()
