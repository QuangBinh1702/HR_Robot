"""
MobileFaceNet Face Recognition - RKNNLite NPU implementation for RK3588S
Extracts face embeddings accelerated on NPU cores.

Usage:
    python src/face_recognition/recognize_mobilefacenet_rknn.py              # Live camera
    python src/face_recognition/recognize_mobilefacenet_rknn.py --register    # Register mode
"""

import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_utils import open_camera

from config.settings import (
    ARCFACE_RKNN_PATH, RECOGNITION_THRESHOLD, EMBEDDING_SIZE,
    FACE_DB_DIR, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    ARCFACE_NPU_CORE_MASK,
)


class MobileFaceNetRKNNRecognizer:
    """
    MobileFaceNet - Face Recognition via embedding extraction on NPU.
    Uses RKNNLite for NPU inference on RK3588S.
    
    Key differences from ONNX ArcFace version:
    - RKNN handles normalization internally (baked into model)
    - Input format: uint8 NHWC (1, 112, 112, 3) - no transpose needed
    - Thread-safe with lock around inference
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
    
    def __init__(self, model_path: str = None, threshold: float = None,
                 core_mask: int = None):
        self.model_path = model_path or ARCFACE_RKNN_PATH
        self.threshold = threshold or RECOGNITION_THRESHOLD
        self.core_mask = core_mask or ARCFACE_NPU_CORE_MASK
        self.model_label = self._infer_model_label(self.model_path)
        
        self.rknn = None
        self._lock = threading.Lock()
        
        # Face database: {name: [embedding1, embedding2, ...]}
        self.face_db = {}
        
        self._load_model()
        self._load_face_db()

    @staticmethod
    def _infer_model_label(model_path: str) -> str:
        model_name = Path(model_path).name.lower()
        if "ghost" in model_name:
            return "GhostFace-RKNN"
        if "mobile" in model_name or "mbf" in model_name:
            return "MobileFaceNet-RKNN"
        return "FaceRec-RKNN"
    
    def _load_model(self):
        """Load RKNN model and init NPU runtime."""
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise ImportError(
                "rknnlite is not available. Install RKNN Toolkit Lite2:\n"
                "  pip install rknn-toolkit-lite2\n"
                "This module only runs on RK3588S boards with NPU."
            )
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run: python scripts/download_models.py"
            )
        
        self.rknn = RKNNLite()
        
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {self.model_path} (ret={ret})")
        
        ret = self.rknn.init_runtime(core_mask=self.core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime (core_mask={self.core_mask}, ret={ret})")
        
        print(f"[{self.model_label}] Model loaded: {self.model_path}")
        print(f"[{self.model_label}] NPU core_mask: {self.core_mask}")
    
    def _load_face_db(self):
        """Load registered face embeddings from disk."""
        db_path = FACE_DB_DIR / "face_database.npz"
        
        if db_path.exists():
            data = np.load(db_path, allow_pickle=True)
            self.face_db = dict(data['database'].item())
            print(f"[{self.model_label}] Loaded {len(self.face_db)} registered faces")
        else:
            self.face_db = {}
            print(f"[{self.model_label}] No face database found (empty)")
    
    def save_face_db(self):
        """Save face database to disk."""
        db_path = FACE_DB_DIR / "face_database.npz"
        np.savez(db_path, database=self.face_db)
        print(f"[{self.model_label}] Saved {len(self.face_db)} faces to {db_path}")
    
    def align_face(self, image: np.ndarray, keypoints: list) -> Optional[np.ndarray]:
        """
        Align face using 5-point landmarks via similarity transform.
        This is crucial for recognition accuracy.
        
        Args:
            image: original BGR image
            keypoints: list of 5 (x, y) tuples from SCRFD
            
        Returns:
            Aligned face image (112x112) or None
        """
        if keypoints is None or len(keypoints) < 5:
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
        Extract embedding from aligned face image.
        
        Args:
            aligned_face: 112x112 BGR face image
            
        Returns:
            Normalized embedding vector
        """
        # BGR to RGB - RKNN model expects RGB
        face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        # NHWC format, keep uint8 - RKNN handles normalization
        input_data = np.expand_dims(face_rgb, axis=0)
        
        # Run inference (thread-safe)
        with self._lock:
            outputs = self.rknn.inference(inputs=[input_data])
        
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
    
    def release(self):
        """Release NPU resources."""
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
            print(f"[{self.model_label}] Released NPU resources")


def run_recognition(detector, recognizer: MobileFaceNetRKNNRecognizer):
    """Run face recognition on live camera feed."""
    cap = open_camera()
    if cap is None:
        print("Error: Cannot open camera")
        return
    
    print(f"\n[Camera] Recognition mode started")
    print(f"[Camera] Registered faces: {list(recognizer.face_db.keys())}")
    print("Press 'q' to quit\n")
    
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    
    try:
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
            
            cv2.imshow("HR Robot - Face Recognition (RKNN)", vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        recognizer.release()


def main():
    parser = argparse.ArgumentParser(description="MobileFaceNet Face Recognition (RKNN)")
    parser.add_argument("--register", action="store_true", help="Enter registration mode")
    parser.add_argument("--model", type=str, default=None, help="RKNN model path")
    parser.add_argument("--threshold", type=float, default=None, help="Recognition threshold")
    parser.add_argument("--core-mask", type=int, default=None, help="NPU core mask")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HR Robot - MobileFaceNet Face Recognition (RKNN NPU)")
    print("=" * 60)
    
    from src.face_detection.detect_scrfd_rknn import SCRFDRKNNDetector
    
    detector = SCRFDRKNNDetector()
    recognizer = MobileFaceNetRKNNRecognizer(
        model_path=args.model,
        threshold=args.threshold,
        core_mask=args.core_mask,
    )
    
    run_recognition(detector, recognizer)


if __name__ == "__main__":
    main()
