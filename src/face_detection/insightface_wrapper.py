"""
InsightFace Wrapper - Simplified API using insightface library.
Alternative approach: use insightface's built-in FaceAnalysis 
which bundles detection + recognition automatically.

This can be used as a quick prototype while we build the 
custom SCRFD + ArcFace pipeline for RKNN deployment.

Usage:
    python src/face_detection/insightface_wrapper.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FACE_DB_DIR


class InsightFaceWrapper:
    """
    High-level wrapper around InsightFace's FaceAnalysis.
    Automatically handles detection + recognition + alignment.
    
    Note: This uses InsightFace's bundled models (buffalo_l or buffalo_sc).
    For RKNN deployment, use the custom SCRFD + ArcFace pipeline instead.
    """
    
    def __init__(self, model_name: str = "buffalo_sc", ctx_id: int = 0):
        """
        Args:
            model_name: "buffalo_l" (accurate) or "buffalo_sc" (small/fast)
            ctx_id: 0=CPU, positive=GPU id
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError("Please install insightface: pip install insightface")
        
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition'],
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # Face database
        self.face_db = {}
        self._load_db()
        
        print(f"[InsightFace] Initialized with model: {model_name}")
    
    def _load_db(self):
        db_path = FACE_DB_DIR / "insightface_db.npz"
        if db_path.exists():
            data = np.load(db_path, allow_pickle=True)
            self.face_db = dict(data['database'].item())
            print(f"[InsightFace] Loaded {len(self.face_db)} faces")
    
    def _save_db(self):
        db_path = FACE_DB_DIR / "insightface_db.npz"
        np.savez(db_path, database=self.face_db)
    
    def detect_and_recognize(self, image: np.ndarray) -> list:
        """
        Detect and recognize all faces in image.
        
        Returns:
            List of dicts: [{
                'bbox': [x1,y1,x2,y2],
                'score': float,
                'embedding': np.array(512,),
                'name': str,
                'confidence': float,
                'keypoints': np.array(5,2)
            }]
        """
        faces = self.app.get(image)
        results = []
        
        for face in faces:
            result = {
                'bbox': face.bbox.tolist(),
                'score': float(face.det_score),
                'embedding': face.normed_embedding,
                'keypoints': face.kps.tolist() if face.kps is not None else None,
            }
            
            # Match against database
            name, confidence = self._match(face.normed_embedding)
            result['name'] = name
            result['confidence'] = confidence
            
            results.append(result)
        
        return results
    
    def _match(self, embedding: np.ndarray) -> tuple:
        """Match embedding against database."""
        if not self.face_db:
            return ("Unknown", 0.0)
        
        best_name = "Unknown"
        best_score = 0.0
        
        for name, embeddings in self.face_db.items():
            for db_emb in embeddings:
                sim = np.dot(embedding, db_emb)
                if sim > best_score:
                    best_score = float(sim)
                    best_name = name
        
        if best_score < 0.45:
            return ("Unknown", best_score)
        
        return (best_name, best_score)
    
    def register(self, name: str, image: np.ndarray) -> bool:
        """Register the largest face in image."""
        faces = self.app.get(image)
        if not faces:
            return False
        
        # Pick largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        if name not in self.face_db:
            self.face_db[name] = []
        
        self.face_db[name].append(face.normed_embedding)
        self._save_db()
        return True


def main():
    """Quick demo with InsightFace wrapper."""
    wrapper = InsightFaceWrapper(model_name="buffalo_sc")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("\nPress 'r' to register, 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = wrapper.detect_and_recognize(frame)
        
        vis = frame.copy()
        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r['bbox']]
            color = (0, 255, 0) if r['name'] != "Unknown" else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{r['name']} ({r['confidence']:.2f})"
            cv2.putText(vis, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("InsightFace Demo", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            name = input("Enter name: ").strip()
            if name and wrapper.register(name, frame):
                print(f"  ✓ Registered {name}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
