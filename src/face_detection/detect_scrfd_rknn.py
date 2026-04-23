"""
SCRFD Face Detector - RKNNLite NPU implementation for RK3588S
Lightweight face detection accelerated on NPU cores.

Usage:
    python src/face_detection/detect_scrfd_rknn.py              # Live camera
    python src/face_detection/detect_scrfd_rknn.py --image path  # Single image
"""

import sys
import time
import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_utils import open_camera

from config.settings import (
    SCRFD_RKNN_PATH, DETECTION_THRESHOLD, NMS_THRESHOLD,
    DETECTION_INPUT_SIZE, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    SCRFD_NPU_CORE_MASK,
)


def _load_font(size: int):
    """Load a font with broad Unicode coverage."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
    return ImageFont.load_default()


_FONT_LABEL = _load_font(18)
_FONT_INFO = _load_font(20)


def _draw_text_unicode(image: np.ndarray, text: str, pos: tuple[int, int],
                       color: tuple[int, int, int], font=None) -> np.ndarray:
    """Draw Unicode text on OpenCV image using PIL."""
    font = font or _FONT_LABEL
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class SCRFDRKNNDetector:
    """
    SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
    Uses RKNNLite for NPU inference on RK3588S.
    
    Key differences from ONNX version:
    - RKNN handles mean/std normalization internally (baked into model)
    - Input format: uint8 NHWC (1, H, W, 3) - no transpose needed
    - Thread-safe with lock around inference
    """
    
    def __init__(self, model_path: str = None, conf_threshold: float = None,
                 nms_threshold: float = None, input_size: tuple = None,
                 core_mask: int = None):
        self.model_path = model_path or SCRFD_RKNN_PATH
        self.conf_threshold = conf_threshold or DETECTION_THRESHOLD
        self.nms_threshold = nms_threshold or NMS_THRESHOLD
        self.input_size = input_size or DETECTION_INPUT_SIZE
        self.core_mask = core_mask or SCRFD_NPU_CORE_MASK
        
        self.rknn = None
        self._lock = threading.Lock()
        
        self._load_model()
    
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
        
        print(f"[SCRFD-RKNN] Model loaded: {self.model_path}")
        print(f"[SCRFD-RKNN] NPU core_mask: {self.core_mask}")
        print(f"[SCRFD-RKNN] Input size: {self.input_size}")
    
    def _preprocess(self, image: np.ndarray) -> tuple:
        """
        Preprocess image for SCRFD RKNN inference.
        RKNN handles normalization internally, so just resize + letterbox.
        
        Returns:
            input_data: uint8 NHWC tensor (1, H, W, 3)
            scale: resize scale for coordinate mapping back
        """
        h, w = image.shape[:2]
        input_w, input_h = self.input_size
        
        # Calculate scale to maintain aspect ratio
        scale = min(input_w / w, input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with aspect ratio preservation
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to input size (letterbox)
        padded = np.full((input_h, input_w, 3), 0, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # BGR to RGB (RKNN model expects RGB)
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # NHWC format, keep uint8 - RKNN handles normalization
        input_data = np.expand_dims(padded, axis=0)
        
        return input_data, scale
    
    def _postprocess(self, outputs: list, scale: float,
                     orig_h: int, orig_w: int) -> list:
        """
        Post-process SCRFD outputs to get face bounding boxes and keypoints.
        
        Returns:
            List of dicts: [{
                'bbox': [x1, y1, x2, y2], 
                'score': float,
                'keypoints': [(x,y), ...] or None  # 5 facial landmarks
            }, ...]
        """
        faces = []
        
        # SCRFD outputs: scores, bboxes, (keypoints) for each stride (8, 16, 32)
        fmc = 3  # feature map count
        strides = [8, 16, 32]
        
        has_kps = len(outputs) == 9  # 3 strides × (score + bbox + kps)
        
        for idx in range(fmc):
            score_blob = outputs[idx]
            bbox_blob = outputs[idx + fmc]
            kps_blob = outputs[idx + fmc * 2] if has_kps else None
            
            stride = strides[idx]
            
            # Get score predictions — RKNN output has no batch dim: (N, 1)
            scores = score_blob.reshape(-1)  # (H*W*anchors,)
            
            # Filter by confidence threshold
            mask = scores > self.conf_threshold
            if not mask.any():
                continue
                
            filtered_scores = scores[mask]
            filtered_bboxes = bbox_blob[mask]  # (N, 4)
            filtered_kps = kps_blob[mask] if kps_blob is not None else None
            
            # Decode bounding boxes
            input_h, input_w = self.input_size
            feat_h = input_h // stride
            feat_w = input_w // stride
            
            # Generate anchor centers
            num_anchors = 2
            indices = np.where(mask)[0]
            anchor_indices = indices // num_anchors
            
            cy = (anchor_indices // feat_w) * stride
            cx = (anchor_indices % feat_w) * stride
            
            # Decode bbox (distance format: left, top, right, bottom)
            x1 = (cx - filtered_bboxes[:, 0] * stride) / scale
            y1 = (cy - filtered_bboxes[:, 1] * stride) / scale
            x2 = (cx + filtered_bboxes[:, 2] * stride) / scale
            y2 = (cy + filtered_bboxes[:, 3] * stride) / scale
            
            # Clip to image bounds
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)
            
            for i in range(len(filtered_scores)):
                face = {
                    'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    'score': float(filtered_scores[i]),
                }
                
                # Decode keypoints if available
                if filtered_kps is not None:
                    kps = filtered_kps[i].reshape(-1, 2)
                    keypoints = []
                    for kp in kps:
                        kx = (cx[i] + kp[0] * stride) / scale
                        ky = (cy[i] + kp[1] * stride) / scale
                        keypoints.append((float(kx), float(ky)))
                    face['keypoints'] = keypoints
                else:
                    face['keypoints'] = None
                
                faces.append(face)
        
        # Apply NMS
        if len(faces) > 0:
            faces = self._nms(faces)
        
        return faces
    
    def _nms(self, faces: list) -> list:
        """Non-Maximum Suppression to remove duplicate detections."""
        if len(faces) == 0:
            return []
        
        scores = np.array([f['score'] for f in faces])
        
        # cv2.dnn.NMSBoxes expects [x, y, w, h] format
        boxes_xywh = []
        for f in faces:
            x1, y1, x2, y2 = f['bbox']
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, scores.tolist(),
            self.conf_threshold, self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [faces[i] for i in indices]
        return []
    
    def detect(self, image: np.ndarray) -> list:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image (numpy array from cv2)
            
        Returns:
            List of face dicts with 'bbox', 'score', 'keypoints'
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        h, w = image.shape[:2]
        input_data, scale = self._preprocess(image)
        
        # Run inference (thread-safe)
        with self._lock:
            outputs = self.rknn.inference(inputs=[input_data])
        
        # Debug: dump raw output stats (remove after confirming detection works)
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[SCRFD-RKNN DEBUG] input: shape={input_data.shape}, dtype={input_data.dtype}, "
                  f"min={input_data.min()}, max={input_data.max()}")
            print(f"[SCRFD-RKNN DEBUG] num_outputs={len(outputs)}")
            for i, out in enumerate(outputs):
                arr = np.asarray(out)
                print(f"[SCRFD-RKNN DEBUG] out[{i}]: shape={arr.shape}, dtype={arr.dtype}, "
                      f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
        
        # Post-process
        faces = self._postprocess(outputs, scale, h, w)
        
        return faces
    
    def draw_faces(self, image: np.ndarray, faces: list) -> np.ndarray:
        """Draw detected faces on image for visualization."""
        vis = image.copy()
        
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face['bbox']]
            score = face['score']
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw score
            label = f"{score:.2f}"
            vis = _draw_text_unicode(
                vis,
                label,
                (x1, max(0, y1 - 24)),
                (0, 255, 0),
                _FONT_LABEL,
            )
            
            # Draw keypoints
            if face.get('keypoints'):
                colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0),
                         (255, 0, 0), (255, 0, 255)]
                for i, (kx, ky) in enumerate(face['keypoints']):
                    color = colors[i % len(colors)]
                    cv2.circle(vis, (int(kx), int(ky)), 3, color, -1)
        
        return vis
    
    def release(self):
        """Release NPU resources."""
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
            print("[SCRFD-RKNN] Released NPU resources")


def run_camera(detector: SCRFDRKNNDetector):
    """Run face detection on live camera feed."""
    cap = open_camera()
    if cap is None:
        print("Error: Cannot open camera")
        return
    
    print(f"\n[Camera] Started (index={CAMERA_INDEX}, {CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    print("Press 'q' to quit\n")
    
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            t0 = time.time()
            faces = detector.detect(frame)
            dt = time.time() - t0
            
            # Draw results
            vis = detector.draw_faces(frame, faces)
            
            # FPS counter
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Info overlay
            info = f"FPS: {fps_display} | Faces: {len(faces)} | Inference: {dt*1000:.1f}ms"
            vis = _draw_text_unicode(vis, info, (10, 10), (0, 0, 255), _FONT_INFO)
            
            cv2.imshow("HR Robot - Face Detection (RKNN)", vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


def run_image(detector: SCRFDRKNNDetector, image_path: str):
    """Run face detection on a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image: {image_path}")
        return
    
    t0 = time.time()
    faces = detector.detect(image)
    dt = time.time() - t0
    
    print(f"\nDetected {len(faces)} face(s) in {dt*1000:.1f}ms")
    for i, face in enumerate(faces):
        print(f"  Face {i+1}: score={face['score']:.3f}, bbox={face['bbox']}")
    
    vis = detector.draw_faces(image, faces)
    cv2.imshow("Detection Result (RKNN)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector.release()


def main():
    parser = argparse.ArgumentParser(description="SCRFD Face Detection (RKNN)")
    parser.add_argument("--image", type=str, help="Path to test image (omit for camera)")
    parser.add_argument("--model", type=str, default=None, help="Path to RKNN model")
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold")
    parser.add_argument("--core-mask", type=int, default=None, help="NPU core mask")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HR Robot - SCRFD Face Detection (RKNN NPU)")
    print("=" * 60)
    
    detector = SCRFDRKNNDetector(
        model_path=args.model,
        conf_threshold=args.threshold,
        core_mask=args.core_mask,
    )
    
    if args.image:
        run_image(detector, args.image)
    else:
        run_camera(detector)


if __name__ == "__main__":
    main()
