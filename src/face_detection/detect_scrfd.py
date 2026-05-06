"""
SCRFD Face Detector - ONNX Runtime implementation
Lightweight face detection optimized for edge deployment.

Usage:
    python src/face_detection/detect_scrfd.py              # Live camera
    python src/face_detection/detect_scrfd.py --image path  # Single image
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    SCRFD_MODEL_PATH, DETECTION_THRESHOLD, NMS_THRESHOLD,
    DETECTION_INPUT_SIZE, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
)
from src.onnxruntime_cuda import configure_onnxruntime_cuda_dll_paths


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


class SCRFDDetector:
    """
    SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
    Supports ONNX Runtime inference for CPU/GPU.
    
    For NPU (RKNN) inference, see Phase 2: detect_scrfd_rknn.py
    """
    
    def __init__(self, model_path: str = None, conf_threshold: float = None, 
                 nms_threshold: float = None, input_size: tuple = None):
        self.model_path = model_path or SCRFD_MODEL_PATH
        self.conf_threshold = conf_threshold or DETECTION_THRESHOLD
        self.nms_threshold = nms_threshold or NMS_THRESHOLD
        self.input_size = input_size or DETECTION_INPUT_SIZE
        
        self.session = None
        self.input_name = None
        self.output_names = None
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model with onnxruntime."""
        try:
            configure_onnxruntime_cuda_dll_paths()
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run: python scripts/download_models.py"
            )
        
        # Create session with CPU provider
        # providers = ['CPUExecutionProvider']
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"[SCRFD] Model loaded: {self.model_path}")
        print(f"[SCRFD] Input: {self.input_name}, shape: {self.session.get_inputs()[0].shape}")
        print(f"[SCRFD] Outputs: {self.output_names}")
    
    def _preprocess(self, image: np.ndarray) -> tuple:
        """
        Preprocess image for SCRFD inference.
        
        Returns:
            blob: preprocessed input tensor
            scale: (scale_w, scale_h) for coordinate mapping back
        """
        h, w = image.shape[:2]
        input_w, input_h = self.input_size
        
        # Calculate scale to maintain aspect ratio
        scale = min(input_w / w, input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with aspect ratio preservation
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to input size
        padded = np.full((input_h, input_w, 3), 0, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # BGR to RGB, HWC to CHW, normalize
        blob = cv2.dnn.blobFromImage(
            padded, 
            scalefactor=1.0 / 128.0, 
            size=(input_w, input_h),
            mean=(127.5, 127.5, 127.5), 
            swapRB=True, 
            crop=False
        )
        
        return blob, scale
    
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
            score_blob = np.asarray(outputs[idx])
            bbox_blob = np.asarray(outputs[idx + fmc])
            kps_blob = np.asarray(outputs[idx + fmc * 2]) if has_kps else None
            
            stride = strides[idx]
            
            # Support both ORT layouts: (1, N, C) and (N, C)
            if score_blob.ndim == 3 and score_blob.shape[0] == 1:
                score_blob = score_blob[0]
            if bbox_blob.ndim == 3 and bbox_blob.shape[0] == 1:
                bbox_blob = bbox_blob[0]
            if kps_blob is not None and kps_blob.ndim == 3 and kps_blob.shape[0] == 1:
                kps_blob = kps_blob[0]

            # Get score predictions
            scores = score_blob.reshape(-1)
            
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
        
        bboxes = np.array([f['bbox'] for f in faces])
        scores = np.array([f['score'] for f in faces])
        boxes_xywh = []
        for x1, y1, x2, y2 in bboxes:
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
        h, w = image.shape[:2]
        blob, scale = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        
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


def run_camera(detector: SCRFDDetector):
    """Run face detection on live camera feed."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print(f"\n[Camera] Started (index={CAMERA_INDEX}, {CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    print("Press 'q' to quit\n")
    
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    
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
        
        cv2.imshow("HR Robot - Face Detection", vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def run_image(detector: SCRFDDetector, image_path: str):
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
    cv2.imshow("Detection Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SCRFD Face Detection")
    parser.add_argument("--image", type=str, help="Path to test image (omit for camera)")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HR Robot - SCRFD Face Detection (ONNX)")
    print("=" * 60)
    
    detector = SCRFDDetector(
        model_path=args.model,
        conf_threshold=args.threshold,
    )
    
    if args.image:
        run_image(detector, args.image)
    else:
        run_camera(detector)


if __name__ == "__main__":
    main()
