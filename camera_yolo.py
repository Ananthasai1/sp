#!/usr/bin/env python3
"""
Enhanced Camera and YOLOv12 Object Detection Module
Optimized for OV5647 Camera Module
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import config

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv12 not available - install with: pip install ultralytics")

class EnhancedCameraYOLO:
    def __init__(self):
        """Initialize camera and YOLO with optimization"""
        print("  📷 Initializing enhanced camera system...")
        
        self.camera = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.detections = []
        self.detection_lock = threading.Lock()
        
        # Detection settings
        self.is_running = False
        self.capture_running = False
        self.detection_running = False
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_count = 0
        
        # Initialize camera
        self._init_camera()
        
        # Load YOLO model
        self.model = None
        self.model_loaded = False
        if YOLO_AVAILABLE:
            try:
                print("  🧠 Loading YOLOv12 model...")
                self.model = YOLO(config.YOLO_MODEL_PATH)
                self.model.to('cpu')
                self.model_loaded = True
                print("  ✅ YOLOv12 model loaded successfully")
            except Exception as e:
                print(f"  ⚠️  YOLO loading error: {e}")
                self.model_loaded = False
        
        print("  ✅ Enhanced camera initialized")
    
    def _init_camera(self):
        """Initialize OV5647 camera with optimal settings"""
        camera_found = False
        
        # Try PiCamera2 first (best for OV5647)
        try:
            print("  📹 Trying PiCamera2...")
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Configure for best performance
            config_dict = self.camera.create_video_configuration(
                main={"format": 'BGR888', "size": config.CAMERA_RESOLUTION},
                buffer_count=2
            )
            
            self.camera.configure(config_dict)
            self.camera.start()
            
            print("  ✅ PiCamera2 initialized (OV5647)")
            camera_found = True
            
        except Exception as e:
            print(f"  ⚠️  PiCamera2 error: {e}")
        
        # Fallback to OpenCV
        if not camera_found:
            try:
                print("  📹 Trying OpenCV VideoCapture...")
                self.camera = cv2.VideoCapture(0)
                
                if not self.camera.isOpened():
                    raise Exception("Camera not accessible")
                
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
                self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                
                print("  ✅ OpenCV camera initialized")
                camera_found = True
                
            except Exception as e:
                print(f"  ❌ OpenCV error: {e}")
        
        if not camera_found:
            print("  ❌ No camera found! Using placeholder frames")
            self.camera = None
    
    def _capture_frames(self):
        """Continuous frame capture thread"""
        print("  🎥 Frame capture thread started")
        frame_errors = 0
        
        while self.capture_running:
            try:
                if self.camera is None:
                    frame = self._generate_placeholder_frame()
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(1/config.CAMERA_FPS)
                    continue
                
                # Capture frame
                if hasattr(self.camera, 'capture_array'):
                    # PiCamera2 method
                    frame = self.camera.capture_array()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                else:
                    # OpenCV method
                    ret, frame = self.camera.read()
                    if not ret:
                        frame_errors += 1
                        if frame_errors > 20:
                            print("  ❌ Too many frame capture errors")
                            self.camera = None
                        time.sleep(0.05)
                        continue
                
                frame_errors = 0
                
                # Ensure frame is BGR
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
            except Exception as e:
                print(f"  Capture error: {e}")
                time.sleep(0.1)
        
        print("  🎥 Frame capture thread stopped")
    
    def _generate_placeholder_frame(self):
        """Generate placeholder frame when camera unavailable"""
        frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                         config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(frame.shape[0]):
            frame[i, :] = [30 + i//8, 20 + i//10, 40 + i//12]
        
        # Center text
        cv2.putText(frame, "Camera Initializing", (80, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        cv2.putText(frame, "Please wait...", (120, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        return frame
    
    def _yolo_detection_thread(self):
        """Optimized YOLOv12 detection thread"""
        print("  🔍 YOLOv12 detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  YOLOv12 not available - detection disabled")
            self.detection_running = False
            return
        
        detection_count = 0
        
        while self.detection_running:
            try:
                if self.frame is None:
                    time.sleep(0.05)
                    continue
                
                with self.frame_lock:
                    frame = self.frame.copy()
                
                # Run YOLOv12
                results = self.model(
                    frame,
                    conf=config.YOLO_CONFIDENCE_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    verbose=False,
                    device=0
                )
                
                # Process detections
                detections = []
                if len(results) > 0:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[cls]
                            
                            detection = {
                                'class': class_name,
                                'confidence': round(conf, 3),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2)
                            }
                            detections.append(detection)
                
                with self.detection_lock:
                    self.detections = detections
                
                detection_count += 1
                if detection_count % 10 == 0:
                    print(f"  🎯 Detected {len(detections)} objects")
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"  Detection error: {e}")
                time.sleep(0.1)
        
        print("  🔍 YOLOv12 detection thread stopped")
    
    def start_detection(self):
        """Start detection threads"""
        if self.is_running:
            return
        
        self.is_running = True
        self.capture_running = True
        self.detection_running = True
        
        capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True,
            name="CameraCapture"
        )
        capture_thread.start()
        
        if self.model_loaded and self.model is not None:
            detection_thread = threading.Thread(
                target=self._yolo_detection_thread,
                daemon=True,
                name="YOLODetection"
            )
            detection_thread.start()
        
        print("  ✅ Detection started")
    
    def stop_detection(self):
        """Stop detection threads"""
        self.detection_running = False
        self.capture_running = False
        time.sleep(0.2)
        self.is_running = False
        print("  🛑 Detection stopped")
    
    def get_frame_with_detections(self):
        """Get frame with drawn detections"""
        with self.frame_lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
        
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Color based on confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.6:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            bg_x1, bg_y1 = x1, y1 - text_size[1] - 8
            bg_x2, bg_y2 = x1 + text_size[0] + 8, y1
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4), font, font_scale, (0, 0, 0), thickness)
        
        # Add info overlay
        frame = self._add_info_overlay(frame, len(detections))
        
        return frame
    
    def _add_info_overlay(self, frame, detection_count):
        """Add info overlay"""
        h, w = frame.shape[:2]
        
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # FPS Counter
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection Counter
        det_text = f"Objects: {detection_count}"
        cv2.putText(frame, det_text, (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        # YOLOv12 indicator
        cv2.putText(frame, "YOLOv12", (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return frame
    
    def get_frame(self):
        """Get current frame"""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def get_detections(self):
        """Get detections"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get performance stats"""
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        return {
            'fps': round(avg_fps, 1),
            'detections_count': len(self.detections)
        }
    
    def cleanup(self):
        """Cleanup"""
        print("  Cleaning up camera...")
        self.stop_detection()
        if self.camera:
            if hasattr(self.camera, 'stop'):
                self.camera.stop()
            elif hasattr(self.camera, 'release'):
                self.camera.release()