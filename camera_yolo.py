#!/usr/bin/env python3
"""
Enhanced Camera and YOLOv8 Object Detection Module
Fixed for libcamera-only systems with proper YOLOv8 support
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import config
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")

class EnhancedCameraYOLO:
    def __init__(self):
        """Initialize camera and YOLO"""
        print("  🔷 Initializing camera system...")
        
        self.camera = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.detections = []
        self.detection_lock = threading.Lock()
        
        self.is_running = False
        self.capture_running = False
        self.detection_running = False
        
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_count = 0
        
        # Initialize camera
        self._init_camera()
        time.sleep(1)
        
        # Load YOLO model
        self.model = None
        self.model_loaded = False
        if YOLO_AVAILABLE:
            self._load_yolo_model()
        
        print("  ✅ Camera initialized")
    
    def _load_yolo_model(self):
        """Load YOLOv8 model with proper error handling"""
        try:
            print("  🧠 Loading YOLOv8 model...")
            
            # Check if model file exists
            model_path = config.YOLO_MODEL_PATH
            
            if os.path.exists(model_path):
                print(f"     ℹ️  Using local model: {model_path}")
            else:
                print(f"     ⚠️  Model not found at {model_path}")
                print("     📥 Downloading YOLOv8n from Ultralytics...")
                model_path = 'yolov8n.pt'
            
            # Load model
            self.model = YOLO(model_path)
            
            # Move to CPU (safer for Raspberry Pi)
            self.model.to('cpu')
            
            # Test inference on dummy image
            dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            
            self.model_loaded = True
            print("  ✅ YOLOv8 model loaded successfully")
            
        except Exception as e:
            print(f"  ❌ YOLO loading error: {e}")
            print("     💡 Fix: Try running: pip install --upgrade ultralytics")
            self.model_loaded = False
    
    def _init_camera(self):
        """Initialize camera - libcamera preferred"""
        camera_found = False
        
        # Try OpenCV with libcamera
        try:
            print("  📹 Initializing OpenCV with libcamera...")
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera with V4L2")
            
            # Set properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Warmup frames
            print("     🔄 Warmup frames...")
            for i in range(10):
                ret, _ = self.camera.read()
                print(f"     {'✅' if ret else '❌'} Warmup {i+1}/10")
                time.sleep(0.1)
            
            print("  ✅ OpenCV camera ready (libcamera)")
            camera_found = True
            self.camera_type = 'opencv'
            
        except Exception as e:
            print(f"  ⚠️  OpenCV initialization failed: {e}")
            self.camera = None
        
        if not camera_found:
            print("  ❌ No camera available!")
            print("  Possible fixes:")
            print("     1. Enable camera: sudo raspi-config → Camera → Enable")
            print("     2. Check: libcamera-hello -t 5000")
            print("     3. Verify: ls -la /dev/video*")
            self.camera = None
            self.camera_type = 'none'
    
    def _capture_frames(self):
        """Continuous frame capture thread"""
        print("  🎥 Capture thread started")
        frame_errors = 0
        success_count = 0
        
        while self.capture_running:
            try:
                if self.camera is None:
                    frame = self._generate_placeholder()
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(1/config.CAMERA_FPS)
                    continue
                
                frame = None
                ret, frame = self.camera.read()
                
                if not ret or frame is None or frame.size == 0:
                    frame_errors += 1
                    if frame_errors > 10:
                        print("  ⚠️  Camera read failed repeatedly - reconnecting...")
                        self.camera.release()
                        time.sleep(1)
                        self._init_camera()
                        frame_errors = 0
                    time.sleep(0.05)
                    continue
                
                frame_errors = 0
                
                # Ensure correct resolution
                if frame.shape[0] != config.CAMERA_RESOLUTION[1] or frame.shape[1] != config.CAMERA_RESOLUTION[0]:
                    frame = cv2.resize(frame, config.CAMERA_RESOLUTION)
                
                # Validate frame (check if not all black/white)
                if frame.max() < 5 or frame.min() > 250:
                    frame_errors += 1
                    if frame_errors > 3:
                        print("  ⚠️  Invalid frame detected")
                        frame_errors = 0
                    continue
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                success_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
                if success_count % 30 == 0:
                    avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
                    print(f"  ✅ Captured {self.frame_count} frames | {avg_fps:.1f} FPS")
                
            except Exception as e:
                print(f"  ❌ Capture error: {e}")
                frame_errors += 1
                time.sleep(0.1)
        
        print("  🛑 Capture thread stopped")
    
    def _generate_placeholder(self):
        """Generate test pattern"""
        frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                         config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        # Create gradient background
        for i in range(frame.shape[0]):
            frame[i, :] = [20 + i//8, 15 + i//10, 35 + i//12]
        
        cv2.putText(frame, "Camera Initializing", (80, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        cv2.putText(frame, "Please wait...", (120, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        return frame
    
    def _yolo_detection_thread(self):
        """YOLOv8 detection thread"""
        print("  🔍 Detection thread started")
        
        if not self.model_loaded or self.model is None:
            print("  ⚠️  Detection unavailable - model not loaded")
            self.detection_running = False
            return
        
        detection_errors = 0
        
        while self.detection_running:
            try:
                if self.frame is None:
                    time.sleep(0.05)
                    continue
                
                with self.frame_lock:
                    frame = self.frame.copy()
                
                # Run YOLOv8 inference
                results = self.model(
                    frame,
                    conf=config.YOLO_CONFIDENCE_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    verbose=False,
                    device=0  # 0 for CPU, or GPU id if available
                )
                
                detections = []
                
                if len(results) > 0:
                    for result in results:
                        boxes = result.boxes
                        
                        for box in boxes:
                            # Extract coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # Get class name
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
                
                detection_errors = 0
                time.sleep(0.05)
                
            except Exception as e:
                detection_errors += 1
                if detection_errors > 5:
                    print(f"  ❌ Detection error: {e}")
                    detection_errors = 0
                time.sleep(0.1)
        
        print("  🛑 Detection thread stopped")
    
    def start_detection(self):
        """Start capture and detection"""
        if self.is_running:
            return
        
        self.is_running = True
        self.capture_running = True
        self.detection_running = True
        
        # Start capture thread
        capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True,
            name="CameraCapture"
        )
        capture_thread.start()
        
        # Start detection thread if model is available
        if self.model_loaded and self.model is not None:
            detection_thread = threading.Thread(
                target=self._yolo_detection_thread,
                daemon=True,
                name="YOLODetection"
            )
            detection_thread.start()
        
        print("  ✅ Detection started")
    
    def stop_detection(self):
        """Stop all threads"""
        self.detection_running = False
        self.capture_running = False
        time.sleep(0.5)
        self.is_running = False
    
    def get_frame_with_detections(self):
        """Get frame with bounding boxes and annotations"""
        with self.frame_lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
        
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw bounding boxes and labels
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Color code by confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.6:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.6, 2)[0]
            
            # Label background
            bg_x1, bg_y1 = x1, y1 - text_size[1] - 8
            bg_x2, bg_y2 = x1 + text_size[0] + 8, y1
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4), font, 0.6, (0, 0, 0), 2)
        
        # Add overlay info
        frame = self._add_overlay(frame, len(detections))
        
        return frame
    
    def _add_overlay(self, frame, det_count):
        """Add FPS and detection count overlay"""
        h, w = frame.shape[:2]
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # FPS counter
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection count
        cv2.putText(frame, f"Objects: {det_count}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        # Model indicator
        cv2.putText(frame, "YOLOv8", (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return frame
    
    def get_frame(self):
        """Get current frame without detections"""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def get_detections(self):
        """Get current detections list"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get performance statistics"""
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        return {
            'fps': round(avg_fps, 1),
            'detections_count': len(self.detections),
            'model_loaded': self.model_loaded
        }
    
    def cleanup(self):
        """Cleanup resources"""
        print("  Cleaning up camera...")
        self.stop_detection()
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            