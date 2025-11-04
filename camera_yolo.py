#!/usr/bin/env python3
"""
Enhanced Camera and YOLOv8 Object Detection Module - DEBUG VERSION
Optimized for OV5647 with Night Vision Support
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
    print("⚠️  YOLOv8 not available - install with: pip install ultralytics")

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
        self.night_vision_enabled = False
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        
        # Night vision calibration
        self.brightness_threshold = config.NIGHT_VISION_THRESHOLD
        self.ir_led_pin = config.NIGHT_VISION_GPIO
        self.night_mode = False
        
        # Initialize camera
        self._init_camera()
        
        # Load YOLO model
        self.model = None
        if YOLO_AVAILABLE:
            try:
                print("  🧠 Loading YOLOv8 Nano...")
                self.model = YOLO(config.YOLO_MODEL_PATH)
                self.model.to('cpu')
                print("  ✅ YOLO model loaded successfully")
            except Exception as e:
                print(f"  ⚠️  YOLO loading error: {e}")
        
        # Detection frame buffer
        self.detection_history = deque(maxlen=3)
        
        print("  ✅ Enhanced camera ready")
    
    def _init_camera(self):
        """Initialize OV5647 camera with optimal settings"""
        camera_found = False
        
        # Try PiCamera2 first
        try:
            print("  🔍 Trying PiCamera2...")
            from picamera2 import Picamera2
            self.camera = Picamera2()
            
            config_dict = self.camera.create_still_configuration(
                main={"format": 'BGR888', "size": config.CAMERA_RESOLUTION},
                buffer_count=4,
                queue=False
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
                print("  🔍 Trying OpenCV VideoCapture...")
                self.camera = cv2.VideoCapture(0)
                
                if not self.camera.isOpened():
                    raise Exception("Camera not accessible")
                
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
                self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                print("  ✅ OpenCV camera initialized")
                camera_found = True
                
            except Exception as e:
                print(f"  ❌ OpenCV error: {e}")
        
        if not camera_found:
            print("  ❌ No camera found! Using placeholder frames")
            self.camera = None
    
    def _capture_frames(self):
        """Continuous frame capture thread"""
        print("  📸 Frame capture thread started")
        frame_errors = 0
        
        while self.capture_running:
            try:
                if self.camera is None:
                    # Generate placeholder
                    frame = self._generate_placeholder_frame()
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(0.033)
                    continue
                
                # Capture frame
                if hasattr(self.camera, 'capture_array'):
                    frame = self.camera.capture_array()
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        frame_errors += 1
                        if frame_errors > 10:
                            print("  ❌ Too many frame capture errors")
                            self.camera = None
                        time.sleep(0.05)
                        continue
                
                frame_errors = 0
                
                # Ensure frame is BGR
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                # Check brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                # Toggle night vision
                if brightness < self.brightness_threshold and not self.night_mode:
                    self.night_mode = True
                    print("  🌙 Night vision activated")
                elif brightness >= self.brightness_threshold and self.night_mode:
                    self.night_mode = False
                    print("  ☀️  Day mode activated")
                
                # Enhance low light
                if self.night_mode:
                    frame = self._enhance_low_light(frame)
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current_time
                
            except Exception as e:
                print(f"  Capture error: {e}")
                time.sleep(0.1)
        
        print("  📸 Frame capture thread stopped")
    
    def _generate_placeholder_frame(self):
        """Generate placeholder frame when camera unavailable"""
        frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                         config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        # Dark background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 30), -1)
        
        # Grid
        for i in range(0, frame.shape[1], 50):
            cv2.line(frame, (i, 0), (i, frame.shape[0]), (50, 50, 70), 1)
        for i in range(0, frame.shape[0], 50):
            cv2.line(frame, (0, i), (frame.shape[1], i), (50, 50, 70), 1)
        
        # Text
        cv2.putText(frame, "Camera Initializing...", (100, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(frame, "Check camera connection", (80, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        
        return frame
    
    def _enhance_low_light(self, frame):
        """Enhance low-light frames"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _yolo_detection_thread(self):
        """Optimized YOLO detection thread"""
        print("  🔍 YOLO detection thread started")
        
        if not YOLO_AVAILABLE or self.model is None:
            print("  ⚠️  YOLO not available - detection disabled")
            self.detection_running = False
            return
        
        while self.detection_running:
            try:
                if self.frame is None:
                    time.sleep(0.05)
                    continue
                
                with self.frame_lock:
                    frame = self.frame.copy()
                
                # Run YOLO
                results = self.model(
                    frame,
                    conf=config.YOLO_CONFIDENCE_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    verbose=False
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
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"  Detection error: {e}")
                time.sleep(0.1)
        
        print("  🔍 YOLO detection thread stopped")
    
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
        
        if YOLO_AVAILABLE and self.model is not None:
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
            
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            bg_x1, bg_y1 = x1, y1 - text_size[1] - 5
            bg_x2, bg_y2 = x1 + text_size[0] + 5, y1
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Add overlay
        frame = self._add_info_overlay(frame)
        
        return frame
    
    def _add_info_overlay(self, frame):
        """Add info overlay"""
        h, w = frame.shape[:2]
        
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        with self.detection_lock:
            det_count = len(self.detections)
        det_text = f"Objects: {det_count}"
        cv2.putText(frame, det_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        if self.night_mode:
            cv2.putText(frame, "NIGHT VISION", (w - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
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
            'night_mode': self.night_mode,
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