#!/usr/bin/env python3
"""
Enhanced Camera and YOLOv8 Object Detection Module
FIXED: Using Picamera2 for Raspberry Pi OS (rpicam)
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  Picamera2 not available - trying OpenCV fallback")

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
        self.camera_ready = False
        
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_count = 0
        
        # Initialize camera
        self._init_camera()
        
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
            
            model_path = config.YOLO_MODEL_PATH
            
            if os.path.exists(model_path):
                print(f"     ℹ️  Using local model: {model_path}")
            else:
                print(f"     ⚠️  Model not found at {model_path}")
                print("     📥 Downloading YOLOv8n from Ultralytics...")
                model_path = 'yolov8n.pt'
            
            print(f"     ⏳ Loading {model_path}...")
            self.model = YOLO(model_path)
            print(f"     ✅ Model loaded")
            
            print(f"     ⏳ Moving to CPU...")
            self.model.to('cpu')
            print(f"     ✅ Model on CPU")
            
            print(f"     ⏳ Running test inference...")
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            results = self.model(dummy_image, verbose=False)
            print(f"     ✅ Test inference OK ({len(results)} result(s))")
            
            self.model_loaded = True
            print("  ✅ YOLOv8 model loaded successfully")
            
        except Exception as e:
            print(f"  ❌ YOLO loading error: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def _init_camera(self):
        """Initialize camera with Picamera2"""
        print("  📹 Initializing camera with Picamera2...")
        
        if not PICAMERA2_AVAILABLE:
            print("  ❌ Picamera2 not available")
            print("  💡 Install: sudo apt install -y python3-picamera2")
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
            return
        
        try:
            # Initialize Picamera2
            self.camera = Picamera2()
            
            # Configure camera
            camera_config = self.camera.create_still_configuration(
                main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"},
                buffer_count=2
            )
            self.camera.configure(camera_config)
            
            # Start camera
            print("     ⏳ Starting camera...")
            self.camera.start()
            
            # Wait for camera to warm up
            print("     ⏳ Camera warming up (5 seconds)...")
            time.sleep(5)
            
            # Capture test frame
            print("     ⏳ Capturing test frame...")
            test_frame = self.camera.capture_array()
            
            if test_frame is not None and test_frame.size > 0:
                mean_brightness = test_frame.mean()
                print(f"     ✅ Test frame captured! (brightness: {mean_brightness:.1f})")
                self.camera_ready = True
            else:
                print("     ⚠️  Test frame failed")
                self.camera_ready = True  # Continue anyway
            
            print("  ✅ Picamera2 initialized")
            self.camera_type = 'picamera2'
            
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            import traceback
            traceback.print_exc()
            print("  Possible fixes:")
            print("     1. Install: sudo apt install -y python3-picamera2")
            print("     2. Enable camera: sudo raspi-config → Interface → Camera")
            print("     3. Test: rpicam-hello -t 5000")
            print("     4. Reboot: sudo reboot")
            
            if self.camera:
                try:
                    self.camera.stop()
                except:
                    pass
            
            self.camera = None
            self.camera_type = 'none'
            self.camera_ready = False
    
    def _capture_frames(self):
        """Continuous frame capture thread"""
        print("  🎥 Capture thread started")
        frame_errors = 0
        success_count = 0
        
        # Wait for camera to be ready
        if not self.camera_ready:
            print("  ⏳ Waiting for camera to be ready...")
            time.sleep(2)
        
        while self.capture_running:
            try:
                if self.camera is None or self.camera_type == 'none':
                    # No camera - show placeholder
                    frame = self._generate_placeholder("Camera not available")
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(1/config.CAMERA_FPS)
                    continue
                
                # Capture frame from Picamera2
                frame = self.camera.capture_array()
                
                if frame is None or frame.size == 0:
                    frame_errors += 1
                    if frame_errors > 10:
                        print("  ⚠️  Camera capture failed - trying to restart...")
                        try:
                            self.camera.stop()
                            time.sleep(1)
                            self.camera.start()
                            time.sleep(2)
                        except:
                            pass
                        frame_errors = 0
                    time.sleep(0.05)
                    continue
                
                # Reset error counter on success
                frame_errors = 0
                
                # Ensure correct resolution
                if frame.shape[0] != config.CAMERA_RESOLUTION[1] or frame.shape[1] != config.CAMERA_RESOLUTION[0]:
                    frame = cv2.resize(frame, config.CAMERA_RESOLUTION)
                
                # Convert RGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
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
                
                # Log progress
                if success_count == 1:
                    mean_brightness = frame.mean()
                    print(f"  ✅ First frame captured! (brightness: {mean_brightness:.1f})")
                elif success_count % 30 == 0:
                    avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
                    mean_brightness = frame.mean()
                    print(f"  ✅ Captured {self.frame_count} frames | {avg_fps:.1f} FPS | brightness: {mean_brightness:.1f}")
                
                # Small delay to control frame rate
                time.sleep(0.01)
                
            except Exception as e:
                print(f"  ❌ Capture error: {e}")
                frame_errors += 1
                time.sleep(0.1)
        
        print("  🛑 Capture thread stopped")
    
    def _generate_placeholder(self, message="Waiting for camera..."):
        """Generate placeholder with custom message"""
        frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                         config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        # Create gradient background
        for i in range(frame.shape[0]):
            frame[i, :] = [20 + i//8, 15 + i//10, 35 + i//12]
        
        # Draw messages
        cv2.putText(frame, message, (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        cv2.putText(frame, "Check camera connection", (80, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        cv2.putText(frame, "rpicam-hello -t 5000", (120, 320),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 1)
        
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
                    device='cpu'
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
                return self._generate_placeholder("Initializing camera...")
            frame = self.frame.copy()
        
        # Validate frame
        if frame is None or frame.size == 0:
            return self._generate_placeholder("No frame available")
        
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
            'model_loaded': self.model_loaded,
            'camera_ready': self.camera_ready
        }
    
    def cleanup(self):
        """Cleanup resources"""
        print("  Cleaning up camera...")
        self.stop_detection()
        if self.camera and self.camera_type == 'picamera2':
            try:
                self.camera.stop()
            except:
                pass