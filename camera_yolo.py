#!/usr/bin/env python3
"""
Camera and YOLOv8 Module for OV5647 on Raspberry Pi OS 64-bit
Uses Picamera2 (new camera stack)
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import camera library
try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  Picamera2 not available")

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not available")

class EnhancedCameraYOLO:
    def __init__(self):
        """Initialize OV5647 camera and YOLO detection"""
        print("  🔷 Initializing OV5647 camera system...")
        
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
        
        print("  ✅ Camera system initialized")
    
    def _load_yolo_model(self):
        """Load YOLOv8 nano model"""
        try:
            print("  🧠 Loading YOLOv8 model...")
            
            model_path = config.YOLO_MODEL_PATH
            
            if not os.path.exists(model_path):
                print("     📥 Downloading YOLOv8n...")
                model_path = 'yolov8n.pt'
            
            self.model = YOLO(model_path)
            self.model.to('cpu')
            
            # Test inference
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            
            self.model_loaded = True
            print("  ✅ YOLOv8 model ready")
            
        except Exception as e:
            print(f"  ❌ YOLO error: {e}")
            self.model_loaded = False
    
    def _init_camera(self):
        """Initialize OV5647 camera with Picamera2"""
        
        if not PICAMERA2_AVAILABLE:
            print("  ❌ Picamera2 not installed")
            print("  💡 Run: sudo apt install -y python3-picamera2")
            self.camera_type = 'none'
            return
        
        try:
            print("  📹 Initializing OV5647 with Picamera2...")
            
            # Create camera instance
            self.camera = Picamera2()
            
            # Configure for video streaming
            config_dict = self.camera.create_video_configuration(
                main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"},
                controls={
                    "FrameRate": config.CAMERA_FPS,
                    "ExposureTime": 20000,  # 20ms exposure
                    "AnalogueGain": 2.0,
                },
                buffer_count=config.FRAME_BUFFER_SIZE
            )
            
            self.camera.configure(config_dict)
            
            # Set additional controls for OV5647
            self.camera.set_controls({
                "AeEnable": True,  # Auto exposure
                "AwbEnable": True,  # Auto white balance
                "Brightness": 0.0,
                "Contrast": 1.0,
            })
            
            # Start camera
            print("     ⏳ Starting camera...")
            self.camera.start()
            
            # Warm-up period
            print("     ⏳ Warming up (3 seconds)...")
            time.sleep(3)
            
            # Capture test frame
            test_frame = self.camera.capture_array()
            
            if test_frame is not None and test_frame.size > 0:
                brightness = test_frame.mean()
                print(f"     ✅ Camera ready! (brightness: {brightness:.1f})")
                self.camera_ready = True
            else:
                print("     ⚠️  Warning: Test frame empty")
                self.camera_ready = True
            
            self.camera_type = 'picamera2'
            print("  ✅ OV5647 camera initialized")
            
        except Exception as e:
            print(f"  ❌ Camera init failed: {e}")
            print("\n  Troubleshooting:")
            print("     1. Check cable connection")
            print("     2. Run: rpicam-hello -t 5000")
            print("     3. Enable camera: sudo raspi-config")
            print("     4. Check /boot/config.txt has camera_auto_detect=1")
            print("     5. Reboot if needed")
            
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
        print("  🎥 Frame capture started")
        
        errors = 0
        success = 0
        
        while self.capture_running:
            try:
                if self.camera is None:
                    frame = self._generate_placeholder("Camera not available")
                    with self.frame_lock:
                        self.frame = frame
                    time.sleep(0.1)
                    continue
                
                # Capture frame
                frame = self.camera.capture_array()
                
                if frame is None or frame.size == 0:
                    errors += 1
                    if errors > 10:
                        print("  ⚠️  Too many errors, restarting camera...")
                        try:
                            self.camera.stop()
                            time.sleep(1)
                            self.camera.start()
                            time.sleep(2)
                        except:
                            pass
                        errors = 0
                    time.sleep(0.05)
                    continue
                
                errors = 0
                
                # Resize if needed
                if frame.shape[:2] != (config.CAMERA_RESOLUTION[1], config.CAMERA_RESOLUTION[0]):
                    frame = cv2.resize(frame, config.CAMERA_RESOLUTION)
                
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Store frame
                with self.frame_lock:
                    self.frame = frame.copy()
                
                self.frame_count += 1
                success += 1
                
                # Calculate FPS
                current = time.time()
                fps = 1.0 / (current - self.last_time + 0.001)
                self.fps_counter.append(fps)
                self.last_time = current
                
                # Log progress
                if success == 1:
                    print(f"  ✅ First frame captured!")
                elif success % 50 == 0:
                    avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
                    print(f"  📊 Frames: {self.frame_count} | FPS: {avg_fps:.1f}")
                
                # Control frame rate
                time.sleep(1.0 / config.CAMERA_FPS)
                
            except Exception as e:
                print(f"  ❌ Capture error: {e}")
                errors += 1
                time.sleep(0.1)
        
        print("  🛑 Frame capture stopped")
    
    def _yolo_detection_thread(self):
        """YOLO detection thread"""
        print("  🔍 YOLO detection started")
        
        if not self.model_loaded:
            print("  ⚠️  YOLO not available")
            return
        
        last_detection = 0
        
        while self.detection_running:
            try:
                # Limit detection frequency
                if time.time() - last_detection < config.DETECTION_INTERVAL:
                    time.sleep(0.01)
                    continue
                
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
                    verbose=False,
                    device='cpu'
                )
                
                detections = []
                
                if len(results) > 0:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            
                            detections.append({
                                'class': self.model.names[cls],
                                'confidence': round(conf, 3),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2)
                            })
                
                with self.detection_lock:
                    self.detections = detections
                
                last_detection = time.time()
                
            except Exception as e:
                print(f"  ❌ Detection error: {e}")
                time.sleep(0.1)
        
        print("  🛑 YOLO detection stopped")
    
    def _generate_placeholder(self, message="Waiting..."):
        """Generate placeholder frame"""
        frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                         config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(frame.shape[0]):
            frame[i, :] = [20 + i//8, 15 + i//10, 35 + i//12]
        
        cv2.putText(frame, message, (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        cv2.putText(frame, "Check camera connection", (80, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        return frame
    
    def start_detection(self):
        """Start camera and detection threads"""
        if self.is_running:
            return
        
        self.is_running = True
        self.capture_running = True
        self.detection_running = True
        
        # Capture thread
        threading.Thread(target=self._capture_frames, daemon=True, name="Capture").start()
        
        # Detection thread
        if self.model_loaded:
            threading.Thread(target=self._yolo_detection_thread, daemon=True, name="YOLO").start()
        
        print("  ✅ Detection system started")
    
    def stop_detection(self):
        """Stop all threads"""
        self.detection_running = False
        self.capture_running = False
        time.sleep(0.5)
        self.is_running = False
    
    def get_frame_with_detections(self):
        """Get annotated frame"""
        with self.frame_lock:
            if self.frame is None:
                return self._generate_placeholder("Initializing...")
            frame = self.frame.copy()
        
        with self.detection_lock:
            detections = self.detections.copy()
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['class']
            
            # Color by confidence
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label}: {conf:.2f}"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-size[1]-8), (x1+size[0]+8, y1), color, -1)
            cv2.putText(frame, text, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add overlay
        frame = self._add_overlay(frame, len(detections))
        
        return frame
    
    def _add_overlay(self, frame, count):
        """Add FPS and detection count"""
        h, w = frame.shape[:2]
        fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {count}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        cv2.putText(frame, "OV5647", (w-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return frame
    
    def get_frame(self):
        """Get raw frame"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_detections(self):
        """Get current detections"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get statistics"""
        fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        return {
            'fps': round(fps, 1),
            'detections_count': len(self.detections),
            'model_loaded': self.model_loaded,
            'camera_ready': self.camera_ready
        }
    
    def cleanup(self):
        """Cleanup resources"""
        print("  🧹 Cleaning up camera...")
        self.stop_detection()
        if self.camera and self.camera_type == 'picamera2':
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass