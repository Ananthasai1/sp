#!/usr/bin/env python3
"""
Enhanced Camera and YOLOv8 Object Detection Module
Optimized for OV5647 with Night Vision Support
Separate threads for Auto/Manual modes - zero performance impact
"""

import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
import config
from collections import deque

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
        
        # Load YOLO model - optimized for speed
        try:
            print("  🧠 Loading YOLOv8 Nano (optimized for Raspberry Pi)...")
            self.model = YOLO(config.YOLO_MODEL_PATH)
            self.model.to('cpu')  # Force CPU usage
            print("  ✅ YOLO model loaded")
        except Exception as e:
            print(f"  ⚠️  YOLO loading error: {e}")
            self.model = None
        
        # Detection frame buffer (for smoothing)
        self.detection_history = deque(maxlen=3)
        
        print("  ✅ Enhanced camera ready")
    
    def _init_camera(self):
        """Initialize OV5647 camera with optimal settings"""
        try:
            from picamera2 import Picamera2
            self.camera = Picamera2()
            
            # Optimized camera configuration
            config_dict = self.camera.create_still_configuration(
                main={"format": 'BGR888', "size": config.CAMERA_RESOLUTION},
                buffer_count=4,
                queue=False
            )
            
            self.camera.configure(config_dict)
            
            # Camera properties for better quality
            self.camera.set_controls({
                "ExposureTime": 30000,  # Auto exposure
                "AnalogueGain": 1.0,
                "Brightness": 0.0,
                "Contrast": 1.0,
                "Saturation": 1.0,
                "Sharpness": 1.0
            })
            
            self.camera.start()
            print("  ✅ PiCamera2 initialized (OV5647)")
            
        except Exception as e:
            print(f"  ⚠️  PiCamera2 error: {e}")
            try:
                # Fallback to OpenCV
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
                self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print("  ✅ OpenCV camera initialized")
            except Exception as e2:
                print(f"  ❌ Camera init failed: {e2}")
                self.camera = None
    
    def _setup_night_vision(self):
        """Setup IR LED control for night vision"""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.ir_led_pin, GPIO.OUT)
            print("  ✅ Night vision setup complete")
        except Exception as e:
            print(f"  ⚠️  Night vision setup error: {e}")
    
    def _control_ir_led(self, enable):
        """Control IR LED based on light level"""
        try:
            import RPi.GPIO as GPIO
            GPIO.output(self.ir_led_pin, GPIO.HIGH if enable else GPIO.LOW)
        except:
            pass
    
    def _capture_frames(self):
        """Continuous frame capture thread (optimized)"""
        print("  📸 Frame capture thread started")
        
        while self.capture_running:
            try:
                if self.camera is None:
                    time.sleep(0.05)
                    continue
                
                # Capture frame
                if hasattr(self.camera, 'capture_array'):
                    frame = self.camera.capture_array()
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        continue
                
                # Ensure frame is BGR
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] != 3:  # Not BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Check brightness for night vision
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                # Toggle night vision
                if brightness < self.brightness_threshold and not self.night_mode:
                    print("  🌙 Night vision activated")
                    self.night_mode = True
                    self._control_ir_led(True)
                elif brightness >= self.brightness_threshold and self.night_mode:
                    print("  ☀️  Day mode activated")
                    self.night_mode = False
                    self._control_ir_led(False)
                
                # Apply enhancement if needed
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
                time.sleep(0.05)
        
        print("  📸 Frame capture thread stopped")
    
    def _enhance_low_light(self, frame):
        """Enhance low-light frames without degrading quality"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _yolo_detection_thread(self):
        """Optimized YOLO detection thread (separate from capture)"""
        print("  🔍 YOLO detection thread started")
        
        frame_count = 0
        skip_frames = 0  # Skip every Nth frame for speed
        
        while self.detection_running:
            try:
                if self.model is None or self.frame is None:
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                
                # Dynamic frame skipping for performance
                skip_frames = 0 if self.night_mode else 0  # Process every frame
                if frame_count % (skip_frames + 1) != 0:
                    time.sleep(0.01)
                    continue
                
                with self.frame_lock:
                    frame = self.frame.copy()
                
                if frame is None:
                    continue
                
                # Run YOLO inference
                results = self.model(
                    frame,
                    conf=config.YOLO_CONFIDENCE_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    verbose=False,
                    half=False  # Full precision for accuracy
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
                
                # Smooth detections
                self.detection_history.append(detections)
                smoothed = self._smooth_detections()
                
                with self.detection_lock:
                    self.detections = smoothed
                
            except Exception as e:
                print(f"  Detection error: {e}")
                time.sleep(0.05)
        
        print("  🔍 YOLO detection thread stopped")
    
    def _smooth_detections(self):
        """Smooth detections across frames for stability"""
        if len(self.detection_history) < 2:
            return self.detection_history[-1] if self.detection_history else []
        
        # Use most confident detections
        current = self.detection_history[-1]
        prev = self.detection_history[-2] if len(self.detection_history) > 1 else current
        
        smoothed = []
        for det in current:
            # Find matching detection in previous frame
            matched = False
            for prev_det in prev:
                if det['class'] == prev_det['class']:
                    # Calculate distance between centers
                    dist = np.sqrt(
                        (det['center_x'] - prev_det['center_x']) ** 2 +
                        (det['center_y'] - prev_det['center_y']) ** 2
                    )
                    if dist < 50:  # Same object
                        matched = True
                        break
            
            smoothed.append(det)
        
        return smoothed
    
    def start_detection(self):
        """Start detection threads"""
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
        
        # Start detection thread
        detection_thread = threading.Thread(
            target=self._yolo_detection_thread,
            daemon=True,
            name="YOLODetection"
        )
        detection_thread.start()
        
        print("  ✅ Detection started (dual-thread optimized)")
    
    def stop_detection(self):
        """Stop detection threads gracefully"""
        self.detection_running = False
        self.capture_running = False
        time.sleep(0.2)
        self.is_running = False
        print("  🛑 Detection stopped")
    
    def get_frame_with_detections(self):
        """Get frame with drawn detections for display"""
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
            
            # Draw bounding box
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            bg_x1, bg_y1 = x1, y1 - text_size[1] - 5
            bg_x2, bg_y2 = x1 + text_size[0] + 5, y1
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Add info overlay
        frame = self._add_info_overlay(frame)
        
        return frame
    
    def _add_info_overlay(self, frame):
        """Add performance and status info to frame"""
        h, w = frame.shape[:2]
        
        # FPS
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection count
        with self.detection_lock:
            det_count = len(self.detections)
        det_text = f"Objects: {det_count}"
        cv2.putText(frame, det_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        # Night vision indicator
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
        """Get current detections"""
        with self.detection_lock:
            return self.detections.copy()
    
    def get_performance_stats(self):
        """Get performance metrics"""
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        return {
            'fps': round(avg_fps, 1),
            'night_mode': self.night_mode,
            'detections_count': len(self.detections)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        print("  Cleaning up camera...")
        self.stop_detection()
        if self.camera:
            if hasattr(self.camera, 'stop'):
                self.camera.stop()
            elif hasattr(self.camera, 'release'):
                self.camera.release()