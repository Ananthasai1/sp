#!/usr/bin/env python3
"""
CyberCrawl Spider Robot - Real-time YOLO with OV5647 Camera
Flask Web Server with PiCamera2 + YOLOv8 detection
"""

from flask import Flask, render_template, Response, jsonify
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
import sys

app = Flask(__name__)

# Try to import PiCamera2
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    print("✅ PiCamera2 available")
except ImportError:
    PICAMERA_AVAILABLE = False
    print("⚠️  PiCamera2 not available - will try OpenCV")

# Global variables
current_mode = "STOPPED"
mode_lock = threading.Lock()

# YOLO Model
print("🧠 Loading YOLOv8 Nano model (fastest)...")
try:
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8 loaded")
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"❌ YOLO load failed: {e}")
    print("   Install: pip install ultralytics torch torchvision")
    model = None
    YOLO_AVAILABLE = False

# Camera setup
camera = None
frame_lock = threading.Lock()
current_frame = None

def init_camera():
    """Initialize camera - try PiCamera2 first, fallback to OpenCV"""
    global camera, PICAMERA_AVAILABLE
    
    if PICAMERA_AVAILABLE:
        try:
            print("📷 Initializing PiCamera2...")
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"format": 'RGB888', "size": (640, 480)}
            )
            camera.configure(config)
            camera.start()
            print("✅ PiCamera2 initialized (OV5647)")
            return True
        except Exception as e:
            print(f"⚠️  PiCamera2 failed: {e}")
            PICAMERA_AVAILABLE = False
    
    # Fallback to OpenCV
    try:
        print("📷 Initializing OpenCV camera...")
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        print("✅ OpenCV camera initialized")
        return True
    except Exception as e:
        print(f"❌ Camera init failed: {e}")
        return False

def capture_and_detect():
    """Capture frame from camera and run YOLO detection"""
    global current_frame, camera
    
    try:
        # Capture frame
        if PICAMERA_AVAILABLE:
            frame = camera.capture_array()
            # Convert RGB to BGR for OpenCV
            if frame is not None and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = camera.read()
            if not ret:
                frame = None
        
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Feed", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            return frame, []
        
        detections = []
        
        # Run YOLO detection
        if YOLO_AVAILABLE and model:
            try:
                results = model(frame, conf=0.5, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Extract data
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        # Store detection
                        detections.append({
                            'class': class_name,
                            'confidence': round(confidence, 2),
                            'bbox': [x1, y1, x2, y2]
                        })
                        
                        # Draw detection box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with background
                        label = f"{class_name} {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - 25), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            except Exception as e:
                print(f"Detection error: {e}")
        
        # Add top info bar
        cv2.rectangle(frame, (0, 0), (640, 35), (0, 0, 0), -1)
        info_text = f"Mode: {current_mode} | Objects: {len(detections)} | FPS: ~30"
        cv2.putText(frame, info_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (500, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Add detection list on side
        y_offset = 50
        for detection in detections[:5]:  # Show top 5
            text = f"{detection['class']} {int(detection['confidence']*100)}%"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        with frame_lock:
            current_frame = frame.copy()
        
        return frame, detections
    
    except Exception as e:
        print(f"Capture error: {e}")
        return None, []

def generate_frames():
    """Video streaming generator"""
    while True:
        try:
            frame, _ = capture_and_detect()
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
        
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)

def detection_thread_loop():
    """Continuous detection loop for API responses"""
    detections_cache = []
    
    while True:
        try:
            _, detections = capture_and_detect()
            detections_cache = detections
            time.sleep(0.1)
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.5)

# ===== Flask Routes =====

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Live video stream with YOLO detections"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get robot status"""
    _, detections = capture_and_detect()
    
    distance = 150
    if detections:
        distance = 50 + len(detections) * 15
    
    return jsonify({
        'mode': current_mode,
        'distance': round(distance, 1),
        'detections': detections,
        'timestamp': time.time()
    })

@app.route('/api/start_auto', methods=['POST'])
def start_auto():
    """Start auto mode"""
    global current_mode
    
    with mode_lock:
        if current_mode != "STOPPED":
            return jsonify({'success': False, 'message': 'Stop current mode first'})
        
        current_mode = "AUTO"
        return jsonify({'success': True, 'message': 'Auto mode started'})

@app.route('/api/stop', methods=['POST'])
def stop():
    """Stop all movements"""
    global current_mode
    
    with mode_lock:
        current_mode = "STOPPED"
        return jsonify({'success': True, 'message': 'Robot stopped'})

@app.route('/api/manual_mode', methods=['POST'])
def manual_mode():
    """Switch to manual mode"""
    global current_mode
    
    with mode_lock:
        if current_mode != "STOPPED":
            return jsonify({'success': False, 'message': 'Stop robot first'})
        
        current_mode = "MANUAL"
        return jsonify({'success': True, 'message': 'Manual mode activated'})

@app.route('/api/manual_control/<action>', methods=['POST'])
def manual_control(action):
    """Execute manual control action"""
    if current_mode != "MANUAL":
        return jsonify({'success': False, 'message': 'Not in manual mode'})
    
    actions_text = {
        'forward': '⬆️ Moving forward',
        'backward': '⬇️ Moving backward',
        'left': '⬅️ Turning left',
        'right': '➡️ Turning right',
        'wave': '👋 Waving',
        'shake': '🤝 Shaking',
        'dance': '💃 Dancing',
        'stand': '🧍 Standing',
        'sit': '💺 Sitting'
    }
    
    if action not in actions_text:
        return jsonify({'success': False, 'message': 'Unknown action'})
    
    print(f"Manual: {actions_text[action]}")
    return jsonify({'success': True, 'message': f'Action {action} executed'})

@app.route('/api/detections')
def get_detections():
    """Get current detections"""
    _, detections = capture_and_detect()
    return jsonify({'detections': detections})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🕷️  CyberCrawl - Real-Time YOLO Object Detection")
    print("="*70)
    print("Camera: ", end="")
    if init_camera():
        print("✅ Ready")
    else:
        print("❌ Failed - Cannot start without camera")
        sys.exit(1)
    
    print("YOLO Model: ", end="")
    print("✅ YOLOv8 Nano" if YOLO_AVAILABLE else "❌ Not loaded")
    
    print("="*70)
    print("\n🌐 Starting web server...")
    print("📍 Access at: http://localhost:5000")
    print("📍 Remote: http://<your-pi-ip>:5000")
    print("\n🎮 Features:")
    print("   ✨ Real-time camera stream")
    print("   ✨ YOLO v8 object detection")
    print("   ✨ Auto mode with live detections")
    print("   ✨ Manual control with keyboard")
    print("   ✨ Detection tracking")
    print("="*70 + "\n")
    
    # Start detection thread
    det_thread = threading.Thread(target=detection_thread_loop, daemon=True)
    det_thread.start()
    
    # Start Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)