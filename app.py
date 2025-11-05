#!/usr/bin/env python3
"""
CyberCrawl Spider Robot - Flask Server with YOLOv8
Live camera feed streaming with object detection
"""

from flask import Flask, render_template, Response, jsonify, request
import threading
import time
import cv2
import numpy as np
from camera.camera_yolo import EnhancedCameraYOLO
import config

app = Flask(__name__)

# Global variables
current_mode = "STOPPED"
mode_lock = threading.Lock()

# Camera and detection
try:
    print("📷 Initializing camera...")
    camera = EnhancedCameraYOLO()
    camera.start_detection()
    print("✅ Camera ready!")
except Exception as e:
    print(f"❌ Camera initialization failed: {e}")
    camera = None

# Detection mode states
detection_state = {
    'auto_detection_active': False,
    'manual_detection_active': False,
}

def generate_frames():
    """Video streaming generator with live camera feed and YOLO detections"""
    print("📹 Starting video stream...")
    frame_count = 0
    last_log = time.time()
    
    while True:
        try:
            if camera is None:
                # No camera available - show error frame
                frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                                config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
                cv2.putText(frame, "NO CAMERA AVAILABLE", (100, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                # Get live camera frame with detections
                frame = camera.get_frame_with_detections()
                
                if frame is None:
                    # Fallback if frame is still None (shouldn't happen now)
                    frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                                    config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting for camera...", (100, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            
            # Validate frame before encoding
            if frame is None or frame.size == 0:
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ret and buffer is not None:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            
            # Log every 5 seconds
            current_time = time.time()
            if current_time - last_log > 5:
                print(f"  📊 Streaming: {frame_count} frames sent")
                last_log = current_time
            
            time.sleep(0.01)  # ~100 FPS capability
            
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            time.sleep(0.1)

def auto_mode_detection():
    """Optimized detection for autonomous mode"""
    print("🤖 Auto mode detection: FULL ACCURACY MODE")
    detection_state['auto_detection_active'] = True
    
    while current_mode == "AUTO" and detection_state['auto_detection_active']:
        try:
            if camera:
                detections = camera.get_detections()
                
                # Process detections for obstacle avoidance
                for det in detections:
                    conf = det['confidence']
                    class_name = det['class']
                    
                    # Critical objects to avoid
                    critical_objects = ['person', 'cat', 'dog', 'car', 'motorcycle', 'bicycle']
                    
                    if class_name in critical_objects and conf > 0.65:
                        print(f"  🎯 Auto: Detected {class_name} ({conf:.2f}) - AVOIDING")
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Auto detection error: {e}")
            time.sleep(0.1)
    
    detection_state['auto_detection_active'] = False
    print("🤖 Auto detection stopped")

def manual_mode_detection():
    """Optimized detection for manual mode"""
    print("⚙️ Manual mode detection: CONTINUOUS TRACKING")
    detection_state['manual_detection_active'] = True
    tracked_objects = {}
    
    while current_mode == "MANUAL" and detection_state['manual_detection_active']:
        try:
            if camera:
                detections = camera.get_detections()
                
                # Update tracking
                current_ids = set()
                for det in detections:
                    class_name = det['class']
                    conf = det['confidence']
                    center = (det['center_x'], det['center_y'])
                    
                    obj_id = f"{class_name}_{int(det['center_x']/50)}"
                    current_ids.add(obj_id)
                    
                    tracked_objects[obj_id] = {
                        'class': class_name,
                        'confidence': conf,
                        'center': center,
                        'bbox': det['bbox'],
                        'timestamp': time.time()
                    }
                
                # Clean old tracks
                now = time.time()
                tracked_objects = {
                    k: v for k, v in tracked_objects.items()
                    if now - v['timestamp'] < 2.0
                }
            
            time.sleep(0.033)
            
        except Exception as e:
            print(f"Manual detection error: {e}")
            time.sleep(0.1)
    
    detection_state['manual_detection_active'] = False
    print("⚙️ Manual detection stopped")

# ===== REST API Routes =====

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Live video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get robot status"""
    if camera:
        detections = camera.get_detections()
        stats = camera.get_performance_stats()
        distance = 150.0  # Placeholder
    else:
        detections = []
        stats = {'fps': 0, 'detections_count': 0, 'model_loaded': False}
        distance = -1
    
    return jsonify({
        'mode': current_mode,
        'distance': distance,
        'detections': detections,
        'fps': stats.get('fps', 0),
        'detection_count': stats.get('detections_count', 0),
        'model_loaded': stats.get('model_loaded', False),
        'timestamp': time.time()
    })

@app.route('/api/start_auto', methods=['POST'])
def start_auto():
    """Start autonomous mode"""
    global current_mode
    
    with mode_lock:
        if current_mode != "STOPPED":
            return jsonify({'success': False, 'message': 'Stop current mode first'})
        
        try:
            current_mode = "AUTO"
            
            auto_thread = threading.Thread(
                target=auto_mode_detection,
                daemon=True,
                name="AutoDetection"
            )
            auto_thread.start()
            
            return jsonify({'success': True, 'message': 'Auto mode started'})
        except Exception as e:
            current_mode = "STOPPED"
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/manual_mode', methods=['POST'])
def manual_mode():
    """Switch to manual mode"""
    global current_mode
    
    with mode_lock:
        if current_mode != "STOPPED":
            return jsonify({'success': False, 'message': 'Stop robot first'})
        
        try:
            current_mode = "MANUAL"
            
            manual_thread = threading.Thread(
                target=manual_mode_detection,
                daemon=True,
                name="ManualDetection"
            )
            manual_thread.start()
            
            return jsonify({'success': True, 'message': 'Manual mode activated'})
        except Exception as e:
            current_mode = "STOPPED"
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop():
    """Stop all modes"""
    global current_mode
    
    with mode_lock:
        try:
            detection_state['auto_detection_active'] = False
            detection_state['manual_detection_active'] = False
            current_mode = "STOPPED"
            
            return jsonify({'success': True, 'message': 'Robot stopped'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/detections')
def get_detections():
    """Get current detections"""
    if camera:
        detections = camera.get_detections()
    else:
        detections = []
    
    return jsonify({
        'detections': detections,
        'count': len(detections),
        'timestamp': time.time()
    })

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    if camera:
        stats = camera.get_performance_stats()
    else:
        stats = {'fps': 0, 'detections_count': 0}
    
    return jsonify(stats)

@app.route('/api/manual_control/<action>', methods=['POST'])
def manual_control(action):
    """Manual control actions"""
    global current_mode
    
    if current_mode != "MANUAL":
        return jsonify({'success': False, 'message': 'Not in manual mode'})
    
    try:
        actions = ['forward', 'backward', 'left', 'right', 'wave', 'shake', 'dance', 'stand', 'sit']
        
        if action not in actions:
            return jsonify({'success': False, 'message': 'Unknown action'})
        
        action_emoji = {
            'forward': '⬆️',
            'backward': '⬇️',
            'left': '⬅️',
            'right': '➡️',
            'wave': '👋',
            'shake': '🤝',
            'dance': '💃',
            'stand': '🧍',
            'sit': '💺'
        }
        
        print(f"Manual: {action_emoji.get(action, '?')} {action}")
        
        return jsonify({'success': True, 'message': f'Action {action} executed'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🕷️  CyberCrawl Spider Robot - YOLOv8 Object Detection")
    print("="*70)
    print("✅ Live camera feed enabled")
    print("✅ Real-time YOLOv8 object detection")
    print("✅ Bounding boxes with confidence scores")
    print("✅ Separate detection modes (Auto/Manual)")
    print("="*70)
    print("\n🌐 Starting Flask server...")
    print("🔗 Access at: http://localhost:5000")
    print("🔗 Or: http://<your-raspberry-pi-ip>:5000")
    print("\n🎮 Features:")
    print("   - Live camera stream with detections")
    print("   - Person, object detection with boxes")
    print("   - FPS counter and performance metrics")
    print("   - Auto & Manual modes")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)