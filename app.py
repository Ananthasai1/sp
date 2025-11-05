#!/usr/bin/env python3
"""
CyberCrawl Spider Robot - Flask Server
FIXED: Proper video streaming with OV5647
"""

from flask import Flask, render_template, Response, jsonify
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
camera = None

# Detection state
detection_state = {
    'auto_detection_active': False,
    'manual_detection_active': False,
}

# Initialize camera
def init_camera():
    global camera
    try:
        print("📷 Initializing camera...")
        camera = EnhancedCameraYOLO()
        camera.start_detection()
        print("✅ Camera ready!")
        return True
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        camera = None
        return False

# Video streaming generator
def generate_frames():
    """Generate video frames with MJPEG streaming"""
    print("📹 Starting video stream...")
    
    frame_count = 0
    last_log = time.time()
    
    while True:
        try:
            if camera is None:
                # No camera - show error frame
                frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                                config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA NOT AVAILABLE", (80, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "Check connections", (140, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                # Get frame with detections
                frame = camera.get_frame_with_detections()
                
                if frame is None or frame.size == 0:
                    # Fallback frame
                    frame = np.zeros((config.CAMERA_RESOLUTION[1], 
                                    config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting for camera...", (80, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if not ret or buffer is None:
                time.sleep(0.05)
                continue
            
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            
            # Yield MJPEG frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            
            # Log status every 5 seconds
            if time.time() - last_log > 5:
                print(f"  📊 Streaming: {frame_count} frames sent")
                last_log = time.time()
            
            # Small delay to control bandwidth
            time.sleep(0.033)  # ~30 FPS max
            
        except GeneratorExit:
            print("  📹 Client disconnected from video feed")
            break
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            time.sleep(0.1)

# Auto mode detection thread
def auto_mode_detection():
    """Detection for autonomous mode"""
    print("🤖 Auto mode detection started")
    detection_state['auto_detection_active'] = True
    
    while current_mode == "AUTO" and detection_state['auto_detection_active']:
        try:
            if camera:
                detections = camera.get_detections()
                
                # Check for obstacles
                critical = ['person', 'cat', 'dog', 'car', 'motorcycle', 'bicycle']
                for det in detections:
                    if det['class'] in critical and det['confidence'] > 0.65:
                        print(f"  🎯 Detected {det['class']} ({det['confidence']:.2f})")
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Auto detection error: {e}")
            time.sleep(0.1)
    
    detection_state['auto_detection_active'] = False
    print("🤖 Auto detection stopped")

# Manual mode detection thread
def manual_mode_detection():
    """Detection for manual mode"""
    print("⚙️ Manual mode detection started")
    detection_state['manual_detection_active'] = True
    
    while current_mode == "MANUAL" and detection_state['manual_detection_active']:
        try:
            if camera:
                detections = camera.get_detections()
                # Just track objects, no action needed
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Manual detection error: {e}")
            time.sleep(0.1)
    
    detection_state['manual_detection_active'] = False
    print("⚙️ Manual detection stopped")

# ===== ROUTES =====

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route - MJPEG"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get robot status"""
    if camera:
        detections = camera.get_detections()
        stats = camera.get_performance_stats()
        distance = 150.0  # Placeholder for ultrasonic
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
        'camera_ready': stats.get('camera_ready', False),
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
            threading.Thread(target=auto_mode_detection, daemon=True).start()
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
            threading.Thread(target=manual_mode_detection, daemon=True).start()
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

@app.route('/api/manual_control/<action>', methods=['POST'])
def manual_control(action):
    """Manual control actions"""
    if current_mode != "MANUAL":
        return jsonify({'success': False, 'message': 'Not in manual mode'})
    
    actions = ['forward', 'backward', 'left', 'right', 'wave', 'shake', 'dance', 'stand', 'sit']
    
    if action not in actions:
        return jsonify({'success': False, 'message': 'Unknown action'})
    
    print(f"Manual: {action}")
    return jsonify({'success': True, 'message': f'Action {action} executed'})

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

# Cleanup on shutdown
def cleanup():
    """Cleanup resources"""
    global camera
    if camera:
        try:
            camera.cleanup()
        except:
            pass

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🕷️  CyberCrawl Spider Robot - OV5647 + YOLOv8")
    print("="*70)
    
    # Initialize camera
    if init_camera():
        print("✅ Live video feed enabled")
        print("✅ Real-time object detection active")
    else:
        print("⚠️  Running without camera")
    
    print("="*70)
    print("\n🌐 Starting Flask server...")
    print("🔗 Access at: http://localhost:5000")
    print("🔗 Or: http://<raspberry-pi-ip>:5000")
    print("\n🎮 Features:")
    print("   - Live OV5647 camera stream")
    print("   - YOLOv8 object detection")
    print("   - Auto & Manual modes")
    print("="*70 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cleanup()