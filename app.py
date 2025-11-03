#!/usr/bin/env python3
"""
CyberCrawl Spider Robot - Flask Web Server (DEMO MODE)
Runs without physical hardware - perfect for testing/development
"""

from flask import Flask, render_template, Response, jsonify, request
import threading
import time
import cv2
import numpy as np
import random

app = Flask(__name__)

# Global variables
current_mode = "STOPPED"
mode_lock = threading.Lock()

# Demo state
demo_state = {
    'distance': 150,
    'detections': [],
    'auto_running': False,
    'robot_status': 'idle'
}

def generate_demo_frame():
    """Generate a demo frame with mock data"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create gradient background
    for i in range(480):
        frame[i, :] = [20, 30, 50]
    
    # Add grid
    for i in range(0, 640, 40):
        cv2.line(frame, (i, 0), (i, 480), (40, 40, 60), 1)
    for i in range(0, 480, 40):
        cv2.line(frame, (0, i), (640, i), (40, 40, 60), 1)
    
    # Title
    cv2.putText(frame, "CyberCrawl DEMO MODE", (80, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 212, 255), 3)
    
    # Status bar
    cv2.rectangle(frame, (20, 100), (620, 140), (50, 60, 80), -1)
    status_text = f"Mode: {current_mode} | Distance: {demo_state['distance']:.1f} cm | Objects: {len(demo_state['detections'])}"
    cv2.putText(frame, status_text, (35, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (6, 255, 165), 2)
    
    # Draw detected objects
    for i, detection in enumerate(demo_state['detections']):
        x = 100 + (i * 120) % 400
        y = 200 + (i * 100) % 200
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+100, y+80), (0, 255, 0), 2)
        
        # Draw label
        label = f"{detection['class']} {int(detection['confidence']*100)}%"
        cv2.putText(frame, label, (x+5, y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Robot status icon
    if current_mode == "AUTO":
        cv2.circle(frame, (600, 400), 15, (0, 212, 255), -1)
        cv2.putText(frame, "WALKING", (540, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
    elif current_mode == "MANUAL":
        cv2.circle(frame, (600, 400), 15, (123, 44, 191), -1)
        cv2.putText(frame, "MANUAL", (545, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (123, 44, 191), 2)
    else:
        cv2.circle(frame, (600, 400), 15, (100, 100, 100), -1)
        cv2.putText(frame, "STOPPED", (535, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    # Timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, timestamp, (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return frame

def generate_frames():
    """Video streaming generator function"""
    while True:
        try:
            frame = generate_demo_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Error generating frame: {e}")
            time.sleep(0.1)

def auto_mode_simulation():
    """Simulate autonomous mode"""
    print("🤖 Auto Mode: Starting simulation...")
    obstacle_count = 0
    
    while demo_state['auto_running'] and current_mode == "AUTO":
        try:
            # Simulate distance sensor readings
            if random.random() < 0.3:
                demo_state['distance'] = random.uniform(20, 200)
            
            # Simulate object detection
            if random.random() < 0.15 and len(demo_state['detections']) < 3:
                demo_state['detections'].append({
                    'class': random.choice(['person', 'cat', 'dog', 'car', 'cup', 'chair', 'plant']),
                    'confidence': round(random.uniform(0.65, 0.99), 2),
                    'bbox': [random.randint(50, 400), random.randint(100, 350), 100, 80]
                })
            
            # Remove detections randomly
            if demo_state['detections'] and random.random() < 0.1:
                demo_state['detections'].pop(0)
            
            # Obstacle avoidance logic
            if demo_state['distance'] < 25:
                print(f"⚠️  Obstacle detected at {demo_state['distance']:.1f} cm - Avoiding")
                obstacle_count += 1
            else:
                obstacle_count = 0
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Auto mode error: {e}")
            time.sleep(0.5)
    
    print("🛑 Auto Mode: Stopped")

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current robot status"""
    return jsonify({
        'mode': current_mode,
        'distance': round(demo_state['distance'], 1),
        'detections': demo_state['detections'],
        'timestamp': time.time()
    })

@app.route('/api/start_auto', methods=['POST'])
def start_auto():
    """Start auto mode"""
    global current_mode
    
    with mode_lock:
        if current_mode != "STOPPED":
            return jsonify({'success': False, 'message': 'Stop current mode first'})
        
        try:
            current_mode = "AUTO"
            demo_state['auto_running'] = True
            auto_thread = threading.Thread(target=auto_mode_simulation, daemon=True)
            auto_thread.start()
            
            return jsonify({'success': True, 'message': 'Auto mode started'})
        except Exception as e:
            current_mode = "STOPPED"
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop():
    """Stop all movements"""
    global current_mode
    
    with mode_lock:
        try:
            demo_state['auto_running'] = False
            demo_state['detections'] = []
            current_mode = "STOPPED"
            
            return jsonify({'success': True, 'message': 'Robot stopped'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

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
    global current_mode
    
    if current_mode != "MANUAL":
        return jsonify({'success': False, 'message': 'Not in manual mode'})
    
    try:
        actions = ['forward', 'backward', 'left', 'right', 'wave', 'shake', 'dance', 'stand', 'sit']
        
        if action not in actions:
            return jsonify({'success': False, 'message': 'Unknown action'})
        
        action_text = {
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
        
        print(f"Manual: {action_text.get(action, action)}")
        
        return jsonify({'success': True, 'message': f'Action {action} executed'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/detections')
def get_detections():
    """Get current object detections"""
    return jsonify({'detections': demo_state['detections']})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🕷️  CyberCrawl Spider Robot - DEMO MODE")
    print("="*60)
    print("✅ Hardware simulation enabled")
    print("✅ Web interface ready")
    print("="*60)
    print("\n🌐 Starting web server...")
    print("📍 Access at: http://localhost:5000")
    print("📍 Or: http://<your-ip>:5000")
    print("\n🎮 Features:")
    print("   - Live video feed with detections")
    print("   - Auto mode with obstacle simulation")
    print("   - Manual control with keyboard support")
    print("   - Object detection display")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)