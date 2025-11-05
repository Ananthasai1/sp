#!/usr/bin/env python3
"""
Configuration settings for CyberCrawl Spider Robot - YOLOv12 Edition
"""

# Flask Server Settings
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000
DEBUG = False

# GPIO Pin Assignments
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN = 24

# PCA9685 Servo Driver Settings
PCA9685_ADDRESS = 0x40
PCA9685_FREQUENCY = 50

# Servo Channel Mapping [leg][joint]
SERVO_CHANNELS = [
    [0, 1, 2],    # Leg 0 (Front-Right)
    [4, 5, 6],    # Leg 1 (Front-Left)
    [8, 9, 10],   # Leg 2 (Rear-Left)
    [12, 13, 14]  # Leg 3 (Rear-Right)
]

# Robot Physical Dimensions (mm)
LENGTH_A = 55.0
LENGTH_B = 77.5
LENGTH_C = 27.5
LENGTH_SIDE = 71.0

# Movement Parameters
Z_DEFAULT = -50.0
Z_UP = -30.0
Z_BOOT = -28.0
X_DEFAULT = 62.0
X_OFFSET = 0.0
Y_START = 0.0
Y_STEP = 40.0

# Movement Speeds
LEG_MOVE_SPEED = 8.0
BODY_MOVE_SPEED = 3.0
SPOT_TURN_SPEED = 4.0
STAND_SEAT_SPEED = 1.0
SPEED_MULTIPLE = 1.2

# Ultrasonic Sensor Settings
OBSTACLE_THRESHOLD = 20
MAX_DISTANCE = 200

# Camera Settings - Optimized for OV5647
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 30
CAMERA_FORMAT = 'BGR888'

# YOLOv12 Detection Settings
YOLO_MODEL_PATH = 'yolov12n.pt'  # YOLOv12 Nano (lightweight)
# Alternative models:
# 'yolov12s.pt'   # Small (better accuracy, slower)
# 'yolov12m.pt'   # Medium (more accurate, slower)
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = 'cpu'  # 'cpu' for Raspberry Pi, 'cuda' for GPU if available

# Auto Mode Settings
AUTO_MODE_LOOP_DELAY = 0.05

# Servo Calibration
SERVO_PULSE_RANGE = [150, 600]

# Performance Settings
MAX_FRAMES_BUFFER = 2  # Keep only latest frames for low latency
DETECTION_CONFIDENCE_MIN = 0.45  # Minimum confidence to show detection
SKIP_FRAMES_FOR_DETECTION = 1  # Process every frame (set to 2 to skip frames)