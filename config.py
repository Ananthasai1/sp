#!/usr/bin/env python3
"""
Configuration settings for CyberCrawl Spider Robot
Updated for Raspberry Pi OS 64-bit with OV5647
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

# Camera Settings (OV5647 optimal settings)
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 20  # Lower FPS for better stability on Pi 3

# YOLOv8 Detection Settings
YOLO_MODEL_PATH = 'yolov8n.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45

# Performance Settings
DETECTION_INTERVAL = 0.1  # Run YOLO every 100ms
FRAME_BUFFER_SIZE = 2

# Night Vision Settings
NIGHT_VISION_THRESHOLD = 50
NIGHT_VISION_GPIO = 18

# Auto Mode Settings
AUTO_MODE_LOOP_DELAY = 0.05

# Servo Calibration
SERVO_PULSE_RANGE = [150, 600]