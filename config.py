#!/usr/bin/env python3
"""
Configuration settings for CyberCrawl Spider Robot
Updated for YOLOv8
"""

# Flask Server Settings
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 5000
DEBUG = False  # Set to False in production

# GPIO Pin Assignments
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN = 24

# PCA9685 Servo Driver Settings
PCA9685_ADDRESS = 0x40
PCA9685_FREQUENCY = 50  # Hz (standard for servos)

# Servo Channel Mapping [leg][joint]
# Leg numbering: 0=Front-Right, 1=Front-Left, 2=Rear-Left, 3=Rear-Right
# Joint numbering: 0=Coxa, 1=Femur, 2=Tibia
SERVO_CHANNELS = [
    [0, 1, 2],    # Leg 0 (Front-Right)
    [4, 5, 6],    # Leg 1 (Front-Left)
    [8, 9, 10],   # Leg 2 (Rear-Left)
    [12, 13, 14]  # Leg 3 (Rear-Right)
]

# Robot Physical Dimensions (mm)
LENGTH_A = 55.0      # Upper leg length
LENGTH_B = 77.5      # Lower leg length
LENGTH_C = 27.5      # Coxa length
LENGTH_SIDE = 71.0   # Body side length

# Movement Parameters
Z_DEFAULT = -50.0    # Default standing height
Z_UP = -30.0         # Leg lift height
Z_BOOT = -28.0       # Boot/sit position
X_DEFAULT = 62.0     # Default X position
X_OFFSET = 0.0       # X offset for body
Y_START = 0.0        # Starting Y position
Y_STEP = 40.0        # Step length

# Movement Speeds
LEG_MOVE_SPEED = 8.0
BODY_MOVE_SPEED = 3.0
SPOT_TURN_SPEED = 4.0
STAND_SEAT_SPEED = 1.0
SPEED_MULTIPLE = 1.2

# Ultrasonic Sensor Settings
OBSTACLE_THRESHOLD = 20  # cm - distance to trigger avoidance
MAX_DISTANCE = 200       # cm - maximum detection range

# Camera Settings
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 30

# ===== YOLOv8 Detection Settings =====
# Use yolov8n.pt (nano - smallest & fastest)
# Other options: yolov8s.pt, yolov8m.pt, yolov8l.pt (larger = slower but more accurate)
YOLO_MODEL_PATH = 'yolov8n.pt'  # Path to YOLO model file
YOLO_CONFIDENCE_THRESHOLD = 0.5   # 0.5 = 50% confidence minimum
YOLO_IOU_THRESHOLD = 0.45         # Intersection over Union threshold

# Night Vision Settings
NIGHT_VISION_THRESHOLD = 50  # Brightness threshold for night mode
NIGHT_VISION_GPIO = 18       # GPIO pin for IR LEDs (if used)

# Auto Mode Settings
AUTO_MODE_LOOP_DELAY = 0.05  # seconds between sensor readings

# Servo Calibration
# Format: [min_pulse, max_pulse] for 0-180 degrees
SERVO_PULSE_RANGE = [150, 600]  # Typical SG90 servo values