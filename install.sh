#!/bin/bash

# CyberCrawl Complete Installation Script
# For Raspberry Pi OS 64-bit with OV5647 Camera

set -e  # Exit on error

echo "🕷️  CyberCrawl Spider Robot - Complete Installation"
echo "========================================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo "⚠️  Please run as normal user (not sudo)"
   exit 1
fi

# Check Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: Not a Raspberry Pi"
    read -p "Continue anyway? (y/N): " confirm
    [[ $confirm != [yY] ]] && exit 1
fi

# Update system
echo ""
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    i2c-tools \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    python3-opencv \
    libcamera-dev \
    libcamera-apps \
    python3-picamera2 \
    python3-libcamera \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev

# Enable I2C
echo ""
echo "⚙️  Enabling I2C..."
sudo raspi-config nonint do_i2c 0
if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
    if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null; then
        echo "dtparam=i2c_arm=on" | sudo tee -a /boot/config.txt
    fi
fi

# Enable Camera
echo ""
echo "📷 Enabling OV5647 camera..."
sudo raspi-config nonint do_camera 0

# Add camera configuration
if ! grep -q "^start_x=1" /boot/config.txt 2>/dev/null; then
    echo "start_x=1" | sudo tee -a /boot/config.txt
fi
if ! grep -q "^gpu_mem=128" /boot/config.txt 2>/dev/null; then
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
fi
if ! grep -q "^camera_auto_detect=1" /boot/config.txt 2>/dev/null; then
    echo "camera_auto_detect=1" | sudo tee -a /boot/config.txt
fi

# Create project directory
echo ""
echo "📁 Creating project structure..."
PROJECT_DIR="$HOME/cybercrawl"

if [ -d "$PROJECT_DIR" ]; then
    echo "⚠️  Directory exists: $PROJECT_DIR"
    read -p "Remove and reinstall? (y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        rm -rf "$PROJECT_DIR"
    else
        echo "Installation cancelled"
        exit 1
    fi
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create subdirectories
mkdir -p movement sensors camera templates static/css static/js static/images

# Create __init__.py files
touch movement/__init__.py
touch sensors/__init__.py
touch camera/__init__.py

# Create virtual environment
echo ""
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python packages
echo ""
echo "📚 Installing Python dependencies..."
echo "⏰ This will take 30-60 minutes on Raspberry Pi 3"
echo ""

# Create requirements.txt
cat > requirements.txt << 'EOF'
Flask==3.0.0
Werkzeug==3.0.1
RPi.GPIO==0.7.1
smbus2==0.4.3
adafruit-circuitpython-pca9685==3.4.15
adafruit-blinka==8.25.0
opencv-python==4.8.1.78
numpy==1.24.4
Pillow==10.1.0
picamera2==0.3.17
EOF

pip install -r requirements.txt

# Install PyTorch (CPU version)
echo ""
echo "🔥 Installing PyTorch (CPU version)..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install YOLO
echo ""
echo "🧠 Installing YOLOv8..."
pip install ultralytics==8.1.0

# Download YOLO model
echo ""
echo "📥 Downloading YOLOv8 nano model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create run script
cat > run.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python app.py
EOF
chmod +x run.sh

# Create systemd service
cat > cybercrawl.service << EOF
[Unit]
Description=CyberCrawl Spider Robot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Test I2C
echo ""
echo "🔍 Testing I2C connection..."
if command -v i2cdetect &> /dev/null; then
    echo "Scanning I2C bus..."
    sudo i2cdetect -y 1 || true
else
    echo "⚠️  i2cdetect not available"
fi

# Create README with next steps
cat > NEXT_STEPS.txt << 'EOF'
🕷️  CyberCrawl Installation Complete!

NEXT STEPS:

1. REBOOT YOUR RASPBERRY PI:
   sudo reboot

2. After reboot, test camera:
   rpicam-hello -t 5000

3. Navigate to project:
   cd ~/cybercrawl
   source venv/bin/activate

4. Copy your code files to ~/cybercrawl:
   - app.py
   - config.py
   - camera/camera_yolo.py
   - templates/index.html
   - static/css/style.css
   - static/js/script.js
   (These should already be there from GitHub)

5. Run the application:
   python app.py
   
   Or use the run script:
   ./run.sh

6. Access web interface:
   Open browser to: http://<raspberry-pi-ip>:5000
   
   Find your IP with: hostname -I

7. (Optional) Enable auto-start on boot:
   sudo cp ~/cybercrawl/cybercrawl.service /etc/systemd/system/
   sudo systemctl enable cybercrawl
   sudo systemctl start cybercrawl

TROUBLESHOOTING:

Camera not working:
  - Check cable connection (blue side to camera, silver to Pi)
  - Run: rpicam-hello -t 5000
  - Check: ls /dev/video*
  - Verify /boot/config.txt has camera settings

I2C not working:
  - Run: sudo i2cdetect -y 1
  - Should see 0x40 (PCA9685)
  - Check wiring connections

Python errors:
  - Make sure virtual environment is activated
  - Reinstall: pip install --upgrade --force-reinstall -r requirements.txt

COMMANDS:

Start server:     ./run.sh
Stop server:      Ctrl+C
Check status:     systemctl status cybercrawl
View logs:        journalctl -u cybercrawl -f
Test camera:      rpicam-hello -t 5000
Test I2C:         sudo i2cdetect -y 1
EOF

echo ""
echo "="*70
echo "✅ Installation Complete!"
echo "="*70
echo ""
echo "📋 IMPORTANT: Read NEXT_STEPS.txt for what to do next"
echo ""
echo "Quick start:"
echo "  1. sudo reboot"
echo "  2. cd ~/cybercrawl"
echo "  3. source venv/bin/activate"
echo "  4. python app.py"
echo ""
echo "Then open: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "🕷️  Happy Crawling!"
echo "="*70