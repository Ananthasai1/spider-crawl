#!/bin/bash

# CyberCrawl Spider Robot - Automated Installation Script
# For Raspberry Pi 3

echo "🕷️  CyberCrawl Spider Robot Installation"
echo "========================================"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/N): " confirm
    if [[ $confirm != [yY] ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    i2c-tools \
    git \
    cmake \
    libopencv-dev \
    python3-opencv

# Enable I2C
echo "⚙️  Enabling I2C..."
if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt; then
    echo "dtparam=i2c_arm=on" | sudo tee -a /boot/config.txt
    echo "✅ I2C enabled (reboot required)"
fi

# Enable Camera
echo "📷 Enabling camera..."
if ! grep -q "^start_x=1" /boot/config.txt; then
    echo "start_x=1" | sudo tee -a /boot/config.txt
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
    echo "✅ Camera enabled (reboot required)"
fi

# Create project directory
echo "📁 Setting up project directory..."
PROJECT_DIR="$HOME/cybercrawl"

if [ -d "$PROJECT_DIR" ]; then
    echo "⚠️  Directory $PROJECT_DIR already exists"
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

# Create directory structure
echo "🗂️  Creating directory structure..."
mkdir -p movement sensors camera templates static/css static/js static/images

# Create __init__.py files
echo "Creating module files..."
touch movement/__init__.py
touch sensors/__init__.py
touch camera/__init__.py

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "📚 Installing Python dependencies (this may take 30-60 minutes)..."
pip install --upgrade pip

# Install requirements
cat > requirements.txt << 'EOF'
Flask==3.0.0
Werkzeug==3.0.1
RPi.GPIO==0.7.1
smbus2==0.4.3
adafruit-circuitpython-pca9685==3.4.15
adafruit-blinka==8.25.0
opencv-python==4.8.1.78
numpy==1.24.4
ultralytics==8.1.0
Pillow==10.1.0
EOF

pip install -r requirements.txt

# Download YOLO model
echo "🧠 Downloading YOLO model..."
wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Test I2C
echo ""
echo "🔍 Testing I2C connection..."
if command -v i2cdetect &> /dev/null; then
    echo "Running i2cdetect..."
    sudo i2cdetect -y 1
else
    echo "⚠️  i2cdetect not available"
fi

# Create run script
cat > run.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python app.py
EOF
chmod +x run.sh

# Create systemd service file
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

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your Python files to $PROJECT_DIR"
echo "   - app.py, config.py"
echo "   - movement/servo_control.py, movement/auto_walk.py"
echo "   - sensors/ultrasonic.py"
echo "   - camera/camera_yolo.py"
echo "   - templates/index.html"
echo "   - static/css/style.css, static/js/script.js"
echo ""
echo "2. Reboot to enable I2C and camera:"
echo "   sudo reboot"
echo ""
echo "3. After reboot, run the application:"
echo "   cd $PROJECT_DIR"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "4. Access web interface at:"
echo "   http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "5. (Optional) Enable auto-start on boot:"
echo "   sudo cp $PROJECT_DIR/cybercrawl.service /etc/systemd/system/"
echo "   sudo systemctl enable cybercrawl"
echo "   sudo systemctl start cybercrawl"
echo ""
echo "🕷️  Happy crawling!"