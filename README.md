# 🕷️ CyberCrawl - Spider Robot with Object Detection

A sophisticated spider robot controlled by Raspberry Pi 3 featuring autonomous navigation, object detection using YOLOv8, and a modern web interface.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%203-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Features

- 🤖 **Autonomous Mode**: Self-navigation with obstacle avoidance
- 🎮 **Manual Control**: Web-based joystick control
- 📹 **Live Camera Feed**: Real-time video streaming
- 🧠 **Object Detection**: YOLOv8-powered computer vision
- 🌙 **Night Vision**: Automatic IR LED activation
- 🕸️ **Spider Movement**: Realistic 12-servo quadruped gait
- 📱 **Responsive Web UI**: Modern, mobile-friendly interface

---

## 🛠️ Hardware Requirements

### Core Components

| Component | Model | Quantity |
|-----------|-------|----------|
| Microcontroller | Raspberry Pi 3 Model B+ | 1 |
| Servo Motors | SG90 or MG90S (180°) | 12 |
| Servo Driver | PCA9685 16-Channel I2C | 1 |
| Camera | OV5647 Camera Module | 1 |
| Ultrasonic Sensor | HC-SR04 | 1 |
| Power Supply | 5V 2A+ (servos) | 1 |
| Power Supply | 5V 2.5A (Raspberry Pi) | 1 |
| IR LEDs | 850nm (optional) | 2-4 |

### Optional Components
- Transistor for IR LED control (2N2222)
- Resistors (220Ω for LEDs)
- Spider robot chassis/frame
- Battery pack (2S LiPo or 5V power bank)

---

## 🔌 Hardware Connections

### PCA9685 Servo Driver
```
PCA9685          Raspberry Pi 3
VCC        →     5V (Pin 2 or 4)
GND        →     GND (Pin 6, 9, 14, 20, 25, 30, 34, 39)
SCL        →     GPIO 3 (SCL, Pin 5)
SDA        →     GPIO 2 (SDA, Pin 3)
V+         →     External 5V Supply (Servos)
```

### Servo Connections
Connect servos to PCA9685 channels:
- **Leg 0** (Front-Right): Channels 0, 1, 2
- **Leg 1** (Front-Left): Channels 4, 5, 6
- **Leg 2** (Rear-Left): Channels 8, 9, 10
- **Leg 3** (Rear-Right): Channels 12, 13, 14

### HC-SR04 Ultrasonic Sensor
```
HC-SR04          Raspberry Pi 3
VCC        →     5V (Pin 2)
TRIG       →     GPIO 23 (Pin 16)
ECHO       →     GPIO 24 (Pin 18)
GND        →     GND (Pin 20)
```

### OV5647 Camera Module
Connect to CSI camera port on Raspberry Pi.

### IR LEDs (Optional)
```
IR LED + → GPIO 18 (Pin 12) → 220Ω resistor → Transistor collector
Transistor base → GPIO (via 1kΩ resistor)
Transistor emitter → GND
```

---

## 💻 Software Installation

### 1. Prepare Raspberry Pi

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    i2c-tools \
    git \
    cmake \
    libopencv-dev \
    python3-opencv
```

### 2. Enable I2C and Camera

```bash
sudo raspi-config
```

Navigate to:
- **Interface Options** → **I2C** → **Enable**
- **Interface Options** → **Camera** → **Enable**
- Reboot: `sudo reboot`

### 3. Verify I2C Connection

```bash
sudo i2cdetect -y 1
```

You should see address `0x40` (PCA9685).

### 4. Clone Project

```bash
cd ~
git clone <your-repo-url> cybercrawl
cd cybercrawl
```

### 5. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 6. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installing PyTorch and YOLO on Raspberry Pi takes 30-60 minutes.

### 7. Download YOLO Model

```bash
# The model will auto-download on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 8. Create Required Directories

```bash
mkdir -p movement sensors camera static/css static/js static/images templates
```

### 9. Add Empty __init__.py Files

```bash
touch movement/__init__.py
touch sensors/__init__.py
touch camera/__init__.py
```

---

## 🚀 Running the Application

### Start the Server

```bash
cd ~/cybercrawl
source venv/bin/activate
python app.py
```

### Access Web Interface

Open your browser and navigate to:
```
http://<raspberry-pi-ip>:5000
```

To find your Raspberry Pi's IP:
```bash
hostname -I
```

---

## 🎮 Usage Instructions

### Auto Mode
1. Click **"Auto Mode"** button
2. Robot will:
   - Stand up and wave
   - Start walking forward
   - Detect obstacles and navigate around them
   - Display detected objects on video feed
   - Activate night vision in low light

### Manual Mode
1. Click **"Stop"** to halt robot
2. Click **"Manual Mode"**
3. Control using:
   - **Direction buttons**: Forward, Backward, Left, Right
   - **Keyboard**: Arrow keys or WASD
   - **Action buttons**: Wave, Shake, Dance, Stand, Sit

### Stop
- Click **"Stop"** at any time to halt all movement
- Press **Spacebar** for emergency stop in manual mode

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Adjust servo channels if your wiring differs
SERVO_CHANNELS = [
    [0, 1, 2],    # Leg 0
    [4, 5, 6],    # Leg 1
    [8, 9, 10],   # Leg 2
    [12, 13, 14]  # Leg 3
]

# Adjust obstacle detection distance
OBSTACLE_THRESHOLD = 20  # cm

# Camera resolution
CAMERA_RESOLUTION = (640, 480)

# YOLO confidence threshold
YOLO_CONFIDENCE_THRESHOLD = 0.5
```

---

## 🐛 Troubleshooting

### I2C Not Detected
```bash
# Check if I2C is enabled
ls /dev/i2c*

# Install I2C tools
sudo apt-get install i2c-tools

# Scan for devices
sudo i2cdetect -y 1
```

### Servos Not Moving
- Check power supply (servos need 2A+)
- Verify PCA9685 connections
- Test with: `sudo i2cdetect -y 1`
- Adjust `SERVO_PULSE_RANGE` in config.py

### Camera Not Working
```bash
# Test camera
libcamera-hello

# Check camera interface
sudo raspi-config
# Interface Options → Camera → Enable
```

### YOLO Detection Slow
- Use lighter model: `yolov8n.pt` (nano)
- Reduce camera resolution in config.py
- Lower FPS: `CAMERA_FPS = 15`

### Import Errors
```bash
# Reinstall in virtual environment
source venv/bin/activate
pip install --upgrade --force-reinstall -r requirements.txt
```

---

## 📁 Project Structure

```
cybercrawl/
├── app.py                      # Flask main server
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── movement/
│   ├── __init__.py
│   ├── servo_control.py       # PCA9685 servo controller
│   └── auto_walk.py           # Autonomous navigation
│
├── sensors/
│   ├── __init__.py
│   └── ultrasonic.py          # HC-SR04 sensor
│
├── camera/
│   ├── __init__.py
│   └── camera_yolo.py         # Camera & YOLO detection
│
├── templates/
│   └── index.html             # Web interface
│
└── static/
    ├── css/
    │   └── style.css          # Styling
    ├── js/
    │   └── script.js          # Frontend logic
    └── images/
        └── spider_icon.png
```

---

## 🔧 Advanced Features

### Auto-Start on Boot

```bash
# Create systemd service
sudo nano /etc/systemd/system/cybercrawl.service
```

Add:
```ini
[Unit]
Description=CyberCrawl Spider Robot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/cybercrawl
ExecStart=/home/pi/cybercrawl/venv/bin/python /home/pi/cybercrawl/app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable cybercrawl
sudo systemctl start cybercrawl
```

### Servo Calibration

If servos are misaligned, adjust in `config.py`:
```python
SERVO_PULSE_RANGE = [150, 600]  # Adjust these values
```

Or add per-servo calibration:
```python
SERVO_OFFSETS = [
    [0, 5, -3],   # Leg 0 offsets (degrees)
    [2, 0, 0],    # Leg 1
    [-1, 3, 2],   # Leg 2
    [0, -2, 1]    # Leg 3
]
```

---

## 📊 Performance Tips

1. **Reduce CPU Load**:
   - Lower camera FPS
   - Use smaller YOLO model (nano)
   - Reduce detection frequency

2. **Improve Battery Life**:
   - Lower servo speeds
   - Disable night vision when not needed
   - Use efficient power supply

3. **Better Detection**:
   - Good lighting conditions
   - Camera at proper angle
   - Clean camera lens

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

CyberCrawl Spider Robot Project
- **Hardware**: Raspberry Pi 3 + PCA9685 + 12 Servos
- **Software**: Python + Flask + YOLOv8
- **Framework**: Quadruped inverse kinematics

---

## 🙏 Acknowledgments

- Arduino Quadruped Robot Project (inverse kinematics reference)
- YOLOv8 by Ultralytics
- Flask web framework
- Adafruit PCA9685 library

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check troubleshooting section
- Review hardware connections

---

**Happy Crawling! 🕷️**