# AI-Powered Drumming Robot

An innovative robotic system that generates and performs drum patterns using artificial intelligence. This project combines a Generative Adversarial Network (GAN) with physical actuators to create an autonomous drumming robot capable of generating novel rhythmic patterns.

## 🎥 Demo Video
[Insert your demo video link here]

## 💻 Code Repository
[Insert your GitHub repository link here]

## 📋 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [File Descriptions](#file-descriptions)
- [Hardware Setup](#hardware-setup)
- [Troubleshooting](#troubleshooting)
- [Team](#team)

## 🎯 Overview

This project demonstrates the integration of AI-generated music with physical robotics. The system processes MIDI drum patterns, trains a GAN to generate new rhythms, and translates them into physical drum performances using Arduino-controlled servo motors with computer vision safety monitoring.

### Key Features
- **AI Pattern Generation**: GAN-based drum pattern creation
- **Computer Vision Safety**: Real-time distance monitoring for safe operation
- **Physical Performance**: Multi-servo coordination for drum strikes
- **Real-time Feedback**: OLED display and RGB LED indicators

## 🔧 System Requirements

### Software Dependencies
```
Python 3.7+
TensorFlow 2.x
OpenCV (cv2)
NumPy
Pandas
Matplotlib
Mido
PIL (Pillow)
scikit-learn
```

### Hardware Requirements
- ESP32 microcontroller
- 3x Servo motors (ST series)
- Camera module
- OLED display (SSD1306)
- RGB LED strip (NeoPixel)
- Drum set with mounting hardware

### Arduino Libraries
```
SCServo
Adafruit_GFX
Adafruit_SSD1306
Adafruit_NeoPixel
Wire
```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tcellsx/robotics_final
   ```

2. **Install Python dependencies**
   ```bash
   pip install tensorflow opencv-python numpy pandas matplotlib mido pillow scikit-learn
   ```

3. **Install Arduino libraries**
   - Open Arduino IDE
   - Install required libraries through Library Manager

## 🚀 Usage Workflow

Follow these steps in order to run the complete system:

### Step 1: Data Processing (Optional - if training new model)
```bash
python midi2csv.py --output drum_patterns.csv
```
- Downloads Google Magenta's Groove MIDI dataset
- Converts MIDI files to CSV format suitable for training
- Extracts patterns for kick, snare, hi-hat, tom1, tom2

### Step 2: Model Training (Optional - if training new model)
```bash
python ganpcg.py --input drum_patterns.csv --epochs 150
```
- Trains the GAN model on drum patterns
- Saves trained generator and discriminator models
- Creates training history visualization

### Step 3: Camera Safety System
```bash
python camera.py
```
- **IMPORTANT: Run this first before physical operation**
- Calibrate the distance measurement system:
  - Press 's' to calibrate camera-to-object distance
  - Place red markers at specified distances
  - Ensure "READY TO HIT" status is displayed
- Keep this running during robot operation for safety monitoring

### Step 4: Generate New Drum Patterns
```bash
python ganpcg.py --num-patterns 1 --temperature 1.0
```
- Uses trained GAN to generate new drum patterns
- Outputs `generated_pattern.json` file
- Adjust temperature (0.5-2.0) for different creativity levels

### Step 5: Update Robot Controller
```bash
python change_ion.py
```
- Automatically reads `generated_pattern.json`
- Updates Arduino code with new drum patterns
- Creates backup of previous controller code

### Step 6: Upload to Arduino
1. Open `sketch_apr24a/sketch_apr24a.ino` in Arduino IDE
2. Connect ESP32 controller
3. Upload the updated code
4. Monitor serial output for initialization status

### Step 7: Robot Operation
- Ensure camera system shows "READY TO HIT"
- Robot will automatically start playing the generated pattern
- Monitor OLED display for current beat and pattern status
- RGB LEDs indicate active drums (Red=Kick, Green=Snare, Blue=Hi-hat)

## 📁 File Descriptions

### Core Python Files

#### `midi2csv.py`
**Purpose**: Data preprocessing pipeline
- Downloads and processes Google Magenta Groove MIDI dataset
- Converts MIDI files to structured CSV format
- Supports multiple output formats (time-series and flat)
- **Usage**: `python midi2csv.py --output drum_patterns.csv --max-patterns 20`

#### `ganpcg.py`
**Purpose**: GAN training and pattern generation
- Implements Generator and Discriminator networks
- Trains on drum pattern data with data augmentation
- Generates new drum patterns from trained model
- **Key Functions**:
  - `train_gan()`: Main training loop
  - `generate_patterns()`: Create new patterns
  - `print_drum_pattern()`: Visualize patterns

#### `camera.py`
**Purpose**: Computer vision safety system
- Real-time distance measurement using red markers
- HSV color space filtering and contour detection
- Dual calibration system (distance and focal length)
- Safety state management with filtering
- **Controls**:
  - Press 's': Calibrate camera distance
  - Press 'c': Calibrate object distance
  - Press 'q': Quit

#### `change_ion.py`
**Purpose**: Arduino code updater
- Reads generated drum patterns from JSON
- Updates hardcoded patterns in Arduino sketch
- Creates automatic backups
- Validates pattern format and length

### Arduino Code

#### `sketch_apr24a.ino`
**Purpose**: Robot controller firmware
- ESP32-based servo motor control
- FreeRTOS multitasking for precise timing
- OLED display and RGB LED feedback
- Near-simultaneous multi-drum coordination
- **Key Components**:
  - Servo configuration and control
  - Pattern playback engine
  - Visual feedback system

## 🔧 Hardware Setup

### Servo Motor Configuration
```cpp
// Servo IDs and positions (customize as needed)
ID 1: Hi-hat    (Start: 1600, Strike: 500)
ID 3: Kick drum (Start: 1900, Strike: 900)  
ID 7: Snare     (Start: 1800, Strike: 2500)
```

### Camera Setup
1. Mount camera with clear view of drum area
2. Place two red markers in drum zone
3. Ensure adequate lighting for color detection
4. Calibrate distance measurements

### Wiring Diagram
```
ESP32 Connections:
- Servo Bus: Pins 18(RX), 19(TX)
- I2C (Display): Pins 21(SDA), 22(SCL)
- RGB LEDs: Pin 23
- Camera: USB connection to computer
```

## 🔍 Troubleshooting

### Common Issues

**"Cannot find generator.h5"**
- Solution: Run training first with `python ganpcg.py --input drum_patterns.csv`

**Camera not detecting red markers**
- Check lighting conditions
- Adjust HSV color ranges in `camera.py`
- Ensure markers are clearly visible

**Servos not responding**
- Verify servo IDs match configuration
- Check power supply (servos require adequate current)
- Ensure proper wiring connections

**"NOT READY" status in camera**
- Verify red markers are at correct distances (44-47cm and 51-56cm)
- Recalibrate distance measurement
- Check for stable detection (8 consecutive frames required)

### Performance Optimization

**Improve Pattern Quality**
- Increase training epochs (150-300)
- Adjust data augmentation parameters
- Experiment with different temperature values

**Enhance Safety**
- Reduce frame confirmation threshold
- Increase filtering window size
- Add additional safety markers

## 👥 Team

**Group: Deepsickers**
- Xiaobin Tang (s4221923) - x.tang.8@umail.leidenuniv.nl
- Loogle Lu (s4212681) - D.lu@umail.leidenuniv.nl  
- Di Xie (s4347978) - d.xie@umail.leidenuniv.nl
- Junze Yang (s4244532) - j.yang.18@umail.leidenuniv.nl
- Ruishu Li (s4331184) - r.li.15@umail.leidenuniv.nl

**Institution**: Leiden University, Netherlands

## 📄 License

This project is developed for academic purposes as part of a robotics course at Leiden University.

## 🙏 Acknowledgments

- Google Magenta team for the Groove MIDI dataset
- TensorFlow team for the deep learning framework
- Arduino and ESP32 communities for hardware support

---

For technical questions or issues, please contact the team members listed above.