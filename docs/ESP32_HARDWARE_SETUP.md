# 🔧 EarthPulse AI - ESP32-S3 Hardware Setup Guide

## 📋 Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Circuit Assembly](#circuit-assembly)
3. [Firmware Installation](#firmware-installation)
4. [Testing Your Device](#testing-your-device)
5. [Running with Real Data](#running-with-real-data)
6. [Troubleshooting](#troubleshooting)

---

## 🛠️ Hardware Requirements

### Required Components
- ✅ **ESP32-S3 Dev Board** (any variant with USB-C)
- ✅ **ADS1115 16-bit ADC Module** (I2C interface)
- ✅ **SM-24 Geophone** or equivalent seismic sensor
- ✅ **Resistors**: 2x 1kΩ (for signal conditioning)
- ✅ **Breadboard** and jumper wires
- ✅ **USB Type-C Cable** (data capable, not charge-only)

### Optional Components
- 🔋 Battery pack (for field deployment)
- 📦 Weatherproof enclosure
- 🔌 Power bank with USB-C output

---

## 🔌 Circuit Assembly

### Pin Connections

#### ESP32-S3 ↔ ADS1115 (I2C ADC)
```
ESP32-S3          ADS1115
--------          -------
GPIO 8    ────→   SDA
GPIO 9    ────→   SCL
3.3V      ────→   VDD
GND       ────→   GND
```

#### Geophone ↔ ADS1115 (with Signal Conditioning)
```
Geophone          Resistor Network          ADS1115
--------          ----------------          -------
Signal Out ───┬──→ 1kΩ resistor ───┬──→    A0
              │                     │
              └──→ 1kΩ resistor ───┴──→    GND
GND       ─────────────────────────────→    GND
```

### Circuit Diagram
Your circuit matches this configuration (just replace Raspberry Pi with ESP32-S3):

```
                    ┌─────────────┐
                    │   SM-24     │
                    │  Geophone   │
                    └──────┬──────┘
                           │ Signal Out
                           │
                    ┌──────┴──────┐
                    │  1kΩ   1kΩ  │  Signal Conditioning
                    │   ╲     ╱   │  (Voltage Divider)
                    │    ╲   ╱    │
                    └─────┬───────┘
                          │
                    ┌─────┴──────────┐
                    │   ADS1115      │
                    │   16-bit ADC   │
                    │   I2C: 0x48    │
                    └────────────────┘
                          │ I2C
                    ┌─────┴──────────┐
                    │   ESP32-S3     │
                    │  GPIO 8 (SDA)  │
                    │  GPIO 9 (SCL)  │
                    └────────────────┘
                          │ USB-C
                    ┌─────┴──────────┐
                    │   Computer     │
                    │  (Python App)  │
                    └────────────────┘
```

### Important Notes
⚠️ **CRITICAL CONNECTIONS:**
1. **ADS1115 ADDR Pin**: Connect to **GND** (sets I2C address to 0x48)
2. **Power**: Use **3.3V only** - do NOT use 5V on ADS1115 VDD!
3. **I2C Pullups**: ADS1115 module has built-in pullup resistors
4. **Geophone Polarity**: Red = Signal, Black = GND

---

## 💻 Firmware Installation

### Step 1: Install Arduino IDE
1. Download from: https://www.arduino.cc/en/software
2. Install Arduino IDE 2.x or later

### Step 2: Add ESP32-S3 Board Support
1. Open Arduino IDE
2. Go to **File → Preferences**
3. Add to "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to **Tools → Board → Boards Manager**
5. Search for "**esp32**" by Espressif Systems
6. Install **esp32** (version 2.0.14 or later)

### Step 3: Install Required Libraries
Go to **Tools → Manage Libraries** and install:
- **Adafruit ADS1X15** (by Adafruit) - for ADS1115 ADC
- **Wire** (built-in) - for I2C communication

### Step 4: Configure Arduino IDE
1. **Select Board**: 
   - **Tools → Board → ESP32 Arduino → ESP32S3 Dev Module**

2. **Configure Board Settings**:
   - USB CDC On Boot: **Enabled**
   - USB Mode: **Hardware CDC and JTAG**
   - Upload Mode: **UART0 / Hardware CDC**
   - Upload Speed: **921600**
   - Port: Select your ESP32 COM port (e.g., COM3)

### Step 5: Upload Firmware
1. Open firmware file:
   ```
   D:\Sliit Projects\Reserach\EarthPulse-AI\hardware_firmware\esp32_s3\esp32_s3_seismic_sensor\esp32_s3_seismic_sensor.ino
   ```

2. Connect ESP32-S3 to computer via USB-C cable

3. Click **Upload** button (→) in Arduino IDE

4. Wait for upload to complete (should see "Done uploading")

5. Open **Serial Monitor** (Tools → Serial Monitor)
   - Set baud rate to **115200**

6. You should see:
   ```
   ====================================
   EarthPulse AI - ESP32-S3 Firmware
   ====================================
   
   I2C initialized (SDA: GPIO8, SCL: GPIO9)
   Initializing ADS1115... SUCCESS!
   Gain set to: +/- 2.048V
   Reading from channel: A0
   
   Sampling rate: 1000 Hz
   Buffer size: 1000 samples
   
   System ready!
   ====================================
   ```

---

## 🧪 Testing Your Device

### Test 1: Check Serial Connection
```bash
# In Arduino Serial Monitor, type:
STATUS
```
Should show:
```
STATUS:
  ADS1115: Connected
  Sampling Rate: 1000 Hz
  Buffer Size: 1000
  Channel: A0
  Uptime: 25 seconds
```

### Test 2: Verify Data Transmission
You should see data packets like:
```
START
0.0012,0.0015,0.0013,0.0018,...
END
Min: 0.0010 V, Max: 0.0025 V, Avg: 0.0015 V
```

### Test 3: Physical Vibration Test
1. Gently tap the surface where geophone is mounted
2. You should see voltage spikes in Serial Monitor
3. Larger taps = larger voltage readings

### Test 4: Python Serial Reader
```bash
cd "D:\Sliit Projects\Reserach\EarthPulse-AI"
python hardware_interface/esp32_serial_reader.py
```

This will:
- List available COM ports
- Connect to your ESP32
- Display real-time signal statistics

---

## 🚀 Running with Real Data

### Method 1: Command Line Interface
```bash
# Start the hardware interface
cd "D:\Sliit Projects\Reserach\EarthPulse-AI"
python hardware_interface/realtime_detection_hardware.py
```

This script:
- ✅ Connects to ESP32 automatically
- ✅ Reads real geophone data
- ✅ Runs detection system
- ✅ Shows results in terminal

### Method 2: Dashboard with Hardware
```bash
# Run dashboard with hardware mode
python dashboard/realtime_dashboard.py --hardware --port COM3
```

Replace `COM3` with your actual port.

### Method 3: Integration Code
To use hardware data in your own scripts:

```python
from hardware_interface.esp32_serial_reader import ESP32SerialReader, ESP32Config
from edge_firmware_simulated.detection_system import ElephantDetectionSystem

# Initialize hardware reader
reader = ESP32SerialReader(ESP32Config(port="COM3"))
reader.connect()

# Initialize detection system
detector = ElephantDetectionSystem(model_path="./models/lstm_model.h5")

# Start reading
reader.start_reading()

try:
    while True:
        # Get signal from hardware
        signal = reader.get_signal_blocking(timeout=5.0)
        
        if signal is not None:
            # Run detection
            result = detector.process_signal(signal, time.time())
            
            if result['detected']:
                print(f"🐘 ELEPHANT DETECTED!")
                print(f"   Confidence: {result['confidence']:.2%}")
                print(f"   Direction: {result['direction']['status']}")
                print(f"   Distance: {result['direction']['distance']:.1f}m")
                print(f"   Behavior: {result['behavior']['type']}")

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    reader.disconnect()
```

---

## 🔧 Troubleshooting

### Problem: "ADS1115 not found"
**Solution:**
1. Check **ADDR pin** on ADS1115 is connected to **GND**
2. Verify I2C wiring:
   - SDA → GPIO 8
   - SCL → GPIO 9
3. Check power: VDD → 3.3V (NOT 5V!)
4. Try I2C scanner in Serial Monitor:
   ```
   SCAN
   ```
   Should show: `Device found at 0x48`

### Problem: "No serial port found"
**Solution:**
1. Install USB drivers:
   - **CH340**: https://sparks.gogo.co.nz/ch340.html
   - **CP210x**: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
2. Check Device Manager (Windows) for "Ports (COM & LPT)"
3. Try different USB cable (must be data cable, not charge-only)
4. Try different USB port on computer

### Problem: "Upload Failed"
**Solution:**
1. Hold **BOOT** button on ESP32 while clicking Upload
2. Check board selection: ESP32S3 Dev Module
3. Enable "USB CDC On Boot" in Tools menu
4. Try lower upload speed (460800 or 115200)

### Problem: Noisy or Unstable Readings
**Solution:**
1. Add 10µF capacitor between VDD and GND on ADS1115
2. Use shielded cable for geophone connection
3. Keep geophone away from electrical noise sources
4. Ensure solid ground connection
5. Try different ADS1115 gain setting (lower gain = less noise but less sensitivity)

### Problem: Serial data corruption
**Solution:**
1. Check baud rate is 115200 in both firmware and Python
2. Reduce USB cable length
3. Close other programs using serial port
4. Try different USB port (USB 3.0 sometimes causes issues, try USB 2.0)

### Problem: Low sensitivity / No detection
**Solution:**
1. Increase ADS1115 gain in firmware:
   ```cpp
   #define ADS_GAIN GAIN_FOUR  // or GAIN_EIGHT
   ```
2. Check geophone mounting:
   - Must be in firm contact with ground
   - Bury 10-15cm underground for best results
   - Avoid loose soil or grass
3. Calibrate geophone sensitivity
4. Check signal conditioning resistors are 1kΩ each

---

## 📊 Data Flow Architecture

```
┌─────────────┐
│  Geophone   │ → Mechanical vibration (ground motion)
└──────┬──────┘
       │ Converts to voltage (28 V/m/s)
       ↓
┌─────────────┐
│  1kΩ + 1kΩ  │ → Signal conditioning (voltage divider)
│  Resistors  │
└──────┬──────┘
       │ Conditioned voltage (±2V range)
       ↓
┌─────────────┐
│  ADS1115    │ → 16-bit digitization (860 samples/sec max)
│  ADC        │
└──────┬──────┘
       │ I2C protocol (address 0x48)
       ↓
┌─────────────┐
│  ESP32-S3   │ → Samples at 1000 Hz
│  (Firmware) │ → Buffers 1000 samples
└──────┬──────┘
       │ USB Serial (115200 baud)
       ↓
┌─────────────┐
│  Python     │ → Parses serial data
│  Reader     │ → Converts to numpy array
└──────┬──────┘
       │ Signal array (1000 samples @ 1000Hz)
       ↓
┌─────────────┐
│  Detection  │ → DSP processing
│  System     │ → LSTM classification
└──────┬──────┘
       │ Detection result
       ↓
┌─────────────┐
│  Dashboard  │ → Real-time visualization
│  / Alert    │ → Direction & behavior analysis
└─────────────┘
```

---

## 🎯 Field Deployment Tips

### 1. Geophone Placement
- **Location**: 50-100m from expected elephant path
- **Depth**: Bury 10-15cm underground
- **Surface**: Firm soil (avoid sand, loose gravel, grass)
- **Orientation**: Vertical axis perpendicular to ground
- **Coupling**: Pack soil firmly around sensor

### 2. Power Management
- Use 5V 2A power bank for portable operation
- ESP32-S3 draws ~80mA when active
- Battery life: ~20 hours with 5000mAh power bank
- Consider solar panel for long-term deployment

### 3. Weather Protection
- Use IP65 or better waterproof enclosure
- Silica gel packets to prevent condensation
- Cable glands for wire entry points
- Mount electronics above ground level

### 4. Data Collection
- Test system in controlled environment first
- Monitor serial output for 24 hours before field deployment
- Keep laptop/computer within 5m of ESP32 (USB cable limit)
- For longer distances, use USB extension cable or Wi-Fi module

---

## 📞 Support

If you encounter issues:
1. Check Arduino Serial Monitor for error messages
2. Run Python test script: `python hardware_interface/esp32_serial_reader.py`
3. Verify all connections match the circuit diagram
4. Review troubleshooting section above

**Hardware Checklist:**
- ✅ ESP32-S3 powered and recognized by computer
- ✅ ADS1115 at I2C address 0x48
- ✅ Geophone connected to A0 with 1kΩ resistors
- ✅ Firmware uploaded and running
- ✅ Serial data flowing (START...END packets visible)
- ✅ Python reader can parse data

---

## 🎓 Next Steps

Once your hardware is working:
1. ✅ Calibrate geophone sensitivity with known vibrations
2. ✅ Collect baseline data (no elephants present)
3. ✅ Test with simulated elephant footfalls (heavy impacts)
4. ✅ Adjust detection thresholds based on field conditions
5. ✅ Deploy in actual monitoring location
6. ✅ Monitor and analyze detection accuracy

**Good luck with your EarthPulse AI deployment! 🐘🌍**
