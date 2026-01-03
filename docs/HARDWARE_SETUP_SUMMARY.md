# 🎯 Hardware Setup - Complete Summary

## Your Hardware Configuration

Based on your circuit diagram (with ESP32-S3 instead of Raspberry Pi 4):

```
Components:
✅ ESP32-S3 Development Board (USB Type-C)
✅ ADS1115 16-bit ADC Module (I2C interface)  
✅ SM-24 Geophone (or equivalent seismic sensor)
✅ 2x 1kΩ Resistors (signal conditioning)
✅ Breadboard + Jumper wires
```

## Complete Wiring

### Connections Table
| Component 1  | Pin/Terminal | →  | Component 2  | Pin/Terminal |
|-------------|--------------|-----|--------------|--------------|
| **ESP32-S3**    | GPIO 8       | →  | **ADS1115**      | SDA          |
| **ESP32-S3**    | GPIO 9       | →  | **ADS1115**      | SCL          |
| **ESP32-S3**    | 3.3V         | →  | **ADS1115**      | VDD          |
| **ESP32-S3**    | GND          | →  | **ADS1115**      | GND          |
| **ADS1115**     | ADDR         | →  | **ADS1115**      | GND          |
| **Geophone**    | Signal (+)   | →  | **1kΩ Resistor** | Terminal 1   |
| **1kΩ Resistor**| Terminal 2   | →  | **ADS1115**      | A0           |
| **Geophone**    | Signal (+)   | →  | **1kΩ Resistor** | Terminal 1   |
| **1kΩ Resistor**| Terminal 2   | →  | **GND**          | —            |
| **Geophone**    | GND (-)      | →  | **GND**          | —            |
| **ESP32-S3**    | USB-C        | →  | **Computer**     | USB Port     |

### Critical Notes
⚠️ **MUST DO:**
1. **ADS1115 ADDR to GND**: This sets I2C address to 0x48 (default in firmware)
2. **Use 3.3V only**: Never connect 5V to ADS1115 VDD
3. **Data cable**: USB-C must support data transfer (not charge-only)

## Software Setup Steps

### 1. Arduino IDE Setup
```bash
# Install Arduino IDE 2.3 or later
# Add ESP32 board support URL:
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json

# Install boards:
Tools → Board Manager → "esp32" by Espressif → Install

# Install libraries:
Tools → Manage Libraries → "Adafruit ADS1X15" → Install
```

### 2. Upload Firmware
```bash
# Location:
D:\Sliit Projects\Reserach\EarthPulse-AI\hardware_firmware\esp32_s3\esp32_s3_seismic_sensor\esp32_s3_seismic_sensor.ino

# Board settings:
Tools → Board → ESP32S3 Dev Module
Tools → USB CDC On Boot → Enabled
Tools → Port → (Select your COM port)
Tools → Upload Speed → 921600

# Click Upload button (→)
```

### 3. Verify Firmware
```bash
# Open Serial Monitor (Ctrl+Shift+M)
# Baud rate: 115200
# Should see:
====================================
EarthPulse AI - ESP32-S3 Firmware
====================================
I2C initialized (SDA: GPIO8, SCL: GPIO9)
Initializing ADS1115... SUCCESS!
System ready!
START
0.0012,0.0015,0.0013,...
END
```

### 4. Python Setup
```bash
# Install dependencies
cd "D:\Sliit Projects\Reserach\EarthPulse-AI"
pip install pyserial colorama

# Test connection
python test_hardware_connection.py

# Expected output:
✅ PASSED: Found serial ports
✅ PASSED: Connected to ESP32
✅ PASSED: Received 10 packets
🎉 SUCCESS! Your hardware is working correctly!
```

## Running Detection with Real Hardware

### Method 1: Command Line (Recommended)
```bash
# Single detection
python hardware_interface/realtime_detection_hardware.py --port COM3

# Continuous monitoring
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous

# With logging
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous --log detections.txt
```

Replace `COM3` with your actual port (check Device Manager on Windows, or use `/dev/ttyUSB0` on Linux).

### Method 2: Python Script Integration
```python
from hardware_interface.esp32_serial_reader import ESP32SerialReader
from edge_firmware_simulated.detection_system import ElephantDetectionSystem

# Connect to hardware
reader = ESP32SerialReader()
reader.connect("COM3")  # Your port
reader.start_reading()

# Initialize detector
detector = ElephantDetectionSystem(model_path="./models/lstm_model.h5")

# Process signals
while True:
    signal = reader.get_signal_blocking(timeout=5.0)
    if signal is not None:
        result = detector.process_signal(signal, time.time())
        if result['detected']:
            print(f"🐘 Elephant detected! Confidence: {result['confidence']:.2%}")
            print(f"   Direction: {result['direction']['status']}")
            print(f"   Behavior: {result['behavior']['type']}")
```

### Method 3: Dashboard Integration (Future)
```bash
# Coming soon - dashboard with hardware mode
python dashboard/realtime_dashboard.py --hardware --port COM3
```

## Expected Output Example

```
──────────────────────────────────────────────────────────────
📊 Signal Statistics:
   Samples:    1000
   RMS:        0.0234 V
   Peak-Peak:  0.1245 V
   Mean:       0.0015 V

🐘 ELEPHANT DETECTED!
   Confidence: 89.5%

📍 Movement Direction:
   Status:     ⬆️ Approaching
   Direction:  North-East
   Distance:   42.7 m
   Velocity:   0.95 m/s
   Confidence: 82.3%

🐘 Behavior Analysis:
   Activity:   🚶 Walking
   Gait Speed: 1.42 m/s
   Activity:   Moderate
   Weight Est: 4100 kg
   Confidence: 85.7%

⚠️  ALERT: Elephant in vicinity - Take precautions!
──────────────────────────────────────────────────────────────
```

## Troubleshooting Guide

### Issue: ADS1115 Not Detected
```bash
# Check in Serial Monitor:
STATUS

# Or type:
SCAN

# Should show:
Device found at 0x48

# If not found:
1. Verify ADDR pin connected to GND
2. Check SDA/SCL wiring (GPIO 8/9)
3. Confirm 3.3V power to VDD
4. Try different I2C address if ADDR is floating:
   ADDR = GND  → 0x48 (default)
   ADDR = VDD  → 0x49
   ADDR = SDA  → 0x4A
   ADDR = SCL  → 0x4B
```

### Issue: No Serial Port Found
```bash
# Windows:
1. Open Device Manager
2. Look for "Ports (COM & LPT)"
3. Check for "USB-SERIAL CH340" or similar
4. If not found, install driver:
   https://sparks.gogo.co.nz/ch340.html

# Port not working:
1. Try different USB cable (must be data cable)
2. Try different USB port on computer
3. Restart ESP32 (press RST button)
4. Close Arduino IDE Serial Monitor
```

### Issue: Noisy or Erratic Readings
```cpp
// In firmware, adjust gain for your signal level:
// Edit line ~46:
#define ADS_GAIN GAIN_FOUR    // Increase sensitivity

// Gain options (smaller range = higher sensitivity):
GAIN_TWOTHIRDS  // ±6.144V (less sensitive, more noise immune)
GAIN_ONE        // ±4.096V
GAIN_TWO        // ±2.048V (default)
GAIN_FOUR       // ±1.024V (recommended for weak signals)
GAIN_EIGHT      // ±0.512V (very sensitive)
GAIN_SIXTEEN    // ±0.256V (maximum sensitivity, more noise)
```

### Issue: No Detections
```bash
# Verify geophone is working:
python test_hardware_connection.py quality

# Should show signal increase when you tap ground
# If no change:
1. Check geophone connections
2. Verify 1kΩ resistors (not 10kΩ or 100Ω!)
3. Test geophone with multimeter:
   - Set to AC voltage mode
   - Connect to geophone terminals
   - Tap geophone - should see voltage spike
4. Ensure geophone is firmly coupled to ground
```

## Calibration Tips

### Geophone Sensitivity Calibration
```bash
# Collect baseline data:
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous --log baseline.txt

# While logging:
1. No activity (baseline noise): 5 minutes
2. Known weight drops (e.g., 10kg from 1m): 10 samples
3. Person walking: 20 steps at various distances
4. Vehicle passing: Multiple vehicles

# Analyze log to determine:
• Detection threshold
• Distance calibration factor
• Weight estimation parameters
```

### Optimal Gain Selection
```
Signal Level          Recommended Gain
────────────────      ────────────────
Very weak (<10mV)     GAIN_EIGHT or GAIN_SIXTEEN
Weak (10-50mV)        GAIN_FOUR
Medium (50-200mV)     GAIN_TWO (default)
Strong (>200mV)       GAIN_ONE
Very strong (>1V)     GAIN_TWOTHIRDS
```

## Performance Expectations

### Detection Range
- **Close range (10-30m)**: High confidence (>85%)
- **Medium range (30-60m)**: Good confidence (70-85%)
- **Far range (60-100m)**: Lower confidence (50-70%)
- **Beyond 100m**: Unreliable, background noise dominant

### Accuracy Factors
✅ **Good conditions:**
- Firm, dry soil
- Geophone buried 10-15cm
- Quiet environment (night time)
- Moderate temperature (15-30°C)

⚠️ **Challenging conditions:**
- Loose or sandy soil
- Surface-mounted geophone
- Windy/rainy weather
- Near roads or machinery

## Next Steps

1. ✅ **Verify hardware works**: Run `test_hardware_connection.py`
2. ✅ **Test detection**: Run single detection with real geophone
3. ✅ **Calibrate system**: Collect data in your deployment environment
4. ✅ **Deploy**: Mount geophone in field location
5. ✅ **Monitor**: Run continuous detection with logging

## Quick Reference Commands

```bash
# Test connection
python test_hardware_connection.py

# Single detection
python hardware_interface/realtime_detection_hardware.py --port COM3

# Continuous monitoring
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous

# With logging
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous --log detections.txt

# Test serial reader
python hardware_interface/esp32_serial_reader.py

# Arduino Serial Monitor commands:
STATUS  # Show system status
RESET   # Reset statistics
SCAN    # Scan I2C devices
```

## Documentation Files

- 📘 **Complete Guide**: `docs/ESP32_HARDWARE_SETUP.md` (detailed 3000+ words)
- 🚀 **Quick Start**: `docs/QUICK_START_HARDWARE.md` (15 minute setup)
- 📝 **This Summary**: `docs/HARDWARE_SETUP_SUMMARY.md` (you are here)
- 🔧 **Test Script**: `test_hardware_connection.py`
- 💻 **Detection Script**: `hardware_interface/realtime_detection_hardware.py`
- 📡 **Serial Reader**: `hardware_interface/esp32_serial_reader.py`
- 🔌 **Firmware**: `hardware_firmware/esp32_s3/esp32_s3_seismic_sensor/`

---

**Your hardware is ready! Connect it and start detecting! 🐘🌍**
