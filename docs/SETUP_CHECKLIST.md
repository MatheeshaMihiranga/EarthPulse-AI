# 🎯 COMPLETE SETUP CHECKLIST - Hardware to Detection

## ✅ Pre-Flight Checklist

### Hardware Inventory
- [ ] ESP32-S3 Development Board (with USB-C port)
- [ ] ADS1115 16-bit ADC Module
- [ ] SM-24 Geophone (or equivalent seismic sensor)
- [ ] 2x 1kΩ resistors (brown-black-red bands)
- [ ] Breadboard
- [ ] Male-to-male jumper wires (at least 10)
- [ ] USB Type-C cable (data-capable, not charge-only!)
- [ ] Computer with Windows/Linux/Mac

### Software Prerequisites
- [ ] Arduino IDE 2.0+ installed
- [ ] Python 3.9+ installed
- [ ] USB drivers installed (CH340/CP210x if needed)

---

## 🔌 STEP 1: Hardware Assembly (10 minutes)

### 1.1 Prepare Components
```
✓ Layout components on breadboard
✓ Identify ESP32-S3 GPIO pins (use pinout diagram)
✓ Verify ADS1115 module (should have 4 screw terminals)
✓ Check resistor values with multimeter (should read 980Ω - 1020Ω)
```

### 1.2 ESP32-S3 to ADS1115 Connections
```
Connection #1: ESP32 GPIO 8  →  ADS1115 SDA    [I2C Data]
Connection #2: ESP32 GPIO 9  →  ADS1115 SCL    [I2C Clock]
Connection #3: ESP32 3.3V    →  ADS1115 VDD    [Power - 3.3V ONLY!]
Connection #4: ESP32 GND     →  ADS1115 GND    [Ground]
Connection #5: ADS1115 ADDR  →  ADS1115 GND    [Set I2C address to 0x48]
```

**⚠️ CRITICAL: Connection #5 (ADDR→GND) is REQUIRED!**

### 1.3 Geophone to ADS1115 Signal Conditioning
```
Geophone Red Wire (+):
  ├──→ 1kΩ Resistor →  ADS1115 A0
  └──→ 1kΩ Resistor →  GND

Geophone Black Wire (-):
  └──→ GND
```

### 1.4 Double-Check All Connections
- [ ] SDA and SCL are not swapped
- [ ] VDD is connected to **3.3V** (NOT 5V!)
- [ ] ADDR pin is connected to GND
- [ ] Geophone has both resistors (voltage divider)
- [ ] All GND connections are common

### 1.5 Power On Test
```
1. Connect USB-C cable to ESP32
2. Connect USB to computer
3. ESP32 LED should light up (red/blue/green depending on model)
4. No smoke, no burning smell = good! ✓
```

---

## 💻 STEP 2: Firmware Upload (15 minutes)

### 2.1 Install Arduino IDE
```bash
# Download from: https://www.arduino.cc/en/software
# Version 2.3 or later recommended
```

### 2.2 Add ESP32 Board Support
```
1. Open Arduino IDE
2. File → Preferences
3. In "Additional Board Manager URLs", add:
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
4. Tools → Board → Boards Manager
5. Search: "esp32"
6. Install: "esp32" by Espressif Systems (version 2.0.14+)
7. Wait for installation (may take 5-10 minutes)
```

### 2.3 Install Required Libraries
```
1. Tools → Manage Libraries (Ctrl+Shift+I)
2. Search: "Adafruit ADS1X15"
3. Install: "Adafruit ADS1X15" by Adafruit
4. Click "Install All" if prompted for dependencies
5. Close Library Manager when done
```

### 2.4 Open Firmware
```
1. File → Open
2. Navigate to:
   D:\Sliit Projects\Reserach\EarthPulse-AI\hardware_firmware\esp32_s3\esp32_s3_seismic_sensor\esp32_s3_seismic_sensor.ino
3. Firmware should open in new window
```

### 2.5 Configure Board
```
Tools → Board → ESP32 Arduino → ESP32S3 Dev Module

Configure these settings:
✓ USB CDC On Boot:         Enabled
✓ USB Mode:                Hardware CDC and JTAG
✓ Upload Mode:             UART0 / Hardware CDC
✓ Upload Speed:            921600
✓ CPU Frequency:           240MHz (default)
✓ Flash Mode:              QIO (default)
✓ Flash Size:              4MB (or your board's size)
✓ Partition Scheme:        Default
✓ PSRAM:                   Disabled (or default)

✓ Port:                    COMx (Windows) or /dev/ttyUSBx (Linux)
                           Select the port that appeared when you plugged in ESP32
```

### 2.6 Upload Firmware
```
1. Click Upload button (→) or press Ctrl+U
2. Wait for "Connecting..." message
3. If stuck, press and hold BOOT button on ESP32, then release
4. Wait for upload (30-60 seconds)
5. Should see: "Hard resetting via RTS pin..."
6. Success message: "Done uploading."
```

### 2.7 Verify Upload
```
1. Tools → Serial Monitor (Ctrl+Shift+M)
2. Set baud rate: 115200 (bottom right)
3. Press RST button on ESP32

Expected output:
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

START
0.0012,0.0015,0.0013,0.0018,...
END
Min: 0.0010 V, Max: 0.0025 V, Avg: 0.0015 V
```

**If you see "ADS1115... FAILED!" → Go to Troubleshooting section**

---

## 🐍 STEP 3: Python Setup (5 minutes)

### 3.1 Install Python Packages
```bash
cd "D:\Sliit Projects\Reserach\EarthPulse-AI"

# Install hardware communication packages
pip install pyserial colorama

# Verify all requirements
pip install -r requirements.txt
```

### 3.2 Test Serial Connection
```bash
# Run connection test
python test_hardware_connection.py
```

**Expected Output:**
```
✅ PASSED: Found serial ports
✅ PASSED: Connected to ESP32
✅ PASSED: Received 10 packets
🎉 SUCCESS! Your hardware is working correctly!
```

**If test fails → Go to Troubleshooting section**

### 3.3 Signal Quality Test (Optional but Recommended)
```bash
python test_hardware_connection.py quality
```

Follow on-screen instructions:
1. Baseline measurement (don't touch anything)
2. TAP the ground near geophone
3. Check signal-to-noise ratio

**Good result:** SNR > 2.0x

---

## 🚀 STEP 4: Run Detection with Real Data

### 4.1 Single Detection Test
```bash
# Replace COM3 with your actual port
python hardware_interface/realtime_detection_hardware.py --port COM3
```

### 4.2 Continuous Monitoring
```bash
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous
```

### 4.3 With Data Logging
```bash
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous --log detections.txt
```

### 4.4 Understanding Output
```
📊 Signal Statistics:
   Samples:    1000          ← Buffer size (1 second @ 1000Hz)
   RMS:        0.0156 V      ← Root-mean-square voltage
   Peak-Peak:  0.0892 V      ← Signal range
   Mean:       0.0012 V      ← DC offset

🐘 ELEPHANT DETECTED!        ← Detection result
   Confidence: 87.3%         ← Model confidence (>70% is high)

📍 Movement Direction:
   Status:     ⬆️ Approaching ← Elephant moving toward sensor
   Direction:  NE            ← Compass direction
   Distance:   45.3 m        ← Estimated distance (10-150m range)
   Velocity:   0.82 m/s      ← Movement speed
   Confidence: 78.5%         ← Direction confidence

🐘 Behavior Analysis:
   Activity:   🚶 Walking    ← Detected behavior
   Gait Speed: 1.35 m/s      ← Step frequency analysis
   Activity:   Moderate      ← Calm/Moderate/Agitated
   Weight Est: 3850 kg       ← Based on signal amplitude
   Confidence: 81.2%         ← Behavior confidence
```

---

## 🧪 STEP 5: Field Deployment

### 5.1 Geophone Installation
```
1. Choose location:
   ✓ 50-100m from expected elephant path
   ✓ Firm soil (avoid sand, loose gravel, grass)
   ✓ Away from roads, buildings, machinery

2. Dig hole:
   ✓ Depth: 10-15 cm
   ✓ Width: Slightly larger than geophone base

3. Place geophone:
   ✓ Vertical orientation (check arrow on geophone)
   ✓ Firm contact with bottom of hole
   ✓ Level (use bubble level if available)

4. Backfill:
   ✓ Pack soil firmly around geophone
   ✓ No air gaps
   ✓ Cable exits horizontally to avoid pulling sensor

5. Protect cable:
   ✓ Bury cable 5cm deep for first 2 meters
   ✓ Mark cable path
   ✓ Strain relief at geophone connection
```

### 5.2 ESP32 Placement
```
✓ Within 3 meters of geophone (cable length limit)
✓ Elevated off ground (at least 30cm)
✓ Weather protection (waterproof box/bag)
✓ Ventilation (ESP32 generates heat)
✓ Access to USB port for computer connection
```

### 5.3 Power Options
```
Option 1: USB Cable to Laptop
  ✓ Simple, reliable
  ✗ Limited range (~5m with USB extension)
  ✗ Requires laptop in field

Option 2: Power Bank
  ✓ Portable
  ✓ ~20 hours runtime (5000mAh)
  ✗ Need laptop nearby for data collection

Option 3: Solar + Battery (Advanced)
  ✓ Autonomous operation
  ✓ Weeks of runtime
  ✗ Requires additional hardware
```

### 5.4 Initial Field Test
```bash
# Run 1-hour test
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous --log field_test.txt

# While running:
1. Monitor for regular data packets (every 1 second)
2. Check signal quality (RMS should be stable)
3. Test with known vibrations (walk near sensor)
4. Verify detections are logged to file

# After 1 hour:
1. Review log file
2. Check detection accuracy
3. Adjust thresholds if needed
```

---

## 🔧 TROUBLESHOOTING

### Problem: "ADS1115 not found"
```
Symptom: Serial Monitor shows "Initializing ADS1115... FAILED!"

Solutions:
1. Check ADDR pin:
   arduino: Type "SCAN" in Serial Monitor
   Should show: "Device found at 0x48"
   If no device:
   ✓ Verify ADDR connected to GND
   ✓ Check SDA/SCL not swapped
   ✓ Verify 3.3V on VDD pin (measure with multimeter)

2. Check I2C wiring:
   ✓ SDA → GPIO 8 (not GPIO 9!)
   ✓ SCL → GPIO 9 (not GPIO 8!)
   ✓ Try different jumper wires
   ✓ Check breadboard connections

3. Test ADS1115 module:
   ✓ Swap with another ADS1115 if available
   ✓ Check for physical damage
   ✓ Verify I2C address with I2C scanner
```

### Problem: "No serial port found"
```
Symptom: Python can't find COM port

Solutions:
1. Install USB drivers:
   Windows: Device Manager → Other Devices → Update Driver
   Driver links:
   • CH340: https://sparks.gogo.co.nz/ch340.html
   • CP210x: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers

2. Check cable:
   ✓ Try different USB-C cable (must support data!)
   ✓ Try different USB port on computer
   ✓ Avoid USB hubs, connect directly to computer

3. Verify ESP32 powered:
   ✓ LED should be on
   ✓ Try pressing RST button
   ✓ Check USB-C cable is fully inserted
```

### Problem: Noisy/Erratic Readings
```
Symptom: RMS constantly changing, random spikes, no stable baseline

Solutions:
1. Electrical noise:
   ✓ Move away from motors, transformers, fluorescent lights
   ✓ Use shielded cable for geophone
   ✓ Add 10µF capacitor between VDD and GND on ADS1115

2. Ground coupling:
   ✓ Ensure geophone is firmly in ground
   ✓ Check for loose connections
   ✓ Verify resistors are actually 1kΩ (not 10kΩ or 100Ω)

3. Adjust ADC gain:
   In firmware (line ~46), try:
   #define ADS_GAIN GAIN_ONE  // Less sensitive to noise
```

### Problem: No Detections
```
Symptom: System runs but never detects elephants

Solutions:
1. Verify signal levels:
   python test_hardware_connection.py quality
   ✓ Baseline RMS should be <0.01V
   ✓ Tap test should show >2x increase
   ✓ If no change → check geophone connection

2. Test with known vibration:
   ✓ Heavy object dropped nearby (5-10kg from 1m)
   ✓ Person jumping near geophone
   ✓ Should see RMS spike >0.05V

3. Calibrate detection threshold:
   ✓ Collect baseline data in field
   ✓ Determine typical background RMS
   ✓ Elephant signals should be 5-10x baseline
```

---

## 📊 Expected Performance

### Detection Accuracy
| Distance | Confidence | Reliability |
|----------|-----------|-------------|
| 10-30m   | >85%      | Excellent   |
| 30-60m   | 70-85%    | Good        |
| 60-100m  | 50-70%    | Fair        |
| >100m    | <50%      | Poor        |

### Environmental Effects
```
Best Conditions:
✓ Firm, dry soil
✓ Night time (less background noise)
✓ Calm weather
✓ Geophone buried 10-15cm

Challenging Conditions:
✗ Loose/sandy soil
✗ Windy/rainy weather
✗ Near roads/construction
✗ Surface-mounted sensor
```

---

## 📞 Quick Reference

### Arduino Serial Monitor Commands
```
STATUS  → Show system status
RESET   → Reset statistics
SCAN    → Scan I2C bus for devices
```

### Python Scripts
```bash
# Test connection
python test_hardware_connection.py

# Signal quality test
python test_hardware_connection.py quality

# Single detection
python hardware_interface/realtime_detection_hardware.py --port COM3

# Continuous monitoring
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous

# With logging
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous --log detections.txt
```

### Port Finding
```bash
# Windows
Device Manager → Ports (COM & LPT) → Look for "USB-SERIAL" or "CH340"

# Linux
ls /dev/ttyUSB*

# Mac
ls /dev/cu.*
```

---

## 📚 Documentation Index

1. **This Checklist**: `docs/SETUP_CHECKLIST.md` ← You are here
2. **Complete Guide**: `docs/ESP32_HARDWARE_SETUP.md` (detailed)
3. **Quick Start**: `docs/QUICK_START_HARDWARE.md` (15 min)
4. **Summary**: `docs/HARDWARE_SETUP_SUMMARY.md` (reference)
5. **Firmware**: `hardware_firmware/esp32_s3/esp32_s3_seismic_sensor/`
6. **Python Interface**: `hardware_interface/`

---

## ✅ Final Checklist

Before field deployment, verify:
- [ ] Hardware connections match circuit diagram
- [ ] Firmware uploads without errors
- [ ] Serial Monitor shows "System ready!"
- [ ] Python test script passes all checks
- [ ] Signal quality test shows SNR > 2.0x
- [ ] Detection script runs without errors
- [ ] Geophone properly installed in ground
- [ ] ESP32 weatherproofed
- [ ] Power source configured (USB/battery/solar)
- [ ] Data logging configured
- [ ] Backup power available
- [ ] Field site documented (GPS coordinates, photos)

---

**Your EarthPulse AI system is ready for elephant detection! 🐘🌍**
**Good luck with your conservation work!**
