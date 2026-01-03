# 🎯 YOUR HARDWARE SETUP - COMPLETE GUIDE

## 📌 What You Have

Based on your circuit diagram:
- ✅ **ESP32-S3** Development Board (replaces Raspberry Pi 4 in your image)
- ✅ **ADS1115** 16-bit ADC Module (I2C interface)
- ✅ **SM-24 Geophone** (seismic sensor)
- ✅ **2x 1kΩ Resistors** (signal conditioning)
- ✅ **USB Type-C** connection to computer

## 🔌 Your Circuit (Corrected from Image)

```
                SM-24 Geophone
                      ↓
              1kΩ ← Signal → 1kΩ
                ↓             ↓
               A0           GND
                ↓
            ADS1115 (I2C ADC)
            • VDD → 3.3V
            • GND → GND
            • SDA → GPIO 8
            • SCL → GPIO 9
            • ADDR → GND ⚠️ IMPORTANT!
                ↓
            ESP32-S3 Dev Board
            • GPIO 8 = SDA
            • GPIO 9 = SCL  
                ↓
           USB Type-C Cable
                ↓
            Your Computer
```

## 🚀 COMPLETE SETUP IN 3 STEPS

### STEP 1: Upload Firmware to ESP32-S3 (15 min)

#### 1.1 Install Arduino IDE
Download: https://www.arduino.cc/en/software (version 2.3+)

#### 1.2 Add ESP32 Support
```
1. Arduino IDE → File → Preferences
2. Additional Board Manager URLs:
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
3. Tools → Board → Boards Manager → Install "esp32" by Espressif
```

#### 1.3 Install Library
```
Tools → Manage Libraries → Install "Adafruit ADS1X15"
```

#### 1.4 Upload Firmware
```
File → Open:
D:\Sliit Projects\Reserach\EarthPulse-AI\hardware_firmware\esp32_s3\esp32_s3_seismic_sensor\esp32_s3_seismic_sensor.ino

Tools → Board → ESP32S3 Dev Module
Tools → USB CDC On Boot → Enabled
Tools → Port → (Select your COM port)
Click Upload (→)
```

#### 1.5 Verify
```
Tools → Serial Monitor (115200 baud)
Should see:
"ADS1115... SUCCESS!"
"System ready!"
"START...END" (data packets)
```

### STEP 2: Test Hardware Connection (5 min)

```bash
cd "D:\Sliit Projects\Reserach\EarthPulse-AI"
python test_hardware_connection.py
```

**Expected:**
```
✅ PASSED: Found serial ports
✅ PASSED: Connected to ESP32  
✅ PASSED: Received 10 packets
🎉 SUCCESS!
```

### STEP 3: Run Real-Time Detection (2 min)

#### Single Detection:
```bash
python hardware_interface/realtime_detection_hardware.py --port COM3
```

#### Continuous Monitoring:
```bash
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous
```

#### With Logging:
```bash
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous --log detections.txt
```

**Replace `COM3` with your actual port**

## 🎯 EXPECTED OUTPUT

When an elephant is detected:

```
──────────────────────────────────────────────
📊 Signal Statistics:
   Samples:    1000
   RMS:        0.0234 V
   Peak-Peak:  0.1245 V

🐘 ELEPHANT DETECTED!
   Confidence: 89.5%

📍 Movement Direction:
   Status:     ⬆️ Approaching
   Distance:   42.7 m
   Velocity:   0.95 m/s

🐘 Behavior Analysis:
   Activity:   🚶 Walking
   Gait Speed: 1.42 m/s
   Weight Est: 4100 kg

⚠️  ALERT: Elephant in vicinity!
──────────────────────────────────────────────
```

## 🔧 COMMON ISSUES & FIXES

### Issue 1: "ADS1115 not found"
```bash
# In Arduino Serial Monitor, type:
SCAN

# Should show: Device found at 0x48
# If not:
1. Check ADDR pin is connected to GND
2. Verify SDA → GPIO 8, SCL → GPIO 9
3. Confirm VDD is 3.3V (NOT 5V!)
```

### Issue 2: "No serial port found"
```bash
# Windows: Install CH340 driver
https://sparks.gogo.co.nz/ch340.html

# Check Device Manager → Ports (COM & LPT)
# Look for "USB-SERIAL CH340" or similar
```

### Issue 3: Noisy readings
```cpp
// In firmware, change line ~46:
#define ADS_GAIN GAIN_FOUR  // Increase to GAIN_FOUR or GAIN_EIGHT
```

### Issue 4: No detections
```bash
# Test signal quality:
python test_hardware_connection.py quality

# Tap ground near geophone
# Should see RMS increase >2x
```

## 📁 FILE LOCATIONS

### Documentation:
```
docs/
├── SETUP_CHECKLIST.md          ← Complete step-by-step
├── ESP32_HARDWARE_SETUP.md     ← Detailed guide (3000+ words)
├── QUICK_START_HARDWARE.md     ← 15-minute setup
├── HARDWARE_SETUP_SUMMARY.md   ← Reference guide
└── YOUR_HARDWARE_SETUP.md      ← This file
```

### Code:
```
hardware_firmware/esp32_s3/
└── esp32_s3_seismic_sensor/
    └── esp32_s3_seismic_sensor.ino  ← Arduino firmware

hardware_interface/
├── esp32_serial_reader.py           ← Serial communication
└── realtime_detection_hardware.py   ← Detection with real data

test_hardware_connection.py          ← Hardware test script
```

## 🎓 HOW IT WORKS

```
1. Geophone senses ground vibration
   ↓ (converts to voltage)
   
2. 1kΩ resistors condition signal
   ↓ (voltage divider)
   
3. ADS1115 digitizes (16-bit, 1000 Hz)
   ↓ (I2C protocol)
   
4. ESP32-S3 reads and buffers data
   ↓ (1000 samples = 1 second)
   
5. USB serial transmits to computer
   ↓ (115200 baud)
   
6. Python parses and processes
   ↓ (numpy array)
   
7. Detection system analyzes
   ↓ (LSTM model + DSP)
   
8. Results displayed
   ↓ (detection, direction, behavior)
```

## 🌍 FIELD DEPLOYMENT

### Geophone Installation:
```
1. Location: 50-100m from elephant path
2. Dig hole: 10-15cm deep
3. Place geophone vertically
4. Pack soil firmly around sensor
5. Bury cable for protection
```

### ESP32 Placement:
```
1. Within 3m of geophone
2. Elevated off ground (30cm+)
3. Weather-protected enclosure
4. Ventilation for heat
```

### Power Options:
```
Option 1: USB to laptop (simple, limited range)
Option 2: 5V power bank (portable, ~20 hours)
Option 3: Solar + battery (long-term, requires setup)
```

## 📊 PERFORMANCE

| Distance | Confidence | Notes |
|----------|-----------|-------|
| 10-30m   | >85%      | Excellent detection |
| 30-60m   | 70-85%    | Good detection |
| 60-100m  | 50-70%    | Fair (noisy conditions) |
| >100m    | <50%      | Unreliable |

**Best conditions:** Firm soil, night time, calm weather, buried sensor

## 🆘 QUICK HELP

### Commands:
```bash
# Test hardware
python test_hardware_connection.py

# Signal quality test  
python test_hardware_connection.py quality

# Run detection
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous
```

### Arduino Serial Monitor:
```
STATUS  → System status
RESET   → Reset statistics
SCAN    → Find I2C devices
```

## ✅ PRE-DEPLOYMENT CHECKLIST

Hardware:
- [ ] ESP32-S3 connected to ADS1115 (GPIO 8/9)
- [ ] ADS1115 powered by 3.3V (not 5V!)
- [ ] ADDR pin connected to GND
- [ ] Geophone connected via 1kΩ resistors
- [ ] USB cable is data-capable

Software:
- [ ] Firmware uploaded successfully
- [ ] Serial Monitor shows "System ready!"
- [ ] Python test passes
- [ ] Detection script runs

Field:
- [ ] Geophone buried 10-15cm
- [ ] ESP32 weather-protected
- [ ] Power source configured
- [ ] Logging enabled

## 🎯 NEXT STEPS

1. ✅ **Complete setup** (follow this guide)
2. ✅ **Test indoors** (verify hardware works)
3. ✅ **Calibrate** (signal quality test)
4. ✅ **Deploy** (install in field)
5. ✅ **Monitor** (run continuous detection)
6. ✅ **Analyze** (review detection logs)

## 📞 DOCUMENTATION LINKS

- **This Guide**: `docs/YOUR_HARDWARE_SETUP.md`
- **Complete Setup**: `docs/SETUP_CHECKLIST.md`
- **Quick Start**: `docs/QUICK_START_HARDWARE.md`
- **Troubleshooting**: `docs/ESP32_HARDWARE_SETUP.md#troubleshooting`

---

## 💡 KEY POINTS TO REMEMBER

1. **ADDR → GND**: Required for I2C address 0x48
2. **Use 3.3V**: NOT 5V on ADS1115 VDD
3. **GPIO 8 = SDA, GPIO 9 = SCL**: Don't swap!
4. **Data cable**: USB-C must support data transfer
5. **Bury geophone**: 10-15cm in firm soil for best results
6. **Test before deploy**: Always run connection test first

---

**🎉 Your hardware is ready! Connect your device and start detecting elephants! 🐘**

**For detailed help, see: `docs/SETUP_CHECKLIST.md`**
