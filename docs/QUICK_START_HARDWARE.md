# 🚀 Quick Start - Hardware Setup

Get your ESP32-S3 geophone system running in 15 minutes!

## ⚡ Fast Track Setup

### 1️⃣ Hardware Assembly (5 minutes)
```
1. Connect ADS1115 to ESP32-S3:
   • SDA → GPIO 8
   • SCL → GPIO 9
   • VDD → 3.3V
   • GND → GND
   • ADDR → GND (important!)

2. Connect Geophone to ADS1115:
   • Signal → 1kΩ → ADS1115 A0
   • Signal → 1kΩ → GND
   • GND → ADS1115 GND
```

### 2️⃣ Firmware Upload (5 minutes)
```
1. Install Arduino IDE 2.x
2. Add ESP32 board support (see full guide)
3. Install library: Adafruit ADS1X15
4. Open firmware:
   EarthPulse-AI/hardware_firmware/esp32_s3/esp32_s3_seismic_sensor/
5. Select Board: ESP32S3 Dev Module
6. Click Upload
```

### 3️⃣ Test Connection (2 minutes)
```bash
# Arduino Serial Monitor (115200 baud)
# Should see:
✓ ADS1115 initialized
✓ System ready!
START,0.0012,0.0015,...,END
```

### 4️⃣ Run Python Detection (3 minutes)
```bash
# Install Python dependencies
pip install pyserial numpy colorama

# Test serial connection
python hardware_interface/esp32_serial_reader.py

# Run detection
python hardware_interface/realtime_detection_hardware.py --port COM3 --continuous
```

## 🎯 Expected Output

```
🐘 EarthPulse AI - Real-Time Hardware Detection
==================================================================
✓ Connected to COM3
✓ ADS1115: Connected
✓ Model loaded: ./models/lstm_model.h5
✓ System ready!

──────────────────────────────────────────────────────────────
📊 Signal Statistics:
   Samples:    1000
   RMS:        0.0156 V
   Peak-Peak:  0.0892 V
   Mean:       0.0012 V

🐘 ELEPHANT DETECTED!
   Confidence: 87.3%

📍 Movement Direction:
   Status:     ⬆️ Approaching
   Direction:  NE
   Distance:   45.3 m
   Velocity:   0.82 m/s
   Confidence: 78.5%

🐘 Behavior Analysis:
   Activity:   🚶 Walking
   Gait Speed: 1.35 m/s
   Activity:   Moderate
   Weight Est: 3850 kg
   Confidence: 81.2%

⚠️  ALERT: Elephant in vicinity - Take precautions!
──────────────────────────────────────────────────────────────
```

## 🔧 Troubleshooting Quick Fixes

### "ADS1115 not found"
```bash
# Check ADDR pin is connected to GND!
# Verify in Serial Monitor:
SCAN
# Should show: Device found at 0x48
```

### "No serial port found"
```bash
# Windows: Install CH340 driver
# Check Device Manager → Ports (COM & LPT)
# Try different USB cable (must support data)
```

### "Noisy readings"
```cpp
// In firmware, increase gain:
#define ADS_GAIN GAIN_FOUR  // Change from GAIN_TWO
```

## 📚 Full Documentation

- **Complete Setup Guide**: `docs/ESP32_HARDWARE_SETUP.md`
- **Circuit Diagram**: Your attached image (ESP32-S3 instead of Raspberry Pi)
- **Firmware Code**: `hardware_firmware/esp32_s3/`
- **Python Interface**: `hardware_interface/`

## 💡 Pro Tips

1. **Best Sensitivity**: Use `GAIN_FOUR` or `GAIN_EIGHT` for weak signals
2. **Field Deployment**: Bury geophone 10-15cm in firm soil
3. **Noise Reduction**: Keep away from motors, vehicles, electrical equipment
4. **Power**: Use 5V 2A power bank for portable operation
5. **Logging**: Add `--log detections.txt` to save all results

## 🆘 Need Help?

1. Check Serial Monitor for error messages
2. Run: `python hardware_interface/esp32_serial_reader.py`
3. Verify all connections match circuit diagram
4. See troubleshooting in `ESP32_HARDWARE_SETUP.md`

---

**Ready to detect elephants! 🐘** Connect your hardware and run the command above.
