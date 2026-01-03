# 🌐 Web Dashboard with Real Hardware

## Overview

The EarthPulse AI dashboard now supports **real-time visualization** of your ESP32-S3 + Geophone hardware! You can see live seismic data and AI detections in your web browser.

## Features

✅ **Real-time vibration waveforms** from your geophone  
✅ **Live AI detection** with confidence scores  
✅ **FFT frequency analysis** of ground vibrations  
✅ **STFT spectrogram** showing time-frequency patterns  
✅ **Detection history** log with timestamps  
✅ **Direction & behavior analysis** when elephants detected  

## Quick Start

### 1. Connect Your Hardware

```powershell
# Make sure your ESP32-S3 is connected via USB
# Check COM port in Device Manager (Windows) or ls /dev/ttyUSB* (Linux)
```

### 2. Launch Dashboard with Hardware

```powershell
# Windows (default COM5):
python dashboard/realtime_dashboard.py --hardware --port COM5

# Windows (different port):
python dashboard/realtime_dashboard.py --hardware --port COM3

# Linux:
python dashboard/realtime_dashboard.py --hardware --port /dev/ttyUSB0
```

### 3. Open in Browser

Open your web browser and go to:
```
http://localhost:8050
```

### 4. Start Detection

1. Click **"Start Detection"** button in the Control Panel
2. Watch real-time data streaming from your geophone!
3. See AI detection results update live

## Command-Line Options

```powershell
python dashboard/realtime_dashboard.py [OPTIONS]

Options:
  --hardware              Use hardware mode (ESP32 + Geophone)
  --port PORT            Serial port (default: COM5)
  --web-port PORT        Web server port (default: 8050)
  --debug                Enable debug mode
  -h, --help             Show help message
```

## Examples

### Hardware Mode (Your IoT Device)
```powershell
# Basic hardware mode
python dashboard/realtime_dashboard.py --hardware --port COM5

# Hardware with custom web port
python dashboard/realtime_dashboard.py --hardware --port COM5 --web-port 8080

# Hardware with debug mode (for development)
python dashboard/realtime_dashboard.py --hardware --port COM5 --debug
```

### Simulation Mode (Testing)
```powershell
# Run with simulated data (no hardware needed)
python dashboard/realtime_dashboard.py

# Simulation with custom port
python dashboard/realtime_dashboard.py --web-port 8080
```

## Dashboard Components

### 📊 Control Panel
- **Hardware Status**: Shows your connected port and device info
- **Soil Moisture**: Adjust for better detection accuracy
- **Start/Stop/Reset**: Control data acquisition

### 🎯 Detection Status
- **Real-time status**: "Elephant Detected!" or "No Detection"
- **Confidence score**: AI confidence percentage
- **Alert messages**: Visual warnings when elephants detected

### 📈 Statistics
- **Signal metrics**: RMS, peak-to-peak, samples
- **Detection rate**: How often elephants are detected
- **Session info**: Runtime, total samples processed

### 🧭 Direction & Behavior
- **Movement direction**: Approaching/Moving Away
- **Distance estimate**: Approximate distance in meters
- **Velocity**: Speed of movement
- **Activity type**: Walking, Running, Drinking, etc.
- **Weight estimate**: Based on vibration intensity

### 📉 Vibration Plot
Real-time waveform showing ground vibration voltages over time

### 🌈 FFT Plot (Frequency Spectrum)
Shows dominant frequencies in the signal:
- **1-5 Hz**: Elephant footfalls
- **10-30 Hz**: Smaller animals
- **High frequencies**: Wind, noise

### 🎨 STFT Spectrogram
Time-frequency heatmap showing how frequency content changes over time

### 📜 Detection History
Scrollable log of recent detections with timestamps and confidence scores

## Troubleshooting

### "Failed to connect to ESP32"
```powershell
# Check if port is correct
python -c "from hardware_interface.esp32_serial_reader import ESP32SerialReader; reader = ESP32SerialReader(); print(reader.list_available_ports())"

# Close Arduino Serial Monitor if open
# Try different USB port
```

### "No data received"
```powershell
# Verify firmware is running
# Check Serial Monitor at 115200 baud shows "START,..." packets
# Press RST button on ESP32
```

### Dashboard not loading
```powershell
# Check firewall isn't blocking port 8050
# Try different port: --web-port 8080
# Check no other application using the port
```

### Slow/laggy dashboard
```powershell
# Close other programs to free resources
# Use better USB cable
# Try USB 2.0 port instead of 3.0
```

## Tips for Best Results

### 🔧 Hardware Setup
1. **Geophone placement**: Bury 10-15cm in firm soil
2. **Stable mounting**: Ensure ESP32 doesn't vibrate itself
3. **Power**: Use quality USB cable with data support
4. **Environment**: Test in location where you expect elephants

### 📊 Detection Tuning
1. **Soil moisture**: Adjust slider based on actual soil conditions
   - Wet soil (30-40%): Better vibration transmission
   - Dry soil (5-15%): Reduced sensitivity
2. **Baseline calibration**: Run for a few minutes to establish baseline
3. **Threshold**: High confidence (>80%) = likely elephant

### 🌐 Remote Access
To access dashboard from other devices on your network:

```powershell
# Find your local IP address
ipconfig  # Windows
ifconfig  # Linux

# Run dashboard with your IP
python dashboard/realtime_dashboard.py --hardware --port COM5
# Then access from other device: http://YOUR_IP:8050
```

⚠️ **Security Note**: Dashboard is not password-protected. Don't expose to public internet.

## Comparison: Hardware vs Simulation

| Feature | Hardware Mode | Simulation Mode |
|---------|--------------|-----------------|
| Data Source | ESP32 + Geophone | Synthetic generator |
| Realism | ✅ Real vibrations | 🎮 Simulated jungle |
| Scenarios | One (your location) | 12+ scenarios |
| Setup | Requires ESP32 hardware | No hardware needed |
| Field Testing | ✅ Yes | ❌ No |
| Development | Best for deployment | Best for testing |

## Performance

- **Update rate**: ~1 Hz (1 packet per second from ESP32)
- **Latency**: <100ms from detection to dashboard update
- **Browser**: Chrome, Firefox, Edge (modern browsers)
- **Concurrent users**: Multiple browsers can view same dashboard

## Next Steps

1. ✅ **Test indoors**: Verify dashboard shows your footsteps
2. ✅ **Calibrate**: Walk at various distances, note confidence scores
3. ✅ **Deploy**: Mount geophone in field location
4. ✅ **Monitor**: Leave dashboard running for continuous surveillance
5. ✅ **Log data**: Add `--log detections.csv` for permanent records

## Example Session

```powershell
PS> python dashboard/realtime_dashboard.py --hardware --port COM5

======================================================================
EarthPulse AI - Real-Time Dashboard (Hardware Mode)
======================================================================
✓ Connected to ESP32 on COM5
✓ Reading real geophone data

Starting dashboard on http://localhost:8050
Press Ctrl+C to stop

Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'realtime_dashboard'
 * Debug mode: off
 
# Open browser to http://localhost:8050
# Click "Start Detection"
# See live data streaming!
```

## Screenshots

When running, you'll see:
- 🟢 **Green badge**: "🔧 HARDWARE MODE" at top
- 📡 **Hardware Status**: "✓ Connected to COM5"
- 📊 **Live plots**: Updating every second with real data
- 🐘 **Detections**: Red alerts when elephants detected

---

**Your IoT device is now web-enabled! Monitor elephants from anywhere on your network! 🐘🌐**
