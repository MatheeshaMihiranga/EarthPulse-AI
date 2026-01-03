# EarthPulse AI - Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EARTHPULSE AI SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    EDGE DEVICE (ESP32-based)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  Geophone    │───▶│  ESP32 ADC   │───▶│   DSP        │    │
│  │  Sensor      │    │  12-bit      │    │   Pipeline   │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                   │             │
│  ┌──────────────┐                                │             │
│  │ Soil Sensor  │─────────────────────────────────┘             │
│  └──────────────┘                                │             │
│                                                   ▼             │
│                                          ┌──────────────┐       │
│                                          │ LSTM Model   │       │
│                                          │ (Quantized)  │       │
│                                          └──────────────┘       │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────────┐                      ┌──────────────┐       │
│  │   LoRa TX    │◀─────────────────────│  Detection   │       │
│  │  433 MHz     │                      │   Logic      │       │
│  └──────────────┘                      └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │ LoRa
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GATEWAY / CLOUD SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  LoRa RX     │───▶│  Alert       │───▶│  Database    │    │
│  │              │    │  Processor   │    │              │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────────┐                      ┌──────────────┐       │
│  │  Dashboard   │◀─────────────────────│  API Server  │       │
│  │  (Web)       │                      │              │       │
│  └──────────────┘                      └──────────────┘       │
│                                                   │             │
│  ┌──────────────┐                                │             │
│  │  Mobile App  │◀───────────────────────────────┘             │
│  │  (Villagers) │                                              │
│  └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Sensor Layer

#### Geophone Sensor

- **Type**: Vertical component geophone
- **Sensitivity**: 28 V/(m/s)
- **Frequency Response**: 1-100 Hz
- **Bandwidth**: DC to 200 Hz
- **Installation**: Buried 10-15 cm underground

#### Soil Moisture Sensor

- **Type**: Capacitive soil moisture sensor
- **Range**: 0-100% volumetric water content
- **Interface**: Analog (ADC)
- **Purpose**: Adaptive signal processing

### 2. Data Acquisition

#### ESP32 ADC

- **Resolution**: 12-bit (0-4095)
- **Voltage Range**: 0-3.3V
- **Sampling Rate**: 1000 Hz
- **Buffer Size**: 1000 samples (1 second)

#### Preprocessing

```python
ADC_VALUE → Voltage → Ground Velocity (m/s)
```

### 3. DSP Pipeline

#### Stage 1: Filtering

```
Raw Signal
    ↓
Notch Filter (50 Hz) - Remove power line interference
    ↓
Bandpass Filter (1-80 Hz) - Retain seismic range
    ↓
Dynamic Range Compression - Handle varying amplitudes
```

#### Stage 2: Feature Extraction

**Time Domain (5 features)**

- RMS energy
- Peak-to-peak amplitude
- Standard deviation
- Mean absolute value
- Zero-crossing rate

**Frequency Domain (5 features)**

- Spectral centroid
- STFT mean power
- STFT max power
- Low frequency energy (1-20 Hz)
- High frequency energy (40-80 Hz)
- Frequency ratio (low/high)

**Temporal Analysis (4 features)**

- Periodicity detection (footstep rhythm)
- Dominant period
- Number of peaks
- Max autocorrelation

**Envelope (3 features)**

- Mean envelope
- Std envelope
- Max envelope

**Total: 18 features**

### 4. Machine Learning Model

#### LSTM Architecture

```
Input Layer (18, 1)
    ↓
LSTM Layer 1 (64 units, return_sequences=True)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
LSTM Layer 2 (32 units)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dropout (0.15)
    ↓
Output Layer (7 units, Softmax)
```

#### Model Specifications

- **Parameters**: ~100K
- **Size (Full)**: 429 KB
- **Size (Quantized)**: 79 KB
- **Inference Time**: ~50-100ms on ESP32
- **Accuracy**: 91.4%

#### Quantization

- **Method**: Post-training float16 quantization
- **Compression**: 5.4x
- **Accuracy Loss**: < 2%
- **Target**: ESP32 with TFLite Micro

### 5. Context-Aware Detection

#### Soil Moisture Adaptation

```python
if moisture < 10%:
    bandpass_high = 60 Hz  # Dry soil: reduce high freq cutoff
    confidence_weight = 0.6  # Lower confidence
elif moisture < 20%:
    bandpass_high = 70 Hz
    confidence_weight = 0.8
elif 15% <= moisture <= 30%:
    bandpass_high = 80 Hz  # Optimal
    confidence_weight = 1.0
else:
    bandpass_high = 80 Hz  # Wet
    confidence_weight = 0.9
```

#### Multi-Frame Confirmation

```
Frame 1: Elephant (0.85) → Wait
Frame 2: Elephant (0.82) → Wait
Frame 3: Elephant (0.88) → CONFIRM! (avg: 0.85)

Alert Sent: "Elephant detected with 85% confidence"
```

### 6. Alert System

#### LoRa Communication

- **Frequency**: 433 MHz (ISM band)
- **Range**: 2-5 km (line of sight)
- **Power**: 20 dBm (100mW)
- **Data Rate**: 1-5 kbps
- **Packet Structure**:

```json
{
  "device_id": "EARTHPULSE_001",
  "timestamp": "2025-11-17T01:00:00Z",
  "alert_type": "elephant_detected",
  "confidence": 0.85,
  "soil_moisture": 22.0,
  "battery": 4.1
}
```

## Data Flow

### Real-Time Processing Pipeline

```
1. Sensor Reading (1000 Hz continuous)
   └─> 1000 samples = 1 second buffer

2. DSP Processing (~10ms)
   ├─> Filtering
   └─> Feature extraction

3. Model Inference (~50ms)
   └─> LSTM prediction

4. Detection Logic (~5ms)
   ├─> Soil moisture weighting
   ├─> Anomaly check
   └─> Multi-frame confirmation

5. Alert Transmission (if confirmed) (~50ms)
   └─> LoRa packet

Total Latency: ~115ms per inference
Processing Rate: ~8-9 Hz (real-time capable)
```

## Power Management

### Energy Budget

**Active Mode (Detection Running)**

- ESP32: ~80mA @ 3.3V = 264mW
- Sensors: ~20mA @ 3.3V = 66mW
- LoRa TX (occasional): ~120mA @ 3.3V = 396mW (50ms)
- **Average**: ~330mW continuous

**Sleep Mode (Between Detections)**

- ESP32 deep sleep: ~10µA
- Sensors off: 0mA
- **Average**: ~0.033mW

**Duty Cycling**

- Active: 10s every minute (16.7%)
- Sleep: 50s every minute (83.3%)
- **Average Power**: 55mW + 0.027mW ≈ 55mW

**Battery Life Calculation**

- Battery: 5000mAh @ 3.7V = 18.5 Wh
- Solar Panel: 6W (peak), ~1W average
- **Autonomy**: Indefinite with solar + battery backup

## Deployment Considerations

### Field Installation

1. **Sensor Placement**

   - Geophone: 10-15cm underground
   - Location: Near known elephant paths
   - Spacing: 50-100m between sensors

2. **Environmental Protection**

   - Waterproof enclosure (IP67)
   - UV-resistant materials
   - Temperature range: -20°C to +60°C

3. **Power System**

   - 6W solar panel
   - 5000mAh LiPo battery
   - MPPT charge controller

4. **Communication**
   - LoRa gateway central location
   - 4G/Satellite backhaul
   - Mesh network for extended range

### Scalability

**Single Node**

- Coverage: ~200m radius
- Power: Solar + battery
- Cost: ~$150/node

**Network (10 nodes)**

- Coverage: ~5-10 km² (depending on topology)
- Coordination: Time-synchronized multi-sensor
- Enhanced: Direction finding, speed estimation

**Cloud Platform**

- Centralized database
- Analytics and reporting
- Mobile app for alerts
- Web dashboard for monitoring

## Future Enhancements

### Hardware

- [ ] Add GPS for precise location
- [ ] Accelerometer for installation quality check
- [ ] Temperature sensor for environmental monitoring

### Software

- [ ] Multi-sensor fusion (triangulation)
- [ ] Direction estimation
- [ ] Distance calculation
- [ ] Number of individuals

### ML Model

- [ ] Transfer learning from real data
- [ ] Online learning capabilities
- [ ] Ensemble models
- [ ] Behavior classification

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Status**: Research Prototype Complete
