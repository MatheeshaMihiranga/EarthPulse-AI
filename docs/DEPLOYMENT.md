# EarthPulse AI - Deployment Guide

## ESP32 Edge Deployment

### Prerequisites

- ESP32 development board (ESP32-WROOM-32)
- Arduino IDE or PlatformIO
- TensorFlow Lite Micro library
- Geophone sensor (SM-24 or equivalent)
- Soil moisture sensor (capacitive)
- LoRa module (RFM95W or SX1278)
- 5V power supply or battery

### Hardware Setup

#### Pin Configuration

```cpp
// Geophone ADC
#define GEOPHONE_PIN 36  // VP pin (ADC1_CH0)

// Soil Moisture Sensor
#define SOIL_MOISTURE_PIN 39  // VN pin (ADC1_CH3)

// LoRa SPI
#define LORA_SCK 5
#define LORA_MISO 19
#define LORA_MOSI 27
#define LORA_CS 18
#define LORA_RST 14
#define LORA_IRQ 26
```

#### Wiring Diagram

```
ESP32                    Geophone
------                   --------
GPIO36 (ADC) ───────────> Signal Out
GND ─────────────────────> GND
3.3V ────────────────────> VCC


ESP32                    Soil Sensor
------                   -----------
GPIO39 (ADC) ───────────> Signal Out
GND ─────────────────────> GND
3.3V ────────────────────> VCC


ESP32                    LoRa Module
------                   -----------
GPIO27 (MOSI) ──────────> MOSI
GPIO19 (MISO) ──────────> MISO
GPIO5  (SCK) ───────────> SCK
GPIO18 (CS) ────────────> NSS
GPIO14 (RST) ───────────> RESET
GPIO26 (IRQ) ───────────> DIO0
GND ─────────────────────> GND
3.3V ────────────────────> VCC
```

### Software Installation

#### Step 1: Install TensorFlow Lite Micro

```bash
# Using Arduino IDE Library Manager
# Search for: "TensorFlow Lite Micro"
# Or add to platformio.ini:
lib_deps =
    tflite-micro
```

#### Step 2: Convert and Deploy Model

```python
# Convert model to C array
import tensorflow as tf

# Load quantized model
with open('models/lstm_model_quantized.tflite', 'rb') as f:
    tflite_model = f.read()

# Convert to C array
c_array = ', '.join([f'0x{byte:02x}' for byte in tflite_model])

# Save to header file
with open('model_data.h', 'w') as f:
    f.write(f'const unsigned char model_data[] = {{\n')
    f.write(f'  {c_array}\n')
    f.write(f'}};\n')
    f.write(f'const unsigned int model_data_len = {len(tflite_model)};\n')
```

#### Step 3: Implement Firmware

```cpp
// main.ino
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include <LoRa.h>

// DSP includes
#include "dsp_filters.h"
#include "feature_extraction.h"

// Globals
const int kTensorArenaSize = 30 * 1024;  // 30KB
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Signal buffer
const int SAMPLE_RATE = 1000;  // Hz
const int BUFFER_SIZE = 1000;  // 1 second
float signal_buffer[BUFFER_SIZE];
int buffer_index = 0;

void setup() {
  Serial.begin(115200);

  // Initialize ADC
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);  // 0-3.3V range

  // Initialize LoRa
  LoRa.setPins(LORA_CS, LORA_RST, LORA_IRQ);
  if (!LoRa.begin(433E6)) {
    Serial.println("LoRa init failed!");
    while (1);
  }
  LoRa.setTxPower(20);

  // Initialize TFLite
  setupModel();

  Serial.println("EarthPulse AI - Ready!");
}

void setupModel() {
  // Load model
  const tflite::Model* model = tflite::GetModel(model_data);

  // Initialize interpreter
  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

  interpreter = &static_interpreter;

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully");
}

void loop() {
  // Read sensors
  float seismic = readSeismic();
  float soil_moisture = readSoilMoisture();

  // Add to buffer
  signal_buffer[buffer_index++] = seismic;

  // Process when buffer full
  if (buffer_index >= BUFFER_SIZE) {
    buffer_index = 0;

    // DSP processing
    float processed[BUFFER_SIZE];
    applyBandpassFilter(signal_buffer, processed, BUFFER_SIZE);

    // Extract features
    float features[18];
    extractFeatures(processed, BUFFER_SIZE, soil_moisture, features);

    // Normalize features
    normalizeFeatures(features, 18);

    // Prepare input tensor
    for (int i = 0; i < 18; i++) {
      input->data.f[i] = features[i];
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    // Get prediction
    int predicted_class = 0;
    float max_confidence = 0.0;

    for (int i = 0; i < 7; i++) {
      float confidence = output->data.f[i];
      if (confidence > max_confidence) {
        max_confidence = confidence;
        predicted_class = i;
      }
    }

    // Check for elephant detection
    if (predicted_class == 0 && max_confidence > 0.70) {
      sendAlert(max_confidence, soil_moisture);
    }

    Serial.printf("Class: %d, Confidence: %.2f\n",
                  predicted_class, max_confidence);
  }

  delay(1);  // 1ms = 1000 Hz
}

float readSeismic() {
  int adc_value = analogRead(GEOPHONE_PIN);
  float voltage = (adc_value / 4095.0) * 3.3;
  // Convert to ground velocity (depends on geophone sensitivity)
  float velocity = voltage / 28.0;  // 28 V/(m/s) sensitivity
  return velocity;
}

float readSoilMoisture() {
  int adc_value = analogRead(SOIL_MOISTURE_PIN);
  // Calibrate these values for your sensor
  const int ADC_WATER = 1000;  // Fully wet
  const int ADC_AIR = 3000;    // Dry
  float moisture = (1.0 - (adc_value - ADC_WATER) /
                   (float)(ADC_AIR - ADC_WATER)) * 100.0;
  return constrain(moisture, 0.0, 100.0);
}

void sendAlert(float confidence, float moisture) {
  // Create JSON payload
  String payload = "{";
  payload += "\"device\":\"EARTHPULSE_001\",";
  payload += "\"alert\":\"elephant_detected\",";
  payload += "\"confidence\":" + String(confidence, 2) + ",";
  payload += "\"moisture\":" + String(moisture, 1) + ",";
  payload += "\"battery\":" + String(readBattery(), 2);
  payload += "}";

  // Send via LoRa
  LoRa.beginPacket();
  LoRa.print(payload);
  LoRa.endPacket();

  Serial.println("ALERT SENT: " + payload);
}

float readBattery() {
  // Read battery voltage (if connected to ADC)
  return 4.2;  // Placeholder
}
```

#### Step 4: Implement DSP Functions

Create `dsp_filters.h`:

```cpp
#ifndef DSP_FILTERS_H
#define DSP_FILTERS_H

// Bandpass filter coefficients (precomputed)
// Butterworth 4th order, 1-80 Hz, 1000 Hz sampling
const int FILTER_ORDER = 4;
const float BP_B[FILTER_ORDER+1] = {0.0123, 0.0492, 0.0738, 0.0492, 0.0123};
const float BP_A[FILTER_ORDER+1] = {1.0000, -2.3695, 2.3140, -1.0547, 0.1874};

void applyBandpassFilter(float* input, float* output, int length) {
  // IIR filter implementation
  float x[FILTER_ORDER+1] = {0};
  float y[FILTER_ORDER+1] = {0};

  for (int n = 0; n < length; n++) {
    // Shift input buffer
    for (int i = FILTER_ORDER; i > 0; i--) {
      x[i] = x[i-1];
    }
    x[0] = input[n];

    // Apply filter
    y[0] = 0;
    for (int i = 0; i <= FILTER_ORDER; i++) {
      y[0] += BP_B[i] * x[i];
    }
    for (int i = 1; i <= FILTER_ORDER; i++) {
      y[0] -= BP_A[i] * y[i];
    }

    output[n] = y[0];

    // Shift output buffer
    for (int i = FILTER_ORDER; i > 0; i--) {
      y[i] = y[i-1];
    }
  }
}

#endif
```

Create `feature_extraction.h`:

```cpp
#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <math.h>

void extractFeatures(float* signal, int length, float moisture, float* features) {
  // Feature 0: RMS
  float sum_sq = 0;
  for (int i = 0; i < length; i++) {
    sum_sq += signal[i] * signal[i];
  }
  features[0] = sqrt(sum_sq / length);

  // Feature 1: Peak-to-peak
  float min_val = signal[0], max_val = signal[0];
  for (int i = 1; i < length; i++) {
    if (signal[i] < min_val) min_val = signal[i];
    if (signal[i] > max_val) max_val = signal[i];
  }
  features[1] = max_val - min_val;

  // Feature 2: Standard deviation
  float mean = 0;
  for (int i = 0; i < length; i++) {
    mean += signal[i];
  }
  mean /= length;

  float variance = 0;
  for (int i = 0; i < length; i++) {
    float diff = signal[i] - mean;
    variance += diff * diff;
  }
  features[2] = sqrt(variance / length);

  // Feature 3: Mean absolute value
  float sum_abs = 0;
  for (int i = 0; i < length; i++) {
    sum_abs += abs(signal[i]);
  }
  features[3] = sum_abs / length;

  // Feature 4: Zero crossing rate
  int zero_crossings = 0;
  for (int i = 1; i < length; i++) {
    if ((signal[i] >= 0 && signal[i-1] < 0) ||
        (signal[i] < 0 && signal[i-1] >= 0)) {
      zero_crossings++;
    }
  }
  features[4] = (float)zero_crossings / length;

  // Features 5-17: Simplified versions (full FFT too expensive)
  // Use approximations or skip if memory/time constrained
  for (int i = 5; i < 18; i++) {
    features[i] = 0.0;
  }
}

void normalizeFeatures(float* features, int count) {
  // Apply pre-computed normalization (from training)
  // Mean and std from training data
  const float means[18] = {0.05, 0.2, 0.03, 0.04, 0.1, /* ... */};
  const float stds[18] = {0.02, 0.1, 0.01, 0.02, 0.05, /* ... */};

  for (int i = 0; i < count; i++) {
    features[i] = (features[i] - means[i]) / stds[i];
  }
}

#endif
```

### Testing and Calibration

#### 1. Sensor Calibration

```cpp
// Calibrate geophone
void calibrateGeophone() {
  Serial.println("Geophone calibration:");
  Serial.println("Tap ground gently...");

  int samples = 100;
  float max_reading = 0;

  for (int i = 0; i < samples; i++) {
    float reading = readSeismic();
    if (abs(reading) > max_reading) {
      max_reading = abs(reading);
    }
    delay(100);
  }

  Serial.printf("Max reading: %.4f m/s\n", max_reading);
}

// Calibrate soil moisture
void calibrateSoilMoisture() {
  Serial.println("Soil moisture calibration:");
  Serial.println("Place in AIR:");
  delay(5000);
  int adc_air = analogRead(SOIL_MOISTURE_PIN);
  Serial.printf("Air: %d\n", adc_air);

  Serial.println("Place in WATER:");
  delay(5000);
  int adc_water = analogRead(SOIL_MOISTURE_PIN);
  Serial.printf("Water: %d\n", adc_water);
}
```

#### 2. Model Validation

Test with known signals:

- Walk near sensor
- Drive vehicle past sensor
- Record actual elephant if possible

Compare predictions with expected classes.

### Field Deployment

#### Pre-Deployment Checklist

- [ ] Sensors calibrated
- [ ] Model validated
- [ ] LoRa communication tested
- [ ] Power system verified
- [ ] Enclosure waterproofed
- [ ] Installation tools prepared

#### Installation Steps

1. **Site Selection**

   - Near known elephant paths
   - Flat, stable ground
   - Good LoRa line-of-sight to gateway

2. **Sensor Installation**

   - Dig 15cm hole
   - Place geophone vertically
   - Compact soil around sensor
   - Mark location with flag

3. **Device Setup**

   - Mount electronics in enclosure
   - Connect solar panel
   - Verify power and charging
   - Test LoRa connectivity

4. **System Activation**
   - Upload firmware
   - Monitor serial output
   - Verify detections
   - Record GPS coordinates

#### Maintenance

**Weekly**

- Check battery voltage
- Clean solar panel
- Verify LoRa connectivity

**Monthly**

- Check sensor connections
- Review detection logs
- Update firmware if needed

**Quarterly**

- Full system calibration
- Replace worn components
- Model retraining if needed

---

## Python Simulation Testing

Before hardware deployment, test with Python simulation:

```bash
# Run detection system demo
python edge_firmware_simulated/detection_system.py

# Launch dashboard for visualization
python dashboard/realtime_dashboard.py
```

---

## Troubleshooting

### Model Not Loading

- Check model size < available RAM
- Verify TFLite version compatibility
- Reduce tensor arena size if needed

### Poor Detection Accuracy

- Calibrate sensors
- Adjust bandpass filter for local conditions
- Check soil moisture effects
- Retrain model with local data

### LoRa Communication Issues

- Verify antenna connection
- Check frequency regulations (433 MHz may not be legal everywhere)
- Reduce TX power if interference
- Increase packet retry count

### Power Problems

- Verify solar panel angle (optimal sunlight)
- Check battery health
- Reduce sampling rate or duty cycle
- Use deep sleep between detections

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Target Platform**: ESP32-WROOM-32
