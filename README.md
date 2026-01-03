# 🐘 EarthPulse AI - Elephant Seismic Detection System

## Full Synthetic IoT Research Component for Elephant Footfall Detection

![Project Status](https://img.shields.io/badge/status-research%20prototype-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [Model Performance](#model-performance)
- [Usage Guide](#usage-guide)
- [Research Methodology](#research-methodology)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Citation](#citation)

---

## 🎯 Overview

**EarthPulse AI** is a comprehensive research system for detecting elephant footfalls using seismic signals. This project implements a complete simulation-based research pipeline, including:

- **Virtual IoT Hardware**: Simulated geophone + ESP32 + soil moisture sensor + LoRa
- **Synthetic Signal Generation**: Physics-based seismic signal simulator
- **Advanced DSP Pipeline**: Real-time signal processing optimized for edge devices
- **LSTM Classifier**: Deep learning model achieving 91.4% accuracy
- **Context-Aware Detection**: Soil moisture-adaptive filtering and confidence weighting
- **Anomaly Suppression**: Multi-frame confirmation and false positive filtering
- **Real-Time Dashboard**: Interactive visualization system

### Why This Matters

Human-elephant conflict causes significant casualties and crop damage. Early detection of elephants approaching human settlements can save lives and property. Seismic detection offers a non-invasive, weather-resistant monitoring solution.

---

## � Project Structure

```
EarthPulse-AI/
│
├── synthetic_generator/          # Signal generation and DSP
│   ├── seismic_signal_generator.py
│   ├── dsp_pipeline.py
│   └── dataset_generator.py
│
├── edge_firmware_simulated/      # Virtual IoT device
│   ├── virtual_iot_device.py
│   └── detection_system.py
│
├── models/                        # Trained models
│   ├── lstm_classifier.py
│   ├── lstm_model.h5
│   ├── lstm_model_quantized.tflite
│   ├── lstm_model.onnx
│   ├── model_card.json
│   ├── confusion_matrix.png
│   └── training_history.png
│
├── data/                          # Datasets
│   ├── raw/
│   ├── processed/
│   ├── train_dataset.csv
│   ├── val_dataset.csv
│   ├── test_dataset.csv
│   └── dataset_metadata.json
│
├── dashboard/                     # Real-time visualization
│   └── realtime_dashboard.py
│
├── notebooks/                     # Jupyter notebooks
│
├── docs/                          # Documentation
│
└── README.md                      # This file
```

---

## ✨ Key Features

### 1. **Physics-Based Synthetic Data Generation**

- **7 Signal Classes**: Elephant footfall, human footsteps, cattle, wind, rain, vehicles, background noise
- **Realistic Parameters**: Frequency content, temporal patterns, attenuation, soil effects
- **Configurable Conditions**: Distance, soil moisture, SNR, weather

### 2. **Complete Virtual IoT Device**

- **Geophone Simulation**: 28 V/(m/s) sensitivity
- **ESP32 ADC**: 12-bit resolution with realistic nonlinearity
- **Soil Moisture Sensor**: Capacitive sensor simulation
- **LoRa Communication**: Alert transmission with packet logging

### 3. **Advanced DSP Pipeline**

- Bandpass filter (1-80 Hz, adaptive)
- Notch filter (50 Hz power line rejection)
- Dynamic range compression
- **18 Feature Extraction**:
  - Time domain: RMS, peak-to-peak, std, mean abs, zero-crossing rate
  - Frequency domain: Spectral centroid, low/high freq energy, freq ratio
  - Time-frequency: STFT features
  - Temporal: Periodicity detection, envelope analysis

### 4. **LSTM Neural Network**

- **Architecture**: 2 LSTM layers + Dense layers
- **Performance**: 91.4% test accuracy
- **Elephant Detection**: 84% precision, 70% recall
- **Quantized**: Float16 quantization (5.4x compression)
- **Formats**: .h5, .tflite, .onnx

### 5. **Context-Aware System**

- **Soil Moisture Adaptation**: Filters and confidence weighting adjust based on soil conditions
- **Multi-Frame Confirmation**: 3-frame confirmation for elephant detections
- **Anomaly Detection**: Outlier filtering and adaptive noise profiling
- **False Positive Suppression**: Threshold gating and signal quality checks

### 6. **Real-Time Dashboard**

- Live seismic waveform streaming
- FFT and STFT visualizations
- Detection status and alerts
- Statistics and history tracking

---

## 🚀 Installation & Complete Setup Guide

### Prerequisites

- **Python 3.11+** (Python 3.8+ also works but 3.11 recommended)
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **4GB RAM minimum** (8GB recommended for training)
- **2GB free disk space** (for datasets and models)

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/MatheeshaMihiranga/EarthPulse-AI.git

# Navigate to project directory
cd EarthPulse-AI
```

#### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Dependencies include:**
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing
- `matplotlib>=3.7.0` - Plotting
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning utilities
- `tensorflow>=2.13.0` - Deep learning framework
- `keras>=2.13.0` - Neural network API
- `seaborn>=0.12.0` - Statistical visualizations
- `plotly>=5.14.0` - Interactive plots
- `dash>=2.9.0` - Web dashboard
- `dash-bootstrap-components>=2.0.0` - Dashboard styling
- `tf2onnx>=1.14.0` - ONNX export
- `tqdm>=4.67.0` - Progress bars

#### 4. Verify Installation

```bash
# Test imports
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import numpy as np; import scipy; import pandas as pd; print('Core libraries OK')"
```

---

## 🎯 Complete Training & Running Guide

### Step 1: Generate Synthetic Dataset

```bash
# Generate complete dataset (train/val/test splits)
python synthetic_generator/dataset_generator.py
```

**What this does:**
- Generates **1,120 labeled samples** across 7 classes
- Creates train (700), validation (210), and test (210) splits
- Saves raw signals in `data/raw/`
- Saves processed features in `data/processed/`
- Creates CSV files: `train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`
- Generates `dataset_metadata.json` with statistics

**Expected time:** 5-10 minutes

**Output structure:**
```
data/
├── raw/                          # Raw seismic signals
├── processed/                    # Processed features
├── train_dataset.csv            # Training data paths
├── val_dataset.csv              # Validation data paths
├── test_dataset.csv             # Test data paths
└── dataset_metadata.json        # Dataset statistics
```

### Step 2: Train the LSTM Model

```bash
# Train model with default settings
python models/lstm_classifier.py
```

**What this does:**
- Loads preprocessed training and validation data
- Builds LSTM neural network (2 LSTM layers + Dense layers)
- Trains with early stopping and learning rate scheduling
- Saves best model to `models/lstm_model.h5`
- Generates quantized model: `models/lstm_model_quantized.tflite`
- Exports ONNX format: `models/lstm_model.onnx`
- Creates evaluation plots: `confusion_matrix.png`, `training_history.png`
- Saves metrics to `models/model_card.json`

**Expected time:** 10-30 minutes (depends on CPU/GPU)

**Training parameters:**
- Epochs: 100 (with early stopping)
- Batch size: 32
- Learning rate: 0.001 (adaptive)
- Optimizer: Adam

**Expected performance:**
- Test Accuracy: ~91%
- Elephant Detection Recall: ~70%

### Step 3: Test the Detection System

```bash
# Run complete detection demo
python test_elephant_detection.py
```

**What this does:**
- Tests detection on various scenarios
- Demonstrates multi-frame confirmation
- Shows false positive suppression
- Validates context-aware filtering

**Alternative tests:**
```bash
# Test jungle environment detection
python test_jungle_detection.py

# Test hardware connection (if available)
python test_hardware_connection.py
```

### Step 4: Launch Real-Time Dashboard

```bash
# Start interactive web dashboard
python dashboard/realtime_dashboard.py
```

**What this does:**
- Starts local web server on `http://localhost:8050`
- Opens browser automatically
- Displays live seismic waveforms
- Shows FFT and STFT visualizations
- Real-time detection status and alerts
- Statistics and detection history

**Dashboard features:**
- Live signal streaming
- Frequency analysis
- Detection alerts
- Historical data
- Interactive controls

---

## ⚡ Quick Start (All-in-One)

Run everything in sequence:

```bash
# 1. Generate dataset
python synthetic_generator/dataset_generator.py

# 2. Train model
python models/lstm_classifier.py

# 3. Test detection
python test_elephant_detection.py

# 4. Launch dashboard
python dashboard/realtime_dashboard.py
```

**Total time:** ~20-45 minutes (first-time setup)

---

## � Troubleshooting

### Common Issues and Solutions

#### Issue: TensorFlow Installation Fails

**Solution:**
```bash
# For Windows with GPU
pip install tensorflow[and-cuda]

# For CPU only (faster install)
pip install tensorflow-cpu

# For older systems
pip install tensorflow==2.13.0
```

#### Issue: "No module named 'xxx'" Error

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or install missing module individually
pip install <module_name>
```

#### Issue: Out of Memory During Training

**Solution:**
- Reduce batch size in `models/lstm_classifier.py`
- Close other applications
- Use fewer training samples

#### Issue: Dashboard Not Opening

**Solution:**
```bash
# Check if port 8050 is available
# Manually open browser to: http://localhost:8050
# Or change port in dashboard/realtime_dashboard.py
```

#### Issue: Dataset Generation is Slow

**Solution:**
- This is normal; generation takes 5-10 minutes
- Reduce sample count in `dataset_generator.py` for testing
- Check CPU usage - should be near 100%

---

## �🔧 Components

### Synthetic Signal Generator

```python
from synthetic_generator.seismic_signal_generator import SeismicSignalGenerator

generator = SeismicSignalGenerator(sampling_rate=1000)
generator.set_soil_conditions(moisture=20.0)

# Generate elephant footfall
signal, metadata = generator.generate_elephant_footfall(
    duration=5.0,
    num_steps=4,
    distance_m=50.0,
    elephant_weight_kg=4000.0
)
```

### DSP Pipeline

```python
from synthetic_generator.dsp_pipeline import AdaptiveDSP

dsp = AdaptiveDSP()
dsp.set_soil_moisture(20.0)

# Process signal
processed_signal, features = dsp.process_signal(raw_signal)

# Get feature vector for ML
feature_vector = dsp.create_feature_vector(features)
```

### Virtual IoT Device

```python
from edge_firmware_simulated.virtual_iot_device import VirtualIoTDevice

device = VirtualIoTDevice()
device.initialize()
device.set_environment(soil_moisture=22.0, temperature=26.0)

# Read sensors
readings = device.read_sensors(vibration_signal)

# Send alert
device.send_alert(
    alert_type="elephant_detected",
    confidence=0.92
)
```

### Detection System

```python
from edge_firmware_simulated.detection_system import ElephantDetectionSystem

system = ElephantDetectionSystem(model_path="./models/lstm_model.h5")

# Process signal
result = system.process_signal(
    raw_signal=seismic_data,
    soil_moisture=20.0
)

if result['detected'] and result['class_name'] == 'elephant_footfall':
    print(f"Elephant detected! Confidence: {result['confidence']:.2%}")
```

---

## 📊 Model Performance

### Overall Performance

- **Test Accuracy**: 91.43%
- **Macro Avg Precision**: 92.05%
- **Macro Avg Recall**: 91.43%
- **Macro Avg F1-Score**: 91.31%

### Class-Specific Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Elephant Footfall** | **84.0%** | **70.0%** | **76.4%** | 30 |
| Human Footsteps | 100.0% | 80.0% | 88.9% | 30 |
| Cattle Movement | 75.7% | 93.3% | 83.6% | 30 |
| Wind Vibration | 100.0% | 100.0% | 100.0% | 30 |
| Rain Impact | 100.0% | 96.7% | 98.3% | 30 |
| Vehicle Passing | 90.9% | 100.0% | 95.2% | 30 |
| Background Noise | 93.8% | 100.0% | 96.8% | 30 |

### Model Artifacts

- **Original Model**: 429 KB (.h5)
- **Quantized Model**: 79 KB (.tflite) - 5.4x compression
- **ONNX Export**: Available for cross-platform deployment

---

## 📖 Usage Guide

### Generating Custom Datasets

```python
from synthetic_generator.dataset_generator import DatasetGenerator

generator = DatasetGenerator(output_dir="./custom_data")

# Generate with custom parameters
train_df, val_df, test_df = generator.generate_all_splits(
    train_samples=500,
    val_samples=100,
    test_samples=100,
    duration=10.0
)
```

### Custom Model Training

```python
from models.lstm_classifier import ElephantDetectionLSTM

lstm = ElephantDetectionLSTM(
    num_classes=7,
    sequence_length=50,
    num_features=18
)

# Load data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = lstm.load_data("./data")

# Train
lstm.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# Evaluate
metrics = lstm.evaluate(X_test, y_test, class_names)

# Export
lstm.quantize_model()
lstm.export_to_onnx()
```

### Real-Time Detection

```python
from edge_firmware_simulated.detection_system import ElephantDetectionSystem

system = ElephantDetectionSystem()

# Process streaming data
for signal_window in signal_stream:
    result = system.process_signal(signal_window, soil_moisture=current_moisture)
    
    if result['detected'] and result['class_name'] == 'elephant_footfall':
        # Trigger alert
        send_alert_to_villagers()
```

---

## 🔬 Research Methodology

### Signal Generation

Based on published research on elephant seismic communication and footfall signatures:

1. **Frequency Content**: Elephants produce dominant frequencies in 1-30 Hz range
2. **Temporal Patterns**: Walking rhythm creates periodic impacts (0.5-2 Hz stride frequency)
3. **Amplitude Scaling**: Proportional to body mass (~4000 kg for adult)
4. **Propagation**: Geometric spreading + material attenuation (soil-dependent)

### Soil Moisture Effects

- **Dry Soil (< 10%)**: High attenuation, reduced propagation velocity
- **Optimal (15-30%)**: Best signal propagation
- **Wet Soil (> 40%)**: Muddy damping, reduced signal quality

### TSTR Evaluation

**Train on Synthetic, Test on Real** methodology:
- Current: 91.4% on synthetic test set
- Future: Validate on real geophone recordings

### Edge Deployment Strategy

1. **Feature Extraction on ESP32**: DSP pipeline runs on device
2. **Model Inference**: Quantized TFLite model
3. **LoRa Alerts**: Low-power long-range communication
4. **Solar Powered**: Self-sustaining field deployment

---

## 🚧 Future Work

### Phase 1: Real Data Validation
- [ ] Collect real elephant footfall recordings
- [ ] Validate model on field data
- [ ] Retrain with mixed synthetic + real data

### Phase 2: Hardware Implementation
- [ ] Deploy on actual ESP32 + geophone
- [ ] Field testing in elephant habitats
- [ ] Power optimization

### Phase 3: System Integration
- [ ] Multi-sensor fusion (acoustic + seismic)
- [ ] Distributed sensor network
- [ ] Cloud-based monitoring platform
- [ ] Mobile app for villagers

### Phase 4: Advanced Features
- [ ] Direction estimation (multiple sensors)
- [ ] Speed and distance calculation
- [ ] Number of individuals estimation
- [ ] Behavior classification

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Real elephant footfall data collection
- Hardware testing and optimization
- Model improvements
- Documentation and tutorials
- Bug fixes and enhancements

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{earthpulse_ai_2026,
  title={EarthPulse AI: Elephant Seismic Detection System},
  author={Matheesha Mihiranga},
  year={2026},
  url={https://github.com/MatheeshaMihiranga/EarthPulse-AI}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Elephant seismic communication research community
- Conservation technology initiatives
- Open-source ML and IoT communities

---

## 📞 Contact

For questions, collaborations, or feedback:

- **GitHub**: [@MatheeshaMihiranga](https://github.com/MatheeshaMihiranga)
- **Project Repository**: https://github.com/MatheeshaMihiranga/EarthPulse-AI
- **Documentation**: [Full Documentation](docs/)

---

## 🌟 Project Status

**Current Version**: 1.0.0 (Research Prototype)

**Last Updated**: January 2026

**Status**: Active Development - Synthetic Data Phase Complete

---

**Built with ❤️ for elephant conservation and human safety**
