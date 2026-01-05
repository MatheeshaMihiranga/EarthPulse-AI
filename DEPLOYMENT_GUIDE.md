# EarthPulse-AI Dev - Quick Setup Guide

## 📦 Repository Information
- **Organization**: WildWatch-60
- **Repository**: EarthPulse-AI-Dev
- **URL**: https://github.com/WildWatch-60/EarthPulse-AI-Dev

## 🚀 Quick Start for End Users

### Installation
```bash
# Clone the repository
git clone https://github.com/WildWatch-60/EarthPulse-AI-Dev.git
cd EarthPulse-AI-Dev

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

#### 1. Test Elephant Detection
```bash
python test_elephant_detection.py
```

#### 2. Test Jungle Environment Detection
```bash
python test_jungle_detection.py
```

#### 3. Run Real-time Dashboard
```bash
# Software simulation mode
python dashboard/realtime_dashboard.py

# Hardware mode (if ESP32 is connected)
python dashboard/realtime_dashboard.py --hardware
```

## 📋 Included Model Files

All trained models are included in the repository for immediate use:

- **`models/lstm_model.h5`** - Main LSTM model (460 KB)
- **`models/lstm_model_quantized.tflite`** - Quantized model for edge devices (80 KB)
- **`models/lstm_model.onnx`** - ONNX format for cross-platform deployment (120 KB)
- **`models/hardware_realistic_model.h5`** - Hardware-optimized model (210 KB)

## 🔧 For Developers

### Project Structure
```
EarthPulse-AI-Dev/
├── models/                  # Trained models (ready to use)
├── synthetic_generator/     # Data generation tools
├── training/               # Training scripts
├── edge_firmware_simulated/ # Edge device simulation
├── hardware/               # ESP32 firmware & integration
├── dashboard/              # Web-based monitoring
├── docs/                   # Detailed documentation
└── data/                   # Dataset files
```

### Training Your Own Model
```bash
python training/train_hardware_realistic_model.py
```

### Hardware Setup
See `docs/DEPLOYMENT.md` for ESP32 setup instructions.

## 📚 Documentation

- **Architecture**: `docs/ARCHITECTURE.md`
- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **Performance Report**: `docs/PERFORMANCE_REPORT.md`
- **Quick Start**: `docs/QUICK_START.md`

## ⚡ System Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib
- Dash (for web dashboard)

All dependencies are listed in `requirements.txt`.

## 🎯 Key Features

✅ Pre-trained models included  
✅ Synthetic data generation  
✅ Real-time detection dashboard  
✅ Edge device deployment ready  
✅ Hardware integration (ESP32)  
✅ Comprehensive testing suite  

## 🤝 Contributing

This is a group project for wildlife monitoring research. For contributions or questions, please contact the WildWatch-60 team.

## 📄 License

See LICENSE file for details.

---

**No additional downloads required** - All model files and datasets are included in the repository. Just install dependencies and run! 🎉
