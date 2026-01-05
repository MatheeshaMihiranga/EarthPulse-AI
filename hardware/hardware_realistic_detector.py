"""
EarthPulse AI - Hardware Realistic Detector
Detection system using hardware-realistic trained model
Optimized for real ESP32+ADS1115+Geophone setup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from typing import Dict, Optional

class HardwareRealisticDetector:
    """
    Detection system trained on hardware-realistic data
    Accurately distinguishes between:
    - Background noise
    - Table taps
    - Floor footsteps (human)
    - Elephant footfalls
    - Random vibrations
    """
    
    def __init__(self, model_path: str = "./models/hardware_realistic_model.h5"):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained model
        """
        print("Initializing Hardware Realistic Detector...")
        
        # Load model
        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded: {model_path}")
        
        # Load metadata (scaler, class names, etc.)
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.class_names = metadata['class_names']
        self.fs = metadata['sampling_rate']
        self.sequence_length = metadata['sequence_length']
        
        print(f"✓ Classes: {self.class_names}")
        print(f"✓ Sampling rate: {self.fs} Hz")
        
        # Detection settings
        self.min_confidence = 0.7  # Minimum confidence for detection
        self.elephant_confidence_threshold = 0.8  # Higher threshold for elephant
        
        # Statistics tracking
        self.total_predictions = 0
        self.detections_by_class = {name: 0 for name in self.class_names}
        
        print("✓ System ready!")
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract features from signal (same as training)"""
        features = []
        
        # Time domain features
        features.append(np.mean(signal))           # Mean
        features.append(np.std(signal))            # Standard deviation
        features.append(np.max(signal))            # Maximum
        features.append(np.min(signal))            # Minimum
        features.append(np.max(signal) - np.min(signal))  # Peak-to-peak
        features.append(np.sqrt(np.mean(signal**2)))      # RMS
        features.append(np.sum(signal**2))         # Energy
        
        # Statistical features
        features.append(np.mean(np.abs(signal - np.mean(signal)))) # MAD
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features.append(zero_crossings / len(signal))
        
        # Frequency domain features (FFT)
        fft = np.fft.rfft(signal)
        fft_mag = np.abs(fft)
        freqs = np.fft.rfftfreq(len(signal), 1/self.fs)
        
        # Dominant frequency
        dominant_idx = np.argmax(fft_mag)
        features.append(freqs[dominant_idx])
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
        features.append(spectral_centroid)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid)**2) * fft_mag) / (np.sum(fft_mag) + 1e-10)
        )
        features.append(spectral_bandwidth)
        
        # Spectral energy in different bands
        low_freq_energy = np.sum(fft_mag[freqs < 10])     # 0-10 Hz (elephant range)
        mid_freq_energy = np.sum(fft_mag[(freqs >= 10) & (freqs < 30)])  # 10-30 Hz
        high_freq_energy = np.sum(fft_mag[freqs >= 30])   # >30 Hz (table tap)
        total_energy = np.sum(fft_mag) + 1e-10
        
        features.append(low_freq_energy / total_energy)
        features.append(mid_freq_energy / total_energy)
        features.append(high_freq_energy / total_energy)
        
        # Peak in first 0.2 seconds (impulse detection)
        first_20_percent = signal[:int(len(signal)*0.2)]
        features.append(np.max(np.abs(first_20_percent)))
        
        # Decay rate
        first_half_energy = np.sum(signal[:len(signal)//2]**2)
        second_half_energy = np.sum(signal[len(signal)//2:]**2)
        decay_ratio = (first_half_energy + 1e-10) / (second_half_energy + 1e-10)
        features.append(decay_ratio)
        
        return np.array(features)
    
    def process_signal(self, signal: np.ndarray) -> Dict:
        """
        Process signal and return detection result
        
        Args:
            signal: Input signal (sequence_length samples)
        
        Returns:
            Dictionary with detection results
        """
        # Ensure correct length
        if len(signal) < self.sequence_length:
            signal = np.pad(signal, (0, self.sequence_length - len(signal)))
        elif len(signal) > self.sequence_length:
            signal = signal[:self.sequence_length]
        
        # Extract features
        features = self.extract_features(signal)
        features = features.reshape(1, -1)
        
        # Normalize
        features_normalized = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_normalized, verbose=0)[0]
        
        # Get top prediction
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        class_name = self.class_names[class_idx]
        
        # Update statistics
        self.total_predictions += 1
        self.detections_by_class[class_name] += 1
        
        # Determine if detection is valid
        detected = False
        detection_type = "none"
        
        if class_name == "elephant_footfall" and confidence >= self.elephant_confidence_threshold:
            detected = True
            detection_type = "elephant"
        elif class_name in ["table_tap", "floor_footstep"] and confidence >= self.min_confidence:
            detected = True
            detection_type = class_name
        
        # Build result
        result = {
            'detected': detected,
            'detection_type': detection_type,
            'class_name': class_name,
            'confidence': float(confidence),
            'all_probabilities': {
                name: float(prob) 
                for name, prob in zip(self.class_names, predictions)
            },
            'signal_stats': {
                'max': float(np.max(signal)),
                'min': float(np.min(signal)),
                'rms': float(np.sqrt(np.mean(signal**2))),
                'peak_to_peak': float(np.max(signal) - np.min(signal))
            }
        }
        
        return result
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_predictions': self.total_predictions,
            'detections_by_class': self.detections_by_class.copy(),
            'elephant_detections': self.detections_by_class.get('elephant_footfall', 0),
            'background_ratio': self.detections_by_class.get('background', 0) / max(self.total_predictions, 1)
        }


if __name__ == "__main__":
    # Test detector
    print("Testing Hardware Realistic Detector...")
    
    detector = HardwareRealisticDetector()
    
    # Generate test signals
    from training.hardware_realistic_generator import HardwareRealisticGenerator
    
    generator = HardwareRealisticGenerator(sampling_rate=100)
    
    print("\nTesting different signal types...")
    
    # Test background
    bg = generator.generate_background_noise()
    result = detector.process_signal(bg)
    print(f"\nBackground noise: {result['class_name']} ({result['confidence']:.2%})")
    
    # Test table tap
    tap = generator.generate_table_tap(intensity='medium')
    result = detector.process_signal(tap)
    print(f"Table tap: {result['class_name']} ({result['confidence']:.2%})")
    
    # Test floor footstep
    step = generator.generate_floor_footstep(person_weight=70)
    result = detector.process_signal(step)
    print(f"Floor footstep: {result['class_name']} ({result['confidence']:.2%})")
    
    # Test elephant
    elephant = generator.generate_elephant_footfall(weight=4000, distance=30)
    result = detector.process_signal(elephant)
    print(f"Elephant footfall: {result['class_name']} ({result['confidence']:.2%})")
    
    # Test random vibration
    random_vib = generator.generate_random_vibration()
    result = detector.process_signal(random_vib)
    print(f"Random vibration: {result['class_name']} ({result['confidence']:.2%})")
    
    print("\n✓ Detector test complete!")
