"""
EarthPulse AI - Integrated Detection System
Combines all components with context-aware detection and anomaly suppression
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
from collections import deque
import tensorflow as tf

from synthetic_generator.dsp_pipeline import AdaptiveDSP
from edge_firmware_simulated.virtual_iot_device import VirtualIoTDevice
from edge_firmware_simulated.direction_and_behavior import DirectionDetector, BehaviorAnalyzer


class AnomalyDetector:
    """
    Anomaly detection and false positive suppression
    """
    
    def __init__(self, threshold_std: float = 3.0, 
                 confirmation_frames: int = 2):
        """
        Initialize anomaly detector
        
        Args:
            threshold_std: Standard deviations for outlier detection
            confirmation_frames: Number of consecutive detections needed (reduced to 2 for real-time)
        """
        self.threshold_std = threshold_std
        self.confirmation_frames = confirmation_frames
        
        # Noise profile (adaptive)
        self.noise_rms = 0.05
        self.noise_history = deque(maxlen=100)
        
        # Detection history for multi-frame confirmation
        self.detection_history = deque(maxlen=confirmation_frames)
        
    def update_noise_profile(self, signal: np.ndarray):
        """Update adaptive noise profile"""
        rms = np.sqrt(np.mean(signal ** 2))
        self.noise_history.append(rms)
        
        if len(self.noise_history) > 10:
            self.noise_rms = np.median(self.noise_history)
    
    def is_outlier(self, features: Dict) -> bool:
        """Check if features indicate anomalous signal"""
        # Check RMS against noise floor (very permissive - only reject extreme noise)
        if features['rms'] < 0.001:  # Absolute minimum threshold
            return True  # Essentially no signal
        
        # Check if signal characteristics are unrealistic
        if features['peak_to_peak'] > 500:  # Extremely unrealistic
            return True
        
        # Check for NaN or inf values
        if not np.isfinite(features['rms']) or not np.isfinite(features['peak_to_peak']):
            return True
        
        return False
    
    def confirm_detection(self, class_name: str, confidence: float,
                         min_confidence: float = 0.5) -> Tuple[bool, float]:
        """
        Multi-frame confirmation with confidence weighting
        
        Args:
            class_name: Detected class
            confidence: Prediction confidence
            min_confidence: Minimum confidence threshold (lowered for real-world detection)
            
        Returns:
            (is_confirmed, average_confidence)
        """
        # Add to history
        self.detection_history.append((class_name, confidence))
        
        # Need enough frames
        if len(self.detection_history) < self.confirmation_frames:
            return False, 0.0
        
        # Check if all recent detections agree
        recent_classes = [det[0] for det in self.detection_history]
        recent_confidences = [det[1] for det in self.detection_history]
        
        # Count occurrences of most common class
        class_counts = {}
        for cls in recent_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        most_common = max(class_counts, key=class_counts.get)
        agreement = class_counts[most_common] / len(recent_classes)
        
        # Average confidence for the agreed class
        agreed_confidences = [conf for cls, conf in self.detection_history 
                            if cls == most_common]
        avg_confidence = np.mean(agreed_confidences)
        
        # Confirm if:
        # 1. Majority agree on same class (60% agreement)
        # 2. Average confidence above threshold
        # 3. Target class is the one being confirmed
        is_confirmed = (
            agreement >= 0.6 and
            avg_confidence >= min_confidence and
            most_common == class_name
        )
        
        return is_confirmed, avg_confidence
    
    def reset_history(self):
        """Reset detection history"""
        self.detection_history.clear()


class ElephantDetectionSystem:
    """
    Complete integrated detection system
    """
    
    def __init__(self, model_path: str = "./models/lstm_model.h5"):
        """Initialize detection system"""
        print("Initializing EarthPulse AI Detection System...")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded: {model_path}")
        
        # Initialize components
        self.dsp = AdaptiveDSP()
        self.anomaly_detector = AnomalyDetector()
        self.direction_detector = DirectionDetector()
        self.behavior_analyzer = BehaviorAnalyzer()
        
        # Class mapping
        self.classes = {
            0: 'elephant_footfall',
            1: 'human_footsteps',
            2: 'cattle_movement',
            3: 'wind_vibration',
            4: 'rain_impact',
            5: 'vehicle_passing',
            6: 'background_noise'
        }
        
        # Detection statistics
        self.stats = {
            'total_predictions': 0,
            'elephant_detections': 0,
            'confirmed_elephants': 0,
            'false_positives_suppressed': 0
        }
        
        print("✓ System ready!")
        
    def process_signal(self, raw_signal: np.ndarray,
                      soil_moisture: float = 20.0) -> Dict:
        """
        Complete signal processing and detection pipeline
        
        Args:
            raw_signal: Raw seismic signal
            soil_moisture: Current soil moisture percentage
            
        Returns:
            Detection results dictionary
        """
        # Update DSP with soil conditions
        self.dsp.set_soil_moisture(soil_moisture)
        
        # Update noise profile
        self.anomaly_detector.update_noise_profile(raw_signal)
        
        # Process signal
        processed_signal, features = self.dsp.process_signal(raw_signal)
        
        # Check for anomalies
        is_anomaly = self.anomaly_detector.is_outlier(features)
        
        if is_anomaly:
            self.stats['false_positives_suppressed'] += 1
            return {
                'detected': False,
                'reason': 'anomaly_detected',
                'class_name': 'anomaly',
                'confidence': 0.0
            }
        
        # Extract feature vector for model
        feature_vector = self.dsp.create_feature_vector(features)
        
        # Reshape for model input
        X = feature_vector.reshape(1, -1, 1)
        
        # Get prediction
        prediction = self.model.predict(X, verbose=0)[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]
        class_name = self.classes[class_idx]
        
        # Store all class probabilities for visualization
        all_predictions = {self.classes[i]: float(prediction[i]) for i in range(len(prediction))}
        
        self.stats['total_predictions'] += 1
        
        # Apply soil moisture confidence weighting
        soil_confidence_weight = self.dsp.get_confidence_weight()
        weighted_confidence = confidence * soil_confidence_weight
        
        # For elephant detection, require multi-frame confirmation
        if class_name == 'elephant_footfall':
            self.stats['elephant_detections'] += 1
            
            is_confirmed, avg_confidence = self.anomaly_detector.confirm_detection(
                class_name, weighted_confidence, min_confidence=0.5
            )
            
            if is_confirmed:
                self.stats['confirmed_elephants'] += 1
                
                # Analyze direction and behavior for confirmed elephants
                import time
                direction = self.direction_detector.detect_direction_single_sensor(
                    raw_signal, time.time()
                )
                behavior = self.behavior_analyzer.analyze_behavior(raw_signal, direction)
                
                return {
                    'detected': True,
                    'class_name': class_name,
                    'confidence': avg_confidence,
                    'soil_moisture': soil_moisture,
                    'soil_confidence_weight': soil_confidence_weight,
                    'features': features,
                    'confirmation': 'multi_frame_confirmed',
                    'all_predictions': all_predictions,
                    'direction': {
                        'cardinal': direction.direction_cardinal,
                        'degrees': direction.direction_degrees,
                        'approaching': direction.approaching,
                        'distance': direction.estimated_distance,
                        'velocity': direction.velocity,
                        'confidence': direction.confidence
                    },
                    'behavior': {
                        'type': behavior.behavior_type,
                        'gait_speed': behavior.gait_speed,
                        'activity_level': behavior.activity_level,
                        'estimated_weight': behavior.estimated_weight,
                        'confidence': behavior.confidence
                    }
                }
            else:
                return {
                    'detected': False,
                    'reason': 'awaiting_confirmation',
                    'class_name': class_name,
                    'confidence': weighted_confidence,
                    'frames_confirmed': len(self.anomaly_detector.detection_history),
                    'all_predictions': all_predictions
                }
        else:
            # Other classes don't need confirmation
            return {
                'detected': True,
                'class_name': class_name,
                'confidence': weighted_confidence,
                'soil_moisture': soil_moisture,
                'soil_confidence_weight': soil_confidence_weight,
                'features': features,
                'confirmation': 'single_frame',
                'all_predictions': all_predictions
            }
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        stats = self.stats.copy()
        
        if stats['elephant_detections'] > 0:
            stats['confirmation_rate'] = (
                stats['confirmed_elephants'] / stats['elephant_detections']
            )
        else:
            stats['confirmation_rate'] = 0.0
        
        return stats


def demo_detection_system():
    """Demo the complete detection system"""
    print("=" * 70)
    print("EarthPulse AI - Detection System Demo")
    print("=" * 70)
    
    # Initialize system
    system = ElephantDetectionSystem(model_path="./models/lstm_model.h5")
    
    # Import signal generator
    from synthetic_generator.seismic_signal_generator import SeismicSignalGenerator
    
    generator = SeismicSignalGenerator(sampling_rate=1000)
    
    print("\nRunning detection scenarios...")
    print("-" * 70)
    
    # Scenario 1: Elephant detection
    print("\n1. Elephant Footfall Detection:")
    generator.set_soil_conditions(moisture=22.0)
    
    for i in range(5):
        elephant_signal, _ = generator.generate_elephant_footfall(
            duration=5.0, num_steps=4, distance_m=40.0
        )
        
        result = system.process_signal(elephant_signal, soil_moisture=22.0)
        
        if result['detected']:
            print(f"   Frame {i+1}: {result['class_name']} "
                  f"(confidence: {result['confidence']:.3f}, "
                  f"status: {result['confirmation']})")
        else:
            print(f"   Frame {i+1}: {result['reason']}")
    
    # Scenario 2: Human footsteps (should not trigger elephant alert)
    print("\n2. Human Footsteps (Non-Elephant):")
    human_signal, _ = generator.generate_human_footsteps(
        duration=5.0, num_steps=15, distance_m=20.0
    )
    
    result = system.process_signal(human_signal, soil_moisture=22.0)
    print(f"   Detected: {result['class_name']} "
          f"(confidence: {result['confidence']:.3f})")
    
    # Scenario 3: Background noise (should be filtered)
    print("\n3. Background Noise:")
    noise_signal, _ = generator.generate_background_noise(
        duration=5.0, noise_level="low"
    )
    
    result = system.process_signal(noise_signal, soil_moisture=22.0)
    if not result['detected']:
        print(f"   Correctly filtered: {result['reason']}")
    else:
        print(f"   Detected: {result['class_name']}")
    
    # Scenario 4: Varying soil moisture
    print("\n4. Soil Moisture Impact:")
    for moisture in [5, 15, 25, 40]:
        generator.set_soil_conditions(moisture=moisture)
        elephant_signal, _ = generator.generate_elephant_footfall(
            duration=5.0, num_steps=4, distance_m=40.0
        )
        
        result = system.process_signal(elephant_signal, soil_moisture=moisture)
        
        if result['detected']:
            print(f"   Moisture {moisture:2d}%: confidence={result['confidence']:.3f}, "
                  f"weight={result['soil_confidence_weight']:.2f}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Detection Statistics:")
    print("-" * 70)
    stats = system.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.3f}")
        else:
            print(f"  {key:30s}: {value}")
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo_detection_system()
