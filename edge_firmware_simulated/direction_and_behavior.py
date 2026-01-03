"""
Elephant Movement Direction Detection and Behavior Analysis
Using seismic signal characteristics from geophone array
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class ElephantBehavior:
    """Detected elephant behavior"""
    behavior_type: str  # 'walking', 'running', 'feeding', 'standing', 'bathing'
    confidence: float
    gait_speed: float  # meters per second
    estimated_weight: float  # kg
    activity_level: str  # 'calm', 'moderate', 'agitated'


@dataclass
class MovementDirection:
    """Estimated movement direction"""
    direction_degrees: float  # 0-360, 0=North
    direction_cardinal: str  # 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'
    confidence: float
    approaching: bool  # True if moving towards sensor
    estimated_distance: float  # meters
    velocity: float  # m/s towards/away from sensor


class DirectionDetector:
    """
    Detect elephant movement direction using signal characteristics
    
    Methods:
    1. Time-of-arrival differences (requires multiple sensors)
    2. Signal amplitude changes over time (single sensor)
    3. Frequency content analysis (Doppler-like effects)
    """
    
    def __init__(self, sampling_rate: int = 1000):
        self.fs = sampling_rate
        self.signal_history = deque(maxlen=10)  # Last 10 seconds
        self.amplitude_history = deque(maxlen=20)
        self.detection_times = deque(maxlen=10)
        
    def detect_direction_single_sensor(self, 
                                      signal: np.ndarray,
                                      timestamp: float) -> MovementDirection:
        """
        Detect direction using single geophone by analyzing amplitude changes
        
        Principle: Signal amplitude increases if approaching, decreases if receding
        """
        # Calculate signal amplitude (RMS)
        rms = np.sqrt(np.mean(signal ** 2))
        current_time = timestamp
        
        self.amplitude_history.append((current_time, rms))
        self.detection_times.append(current_time)
        
        if len(self.amplitude_history) < 3:
            return MovementDirection(
                direction_degrees=0.0,
                direction_cardinal='Unknown',
                confidence=0.0,
                approaching=False,
                estimated_distance=50.0,
                velocity=0.0
            )
        
        # Analyze amplitude trend
        times = [t for t, _ in self.amplitude_history]
        amplitudes = [a for _, a in self.amplitude_history]
        
        # Linear regression on amplitude vs time
        if len(times) >= 3:
            # Fit line: amplitude = slope * time + intercept
            times_array = np.array(times)
            amplitudes_array = np.array(amplitudes)
            
            # Normalize time
            times_normalized = times_array - times_array[0]
            
            # Least squares fit
            A = np.vstack([times_normalized, np.ones(len(times_normalized))]).T
            slope, intercept = np.linalg.lstsq(A, amplitudes_array, rcond=None)[0]
            
            # Positive slope = approaching, negative = receding
            approaching = slope > 0
            
            # Estimate distance based on amplitude (inverse square law approximation)
            # Assuming calibration: amplitude = k / distance^2
            # Use median amplitude for stability
            median_amp = np.median(amplitudes_array)
            
            # Rough calibration: 0.1 amplitude at 50m
            calibration_constant = 0.1 * (50 ** 2)
            estimated_distance = np.sqrt(calibration_constant / max(median_amp, 0.001))
            estimated_distance = np.clip(estimated_distance, 10, 150)
            
            # Estimate velocity (rate of distance change)
            # From amplitude change rate and distance
            velocity_estimate = abs(slope) * estimated_distance / 2
            velocity_estimate = np.clip(velocity_estimate, 0, 5)  # Max 5 m/s for elephant
            
            # Confidence based on consistency of trend
            residuals = amplitudes_array - (slope * times_normalized + intercept)
            r_squared = 1 - (np.sum(residuals**2) / np.sum((amplitudes_array - np.mean(amplitudes_array))**2))
            confidence = max(0.0, min(1.0, r_squared))
            
            # Direction (limited info from single sensor)
            # Assume sensor orientation is known
            if approaching:
                direction_cardinal = "Towards Sensor"
                direction_degrees = 180.0  # Arbitrary, sensor-relative
            else:
                direction_cardinal = "Away from Sensor"
                direction_degrees = 0.0
            
            return MovementDirection(
                direction_degrees=direction_degrees,
                direction_cardinal=direction_cardinal,
                confidence=confidence,
                approaching=approaching,
                estimated_distance=estimated_distance,
                velocity=velocity_estimate
            )
        
        return MovementDirection(
            direction_degrees=0.0,
            direction_cardinal='Unknown',
            confidence=0.0,
            approaching=False,
            estimated_distance=50.0,
            velocity=0.0
        )
    
    def detect_direction_multi_sensor(self,
                                     signals: List[Tuple[np.ndarray, Tuple[float, float]]],
                                     timestamp: float) -> MovementDirection:
        """
        Detect direction using multiple geophones (triangulation)
        
        Args:
            signals: List of (signal_data, (x_position, y_position)) tuples
            timestamp: Current time
            
        Returns:
            MovementDirection with accurate bearing
        """
        if len(signals) < 3:
            return self.detect_direction_single_sensor(signals[0][0], timestamp)
        
        # Time-of-arrival analysis
        arrival_times = []
        positions = []
        
        for signal, position in signals:
            # Detect peak arrival time (cross-correlation with template)
            peak_idx = np.argmax(np.abs(signal))
            arrival_time = peak_idx / self.fs
            arrival_times.append(arrival_time)
            positions.append(position)
        
        # Triangulation using time differences
        # This requires solving hyperbolic equations
        # Simplified approach: use time differences to estimate direction
        
        arrival_times = np.array(arrival_times)
        positions = np.array(positions)
        
        # Reference to first sensor
        time_diffs = arrival_times - arrival_times[0]
        
        # Wave speed in soil (approximately 100-300 m/s depending on type)
        wave_speed = 200.0  # m/s, typical for dry soil
        
        # Distance differences
        distance_diffs = time_diffs * wave_speed
        
        # Use least squares to estimate source position
        # For simplicity, assume 2D plane
        if len(positions) >= 3:
            # Solve for (x, y) position
            A = []
            b = []
            
            for i in range(1, len(positions)):
                x_i, y_i = positions[i]
                x_0, y_0 = positions[0]
                d_i = distance_diffs[i]
                
                # Linearized equation
                A.append([2*(x_i - x_0), 2*(y_i - y_0)])
                b.append(d_i**2 + x_0**2 + y_0**2 - x_i**2 - y_i**2)
            
            A = np.array(A)
            b = np.array(b)
            
            try:
                # Solve least squares
                source_pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                source_x, source_y = source_pos
                
                # Calculate direction from sensor array center
                center_x = np.mean(positions[:, 0])
                center_y = np.mean(positions[:, 1])
                
                # Bearing calculation
                dx = source_x - center_x
                dy = source_y - center_y
                
                # Convert to compass bearing (0 = North, clockwise)
                bearing = np.degrees(np.arctan2(dx, dy))
                if bearing < 0:
                    bearing += 360
                
                # Cardinal direction
                cardinal_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                cardinal_idx = int((bearing + 22.5) / 45) % 8
                cardinal = cardinal_dirs[cardinal_idx]
                
                # Distance to source
                distance = np.sqrt(dx**2 + dy**2)
                
                # Check if approaching (compare with previous detection)
                approaching = False
                if len(self.signal_history) > 0:
                    prev_distance = self.signal_history[-1].get('distance', distance)
                    approaching = distance < prev_distance
                
                # Velocity estimate
                if len(self.detection_times) >= 2:
                    time_delta = timestamp - self.detection_times[-2]
                    if time_delta > 0:
                        distance_delta = distance - self.signal_history[-1].get('distance', distance)
                        velocity = abs(distance_delta) / time_delta
                    else:
                        velocity = 0.0
                else:
                    velocity = 0.0
                
                # Store history
                self.signal_history.append({
                    'distance': distance,
                    'bearing': bearing,
                    'timestamp': timestamp
                })
                
                return MovementDirection(
                    direction_degrees=bearing,
                    direction_cardinal=cardinal,
                    confidence=0.85,  # High confidence with multiple sensors
                    approaching=approaching,
                    estimated_distance=distance,
                    velocity=velocity
                )
                
            except np.linalg.LinAlgError:
                pass
        
        # Fallback to single sensor method
        return self.detect_direction_single_sensor(signals[0][0], timestamp)


class BehaviorAnalyzer:
    """
    Analyze elephant behavior from seismic signature characteristics
    """
    
    def __init__(self, sampling_rate: int = 1000):
        self.fs = sampling_rate
        self.gait_history = deque(maxlen=10)
        
    def analyze_behavior(self, 
                        signal: np.ndarray,
                        direction_info: MovementDirection) -> ElephantBehavior:
        """
        Classify elephant behavior from signal characteristics
        
        Behavior indicators:
        - Walking: Regular footfalls, 1-2 Hz step frequency
        - Running: Faster footfalls, 2-4 Hz, higher amplitude
        - Feeding: Irregular low-frequency movements, stationary
        - Standing: Minimal movement, low amplitude
        - Bathing: Splashing patterns, irregular bursts
        """
        # Extract features
        rms = np.sqrt(np.mean(signal ** 2))
        
        # Detect periodicity (footfall rhythm)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find dominant period
        peaks = []
        for i in range(int(0.3 * self.fs), int(2.0 * self.fs)):  # 0.3 to 2 seconds
            if i > 0 and i < len(autocorr) - 1:
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.3 * np.max(autocorr):
                        peaks.append((i, autocorr[i]))
        
        if peaks:
            dominant_period_idx = max(peaks, key=lambda x: x[1])[0]
            step_frequency = self.fs / dominant_period_idx
        else:
            step_frequency = 0.0
        
        # Spectral analysis
        fft_signal = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/self.fs)
        power_spectrum = np.abs(fft_signal) ** 2
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(power_spectrum[1:100]) + 1  # Skip DC
        dominant_freq = freqs[dominant_freq_idx]
        
        # Signal variability
        signal_std = np.std(signal)
        
        # Classify behavior
        behavior_type = 'unknown'
        confidence = 0.5
        gait_speed = 0.0
        activity_level = 'calm'
        
        # Walking: Regular steps, 1-2 Hz
        if 0.8 <= step_frequency <= 2.0 and signal_std > 0.005:
            behavior_type = 'walking'
            gait_speed = step_frequency * 1.5  # Approximate: step_freq * stride_length
            confidence = 0.8
            activity_level = 'calm' if step_frequency < 1.3 else 'moderate'
        
        # Running: Faster steps, 2-4 Hz, high amplitude
        elif 2.0 < step_frequency <= 4.0 and rms > 0.08:
            behavior_type = 'running'
            gait_speed = step_frequency * 2.0
            confidence = 0.85
            activity_level = 'agitated'
        
        # Feeding: Low frequency, irregular, stationary
        elif step_frequency < 0.5 and dominant_freq < 5.0 and not direction_info.approaching:
            behavior_type = 'feeding'
            gait_speed = 0.0
            confidence = 0.6
            activity_level = 'calm'
        
        # Standing: Very low amplitude, minimal movement
        elif rms < 0.02 and signal_std < 0.01:
            behavior_type = 'standing'
            gait_speed = 0.0
            confidence = 0.7
            activity_level = 'calm'
        
        # Bathing: Burst patterns, mid-frequency content
        elif signal_std > 0.02 and 5 < dominant_freq < 20:
            behavior_type = 'bathing'
            gait_speed = 0.0
            confidence = 0.5
            activity_level = 'moderate'
        
        # Default to walking if moving
        elif direction_info.velocity > 0.5:
            behavior_type = 'walking'
            gait_speed = direction_info.velocity
            confidence = 0.6
            activity_level = 'moderate'
        
        # Estimate weight from signal amplitude (requires calibration)
        # Heavier elephants produce stronger signals
        # Rough approximation: amplitude correlates with weight
        estimated_weight = 3000 + (rms * 10000)  # Very rough estimate
        estimated_weight = np.clip(estimated_weight, 2500, 6000)
        
        self.gait_history.append({
            'behavior': behavior_type,
            'step_frequency': step_frequency,
            'timestamp': time.time()
        })
        
        return ElephantBehavior(
            behavior_type=behavior_type,
            confidence=confidence,
            gait_speed=gait_speed,
            estimated_weight=estimated_weight,
            activity_level=activity_level
        )
    
    def get_behavior_sequence(self) -> List[str]:
        """Get recent behavior sequence"""
        return [item['behavior'] for item in self.gait_history]
    
    def predict_next_behavior(self) -> Tuple[str, float]:
        """
        Predict likely next behavior based on sequence
        
        Behavioral transitions:
        - Walking → Walking (70%), Feeding (15%), Standing (10%), Running (5%)
        - Running → Running (60%), Walking (30%), Standing (10%)
        - Feeding → Feeding (80%), Walking (15%), Standing (5%)
        - Standing → Standing (60%), Walking (30%), Feeding (10%)
        """
        if len(self.gait_history) == 0:
            return 'unknown', 0.0
        
        current_behavior = self.gait_history[-1]['behavior']
        
        # Simple Markov chain
        transitions = {
            'walking': [('walking', 0.70), ('feeding', 0.15), ('standing', 0.10), ('running', 0.05)],
            'running': [('running', 0.60), ('walking', 0.30), ('standing', 0.10)],
            'feeding': [('feeding', 0.80), ('walking', 0.15), ('standing', 0.05)],
            'standing': [('standing', 0.60), ('walking', 0.30), ('feeding', 0.10)],
            'bathing': [('bathing', 0.70), ('walking', 0.20), ('standing', 0.10)]
        }
        
        if current_behavior in transitions:
            next_behavior, prob = transitions[current_behavior][0]
            return next_behavior, prob
        
        return 'unknown', 0.0


def demo_direction_and_behavior():
    """Demonstrate direction detection and behavior analysis"""
    
    print("=" * 70)
    print("ELEPHANT DIRECTION & BEHAVIOR ANALYSIS - Demo")
    print("=" * 70)
    print()
    
    # Initialize
    direction_detector = DirectionDetector()
    behavior_analyzer = BehaviorAnalyzer()
    
    # Simulate elephant approaching sensor
    print("Scenario 1: Elephant Approaching Sensor")
    print("-" * 70)
    
    distances = [80, 70, 60, 50, 40, 30]  # Approaching
    for i, dist in enumerate(distances):
        # Simulate signal (amplitude increases as elephant approaches)
        amplitude = 0.1 * (50 / dist) ** 2
        t = np.linspace(0, 1, 1000)
        # Walking pattern: 1.2 Hz footfall
        signal = amplitude * (np.sin(2 * np.pi * 10 * t) * 
                             (1 + 0.5 * np.sin(2 * np.pi * 1.2 * t)))
        signal += np.random.normal(0, amplitude * 0.1, len(signal))
        
        timestamp = time.time() + i
        
        # Detect direction
        direction = direction_detector.detect_direction_single_sensor(signal, timestamp)
        
        # Analyze behavior
        behavior = behavior_analyzer.analyze_behavior(signal, direction)
        
        print(f"\nTime {i+1}s:")
        print(f"  Direction: {direction.direction_cardinal}")
        print(f"  Approaching: {direction.approaching}")
        print(f"  Estimated Distance: {direction.estimated_distance:.1f}m")
        print(f"  Velocity: {direction.velocity:.2f} m/s")
        print(f"  Behavior: {behavior.behavior_type} ({behavior.confidence:.1%} confidence)")
        print(f"  Gait Speed: {behavior.gait_speed:.2f} m/s")
        print(f"  Activity: {behavior.activity_level}")
        print(f"  Est. Weight: {behavior.estimated_weight:.0f} kg")
    
    # Predict next behavior
    next_behavior, prob = behavior_analyzer.predict_next_behavior()
    print(f"\n  Predicted Next: {next_behavior} ({prob:.0%} probability)")
    
    print("\n" + "=" * 70)
    print("Scenario 2: Elephant Moving Away")
    print("-" * 70)
    
    # Reset
    direction_detector = DirectionDetector()
    
    distances = [30, 40, 50, 60, 70]  # Moving away
    for i, dist in enumerate(distances):
        amplitude = 0.1 * (50 / dist) ** 2
        t = np.linspace(0, 1, 1000)
        signal = amplitude * (np.sin(2 * np.pi * 10 * t) * 
                             (1 + 0.5 * np.sin(2 * np.pi * 1.2 * t)))
        signal += np.random.normal(0, amplitude * 0.1, len(signal))
        
        timestamp = time.time() + i + 10
        direction = direction_detector.detect_direction_single_sensor(signal, timestamp)
        
        print(f"\nTime {i+1}s:")
        print(f"  Approaching: {direction.approaching}")
        print(f"  Distance: {direction.estimated_distance:.1f}m")
        print(f"  Velocity: {direction.velocity:.2f} m/s away")
    
    print("\n" + "=" * 70)
    print("✓ Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    demo_direction_and_behavior()
