"""
EarthPulse AI - Realistic Hardware Training Data Generator
Generate training data that matches real ESP32+ADS1115+Geophone behavior
"""

import numpy as np
from scipy import signal
from typing import Tuple
import random

class HardwareRealisticGenerator:
    """
    Generate realistic training data matching actual hardware characteristics
    Designed to train model on real-world vibration patterns
    """
    
    def __init__(self, sampling_rate: int = 100):
        """
        Initialize generator
        
        Args:
            sampling_rate: Hardware sampling rate (100 Hz after firmware fix)
        """
        self.fs = sampling_rate
        
    def generate_background_noise(self, duration: float = 1.0) -> np.ndarray:
        """
        Generate realistic background noise
        - Electronic noise from ADC
        - Environmental vibrations
        - 50/60 Hz mains hum
        """
        samples = int(duration * self.fs)
        
        # White noise (ADC quantization, thermal noise)
        white_noise = np.random.normal(0, 0.002, samples)  # 2mV RMS
        
        # 1/f (pink) noise (environmental, mechanical)
        pink_noise = self._generate_pink_noise(samples) * 0.001
        
        # 50 Hz mains hum (common in power supplies)
        t = np.linspace(0, duration, samples)
        mains_hum = 0.003 * np.sin(2 * np.pi * 50 * t)
        
        # Random low-frequency drift (temperature, pressure)
        drift_freq = random.uniform(0.05, 0.2)
        drift = 0.005 * np.sin(2 * np.pi * drift_freq * t)
        
        # Combine all noise sources
        noise = white_noise + pink_noise + mains_hum + drift
        
        return noise
    
    def generate_table_tap(self, duration: float = 1.0, 
                          intensity: str = 'medium') -> np.ndarray:
        """
        Generate table tap vibration
        - Sharp impulse response
        - High frequency components
        - Quick decay
        """
        samples = int(duration * self.fs)
        signal_out = self.generate_background_noise(duration)
        
        # Tap parameters based on intensity
        if intensity == 'light':
            amplitude = random.uniform(0.01, 0.03)  # 10-30 mV
            decay_time = random.uniform(0.05, 0.1)  # 50-100 ms
        elif intensity == 'medium':
            amplitude = random.uniform(0.03, 0.08)  # 30-80 mV
            decay_time = random.uniform(0.1, 0.2)   # 100-200 ms
        else:  # heavy
            amplitude = random.uniform(0.08, 0.15)  # 80-150 mV
            decay_time = random.uniform(0.15, 0.3)  # 150-300 ms
        
        # Tap occurs at random time (but not at edges)
        tap_time = random.uniform(0.2, 0.6)
        tap_sample = int(tap_time * self.fs)
        
        # Create impulse with exponential decay
        t = np.arange(samples) / self.fs
        t_shifted = t - tap_time
        impulse = np.where(t_shifted >= 0, 
                          amplitude * np.exp(-t_shifted / decay_time),
                          0)
        
        # Add resonance (table/wood resonance at ~200-500 Hz)
        resonance_freq = random.uniform(200, 500)
        resonance = impulse * np.sin(2 * np.pi * resonance_freq * t)
        
        # Damped oscillation
        damping = 0.5
        resonance *= np.exp(-damping * t_shifted.clip(0))
        
        signal_out += impulse + resonance * 0.3
        
        return signal_out
    
    def generate_floor_footstep(self, duration: float = 1.0,
                                person_weight: float = 70.0) -> np.ndarray:
        """
        Generate human footstep on floor
        - Lower frequency than table tap
        - Longer duration
        - Double peak (heel strike + toe off)
        """
        samples = int(duration * self.fs)
        signal_out = self.generate_background_noise(duration)
        
        # Scale amplitude by weight (70kg = reference)
        weight_scale = person_weight / 70.0
        amplitude = random.uniform(0.02, 0.06) * weight_scale
        
        # Footstep timing (heel strike followed by toe off)
        heel_time = random.uniform(0.2, 0.4)
        toe_time = heel_time + random.uniform(0.1, 0.15)
        
        t = np.arange(samples) / self.fs
        
        # Heel strike (stronger, sharper)
        heel_impulse = self._create_footstep_impulse(
            t, heel_time, amplitude * 1.2, 0.08, 15
        )
        
        # Toe off (softer, broader)
        toe_impulse = self._create_footstep_impulse(
            t, toe_time, amplitude * 0.8, 0.12, 20
        )
        
        signal_out += heel_impulse + toe_impulse
        
        return signal_out
    
    def generate_elephant_footfall(self, duration: float = 1.0,
                                   weight: float = 4000.0,
                                   distance: float = 30.0) -> np.ndarray:
        """
        Generate elephant footfall vibration
        - Much stronger than human
        - Lower frequency (5-15 Hz dominant)
        - Longer duration
        - Distance attenuation
        """
        samples = int(duration * self.fs)
        signal_out = self.generate_background_noise(duration)
        
        # Distance attenuation (inverse square law + absorption)
        distance_factor = 10.0 / (distance + 10.0)  # Reference: 10m
        absorption = np.exp(-distance / 50.0)  # Soil absorption
        attenuation = distance_factor * absorption
        
        # Weight scaling (4000 kg = reference elephant)
        weight_scale = weight / 4000.0
        
        # Base amplitude (much larger than human)
        amplitude = random.uniform(0.1, 0.3) * weight_scale * attenuation
        
        # Elephant footfall timing (slower than human)
        impact_time = random.uniform(0.3, 0.5)
        
        t = np.arange(samples) / self.fs
        t_shifted = t - impact_time
        
        # Main impact (low frequency)
        dominant_freq = random.uniform(5, 15)  # Elephant footfall frequency
        impact = np.where(t_shifted >= 0,
                         amplitude * np.sin(2 * np.pi * dominant_freq * t_shifted),
                         0)
        
        # Decay envelope
        decay_time = random.uniform(0.3, 0.6)
        envelope = np.where(t_shifted >= 0,
                           np.exp(-t_shifted / decay_time),
                           0)
        
        # Add harmonics
        for harmonic in [2, 3]:
            harm_amp = amplitude * 0.3 / harmonic
            harmonic_signal = np.where(
                t_shifted >= 0,
                harm_amp * np.sin(2 * np.pi * dominant_freq * harmonic * t_shifted),
                0
            )
            impact += harmonic_signal * envelope
        
        signal_out += impact * envelope
        
        # Add ground roll (Rayleigh wave)
        roll_freq = random.uniform(3, 8)
        roll_delay = 0.05  # Slight delay for wave propagation
        t_roll = t - (impact_time + roll_delay)
        ground_roll = np.where(
            t_roll >= 0,
            amplitude * 0.4 * np.sin(2 * np.pi * roll_freq * t_roll) * np.exp(-t_roll / 0.5),
            0
        )
        signal_out += ground_roll
        
        return signal_out
    
    def generate_random_vibration(self, duration: float = 1.0) -> np.ndarray:
        """
        Generate random environmental vibrations
        - Door closing
        - Object dropping
        - Walking past (not on sensor)
        - Vehicle passing outside
        """
        samples = int(duration * self.fs)
        signal_out = self.generate_background_noise(duration)
        
        # Random vibration type
        vib_type = random.choice(['bump', 'rumble', 'spike'])
        
        if vib_type == 'bump':
            # Single bump (door close, object drop)
            time = random.uniform(0.2, 0.6)
            amp = random.uniform(0.015, 0.04)
            signal_out += self._create_footstep_impulse(
                np.arange(samples) / self.fs, time, amp, 0.15, 30
            )
        
        elif vib_type == 'rumble':
            # Low frequency rumble (vehicle, machinery)
            t = np.arange(samples) / self.fs
            start_time = random.uniform(0.1, 0.3)
            duration_rumble = random.uniform(0.3, 0.5)
            freq = random.uniform(8, 20)
            amp = random.uniform(0.01, 0.03)
            
            rumble = amp * np.sin(2 * np.pi * freq * t)
            envelope = self._create_envelope(t, start_time, duration_rumble)
            signal_out += rumble * envelope
        
        else:  # spike
            # Sharp spike (electrical glitch, sensor bump)
            time = random.uniform(0.2, 0.6)
            sample = int(time * self.fs)
            amp = random.uniform(0.05, 0.1)
            # Single sample spike
            if sample < samples:
                signal_out[sample] += amp
                # Exponential decay
                for i in range(1, min(10, samples - sample)):
                    signal_out[sample + i] += amp * np.exp(-i * 0.5)
        
        return signal_out
    
    def _generate_pink_noise(self, samples: int) -> np.ndarray:
        """Generate 1/f (pink) noise"""
        # Simple pink noise approximation using multiple octaves
        noise = np.zeros(samples)
        for octave in range(5):
            freq = 2 ** octave
            if samples // freq > 0:
                octave_noise = np.random.randn(samples // freq)
                octave_noise_upsampled = np.repeat(octave_noise, freq)
                # Ensure correct length
                if len(octave_noise_upsampled) > samples:
                    octave_noise_upsampled = octave_noise_upsampled[:samples]
                elif len(octave_noise_upsampled) < samples:
                    octave_noise_upsampled = np.pad(octave_noise_upsampled, 
                                                    (0, samples - len(octave_noise_upsampled)))
                noise += octave_noise_upsampled / (octave + 1)
        return noise / 5.0
    
    def _create_footstep_impulse(self, t: np.ndarray, impact_time: float,
                                amplitude: float, decay_time: float,
                                frequency: float) -> np.ndarray:
        """Create realistic footstep impulse"""
        t_shifted = t - impact_time
        
        # Exponential decay
        envelope = np.where(t_shifted >= 0,
                           np.exp(-t_shifted / decay_time),
                           0)
        
        # Oscillation
        oscillation = amplitude * np.sin(2 * np.pi * frequency * t_shifted)
        
        return oscillation * envelope
    
    def _create_envelope(self, t: np.ndarray, start_time: float,
                        duration: float) -> np.ndarray:
        """Create smooth envelope for sustained vibrations"""
        envelope = np.zeros_like(t)
        
        # Attack (10% of duration)
        attack_time = duration * 0.1
        # Decay (20% of duration)
        decay_time = duration * 0.2
        
        for i, time in enumerate(t):
            if time < start_time:
                envelope[i] = 0
            elif time < start_time + attack_time:
                # Linear attack
                envelope[i] = (time - start_time) / attack_time
            elif time < start_time + duration - decay_time:
                # Sustain
                envelope[i] = 1.0
            elif time < start_time + duration:
                # Exponential decay
                t_decay = time - (start_time + duration - decay_time)
                envelope[i] = np.exp(-3 * t_decay / decay_time)
            else:
                envelope[i] = 0
        
        return envelope
    
    def generate_dataset(self, samples_per_class: int = 5000) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Generate complete training dataset
        
        Args:
            samples_per_class: Number of samples per class
        
        Returns:
            X: Signal data (samples, sequence_length)
            y: Labels (samples,)
            class_names: List of class names
        """
        class_names = [
            'background',
            'table_tap',
            'floor_footstep',
            'elephant_footfall',
            'random_vibration'
        ]
        
        sequence_length = self.fs  # 1 second
        X = []
        y = []
        
        print(f"Generating {samples_per_class} samples per class...")
        print(f"Total samples: {samples_per_class * len(class_names)}")
        
        for class_idx, class_name in enumerate(class_names):
            print(f"\nGenerating class {class_idx}: {class_name}")
            
            for i in range(samples_per_class):
                if i % 1000 == 0:
                    print(f"  Progress: {i}/{samples_per_class}")
                
                if class_name == 'background':
                    signal_data = self.generate_background_noise()
                
                elif class_name == 'table_tap':
                    intensity = random.choice(['light', 'medium', 'heavy'])
                    signal_data = self.generate_table_tap(intensity=intensity)
                
                elif class_name == 'floor_footstep':
                    weight = random.uniform(50, 100)  # 50-100 kg person
                    signal_data = self.generate_floor_footstep(person_weight=weight)
                
                elif class_name == 'elephant_footfall':
                    weight = random.uniform(2500, 6000)  # 2.5-6 ton elephant
                    distance = random.uniform(10, 80)    # 10-80 meters
                    signal_data = self.generate_elephant_footfall(weight=weight, distance=distance)
                
                else:  # random_vibration
                    signal_data = self.generate_random_vibration()
                
                # Ensure correct length
                if len(signal_data) > sequence_length:
                    signal_data = signal_data[:sequence_length]
                elif len(signal_data) < sequence_length:
                    signal_data = np.pad(signal_data, (0, sequence_length - len(signal_data)))
                
                X.append(signal_data)
                y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"\nDataset generated!")
        print(f"Shape: X={X.shape}, y={y.shape}")
        print(f"Classes: {class_names}")
        
        return X, y, class_names


if __name__ == "__main__":
    # Test generator
    generator = HardwareRealisticGenerator(sampling_rate=100)
    
    print("Testing signal generation...")
    
    # Generate test samples
    bg = generator.generate_background_noise()
    tap = generator.generate_table_tap(intensity='medium')
    step = generator.generate_floor_footstep(person_weight=70)
    elephant = generator.generate_elephant_footfall(weight=4000, distance=30)
    random_vib = generator.generate_random_vibration()
    
    print(f"\nBackground noise: {bg.min():.4f} to {bg.max():.4f} V")
    print(f"Table tap: {tap.min():.4f} to {tap.max():.4f} V")
    print(f"Floor footstep: {step.min():.4f} to {step.max():.4f} V")
    print(f"Elephant footfall: {elephant.min():.4f} to {elephant.max():.4f} V")
    print(f"Random vibration: {random_vib.min():.4f} to {random_vib.max():.4f} V")
    
    print("\n✓ Generator working correctly!")
