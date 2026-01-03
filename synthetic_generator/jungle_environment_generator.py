"""
Real-world jungle environment signal generator
Combines multiple realistic vibration sources
"""

import numpy as np
from typing import Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_generator.seismic_signal_generator import SeismicSignalGenerator, SoilConditions


class JungleEnvironmentGenerator:
    """
    Generate realistic jungle environment with multiple vibration sources
    """
    
    def __init__(self, fs: int = 1000):
        self.fs = fs
        self.generator = SeismicSignalGenerator(sampling_rate=fs)
        
    def generate_realistic_jungle_with_elephant(self, 
                                                duration: float = 1.0,
                                                elephant_distance: float = 40.0,
                                                elephant_weight: float = 4000.0,
                                                soil_moisture: float = 20.0,
                                                time_of_day: str = 'day') -> Tuple[np.ndarray, Dict]:
        """
        Generate realistic jungle environment with elephant present
        
        Args:
            duration: Signal duration in seconds
            elephant_distance: Distance to elephant in meters
            elephant_weight: Elephant weight in kg
            soil_moisture: Soil moisture percentage
            time_of_day: 'day', 'night', 'dawn', 'dusk'
        """
        # Set soil conditions
        self.generator.set_soil_conditions(moisture=soil_moisture)
        
        # Generate base signal with elephant
        num_steps = max(2, int(duration * 1.5))  # Elephant walking speed
        elephant_signal, _ = self.generator.generate_elephant_footfall(
            duration=duration,
            num_steps=num_steps,
            distance_m=elephant_distance,
            elephant_weight_kg=elephant_weight,
            snr_db=18.0  # Higher SNR - elephants create significant ground vibration
        )
        
        # Add ambient jungle noise layers
        jungle_noise = self._generate_jungle_ambience(duration, time_of_day, soil_moisture)
        
        # Add random environmental vibrations
        environmental = self._generate_environmental_events(duration, soil_moisture)
        
        # Combine all sources
        combined_signal = elephant_signal + jungle_noise + environmental
        
        # Add sensor noise (ADC quantization, thermal noise)
        sensor_noise = np.random.normal(0, 0.0005, len(combined_signal))
        combined_signal += sensor_noise
        
        metadata = {
            'type': 'jungle_with_elephant',
            'elephant_distance': elephant_distance,
            'elephant_weight': elephant_weight,
            'soil_moisture': soil_moisture,
            'time_of_day': time_of_day,
            'dominant_frequency': '8-15 Hz (elephant)',
            'snr_db': 12.0,
            'has_elephant': True
        }
        
        return combined_signal, metadata
    
    def generate_realistic_jungle_without_elephant(self,
                                                   duration: float = 1.0,
                                                   soil_moisture: float = 20.0,
                                                   time_of_day: str = 'day',
                                                   activity_level: str = 'medium') -> Tuple[np.ndarray, Dict]:
        """
        Generate realistic jungle environment WITHOUT elephant
        
        Args:
            duration: Signal duration in seconds
            soil_moisture: Soil moisture percentage
            time_of_day: 'day', 'night', 'dawn', 'dusk'
            activity_level: 'low', 'medium', 'high' - amount of animal/environmental activity
        """
        # Set soil conditions
        self.generator.set_soil_conditions(moisture=soil_moisture)
        
        # Base jungle ambience
        jungle_noise = self._generate_jungle_ambience(duration, time_of_day, soil_moisture)
        
        # Environmental vibrations (more varied without elephant)
        environmental = self._generate_environmental_events(duration, soil_moisture, activity_level)
        
        # Possible small animals (birds landing, monkeys jumping, deer, etc.)
        small_animals = self._generate_small_animal_vibrations(duration, activity_level)
        
        # Sensor noise
        sensor_noise = np.random.normal(0, 0.0005, len(jungle_noise))
        
        # Combine
        combined_signal = jungle_noise + environmental + small_animals + sensor_noise
        
        metadata = {
            'type': 'jungle_without_elephant',
            'soil_moisture': soil_moisture,
            'time_of_day': time_of_day,
            'activity_level': activity_level,
            'has_elephant': False
        }
        
        return combined_signal, metadata
    
    def _generate_jungle_ambience(self, duration: float, time_of_day: str, 
                                 soil_moisture: float) -> np.ndarray:
        """
        Generate continuous jungle background vibration
        Includes: wind through trees, distant waterfalls, ground settling
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        
        # Wind-induced vibrations (1-5 Hz) - reduced to not mask elephant signals
        wind_base, _ = self.generator.generate_wind_vibration(duration=duration)
        wind_amplitude = 0.15 if time_of_day == 'night' else 0.25
        wind_component = wind_base * wind_amplitude
        
        # Rain if wet soil (random probability) - light rain, not heavy downpour
        rain_component = np.zeros_like(t)
        if soil_moisture > 25 and np.random.random() < 0.3:
            rain_signal, _ = self.generator.generate_rain_impact(duration=duration)
            rain_component = rain_signal * 0.15  # Light rain only
        
        # Low frequency ground rumble (very low amplitude continuous)
        rumble_freq = np.random.uniform(0.5, 2.0)
        rumble = np.random.normal(0, 0.001, len(t)) * np.sin(2 * np.pi * rumble_freq * t)
        
        # Brownian motion (1/f noise - characteristic of natural environments)
        brownian = self._generate_1f_noise(len(t)) * 0.001  # Reduced ambient
        
        return wind_component + rain_component + rumble + brownian
    
    def _generate_environmental_events(self, duration: float, soil_moisture: float,
                                      activity_level: str = 'medium') -> np.ndarray:
        """
        Generate random environmental vibration events
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        signal = np.zeros_like(t)
        
        # Activity level determines number of events
        activity_rates = {'low': 0.5, 'medium': 1.5, 'high': 3.0}
        num_events = int(np.random.poisson(activity_rates.get(activity_level, 1.5)))
        
        for _ in range(num_events):
            event_type = np.random.choice([
                'branch_fall',
                'fruit_drop', 
                'tree_creak',
                'distant_animal',
                'ground_shift'
            ])
            
            # Random time for event
            event_time = np.random.uniform(0.1, duration - 0.1)
            event_idx = int(event_time * self.fs)
            
            if event_type == 'branch_fall':
                # Impact + settling
                impact_duration = 0.1
                impact_len = int(impact_duration * self.fs)
                impact = np.exp(-np.linspace(0, 10, impact_len)) * np.random.uniform(0.002, 0.008)
                if event_idx + impact_len < len(signal):
                    signal[event_idx:event_idx+impact_len] += impact
                    
            elif event_type == 'fruit_drop':
                # Small impact
                impact_len = int(0.05 * self.fs)
                impact = np.exp(-np.linspace(0, 20, impact_len)) * np.random.uniform(0.001, 0.003)
                if event_idx + impact_len < len(signal):
                    signal[event_idx:event_idx+impact_len] += impact
                    
            elif event_type == 'tree_creak':
                # Low frequency oscillation
                creak_duration = 0.3
                creak_len = int(creak_duration * self.fs)
                creak_t = np.linspace(0, creak_duration, creak_len)
                creak_freq = np.random.uniform(3, 8)
                creak = np.sin(2 * np.pi * creak_freq * creak_t) * \
                       np.exp(-creak_t * 5) * np.random.uniform(0.001, 0.004)
                if event_idx + creak_len < len(signal):
                    signal[event_idx:event_idx+creak_len] += creak
                    
            elif event_type == 'distant_animal':
                # Random vibration burst
                burst_len = int(0.2 * self.fs)
                burst = np.random.normal(0, 0.002, burst_len) * \
                       np.hamming(burst_len)
                if event_idx + burst_len < len(signal):
                    signal[event_idx:event_idx+burst_len] += burst
        
        return signal
    
    def _generate_small_animal_vibrations(self, duration: float, 
                                         activity_level: str) -> np.ndarray:
        """
        Generate vibrations from small animals (birds, monkeys, rodents, etc.)
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        signal = np.zeros_like(t)
        
        # Activity determines frequency
        if activity_level == 'high':
            num_animals = np.random.randint(2, 5)
        elif activity_level == 'medium':
            num_animals = np.random.randint(0, 3)
        else:
            num_animals = np.random.randint(0, 2)
        
        for _ in range(num_animals):
            animal_type = np.random.choice([
                'bird_landing',
                'monkey_jump',
                'rodent_scurry',
                'lizard_movement'
            ])
            
            event_time = np.random.uniform(0, duration - 0.2)
            event_idx = int(event_time * self.fs)
            
            if animal_type == 'bird_landing':
                # Light impact, high frequency
                impact_len = int(0.03 * self.fs)
                impact = np.exp(-np.linspace(0, 30, impact_len)) * np.random.uniform(0.0005, 0.002)
                freq = np.random.uniform(40, 80)
                t_impact = np.linspace(0, 0.03, impact_len)
                impact *= np.sin(2 * np.pi * freq * t_impact)
                if event_idx + impact_len < len(signal):
                    signal[event_idx:event_idx+impact_len] += impact
                    
            elif animal_type == 'monkey_jump':
                # Medium impact
                impact_len = int(0.08 * self.fs)
                impact = np.exp(-np.linspace(0, 15, impact_len)) * np.random.uniform(0.002, 0.005)
                if event_idx + impact_len < len(signal):
                    signal[event_idx:event_idx+impact_len] += impact
                    
            elif animal_type in ['rodent_scurry', 'lizard_movement']:
                # Multiple small impacts
                num_steps = np.random.randint(3, 8)
                step_interval = int(0.05 * self.fs)
                for i in range(num_steps):
                    step_idx = event_idx + i * step_interval
                    impact_len = int(0.02 * self.fs)
                    if step_idx + impact_len < len(signal):
                        impact = np.exp(-np.linspace(0, 25, impact_len)) * \
                                np.random.uniform(0.0003, 0.001)
                        signal[step_idx:step_idx+impact_len] += impact
        
        return signal
    
    def _generate_1f_noise(self, length: int) -> np.ndarray:
        """
        Generate 1/f (pink) noise - characteristic of natural processes
        """
        # Generate white noise in frequency domain
        white = np.fft.rfft(np.random.randn(length))
        
        # Create 1/f spectrum
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1  # Avoid division by zero
        pink_spectrum = white / np.sqrt(freqs)
        
        # Convert back to time domain
        pink = np.fft.irfft(pink_spectrum, n=length)
        
        # Normalize
        pink = pink / np.std(pink)
        
        return pink
    
    def generate_realistic_cattle_near_elephant_path(self, 
                                                     duration: float = 1.0,
                                                     num_cattle: int = 3,
                                                     soil_moisture: float = 20.0) -> Tuple[np.ndarray, Dict]:
        """
        Generate cattle movement in area where elephants also roam
        This is a challenging scenario for false positive testing
        """
        self.generator.set_soil_conditions(moisture=soil_moisture)
        
        # Generate cattle movement
        cattle_signal, _ = self.generator.generate_cattle_movement(duration=duration)
        
        # Add jungle ambience
        jungle_noise = self._generate_jungle_ambience(duration, 'day', soil_moisture)
        
        # Add environmental sounds
        environmental = self._generate_environmental_events(duration, soil_moisture, 'medium')
        
        # Sensor noise
        sensor_noise = np.random.normal(0, 0.0005, len(cattle_signal))
        
        combined = cattle_signal + jungle_noise + environmental + sensor_noise
        
        metadata = {
            'type': 'cattle_in_jungle',
            'num_cattle': num_cattle,
            'soil_moisture': soil_moisture,
            'has_elephant': False,
            'challenging_case': True
        }
        
        return combined, metadata


def demo_jungle_scenarios():
    """Demonstrate various jungle scenarios"""
    
    print("=" * 70)
    print("JUNGLE ENVIRONMENT GENERATOR - Demo")
    print("=" * 70)
    print()
    
    generator = JungleEnvironmentGenerator()
    
    scenarios = [
        ("Daytime jungle with elephant (40m away)", 
         lambda: generator.generate_realistic_jungle_with_elephant(
             duration=2.0, elephant_distance=40, soil_moisture=20, time_of_day='day')),
        
        ("Nighttime jungle with close elephant (25m)", 
         lambda: generator.generate_realistic_jungle_with_elephant(
             duration=2.0, elephant_distance=25, soil_moisture=15, time_of_day='night')),
        
        ("Wet jungle with distant elephant (70m)", 
         lambda: generator.generate_realistic_jungle_with_elephant(
             duration=2.0, elephant_distance=70, soil_moisture=35, time_of_day='day')),
        
        ("Active jungle WITHOUT elephant", 
         lambda: generator.generate_realistic_jungle_without_elephant(
             duration=2.0, soil_moisture=20, time_of_day='day', activity_level='high')),
        
        ("Quiet night jungle WITHOUT elephant", 
         lambda: generator.generate_realistic_jungle_without_elephant(
             duration=2.0, soil_moisture=18, time_of_day='night', activity_level='low')),
        
        ("Cattle near elephant path (challenging)", 
         lambda: generator.generate_realistic_cattle_near_elephant_path(
             duration=2.0, num_cattle=3, soil_moisture=22))
    ]
    
    for desc, gen_func in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {desc}")
        print(f"{'='*70}")
        
        signal, metadata = gen_func()
        
        print(f"\nSignal Statistics:")
        print(f"  Length: {len(signal)} samples ({len(signal)/1000:.2f} seconds)")
        print(f"  RMS: {np.sqrt(np.mean(signal**2)):.6f}")
        print(f"  Peak-to-Peak: {np.max(signal) - np.min(signal):.6f}")
        print(f"  Peak: {np.max(np.abs(signal)):.6f}")
        
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Spectral content
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/1000)
        dominant_freq_idx = np.argmax(np.abs(fft[1:100])) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        print(f"\nSpectral Analysis:")
        print(f"  Dominant Frequency: {dominant_freq:.2f} Hz")
        
        # Energy in elephant range (5-20 Hz)
        elephant_range = (freqs >= 5) & (freqs <= 20)
        elephant_energy = np.sum(np.abs(fft[elephant_range])**2)
        total_energy = np.sum(np.abs(fft)**2)
        elephant_pct = (elephant_energy / total_energy) * 100
        
        print(f"  Energy in elephant range (5-20 Hz): {elephant_pct:.1f}%")


if __name__ == "__main__":
    demo_jungle_scenarios()
