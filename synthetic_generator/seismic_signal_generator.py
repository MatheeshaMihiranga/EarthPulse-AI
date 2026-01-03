"""
EarthPulse AI - Synthetic Seismic Signal Generator
Simulates realistic geophone signals for elephant detection research
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SoilConditions:
    """Soil moisture and properties affecting signal propagation"""
    moisture_percent: float = 15.0  # 0-100%
    density_kg_m3: float = 1600.0
    clay_content: float = 0.3  # 0-1
    sand_content: float = 0.4  # 0-1
    
    @property
    def attenuation_factor(self) -> float:
        """Higher moisture = less attenuation, better propagation"""
        # Optimal moisture around 20-30%
        if self.moisture_percent < 5:
            return 0.3  # Very dry, poor propagation
        elif self.moisture_percent < 15:
            return 0.6
        elif self.moisture_percent < 30:
            return 1.0  # Optimal
        elif self.moisture_percent < 50:
            return 0.85
        else:
            return 0.6  # Too wet, muddy damping
    
    @property
    def velocity_m_s(self) -> float:
        """Seismic wave velocity through soil (m/s)"""
        # Increases with moisture up to saturation
        base_velocity = 200.0
        moisture_factor = 1.0 + (self.moisture_percent / 100.0) * 0.5
        return base_velocity * moisture_factor * (self.density_kg_m3 / 1600.0)


@dataclass
class SignalMetadata:
    """Metadata for each generated signal"""
    signal_type: str
    duration: float
    sampling_rate: int
    soil_conditions: Dict
    parameters: Dict
    timestamp: str
    snr_db: float


class SeismicSignalGenerator:
    """
    Generates synthetic seismic signals mimicking real geophone output
    Based on research literature on elephant seismic communication and detection
    """
    
    def __init__(self, sampling_rate: int = 1000):
        """
        Initialize generator
        
        Args:
            sampling_rate: Samples per second (Hz). 1000 Hz typical for geophones
        """
        self.fs = sampling_rate
        self.soil = SoilConditions()
        
    def set_soil_conditions(self, moisture: float, density: float = 1600.0):
        """Update soil conditions"""
        self.soil.moisture_percent = moisture
        self.soil.density_kg_m3 = density
        
    def _add_noise(self, signal_data: np.ndarray, snr_db: float) -> np.ndarray:
        """Add realistic noise to signal"""
        signal_power = np.mean(signal_data ** 2)
        signal_power_db = 10 * np.log10(signal_power)
        noise_power_db = signal_power_db - snr_db
        noise_power = 10 ** (noise_power_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal_data))
        return signal_data + noise
    
    def _apply_soil_attenuation(self, signal_data: np.ndarray, 
                                 distance_m: float) -> np.ndarray:
        """Apply distance and soil-dependent attenuation"""
        # Geometric spreading
        geometric_atten = 1.0 / (distance_m + 1.0)
        
        # Material absorption (frequency dependent)
        # Higher frequencies attenuate more
        freqs = fftfreq(len(signal_data), 1/self.fs)
        spectrum = fft(signal_data)
        
        # Attenuation increases with frequency and distance
        alpha = 0.01 * (1.0 / (self.soil.attenuation_factor + 0.1))
        atten_profile = np.exp(-alpha * distance_m * np.abs(freqs) / 100.0)
        
        spectrum_atten = spectrum * atten_profile
        signal_atten = np.real(np.fft.ifft(spectrum_atten))
        
        return signal_atten * geometric_atten * self.soil.attenuation_factor
    
    def generate_elephant_footfall(self, 
                                   duration: float = 10.0,
                                   num_steps: int = 8,
                                   distance_m: float = 50.0,
                                   elephant_weight_kg: float = 4000.0,
                                   gait_speed_m_s: float = 1.5,
                                   snr_db: float = 15.0) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Generate elephant footfall signature
        
        Elephants produce:
        - Low frequency content (1-30 Hz dominant)
        - Regular temporal periodicity (walking rhythm)
        - High amplitude impacts
        - Characteristic spectral signature
        
        Args:
            duration: Signal duration in seconds
            num_steps: Number of footfalls
            distance_m: Distance from sensor
            elephant_weight_kg: Elephant weight
            gait_speed_m_s: Walking speed
            snr_db: Signal-to-noise ratio
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        signal_data = np.zeros_like(t)
        
        # Step timing with natural variation
        step_interval = duration / num_steps
        step_times = [step_interval * i + np.random.normal(0, 0.05) 
                      for i in range(num_steps)]
        
        for step_time in step_times:
            # Each footfall is a decaying oscillation
            # Dominant frequency 5-15 Hz (elephant footfall resonance)
            f_dominant = np.random.uniform(8, 15)
            f_harmonics = [f_dominant * i for i in [1, 2, 3]]
            
            # Impact duration ~0.2-0.5 seconds
            impact_duration = np.random.uniform(0.2, 0.5)
            decay_rate = np.random.uniform(4, 8)
            
            # Time vector for this impact
            impact_mask = (t >= step_time) & (t < step_time + impact_duration * 3)
            t_impact = t[impact_mask] - step_time
            
            # Amplitude scales with weight
            amplitude = (elephant_weight_kg / 4000.0) * np.random.uniform(0.8, 1.2)
            
            # Generate multi-frequency impact
            impact_signal = np.zeros_like(t_impact)
            for i, freq in enumerate(f_harmonics):
                weight = 1.0 / (i + 1)  # Harmonics decrease in amplitude
                impact_signal += weight * np.sin(2 * np.pi * freq * t_impact)
            
            # Apply exponential decay envelope
            envelope = amplitude * np.exp(-decay_rate * t_impact)
            impact_signal *= envelope
            
            signal_data[impact_mask] += impact_signal
        
        # Apply soil and distance effects
        signal_data = self._apply_soil_attenuation(signal_data, distance_m)
        
        # Add realistic noise
        signal_data = self._add_noise(signal_data, snr_db)
        
        metadata = SignalMetadata(
            signal_type="elephant_footfall",
            duration=duration,
            sampling_rate=self.fs,
            soil_conditions=asdict(self.soil),
            parameters={
                "num_steps": num_steps,
                "distance_m": distance_m,
                "elephant_weight_kg": elephant_weight_kg,
                "gait_speed_m_s": gait_speed_m_s
            },
            timestamp=np.datetime64('now').astype(str),
            snr_db=snr_db
        )
        
        return signal_data, metadata
    
    def generate_human_footsteps(self,
                                 duration: float = 10.0,
                                 num_steps: int = 15,
                                 distance_m: float = 20.0,
                                 snr_db: float = 10.0) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Generate human footstep signature
        
        Humans produce:
        - Higher frequency content (10-50 Hz)
        - Lighter impacts
        - Faster cadence
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        signal_data = np.zeros_like(t)
        
        step_interval = duration / num_steps
        step_times = [step_interval * i + np.random.normal(0, 0.03) 
                      for i in range(num_steps)]
        
        for step_time in step_times:
            # Higher frequency impact
            f_dominant = np.random.uniform(20, 40)
            impact_duration = np.random.uniform(0.05, 0.15)
            decay_rate = np.random.uniform(10, 15)
            
            impact_mask = (t >= step_time) & (t < step_time + impact_duration * 2)
            t_impact = t[impact_mask] - step_time
            
            # Much lighter amplitude
            amplitude = np.random.uniform(0.1, 0.2)
            
            impact_signal = amplitude * np.sin(2 * np.pi * f_dominant * t_impact)
            envelope = np.exp(-decay_rate * t_impact)
            impact_signal *= envelope
            
            signal_data[impact_mask] += impact_signal
        
        signal_data = self._apply_soil_attenuation(signal_data, distance_m)
        signal_data = self._add_noise(signal_data, snr_db)
        
        metadata = SignalMetadata(
            signal_type="human_footsteps",
            duration=duration,
            sampling_rate=self.fs,
            soil_conditions=asdict(self.soil),
            parameters={"num_steps": num_steps, "distance_m": distance_m},
            timestamp=np.datetime64('now').astype(str),
            snr_db=snr_db
        )
        
        return signal_data, metadata
    
    def generate_cattle_movement(self,
                                 duration: float = 10.0,
                                 num_animals: int = 3,
                                 distance_m: float = 30.0,
                                 snr_db: float = 8.0) -> Tuple[np.ndarray, SignalMetadata]:
        """Generate cattle movement signature"""
        t = np.linspace(0, duration, int(self.fs * duration))
        signal_data = np.zeros_like(t)
        
        # Multiple animals with overlapping steps
        for animal in range(num_animals):
            num_steps = np.random.randint(10, 20)
            step_times = sorted([np.random.uniform(0, duration) 
                               for _ in range(num_steps)])
            
            for step_time in step_times:
                f_dominant = np.random.uniform(12, 25)
                impact_duration = np.random.uniform(0.1, 0.3)
                decay_rate = np.random.uniform(6, 10)
                
                impact_mask = (t >= step_time) & (t < step_time + impact_duration * 2)
                t_impact = t[impact_mask] - step_time
                
                amplitude = np.random.uniform(0.2, 0.4)
                
                impact_signal = amplitude * np.sin(2 * np.pi * f_dominant * t_impact)
                envelope = np.exp(-decay_rate * t_impact)
                impact_signal *= envelope
                
                signal_data[impact_mask] += impact_signal
        
        signal_data = self._apply_soil_attenuation(signal_data, distance_m)
        signal_data = self._add_noise(signal_data, snr_db)
        
        metadata = SignalMetadata(
            signal_type="cattle_movement",
            duration=duration,
            sampling_rate=self.fs,
            soil_conditions=asdict(self.soil),
            parameters={"num_animals": num_animals, "distance_m": distance_m},
            timestamp=np.datetime64('now').astype(str),
            snr_db=snr_db
        )
        
        return signal_data, metadata
    
    def generate_wind_vibration(self,
                                duration: float = 10.0,
                                wind_speed_m_s: float = 5.0,
                                snr_db: float = 5.0) -> Tuple[np.ndarray, SignalMetadata]:
        """Generate wind-induced vibration"""
        t = np.linspace(0, duration, int(self.fs * duration))
        
        # Wind creates broadband low-frequency oscillation
        # Multiple frequency components
        freqs = [0.5, 1.2, 2.5, 4.0, 7.0]
        signal_data = np.zeros_like(t)
        
        amplitude_scale = wind_speed_m_s / 10.0
        
        for freq in freqs:
            amplitude = amplitude_scale * np.random.uniform(0.05, 0.15)
            phase = np.random.uniform(0, 2*np.pi)
            signal_data += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add slow amplitude modulation
        modulation = 1.0 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
        signal_data *= modulation
        
        signal_data = self._add_noise(signal_data, snr_db)
        
        metadata = SignalMetadata(
            signal_type="wind_vibration",
            duration=duration,
            sampling_rate=self.fs,
            soil_conditions=asdict(self.soil),
            parameters={"wind_speed_m_s": wind_speed_m_s},
            timestamp=np.datetime64('now').astype(str),
            snr_db=snr_db
        )
        
        return signal_data, metadata
    
    def generate_rain_impact(self,
                            duration: float = 10.0,
                            intensity_mm_hr: float = 20.0,
                            snr_db: float = 3.0) -> Tuple[np.ndarray, SignalMetadata]:
        """Generate rain impact signature"""
        t = np.linspace(0, duration, int(self.fs * duration))
        signal_data = np.zeros_like(t)
        
        # Number of drops scales with intensity
        num_drops = int(intensity_mm_hr * duration * 5)
        
        for _ in range(num_drops):
            drop_time = np.random.uniform(0, duration)
            f_impact = np.random.uniform(50, 150)
            
            impact_mask = (t >= drop_time) & (t < drop_time + 0.05)
            t_impact = t[impact_mask] - drop_time
            
            amplitude = np.random.uniform(0.01, 0.05)
            
            impact_signal = amplitude * np.sin(2 * np.pi * f_impact * t_impact)
            envelope = np.exp(-50 * t_impact)
            impact_signal *= envelope
            
            signal_data[impact_mask] += impact_signal
        
        signal_data = self._add_noise(signal_data, snr_db)
        
        metadata = SignalMetadata(
            signal_type="rain_impact",
            duration=duration,
            sampling_rate=self.fs,
            soil_conditions=asdict(self.soil),
            parameters={"intensity_mm_hr": intensity_mm_hr},
            timestamp=np.datetime64('now').astype(str),
            snr_db=snr_db
        )
        
        return signal_data, metadata
    
    def generate_vehicle_passing(self,
                                 duration: float = 10.0,
                                 vehicle_type: str = "truck",
                                 distance_m: float = 100.0,
                                 speed_m_s: float = 15.0,
                                 snr_db: float = 12.0) -> Tuple[np.ndarray, SignalMetadata]:
        """Generate vehicle passing signature"""
        t = np.linspace(0, duration, int(self.fs * duration))
        
        # Vehicle creates sustained vibration with engine harmonics
        if vehicle_type == "truck":
            base_freq = 25.0
            amplitude = 0.5
        elif vehicle_type == "car":
            base_freq = 35.0
            amplitude = 0.3
        else:
            base_freq = 30.0
            amplitude = 0.4
        
        # Doppler effect as vehicle approaches and recedes
        t_peak = duration / 2
        signal_data = np.zeros_like(t)
        
        for harmonic in [1, 2, 3]:
            freq_modulation = 1.0 + 0.05 * np.sin(2 * np.pi * 0.2 * t)
            freq = base_freq * harmonic * freq_modulation
            
            # Amplitude envelope (closer = louder)
            amp_envelope = amplitude / (harmonic) * np.exp(-((t - t_peak) ** 2) / (duration/3)**2)
            
            signal_data += amp_envelope * np.sin(2 * np.pi * freq * t)
        
        signal_data = self._apply_soil_attenuation(signal_data, distance_m)
        signal_data = self._add_noise(signal_data, snr_db)
        
        metadata = SignalMetadata(
            signal_type="vehicle_passing",
            duration=duration,
            sampling_rate=self.fs,
            soil_conditions=asdict(self.soil),
            parameters={
                "vehicle_type": vehicle_type,
                "distance_m": distance_m,
                "speed_m_s": speed_m_s
            },
            timestamp=np.datetime64('now').astype(str),
            snr_db=snr_db
        )
        
        return signal_data, metadata
    
    def generate_background_noise(self,
                                  duration: float = 10.0,
                                  noise_level: str = "medium") -> Tuple[np.ndarray, SignalMetadata]:
        """Generate background seismic noise"""
        t = np.linspace(0, duration, int(self.fs * duration))
        
        if noise_level == "low":
            amplitude = 0.02
        elif noise_level == "medium":
            amplitude = 0.05
        else:  # high
            amplitude = 0.10
        
        # Multiple noise sources
        # 1. White noise
        white_noise = amplitude * np.random.randn(len(t))
        
        # 2. Colored noise (1/f characteristic)
        freqs = fftfreq(len(t), 1/self.fs)
        white_spectrum = fft(white_noise)
        # Apply 1/f filter
        colored_spectrum = white_spectrum / (np.abs(freqs) + 1.0)
        colored_noise = np.real(np.fft.ifft(colored_spectrum))
        
        # 3. Microseismic noise (ocean waves, distant activity)
        microseismic = 0.03 * amplitude * np.sin(2 * np.pi * 0.2 * t)
        
        signal_data = white_noise * 0.3 + colored_noise * 0.5 + microseismic
        
        metadata = SignalMetadata(
            signal_type="background_noise",
            duration=duration,
            sampling_rate=self.fs,
            soil_conditions=asdict(self.soil),
            parameters={"noise_level": noise_level},
            timestamp=np.datetime64('now').astype(str),
            snr_db=0.0  # Pure noise, no signal
        )
        
        return signal_data, metadata
    
    def save_signal(self, signal_data: np.ndarray, metadata: SignalMetadata, 
                   filename: str):
        """Save signal and metadata to files"""
        np.save(filename + '.npy', signal_data)
        with open(filename + '_metadata.json', 'w') as f:
            json.dump(asdict(metadata), f, indent=2)


if __name__ == "__main__":
    # Demo: Generate sample signals
    print("EarthPulse AI - Seismic Signal Generator Demo")
    print("=" * 60)
    
    generator = SeismicSignalGenerator(sampling_rate=1000)
    
    # Generate each signal type
    print("\n1. Generating Elephant Footfall Signal...")
    generator.set_soil_conditions(moisture=20.0)
    elephant_signal, elephant_meta = generator.generate_elephant_footfall(
        duration=5.0, num_steps=4, distance_m=30.0
    )
    print(f"   Generated {len(elephant_signal)} samples")
    print(f"   Signal power: {np.std(elephant_signal):.4f}")
    
    print("\n2. Generating Human Footsteps...")
    human_signal, human_meta = generator.generate_human_footsteps(
        duration=5.0, num_steps=10, distance_m=15.0
    )
    print(f"   Generated {len(human_signal)} samples")
    
    print("\n3. Generating Cattle Movement...")
    cattle_signal, cattle_meta = generator.generate_cattle_movement(
        duration=5.0, num_animals=2, distance_m=25.0
    )
    print(f"   Generated {len(cattle_signal)} samples")
    
    print("\n4. Generating Wind Vibration...")
    wind_signal, wind_meta = generator.generate_wind_vibration(
        duration=5.0, wind_speed_m_s=7.0
    )
    print(f"   Generated {len(wind_signal)} samples")
    
    print("\n5. Generating Rain Impact...")
    rain_signal, rain_meta = generator.generate_rain_impact(
        duration=5.0, intensity_mm_hr=15.0
    )
    print(f"   Generated {len(rain_signal)} samples")
    
    print("\n6. Generating Vehicle Passing...")
    vehicle_signal, vehicle_meta = generator.generate_vehicle_passing(
        duration=5.0, vehicle_type="truck", distance_m=80.0
    )
    print(f"   Generated {len(vehicle_signal)} samples")
    
    print("\n7. Generating Background Noise...")
    noise_signal, noise_meta = generator.generate_background_noise(
        duration=5.0, noise_level="medium"
    )
    print(f"   Generated {len(noise_signal)} samples")
    
    print("\n" + "=" * 60)
    print("Signal generation complete!")
    print("\nSoil Conditions:")
    print(f"  Moisture: {generator.soil.moisture_percent}%")
    print(f"  Attenuation Factor: {generator.soil.attenuation_factor:.2f}")
    print(f"  Wave Velocity: {generator.soil.velocity_m_s:.1f} m/s")
