"""
EarthPulse AI - Digital Signal Processing Pipeline
ESP32-compatible DSP operations for seismic signal analysis
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DSPConfig:
    """Configuration for DSP pipeline"""
    sampling_rate: int = 1000  # Hz
    bandpass_low: float = 1.0  # Hz
    bandpass_high: float = 80.0  # Hz
    notch_freq: float = 50.0  # Hz (power line interference)
    notch_q: float = 30.0
    window_size: int = 256  # samples for STFT
    hop_length: int = 128  # STFT hop
    compression_threshold: float = 0.5
    compression_ratio: float = 3.0


class SeismicDSP:
    """
    Digital Signal Processing pipeline for seismic signals
    Mimics ESP32 edge processing capabilities
    """
    
    def __init__(self, config: Optional[DSPConfig] = None):
        """Initialize DSP pipeline"""
        self.config = config or DSPConfig()
        self.fs = self.config.sampling_rate
        
        # Pre-design filters for efficiency
        self._design_filters()
        
    def _design_filters(self):
        """Pre-design filters (would be done once on ESP32)"""
        # Bandpass filter (Butterworth, 4th order)
        nyquist = self.fs / 2
        low_norm = self.config.bandpass_low / nyquist
        high_norm = self.config.bandpass_high / nyquist
        
        self.bp_b, self.bp_a = signal.butter(
            4, [low_norm, high_norm], btype='band'
        )
        
        # Notch filter for power line interference
        notch_norm = self.config.notch_freq / nyquist
        self.notch_b, self.notch_a = signal.iirnotch(
            notch_norm, self.config.notch_q
        )
        
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter (1-80 Hz typical for elephant detection)
        
        Args:
            data: Input signal
            
        Returns:
            Filtered signal
        """
        # Use filtfilt for zero-phase filtering
        filtered = signal.filtfilt(self.bp_b, self.bp_a, data)
        return filtered
    
    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Remove power line interference (50/60 Hz)
        
        Args:
            data: Input signal
            
        Returns:
            Notch-filtered signal
        """
        filtered = signal.filtfilt(self.notch_b, self.notch_a, data)
        return filtered
    
    def dynamic_range_compression(self, data: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression to handle varying amplitudes
        
        Args:
            data: Input signal
            
        Returns:
            Compressed signal
        """
        threshold = self.config.compression_threshold
        ratio = self.config.compression_ratio
        
        # Soft-knee compression
        compressed = np.zeros_like(data)
        
        for i, sample in enumerate(data):
            abs_sample = np.abs(sample)
            
            if abs_sample <= threshold:
                compressed[i] = sample
            else:
                # Compress values above threshold
                excess = abs_sample - threshold
                compressed_excess = excess / ratio
                new_abs = threshold + compressed_excess
                compressed[i] = np.sign(sample) * new_abs
        
        return compressed
    
    def extract_rms(self, data: np.ndarray, window_size: int = 100) -> float:
        """
        Root Mean Square (RMS) energy
        
        Args:
            data: Input signal
            window_size: Window for RMS calculation
            
        Returns:
            RMS value
        """
        squared = data ** 2
        windowed = np.convolve(squared, np.ones(window_size)/window_size, mode='valid')
        rms = np.sqrt(np.mean(windowed))
        return rms
    
    def extract_peak_to_peak(self, data: np.ndarray) -> float:
        """
        Peak-to-peak amplitude
        
        Args:
            data: Input signal
            
        Returns:
            Peak-to-peak value
        """
        return np.max(data) - np.min(data)
    
    def extract_stft_features(self, data: np.ndarray) -> Dict:
        """
        Short-Time Fourier Transform features
        
        Args:
            data: Input signal
            
        Returns:
            Dictionary with STFT, frequencies, and times
        """
        f, t, Zxx = stft(
            data, 
            fs=self.fs,
            window='hann',
            nperseg=self.config.window_size,
            noverlap=self.config.window_size - self.config.hop_length
        )
        
        magnitude = np.abs(Zxx)
        
        return {
            'frequencies': f,
            'times': t,
            'magnitude': magnitude,
            'phase': np.angle(Zxx)
        }
    
    def extract_spectral_centroid(self, data: np.ndarray) -> float:
        """
        Spectral centroid (center of mass of spectrum)
        Indicates "brightness" of sound
        
        Args:
            data: Input signal
            
        Returns:
            Spectral centroid in Hz
        """
        # Compute FFT
        spectrum = np.abs(fft(data))
        freqs = fftfreq(len(data), 1/self.fs)
        
        # Only use positive frequencies
        pos_mask = freqs > 0
        spectrum_pos = spectrum[pos_mask]
        freqs_pos = freqs[pos_mask]
        
        # Weighted average of frequencies
        if np.sum(spectrum_pos) > 0:
            centroid = np.sum(freqs_pos * spectrum_pos) / np.sum(spectrum_pos)
        else:
            centroid = 0.0
        
        return centroid
    
    def extract_temporal_periodicity(self, data: np.ndarray) -> Dict:
        """
        Analyze temporal periodicity (footfall rhythm)
        
        Args:
            data: Input signal
            
        Returns:
            Dictionary with periodicity metrics
        """
        # Autocorrelation to find periodicity
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Find peaks (excluding the zero-lag peak)
        min_lag = int(0.5 * self.fs)  # At least 0.5 second between steps
        max_lag = int(3.0 * self.fs)  # At most 3 seconds between steps
        
        search_window = autocorr[min_lag:max_lag]
        
        if len(search_window) > 0:
            peaks, properties = signal.find_peaks(search_window, height=0.3)
            
            if len(peaks) > 0:
                # Most prominent peak
                main_peak_idx = peaks[np.argmax(properties['peak_heights'])]
                period_samples = main_peak_idx + min_lag
                period_seconds = period_samples / self.fs
                frequency_hz = 1.0 / period_seconds if period_seconds > 0 else 0.0
                
                return {
                    'periodic': True,
                    'period_seconds': period_seconds,
                    'frequency_hz': frequency_hz,
                    'num_peaks': len(peaks),
                    'max_correlation': np.max(properties['peak_heights'])
                }
        
        return {
            'periodic': False,
            'period_seconds': 0.0,
            'frequency_hz': 0.0,
            'num_peaks': 0,
            'max_correlation': 0.0
        }
    
    def extract_energy_envelope(self, data: np.ndarray) -> np.ndarray:
        """
        Extract energy envelope of signal
        
        Args:
            data: Input signal
            
        Returns:
            Energy envelope
        """
        # Hilbert transform for analytic signal
        analytic_signal = signal.hilbert(data)
        envelope = np.abs(analytic_signal)
        
        # Smooth envelope
        window_size = int(0.05 * self.fs)  # 50ms smoothing
        if window_size % 2 == 0:
            window_size += 1
        
        envelope_smooth = signal.savgol_filter(envelope, window_size, 3)
        
        return envelope_smooth
    
    def extract_zero_crossing_rate(self, data: np.ndarray) -> float:
        """
        Zero crossing rate
        
        Args:
            data: Input signal
            
        Returns:
            Zero crossing rate
        """
        zero_crossings = np.where(np.diff(np.sign(data)))[0]
        zcr = len(zero_crossings) / len(data)
        return zcr
    
    def extract_all_features(self, data: np.ndarray) -> Dict:
        """
        Extract comprehensive feature set
        
        Args:
            data: Input signal (should be pre-filtered)
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Time domain features
        features['rms'] = self.extract_rms(data)
        features['peak_to_peak'] = self.extract_peak_to_peak(data)
        features['std'] = np.std(data)
        features['mean_abs'] = np.mean(np.abs(data))
        features['zero_crossing_rate'] = self.extract_zero_crossing_rate(data)
        
        # Frequency domain features
        features['spectral_centroid'] = self.extract_spectral_centroid(data)
        
        # Time-frequency features
        stft_features = self.extract_stft_features(data)
        features['stft_mean_power'] = np.mean(stft_features['magnitude'])
        features['stft_max_power'] = np.max(stft_features['magnitude'])
        
        # Low frequency energy (1-20 Hz - elephant range)
        low_freq_mask = (stft_features['frequencies'] >= 1) & \
                       (stft_features['frequencies'] <= 20)
        if np.any(low_freq_mask):
            features['low_freq_energy'] = np.mean(
                stft_features['magnitude'][low_freq_mask, :]
            )
        else:
            features['low_freq_energy'] = 0.0
        
        # High frequency energy (40-80 Hz)
        high_freq_mask = (stft_features['frequencies'] >= 40) & \
                        (stft_features['frequencies'] <= 80)
        if np.any(high_freq_mask):
            features['high_freq_energy'] = np.mean(
                stft_features['magnitude'][high_freq_mask, :]
            )
        else:
            features['high_freq_energy'] = 0.0
        
        # Ratio of low to high frequency energy
        if features['high_freq_energy'] > 0:
            features['freq_ratio'] = features['low_freq_energy'] / \
                                    features['high_freq_energy']
        else:
            features['freq_ratio'] = 0.0
        
        # Temporal periodicity
        periodicity = self.extract_temporal_periodicity(data)
        features.update({
            f'periodicity_{k}': v for k, v in periodicity.items()
        })
        
        # Energy envelope stats
        envelope = self.extract_energy_envelope(data)
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_max'] = np.max(envelope)
        
        return features
    
    def process_signal(self, data: np.ndarray, 
                      full_pipeline: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Complete DSP pipeline processing
        
        Args:
            data: Raw input signal
            full_pipeline: Apply all preprocessing steps
            
        Returns:
            Tuple of (processed_signal, features)
        """
        processed = data.copy()
        
        if full_pipeline:
            # Step 1: Notch filter (remove power line interference)
            processed = self.notch_filter(processed)
            
            # Step 2: Bandpass filter (1-80 Hz)
            processed = self.bandpass_filter(processed)
            
            # Step 3: Dynamic range compression
            processed = self.dynamic_range_compression(processed)
        
        # Extract features
        features = self.extract_all_features(processed)
        
        return processed, features
    
    def create_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Create fixed-size feature vector for ML model
        
        Args:
            features: Dictionary of features
            
        Returns:
            Feature vector as numpy array
        """
        # Define order of features for consistency
        feature_order = [
            'rms', 'peak_to_peak', 'std', 'mean_abs',
            'zero_crossing_rate', 'spectral_centroid',
            'stft_mean_power', 'stft_max_power',
            'low_freq_energy', 'high_freq_energy', 'freq_ratio',
            'periodicity_period_seconds', 'periodicity_frequency_hz',
            'periodicity_num_peaks', 'periodicity_max_correlation',
            'envelope_mean', 'envelope_std', 'envelope_max'
        ]
        
        # Extract values in order
        vector = []
        for key in feature_order:
            value = features.get(key, 0.0)
            # Handle boolean
            if isinstance(value, bool):
                value = float(value)
            vector.append(value)
        
        return np.array(vector)


class AdaptiveDSP(SeismicDSP):
    """
    Adaptive DSP that adjusts parameters based on soil moisture
    """
    
    def __init__(self, config: Optional[DSPConfig] = None):
        super().__init__(config)
        self.soil_moisture = 15.0  # Default
        
    def set_soil_moisture(self, moisture_percent: float):
        """Update soil moisture and adjust DSP parameters"""
        self.soil_moisture = moisture_percent
        
        # Adjust filter cutoffs based on soil conditions
        # Dry soil: signals attenuate more, reduce high-freq cutoff
        # Wet soil: better propagation, can use wider bandwidth
        
        if moisture_percent < 10:  # Very dry
            self.config.bandpass_high = 60.0
        elif moisture_percent < 20:
            self.config.bandpass_high = 70.0
        else:  # Optimal to wet
            self.config.bandpass_high = 80.0
        
        # Re-design filters with new parameters
        self._design_filters()
    
    def get_confidence_weight(self) -> float:
        """
        Get confidence weight based on soil conditions
        
        Returns:
            Weight factor (0-1) for classifier confidence
        """
        # Optimal moisture gives highest confidence
        if 15 <= self.soil_moisture <= 30:
            return 1.0
        elif self.soil_moisture < 10:
            return 0.6  # Very dry, poor signal quality
        elif self.soil_moisture < 15:
            return 0.8
        elif self.soil_moisture < 40:
            return 0.9
        else:
            return 0.7  # Too wet, muddy


if __name__ == "__main__":
    # Demo: Process sample signals
    print("EarthPulse AI - DSP Pipeline Demo")
    print("=" * 60)
    
    # Create sample signal (synthetic elephant footfall)
    fs = 1000
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Simulate elephant footfall
    signal_data = np.zeros_like(t)
    step_times = [1.0, 2.2, 3.4, 4.6]
    for step_time in step_times:
        mask = (t >= step_time) & (t < step_time + 0.5)
        t_step = t[mask] - step_time
        signal_data[mask] += 1.0 * np.sin(2 * np.pi * 10 * t_step) * np.exp(-5 * t_step)
    
    # Add noise
    signal_data += 0.1 * np.random.randn(len(signal_data))
    
    # Process with DSP pipeline
    dsp = SeismicDSP()
    
    print("\nProcessing signal...")
    processed_signal, features = dsp.process_signal(signal_data)
    
    print("\nExtracted Features:")
    print("-" * 60)
    for key, value in features.items():
        if isinstance(value, (int, float)):
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value}")
    
    print("\nFeature Vector:")
    feature_vec = dsp.create_feature_vector(features)
    print(f"  Shape: {feature_vec.shape}")
    print(f"  Values: {feature_vec[:5]}... (first 5)")
    
    # Test adaptive DSP
    print("\n" + "=" * 60)
    print("Adaptive DSP Demo")
    print("-" * 60)
    
    adaptive_dsp = AdaptiveDSP()
    
    for moisture in [5, 15, 25, 40]:
        adaptive_dsp.set_soil_moisture(moisture)
        confidence = adaptive_dsp.get_confidence_weight()
        print(f"Moisture {moisture:2d}% -> Bandpass cutoff: {adaptive_dsp.config.bandpass_high:.1f} Hz, "
              f"Confidence weight: {confidence:.2f}")
    
    print("\n" + "=" * 60)
    print("DSP pipeline ready!")
