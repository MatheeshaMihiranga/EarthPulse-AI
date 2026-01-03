"""
EarthPulse AI - Dataset Generator
Generate comprehensive labeled synthetic datasets for training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from synthetic_generator.seismic_signal_generator import SeismicSignalGenerator, SoilConditions
from synthetic_generator.dsp_pipeline import SeismicDSP


class DatasetGenerator:
    """Generate comprehensive training/validation/test datasets"""
    
    def __init__(self, output_dir: str = "./data"):
        """Initialize dataset generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = SeismicSignalGenerator(sampling_rate=1000)
        self.dsp = SeismicDSP()
        
        # Define class labels
        self.classes = {
            0: 'elephant_footfall',
            1: 'human_footsteps',
            2: 'cattle_movement',
            3: 'wind_vibration',
            4: 'rain_impact',
            5: 'vehicle_passing',
            6: 'background_noise'
        }
        
    def generate_single_sample(self, signal_type: str, 
                              soil_moisture: float,
                              **kwargs) -> dict:
        """
        Generate a single labeled sample with features
        
        Args:
            signal_type: Type of signal to generate
            soil_moisture: Soil moisture percentage
            **kwargs: Additional parameters for signal generation
            
        Returns:
            Dictionary with signal, features, and metadata
        """
        # Set soil conditions
        self.generator.set_soil_conditions(moisture=soil_moisture)
        
        # Generate signal based on type
        if signal_type == 'elephant_footfall':
            signal_data, metadata = self.generator.generate_elephant_footfall(**kwargs)
        elif signal_type == 'human_footsteps':
            signal_data, metadata = self.generator.generate_human_footsteps(**kwargs)
        elif signal_type == 'cattle_movement':
            signal_data, metadata = self.generator.generate_cattle_movement(**kwargs)
        elif signal_type == 'wind_vibration':
            signal_data, metadata = self.generator.generate_wind_vibration(**kwargs)
        elif signal_type == 'rain_impact':
            signal_data, metadata = self.generator.generate_rain_impact(**kwargs)
        elif signal_type == 'vehicle_passing':
            signal_data, metadata = self.generator.generate_vehicle_passing(**kwargs)
        elif signal_type == 'background_noise':
            signal_data, metadata = self.generator.generate_background_noise(**kwargs)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        # Process with DSP pipeline
        processed_signal, features = self.dsp.process_signal(signal_data)
        
        # Get class label
        class_label = [k for k, v in self.classes.items() if v == signal_type][0]
        
        return {
            'raw_signal': signal_data,
            'processed_signal': processed_signal,
            'features': features,
            'metadata': metadata,
            'class_label': class_label,
            'class_name': signal_type
        }
    
    def generate_dataset(self, 
                        samples_per_class: int = 100,
                        duration: float = 5.0,
                        split: str = 'train') -> pd.DataFrame:
        """
        Generate complete dataset
        
        Args:
            samples_per_class: Number of samples per class
            duration: Signal duration in seconds
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            DataFrame with all samples and features
        """
        print(f"\nGenerating {split} dataset...")
        print(f"Samples per class: {samples_per_class}")
        print(f"Signal duration: {duration}s")
        
        all_samples = []
        
        # Define parameter ranges for variation
        soil_moisture_range = [5, 10, 15, 20, 25, 30, 35, 40]
        
        for class_label, class_name in tqdm(self.classes.items(), desc="Classes"):
            
            for i in range(samples_per_class):
                # Vary soil moisture
                soil_moisture = np.random.choice(soil_moisture_range)
                
                # Generate signal with varied parameters
                if class_name == 'elephant_footfall':
                    kwargs = {
                        'duration': duration,
                        'num_steps': np.random.randint(3, 8),
                        'distance_m': np.random.uniform(20, 100),
                        'elephant_weight_kg': np.random.uniform(3000, 5000),
                        'gait_speed_m_s': np.random.uniform(1.0, 2.5),
                        'snr_db': np.random.uniform(10, 20)
                    }
                elif class_name == 'human_footsteps':
                    kwargs = {
                        'duration': duration,
                        'num_steps': np.random.randint(8, 20),
                        'distance_m': np.random.uniform(5, 40),
                        'snr_db': np.random.uniform(5, 15)
                    }
                elif class_name == 'cattle_movement':
                    kwargs = {
                        'duration': duration,
                        'num_animals': np.random.randint(1, 5),
                        'distance_m': np.random.uniform(10, 60),
                        'snr_db': np.random.uniform(6, 14)
                    }
                elif class_name == 'wind_vibration':
                    kwargs = {
                        'duration': duration,
                        'wind_speed_m_s': np.random.uniform(2, 15),
                        'snr_db': np.random.uniform(3, 10)
                    }
                elif class_name == 'rain_impact':
                    kwargs = {
                        'duration': duration,
                        'intensity_mm_hr': np.random.uniform(5, 50),
                        'snr_db': np.random.uniform(2, 8)
                    }
                elif class_name == 'vehicle_passing':
                    kwargs = {
                        'duration': duration,
                        'vehicle_type': np.random.choice(['car', 'truck']),
                        'distance_m': np.random.uniform(50, 200),
                        'speed_m_s': np.random.uniform(10, 25),
                        'snr_db': np.random.uniform(8, 16)
                    }
                else:  # background_noise
                    kwargs = {
                        'duration': duration,
                        'noise_level': np.random.choice(['low', 'medium', 'high'])
                    }
                
                # Generate sample
                sample = self.generate_single_sample(
                    class_name, 
                    soil_moisture,
                    **kwargs
                )
                
                # Extract feature vector
                feature_vector = self.dsp.create_feature_vector(sample['features'])
                
                # Create sample record
                sample_record = {
                    'sample_id': f"{split}_{class_name}_{i:04d}",
                    'class_label': class_label,
                    'class_name': class_name,
                    'soil_moisture': soil_moisture,
                    'duration': duration,
                    'snr_db': kwargs.get('snr_db', 0.0)
                }
                
                # Add all features
                feature_names = [
                    'rms', 'peak_to_peak', 'std', 'mean_abs',
                    'zero_crossing_rate', 'spectral_centroid',
                    'stft_mean_power', 'stft_max_power',
                    'low_freq_energy', 'high_freq_energy', 'freq_ratio',
                    'periodicity_period_seconds', 'periodicity_frequency_hz',
                    'periodicity_num_peaks', 'periodicity_max_correlation',
                    'envelope_mean', 'envelope_std', 'envelope_max'
                ]
                
                for feat_name, feat_value in zip(feature_names, feature_vector):
                    sample_record[f'feature_{feat_name}'] = feat_value
                
                # Save raw signal
                signal_file = self.output_dir / 'raw' / f"{sample_record['sample_id']}.npy"
                signal_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(signal_file, sample['raw_signal'])
                
                # Save processed signal
                processed_file = self.output_dir / 'processed' / f"{sample_record['sample_id']}_processed.npy"
                processed_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(processed_file, sample['processed_signal'])
                
                all_samples.append(sample_record)
        
        # Create DataFrame
        df = pd.DataFrame(all_samples)
        
        # Save to CSV
        csv_file = self.output_dir / f"{split}_dataset.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n{split.capitalize()} dataset saved:")
        print(f"  CSV: {csv_file}")
        print(f"  Total samples: {len(df)}")
        print(f"  Class distribution:")
        print(df['class_name'].value_counts())
        
        return df
    
    def generate_all_splits(self,
                          train_samples: int = 200,
                          val_samples: int = 50,
                          test_samples: int = 50,
                          duration: float = 5.0):
        """Generate train, validation, and test splits"""
        
        print("=" * 70)
        print("EarthPulse AI - Dataset Generation")
        print("=" * 70)
        
        # Generate datasets
        train_df = self.generate_dataset(train_samples, duration, 'train')
        val_df = self.generate_dataset(val_samples, duration, 'val')
        test_df = self.generate_dataset(test_samples, duration, 'test')
        
        # Generate dataset metadata
        metadata = {
            'dataset_name': 'EarthPulse_Synthetic_Seismic_v1',
            'description': 'Synthetic seismic signals for elephant detection',
            'num_classes': len(self.classes),
            'classes': self.classes,
            'sampling_rate': self.generator.fs,
            'signal_duration': duration,
            'splits': {
                'train': {
                    'num_samples': len(train_df),
                    'samples_per_class': train_samples
                },
                'val': {
                    'num_samples': len(val_df),
                    'samples_per_class': val_samples
                },
                'test': {
                    'num_samples': len(test_df),
                    'samples_per_class': test_samples
                }
            },
            'features': [
                'rms', 'peak_to_peak', 'std', 'mean_abs',
                'zero_crossing_rate', 'spectral_centroid',
                'stft_mean_power', 'stft_max_power',
                'low_freq_energy', 'high_freq_energy', 'freq_ratio',
                'periodicity_period_seconds', 'periodicity_frequency_hz',
                'periodicity_num_peaks', 'periodicity_max_correlation',
                'envelope_mean', 'envelope_std', 'envelope_max'
            ],
            'soil_moisture_range': [5, 40],
            'generation_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = self.output_dir / 'dataset_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'=' * 70}")
        print("Dataset generation complete!")
        print(f"{'=' * 70}")
        print(f"Metadata: {metadata_file}")
        print(f"\nDataset summary:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} samples")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Generate datasets
    generator = DatasetGenerator(output_dir="./data")
    
    # Generate with moderate sample sizes for demo
    # For full research, increase to 500+ per class
    train_df, val_df, test_df = generator.generate_all_splits(
        train_samples=100,  # 100 samples per class for training
        val_samples=30,     # 30 per class for validation
        test_samples=30,    # 30 per class for testing
        duration=5.0        # 5 second signals
    )
    
    print("\n✓ Dataset generation complete!")
    print("\nNext steps:")
    print("  1. Train LSTM model on generated data")
    print("  2. Evaluate on test set (TSTR)")
    print("  3. Quantize for ESP32 deployment")
