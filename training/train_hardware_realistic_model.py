"""
EarthPulse AI - Train Hardware-Realistic Detection Model
Train model on realistic hardware vibration data
Designed for accurate real-world detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from training.hardware_realistic_generator import HardwareRealisticGenerator

class HardwareRealisticTrainer:
    """Train model on hardware-realistic vibration data"""
    
    def __init__(self, sampling_rate: int = 100):
        """
        Initialize trainer
        
        Args:
            sampling_rate: Hardware sampling rate
        """
        self.fs = sampling_rate
        self.sequence_length = sampling_rate  # 1 second
        self.generator = HardwareRealisticGenerator(sampling_rate=sampling_rate)
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = []
        
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract features from signal
        
        Features:
        - Time domain: mean, std, max, min, rms, peak-to-peak
        - Frequency domain: dominant frequency, spectral centroid, bandwidth
        - Statistical: skewness, kurtosis, energy
        """
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
        
        # Decay rate (measure how fast signal decreases)
        first_half_energy = np.sum(signal[:len(signal)//2]**2)
        second_half_energy = np.sum(signal[len(signal)//2:]**2)
        decay_ratio = (first_half_energy + 1e-10) / (second_half_energy + 1e-10)
        features.append(decay_ratio)
        
        return np.array(features)
    
    def prepare_dataset(self, X_raw: np.ndarray, y: np.ndarray):
        """Extract features from raw signals"""
        print("\nExtracting features...")
        X_features = []
        
        for i, signal in enumerate(X_raw):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{len(X_raw)}")
            features = self.extract_features(signal)
            X_features.append(features)
        
        X_features = np.array(X_features)
        
        print(f"\nFeatures extracted: {X_features.shape[1]} features per sample")
        
        return X_features
    
    def build_model(self, input_shape: tuple, num_classes: int):
        """
        Build neural network model
        Using dense layers for feature-based classification
        """
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            
            # Dense layers with dropout
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, samples_per_class: int = 5000, epochs: int = 100,
             batch_size: int = 64, validation_split: float = 0.2):
        """
        Train the model
        
        Args:
            samples_per_class: Number of samples per class
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation data fraction
        """
        print("="*70)
        print("EarthPulse AI - Hardware Realistic Model Training")
        print("="*70)
        
        # Generate dataset
        print("\n1. Generating training dataset...")
        X_raw, y, self.class_names = self.generator.generate_dataset(samples_per_class)
        
        # Extract features
        print("\n2. Extracting features...")
        X_features = self.prepare_dataset(X_raw, y)
        
        # Normalize features
        print("\n3. Normalizing features...")
        X_normalized = self.scaler.fit_transform(X_features)
        
        # Split dataset
        print("\n4. Splitting dataset...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_normalized, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model
        print("\n5. Building model...")
        self.model = self.build_model(
            input_shape=(X_features.shape[1],),
            num_classes=len(self.class_names)
        )
        
        print(self.model.summary())
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        print("\n6. Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        print("\n7. Evaluating model...")
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val)
        
        print(f"\n{'='*70}")
        print(f"Final Validation Accuracy: {val_accuracy*100:.2f}%")
        print(f"Final Validation Loss: {val_loss:.4f}")
        print(f"{'='*70}")
        
        # Per-class accuracy
        y_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        
        print("\nPer-Class Accuracy:")
        for i, class_name in enumerate(self.class_names):
            class_mask = y_val == i
            class_acc = np.mean(y_pred[class_mask] == y_val[class_mask])
            print(f"  {class_name:20s}: {class_acc*100:.2f}%")
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        confusion = np.zeros((len(self.class_names), len(self.class_names)), dtype=int)
        for true, pred in zip(y_val, y_pred):
            confusion[true][pred] += 1
        
        print("\n          ", end="")
        for name in self.class_names:
            print(f"{name[:12]:>12s}", end=" ")
        print()
        
        for i, name in enumerate(self.class_names):
            print(f"{name[:12]:>12s}:", end=" ")
            for j in range(len(self.class_names)):
                print(f"{confusion[i][j]:>12d}", end=" ")
            print()
        
        return history
    
    def save_model(self, model_path: str = "./models/hardware_realistic_model.h5"):
        """Save trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model.save(model_path)
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save scaler and class names
        import pickle
        metadata = {
            'scaler': self.scaler,
            'class_names': self.class_names,
            'sampling_rate': self.fs,
            'sequence_length': self.sequence_length
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def plot_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./models/training_history.png', dpi=150)
        print("\n✓ Training history plot saved to: ./models/training_history.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hardware Realistic Detection Model')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Samples per class (default: 5000, use 5000+ for 25k+ total)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    
    args = parser.parse_args()
    
    print(f"\nTraining with {args.samples} samples per class")
    print(f"Total dataset size: {args.samples * 5} samples")
    print(f"Training for {args.epochs} epochs\n")
    
    # Train model
    trainer = HardwareRealisticTrainer(sampling_rate=100)
    history = trainer.train(
        samples_per_class=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    trainer.save_model()
    
    # Plot results
    trainer.plot_history(history)
    
    print("\n" + "="*70)
    print("✓ Training Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Test model: python training/test_hardware_model.py")
    print("2. Run dashboard: python dashboard/realtime_dashboard.py --hardware")
    print("="*70)
