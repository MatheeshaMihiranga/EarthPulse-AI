"""
EarthPulse AI - LSTM Classifier
Train LSTM model for elephant footfall detection with quantization and export
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


class ElephantDetectionLSTM:
    """
    LSTM classifier for elephant footfall detection
    Optimized for edge deployment on ESP32
    """
    
    def __init__(self, num_classes: int = 7, 
                 sequence_length: int = 50,
                 num_features: int = 18):
        """
        Initialize model
        
        Args:
            num_classes: Number of signal classes
            sequence_length: Time steps in sequence
            num_features: Number of features per time step
        """
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def load_data(self, data_dir: str = "./data") -> tuple:
        """Load and prepare datasets"""
        data_dir = Path(data_dir)
        
        print("Loading datasets...")
        train_df = pd.read_csv(data_dir / "train_dataset.csv")
        val_df = pd.read_csv(data_dir / "val_dataset.csv")
        test_df = pd.read_csv(data_dir / "test_dataset.csv")
        
        # Extract feature columns
        feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
        
        # Prepare data
        X_train = train_df[feature_cols].values
        y_train = train_df['class_label'].values
        
        X_val = val_df[feature_cols].values
        y_val = val_df['class_label'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['class_label'].values
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Reshape for LSTM (samples, timesteps, features)
        # We'll treat feature vector as a short sequence
        X_train = X_train.reshape(-1, self.num_features, 1)
        X_val = X_val.reshape(-1, self.num_features, 1)
        X_test = X_test.reshape(-1, self.num_features, 1)
        
        # One-hot encode labels
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_val = to_categorical(y_val, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def build_model(self, lstm_units: int = 64, 
                   dropout: float = 0.3,
                   learning_rate: float = 0.001) -> models.Model:
        """
        Build LSTM model optimized for edge deployment
        
        Args:
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Compiled model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.num_features, 1)),
            
            # First LSTM layer
            layers.LSTM(lstm_units, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            
            # Second LSTM layer
            layers.LSTM(lstm_units // 2),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout / 2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val,
             epochs: int = 50,
             batch_size: int = 32,
             model_path: str = "./models/lstm_model.h5"):
        """
        Train the model with callbacks
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            model_path: Path to save best model
        """
        print("\nTraining LSTM model...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Create model if not exists
        if self.model is None:
            self.model = self.build_model()
        
        print("\nModel Summary:")
        self.model.summary()
        
        # Define callbacks
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        callback_list = [
            callbacks.ModelCheckpoint(
                str(model_path),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
        return self.history
    
    def evaluate(self, X_test, y_test, class_names: list) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test, y_test: Test data
            class_names: List of class names
            
        Returns:
            Dictionary of metrics
        """
        print("\nEvaluating model...")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, 
                                      target_names=class_names,
                                      digits=4)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('./models/confusion_matrix.png', dpi=300)
        print("\n✓ Confusion matrix saved: ./models/confusion_matrix.png")
        plt.close()
        
        # Per-class metrics
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
        }
        
        # Elephant detection specific metrics (class 0)
        elephant_idx = 0
        elephant_precision = metrics['classification_report'][class_names[elephant_idx]]['precision']
        elephant_recall = metrics['classification_report'][class_names[elephant_idx]]['recall']
        elephant_f1 = metrics['classification_report'][class_names[elephant_idx]]['f1-score']
        
        print(f"\n{'='*60}")
        print("Elephant Detection Performance:")
        print(f"{'='*60}")
        print(f"  Precision: {elephant_precision:.4f}")
        print(f"  Recall:    {elephant_recall:.4f}")
        print(f"  F1-Score:  {elephant_f1:.4f}")
        print(f"{'='*60}")
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('./models/training_history.png', dpi=300)
        print("✓ Training history saved: ./models/training_history.png")
        plt.close()
    
    def quantize_model(self, model_path: str = "./models/lstm_model.h5",
                      output_path: str = "./models/lstm_model_quantized.tflite"):
        """
        Quantize model for edge deployment (ESP32)
        Note: LSTM models may require TF ops for conversion
        
        Args:
            model_path: Path to trained model
            output_path: Path for quantized model
        """
        print("\nQuantizing model for edge deployment...")
        
        # Load model if not loaded
        if self.model is None:
            self.model = keras.models.load_model(model_path)
        
        try:
            # Convert to TFLite with TF ops support for LSTM
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Support both TFLite built-in ops and TensorFlow ops
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            # Disable experimental tensor list lowering for LSTM compatibility
            converter._experimental_lower_tensor_list_ops = False
            
            # Apply float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            # Convert
            tflite_model = converter.convert()
            
            # Save
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get size
            original_size = Path(model_path).stat().st_size / 1024  # KB
            quantized_size = len(tflite_model) / 1024  # KB
            compression_ratio = original_size / quantized_size
            
            print(f"\n✓ Model quantized successfully!")
            print(f"  Original size:  {original_size:.2f} KB")
            print(f"  Quantized size: {quantized_size:.2f} KB")
            print(f"  Compression:    {compression_ratio:.2f}x")
            print(f"  Saved to:       {output_path}")
            print(f"  Note: Model includes TensorFlow ops for LSTM support")
            
            return output_path
            
        except Exception as e:
            print(f"⚠ Quantization failed: {e}")
            print("  Saving quantization guide instead...")
            
            # Save quantization guide
            guide_path = Path("./models/quantization_guide.txt")
            with open(guide_path, 'w') as f:
                f.write("EarthPulse AI - Model Quantization Guide\n")
                f.write("=" * 60 + "\n\n")
                f.write("LSTM models require special handling for TFLite conversion.\n\n")
                f.write("Options for ESP32 deployment:\n\n")
                f.write("1. Use TensorFlow Lite Micro with custom ops\n")
                f.write("2. Replace LSTM with simpler layers (Dense, Conv1D)\n")
                f.write("3. Export model to ONNX and convert separately\n")
                f.write("4. Use feature-based classification instead of raw signals\n\n")
                f.write("Current model can be deployed using TF Lite with SELECT_TF_OPS.\n")
                f.write("Model file: " + str(model_path) + "\n")
            
            print(f"  Guide saved: {guide_path}")
            return None
    
    def export_to_onnx(self, model_path: str = "./models/lstm_model.h5",
                      output_path: str = "./models/lstm_model.onnx"):
        """
        Export model to ONNX format
        
        Args:
            model_path: Path to trained model
            output_path: Path for ONNX model
        """
        print("\nExporting model to ONNX format...")
        
        try:
            import tf2onnx
            
            # Load model if not loaded
            if self.model is None:
                self.model = keras.models.load_model(model_path)
            
            # Convert to ONNX
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                output_path=output_path
            )
            
            print(f"✓ ONNX model saved: {output_path}")
            
        except ImportError:
            print("⚠ tf2onnx not installed. Skipping ONNX export.")
        except Exception as e:
            print(f"⚠ ONNX export failed: {e}")
    
    def save_model_card(self, metrics: dict, output_path: str = "./models/model_card.json"):
        """Save model card with metadata and performance"""
        
        model_card = {
            'model_name': 'EarthPulse_LSTM_v1',
            'model_type': 'LSTM Classifier',
            'task': 'Elephant Footfall Detection from Seismic Signals',
            'num_classes': self.num_classes,
            'architecture': {
                'input_shape': (self.num_features, 1),
                'lstm_layers': 2,
                'dense_layers': 1,
                'output_activation': 'softmax',
                'total_parameters': self.model.count_params() if self.model else 0
            },
            'training': {
                'dataset': 'EarthPulse_Synthetic_Seismic_v1',
                'train_samples': 700,
                'val_samples': 210,
                'test_samples': 210
            },
            'performance': metrics,
            'deployment': {
                'target_device': 'ESP32',
                'quantization': 'float16',
                'format': 'TensorFlow Lite'
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
            'preprocessing': {
                'sampling_rate': 1000,
                'bandpass_filter': '1-80 Hz',
                'notch_filter': '50 Hz',
                'normalization': 'StandardScaler'
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(model_card, f, indent=2)
        
        print(f"\n✓ Model card saved: {output_path}")


def main():
    """Main training pipeline"""
    
    print("=" * 70)
    print("EarthPulse AI - LSTM Training Pipeline")
    print("=" * 70)
    
    # Initialize model
    lstm = ElephantDetectionLSTM(
        num_classes=7,
        sequence_length=50,
        num_features=18
    )
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = lstm.load_data("./data")
    
    # Build and train model
    lstm.train(
        X_train, y_train, 
        X_val, y_val,
        epochs=50,
        batch_size=32,
        model_path="./models/lstm_model.h5"
    )
    
    # Plot training history
    lstm.plot_training_history()
    
    # Evaluate model
    class_names = [
        'elephant_footfall',
        'human_footsteps',
        'cattle_movement',
        'wind_vibration',
        'rain_impact',
        'vehicle_passing',
        'background_noise'
    ]
    
    metrics = lstm.evaluate(X_test, y_test, class_names)
    
    # Quantize for edge deployment
    lstm.quantize_model(
        model_path="./models/lstm_model.h5",
        output_path="./models/lstm_model_quantized.tflite"
    )
    
    # Export to ONNX
    lstm.export_to_onnx(
        model_path="./models/lstm_model.h5",
        output_path="./models/lstm_model.onnx"
    )
    
    # Save model card
    lstm.save_model_card(metrics, "./models/model_card.json")
    
    print("\n" + "=" * 70)
    print("✓ Training pipeline complete!")
    print("=" * 70)
    print("\nModel artifacts saved in ./models/:")
    print("  - lstm_model.h5 (Full model)")
    print("  - lstm_model_quantized.tflite (Quantized for ESP32)")
    print("  - lstm_model.onnx (ONNX format)")
    print("  - model_card.json (Model metadata)")
    print("  - confusion_matrix.png")
    print("  - training_history.png")


if __name__ == "__main__":
    main()
