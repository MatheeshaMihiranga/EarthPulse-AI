"""
EarthPulse AI - Hardware Integration
Real-time elephant detection using ESP32-S3 geophone sensor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.hardware_receiver import GeophoneReceiver, find_serial_port
from edge_firmware_simulated.detection_system import ElephantDetectionSystem
import numpy as np
import time
from collections import deque
from datetime import datetime

class RealTimeDetector:
    """
    Real-time elephant detection system using hardware geophone
    
    Integrates:
    - ESP32-S3 geophone sensor (hardware)
    - ADS1115 ADC (hardware)
    - LSTM AI model (software)
    - Direction detection (software)
    - Behavior analysis (software)
    """
    
    def __init__(self, com_port: str = None, model_path: str = None):
        """
        Initialize real-time detection system
        
        Args:
            com_port: Serial port for ESP32-S3 (None = auto-detect)
            model_path: Path to LSTM model (None = use default)
        """
        print("="*70)
        print("🐘 EarthPulse AI - Real-Time Elephant Detection System")
        print("="*70)
        print("\nInitializing hardware...")
        
        # Auto-detect serial port if not specified
        if com_port is None:
            com_port = find_serial_port()
            if com_port is None:
                print("\n❌ Could not auto-detect ESP32-S3")
                com_port = input("Enter COM port manually (e.g., COM3): ")
        
        # Connect to hardware
        self.receiver = GeophoneReceiver(port=com_port)
        
        print("\nInitializing AI detection system...")
        
        # Initialize detection system
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models", "lstm_model.h5"
            )
        
        self.detector = ElephantDetectionSystem(model_path=model_path)
        
        # Buffer configuration
        self.sample_rate = 1000  # Hz
        self.window_size = 1000  # 1 second windows
        self.window_overlap = 500  # 50% overlap
        
        # Signal buffer
        self.signal_buffer = deque(maxlen=self.window_size)
        
        # Detection history
        self.detection_history = []
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds
        
        # Statistics
        self.samples_processed = 0
        self.detections_count = 0
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("✓ System initialized successfully!")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Serial port: {com_port}")
        print(f"  Model: {model_path}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Window size: {self.window_size} samples ({self.window_size/self.sample_rate:.1f}s)")
        print(f"  Overlap: {self.window_overlap} samples (50%)")
    
    def process_sample(self, sample: dict):
        """
        Process each incoming sample
        
        Args:
            sample: Dictionary with timestamp, voltage, adc values
        """
        # Add to buffer
        self.signal_buffer.append(sample['voltage'])
        self.samples_processed += 1
        
        # When buffer is full, run detection
        if len(self.signal_buffer) >= self.window_size:
            self._run_detection()
            
            # Slide window (remove oldest samples)
            for _ in range(self.window_overlap):
                self.signal_buffer.popleft()
    
    def _run_detection(self):
        """Run elephant detection on current buffer"""
        # Convert buffer to numpy array
        signal = np.array(list(self.signal_buffer))
        
        # Check if enough variation (not just noise)
        signal_std = np.std(signal)
        if signal_std < 0.005:  # Too quiet (lowered threshold for real hardware)
            return
        
        # Run detection
        try:
            result = self.detector.process_signal(signal)
            
            # Check cooldown (avoid duplicate detections)
            current_time = time.time()
            if current_time - self.last_detection_time < self.detection_cooldown:
                return
            
            # If elephant detected (check 'detected' key and class_name)
            if result.get('detected', False) and result.get('class_name') == 'elephant_footfall':
                # Only print if multi-frame confirmed (not just awaiting confirmation)
                if result.get('confirmation') == 'multi_frame_confirmed':
                    self.last_detection_time = current_time
                    self.detections_count += 1
                    self._print_detection(result)
                    
                    # Save to history
                    self.detection_history.append({
                        'timestamp': datetime.now(),
                        'result': result
                    })
        
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_detection(self, result: dict):
        """Print formatted detection result"""
        print("\n" + "="*70)
        print("🐘 ELEPHANT DETECTED!")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        # Direction information
        if 'direction' in result and result['direction']:
            direction = result['direction']
            print(f"\n📍 Direction:")
            print(f"   Status: {direction.get('cardinal', 'Unknown')}")
            print(f"   Approaching: {'Yes' if direction.get('approaching', False) else 'No'}")
            print(f"   Distance: {direction.get('distance', 0):.1f} m")
            print(f"   Velocity: {direction.get('velocity', 0):.2f} m/s")
        
        # Behavior information
        if 'behavior' in result and result['behavior']:
            behavior = result['behavior']
            print(f"\n🐘 Behavior:")
            print(f"   Type: {behavior.get('type', 'Unknown')}")
            print(f"   Activity Level: {behavior.get('activity_level', 'Unknown')}")
            print(f"   Gait Speed: {behavior.get('gait_speed', 0):.2f} m/s")
            print(f"   Estimated Weight: {behavior.get('estimated_weight', 0):.0f} kg")
        
        print("="*70 + "\n")
    
    def print_stats(self):
        """Print system statistics"""
        runtime = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("📊 System Statistics")
        print("="*70)
        print(f"Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
        print(f"Samples processed: {self.samples_processed:,}")
        print(f"Sample rate: {self.samples_processed/runtime:.1f} Hz (avg)")
        print(f"Detections: {self.detections_count}")
        print(f"Detection rate: {self.detections_count/runtime*60:.2f} per minute")
        print("="*70)
    
    def start(self):
        """Start real-time detection"""
        print("\n" + "="*70)
        print("🎯 Starting Real-Time Detection")
        print("="*70)
        print("\nListening for elephant footfalls...")
        print("(Press Ctrl+C to stop)\n")
        
        try:
            # Start streaming with our callback
            self.receiver.start_streaming(self.process_sample)
        
        except KeyboardInterrupt:
            print("\n\nStopping detector...")
            self.print_stats()
            
            if self.detection_history:
                print("\n📋 Detection Summary:")
                for i, detection in enumerate(self.detection_history, 1):
                    ts = detection['timestamp'].strftime('%H:%M:%S')
                    conf = detection['result']['confidence']
                    print(f"  {i}. {ts} - Confidence: {conf:.1%}")
            
            self.receiver.close()
            print("\n✓ System shutdown complete")


class TestDetector:
    """Test detector with simulated taps"""
    
    def __init__(self, com_port: str = None):
        """Initialize test detector"""
        print("="*70)
        print("🧪 EarthPulse AI - Hardware Test Mode")
        print("="*70)
        
        if com_port is None:
            com_port = find_serial_port()
            if com_port is None:
                com_port = input("Enter COM port: ")
        
        self.receiver = GeophoneReceiver(port=com_port)
        
        # Detection parameters
        self.baseline = None
        self.threshold = 0.05  # 50mV threshold
        self.in_event = False
        
        print("\n✓ Test mode initialized")
        print(f"Threshold: ±{self.threshold}V\n")
    
    def detect_tap(self, sample: dict):
        """Simple tap detection"""
        voltage = sample['voltage']
        
        # Calculate baseline (running average)
        if self.baseline is None:
            self.baseline = voltage
        else:
            self.baseline = 0.99 * self.baseline + 0.01 * voltage
        
        # Check if exceeds threshold
        deviation = abs(voltage - self.baseline)
        
        if deviation > self.threshold:
            if not self.in_event:
                self.in_event = True
                print(f"\n{'='*60}")
                print(f"👆 TAP DETECTED!")
                print(f"{'='*60}")
                print(f"Time: {sample['timestamp']} ms")
                print(f"Voltage: {voltage:.6f} V")
                print(f"Baseline: {self.baseline:.6f} V")
                print(f"Deviation: {deviation:.6f} V ({deviation/self.baseline*100:.1f}%)")
                print(f"{'='*60}\n")
        else:
            self.in_event = False
    
    def start(self):
        """Start test"""
        print("Tap table to test detection...\n")
        try:
            self.receiver.start_streaming(self.detect_tap)
        except KeyboardInterrupt:
            print("\nTest stopped")
            self.receiver.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EarthPulse AI Hardware Integration')
    parser.add_argument('--port', type=str, default=None,
                       help='Serial port (e.g., COM3)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to LSTM model')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (simple tap detection)')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            # Test mode: simple tap detection
            detector = TestDetector(com_port=args.port)
            detector.start()
        else:
            # Full detection mode
            detector = RealTimeDetector(
                com_port=args.port,
                model_path=args.model
            )
            detector.start()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check hardware connections (see BREADBOARD_WIRING_GUIDE.md)")
        print("2. Verify ESP32 firmware is uploaded")
        print("3. Check Serial Monitor shows 'ADS1115 configured'")
        print("4. Close Arduino Serial Monitor before running this")
        print("5. Verify correct COM port")
