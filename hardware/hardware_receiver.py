"""
EarthPulse AI - Hardware Receiver
Connect to ESP32-S3 geophone sensor via serial port
"""

import serial
import json
import numpy as np
from collections import deque
import time
from typing import Optional, Dict, Callable

class GeophoneReceiver:
    """
    Serial communication interface for ESP32-S3 geophone sensor
    
    Receives real-time seismic data from hardware device
    and provides it to the detection system
    """
    
    def __init__(self, port: str = 'COM3', baudrate: int = 115200):
        """
        Initialize serial connection to ESP32-S3
        
        Args:
            port: Serial port (check Arduino IDE -> Tools -> Port)
                  Windows: COM3, COM4, etc.
                  Linux/Mac: /dev/ttyUSB0, /dev/cu.usbserial, etc.
            baudrate: Must match Arduino sketch (115200)
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print(f"✓ Connected to {port} at {baudrate} baud")
            
            # Clear any existing data in buffer
            self.ser.flushInput()
            
        except serial.SerialException as e:
            print(f"❌ Failed to connect to {port}")
            print(f"Error: {e}")
            print("\nTroubleshooting:")
            print("1. Check Arduino IDE -> Tools -> Port for correct COM port")
            print("2. Close Arduino Serial Monitor if open")
            print("3. Unplug and replug USB cable")
            print("4. Install CP210x or CH340 drivers")
            raise
        
        # Data buffer
        self.buffer = deque(maxlen=1000)  # 1 second at 1000 Hz
        
        # Statistics
        self.samples_received = 0
        self.errors = 0
        
    def read_sample(self) -> Optional[Dict]:
        """
        Read one sample from serial port
        
        Returns:
            Dictionary with timestamp, voltage, and ADC value
            None if no valid data available
        """
        try:
            line = self.ser.readline().decode('utf-8').strip()
            
            # Skip non-JSON lines (debug messages, etc.)
            if not line.startswith('{'):
                return None
            
            # Parse JSON
            data = json.loads(line)
            
            # Extract values (supports both old and new firmware formats)
            if 'voltage' in data:
                # New firmware format (filtered differential)
                voltage = data.get('voltage', 0.0)
                adc = data.get('adc', 0)
            else:
                # Old firmware format (separate channels)
                voltage = data.get('diff', 0.0)
                adc = data.get('adc0', 0) - data.get('adc1', 0)
            
            sample = {
                'timestamp': data.get('timestamp', 0),
                'voltage': voltage,
                'adc': adc,
                'raw': data.get('raw', voltage),  # Raw unfiltered (if available)
                'a0': data.get('a0', 0.0),
                'a1': data.get('a1', 0.0)
            }
            
            self.samples_received += 1
            return sample
            
        except json.JSONDecodeError as e:
            self.errors += 1
            if self.errors < 10:  # Only print first few errors
                print(f"JSON parse error: {e}")
            return None
            
        except Exception as e:
            self.errors += 1
            if self.errors < 10:
                print(f"Read error: {e}")
            return None
    
    def read_buffer(self, num_samples: int = 1000) -> np.ndarray:
        """
        Read multiple samples into buffer
        
        Args:
            num_samples: Number of samples to read
        
        Returns:
            NumPy array of voltage values
        """
        samples = []
        start_time = time.time()
        
        print(f"Reading {num_samples} samples...")
        
        while len(samples) < num_samples:
            sample = self.read_sample()
            if sample:
                samples.append(sample['voltage'])
            
            # Timeout after 5 seconds
            if time.time() - start_time > 5:
                print(f"Warning: Only received {len(samples)} samples in 5 seconds")
                break
        
        return np.array(samples)
    
    def start_streaming(self, callback: Callable[[Dict], None]):
        """
        Stream data continuously with callback function
        
        Args:
            callback: Function to call for each sample
                      signature: callback(sample: dict) -> None
        """
        print("\n" + "="*60)
        print("Starting data stream... (Ctrl+C to stop)")
        print("="*60 + "\n")
        
        try:
            while True:
                sample = self.read_sample()
                if sample:
                    callback(sample)
                    
                    # Print statistics every 1000 samples
                    if self.samples_received % 1000 == 0:
                        error_rate = (self.errors / self.samples_received) * 100
                        print(f"\n[Stats] Samples: {self.samples_received:,} | "
                              f"Errors: {self.errors} ({error_rate:.1f}%)\n")
        
        except KeyboardInterrupt:
            print("\n\nStopped streaming")
            self.print_stats()
            self.close()
    
    def print_stats(self):
        """Print connection statistics"""
        print("\n" + "="*60)
        print("Connection Statistics")
        print("="*60)
        print(f"Samples received: {self.samples_received:,}")
        print(f"Errors: {self.errors}")
        if self.samples_received > 0:
            error_rate = (self.errors / self.samples_received) * 100
            print(f"Error rate: {error_rate:.2f}%")
        print("="*60)
    
    def close(self):
        """Close serial connection"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            print("✓ Serial connection closed")


def find_serial_port():
    """
    Auto-detect ESP32-S3 serial port
    
    Returns:
        Port name if found, None otherwise
    """
    import serial.tools.list_ports
    
    ports = list(serial.tools.list_ports.comports())
    
    print("Available serial ports:")
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port.device} - {port.description}")
    
    # Look for common ESP32 identifiers
    esp32_keywords = ['CP210', 'CH340', 'USB Serial', 'ESP32']
    
    for port in ports:
        for keyword in esp32_keywords:
            if keyword.lower() in port.description.lower():
                print(f"\n✓ Detected ESP32 at: {port.device}")
                return port.device
    
    return None


# Test / Demo script
if __name__ == "__main__":
    print("="*60)
    print("EarthPulse AI - Hardware Receiver Test")
    print("="*60)
    
    # Auto-detect port
    port = find_serial_port()
    
    if not port:
        # Manual input
        print("\nCouldn't auto-detect ESP32")
        port = input("Enter COM port (e.g., COM3 or /dev/ttyUSB0): ")
    
    try:
        # Connect to receiver
        receiver = GeophoneReceiver(port=port)
        
        print("\nTest 1: Reading 10 samples...")
        buffer = receiver.read_buffer(num_samples=10)
        print(f"Received {len(buffer)} samples")
        print(f"Values: {buffer[:5]} ...")
        
        print("\n" + "-"*60)
        print("Test 2: Real-time streaming")
        print("         Tap table near geophone to see changes")
        print("-"*60)
        
        # Define callback to print each sample
        def print_sample(sample: Dict):
            """Print formatted sample data"""
            print(f"Time: {sample['timestamp']:8d} ms | "
                  f"Voltage: {sample['voltage']:+.6f} V | "
                  f"ADC: {sample['adc']:6d} | "
                  f"A0: {sample['a0']:.4f} V | "
                  f"A1: {sample['a1']:.4f} V")
        
        # Start streaming
        receiver.start_streaming(print_sample)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. ESP32-S3 is connected via USB")
        print("2. Arduino firmware is uploaded")
        print("3. Correct COM port is selected")
        print("4. Arduino Serial Monitor is closed")
