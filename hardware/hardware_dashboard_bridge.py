"""
EarthPulse AI - Real-Time Hardware Dashboard Integration
Connect hardware geophone sensor to the dashboard for live visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.hardware_receiver import GeophoneReceiver, find_serial_port
import numpy as np
import time
from collections import deque
import threading
import json

class HardwareDashboardBridge:
    """
    Bridge between hardware geophone and dashboard
    Provides real-time signal data to dashboard callbacks
    """
    
    def __init__(self, com_port: str = None):
        """
        Initialize hardware bridge
        
        Args:
            com_port: Serial port (None = auto-detect)
        """
        print("="*70)
        print("🔗 EarthPulse AI - Hardware Dashboard Bridge")
        print("="*70)
        
        # Auto-detect port
        if com_port is None:
            com_port = find_serial_port()
            if com_port is None:
                print("\n❌ Could not auto-detect ESP32")
                com_port = input("Enter COM port (e.g., COM5): ")
        
        # Connect to hardware
        self.receiver = GeophoneReceiver(port=com_port)
        
        # Signal buffer (shared between threads)
        self.buffer_size = 1000  # 1 second at 1000 Hz
        self.signal_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Metadata
        self.samples_received = 0
        self.last_sample_time = time.time()
        
        # Streaming thread
        self.streaming = False
        self.stream_thread = None
        
        print("\n✓ Hardware bridge initialized")
        print(f"✓ Connected to: {com_port}")
        
    def _stream_worker(self):
        """Background thread worker for streaming data"""
        print("Starting hardware data stream...")
        
        def process_sample(sample):
            """Process each sample from hardware"""
            with self.buffer_lock:
                self.signal_buffer.append(sample['voltage'])
                self.samples_received += 1
                self.last_sample_time = time.time()
        
        try:
            self.receiver.start_streaming(process_sample)
        except Exception as e:
            print(f"Stream error: {e}")
            self.streaming = False
    
    def start_streaming(self):
        """Start background streaming thread"""
        if not self.streaming:
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.stream_thread.start()
            print("✓ Hardware streaming started")
    
    def stop_streaming(self):
        """Stop streaming"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        self.receiver.close()
        print("✓ Hardware streaming stopped")
    
    def get_signal(self, num_samples: int = 1000) -> np.ndarray:
        """
        Get recent signal data
        
        Args:
            num_samples: Number of samples to return
        
        Returns:
            NumPy array of voltage values
        """
        with self.buffer_lock:
            if len(self.signal_buffer) == 0:
                # Return zeros if no data yet
                return np.zeros(num_samples)
            
            # Get most recent samples
            buffer_array = np.array(list(self.signal_buffer))
            
            if len(buffer_array) < num_samples:
                # Pad with zeros if not enough samples
                padding = np.zeros(num_samples - len(buffer_array))
                return np.concatenate([padding, buffer_array])
            else:
                # Return most recent num_samples
                return buffer_array[-num_samples:]
    
    def get_stats(self) -> dict:
        """Get streaming statistics"""
        return {
            'samples_received': self.samples_received,
            'buffer_size': len(self.signal_buffer),
            'sample_rate': self.samples_received / (time.time() - self.receiver.samples_received + 1),
            'is_streaming': self.streaming
        }


# Global hardware bridge instance (shared with dashboard)
_hardware_bridge = None

def get_hardware_bridge(com_port: str = None) -> HardwareDashboardBridge:
    """Get or create hardware bridge singleton"""
    global _hardware_bridge
    if _hardware_bridge is None:
        _hardware_bridge = HardwareDashboardBridge(com_port)
        _hardware_bridge.start_streaming()
    return _hardware_bridge


if __name__ == "__main__":
    """Test hardware bridge"""
    print("Testing Hardware Bridge...\n")
    
    # Create bridge
    bridge = get_hardware_bridge()
    
    print("\nCollecting data for 5 seconds...")
    print("Tap table to see signal changes!\n")
    
    for i in range(5):
        time.sleep(1)
        signal = bridge.get_signal(1000)
        stats = bridge.get_stats()
        
        print(f"Second {i+1}:")
        print(f"  Samples: {stats['samples_received']}")
        print(f"  Buffer: {stats['buffer_size']}")
        print(f"  Signal RMS: {np.sqrt(np.mean(signal**2)):.6f} V")
        print(f"  Signal range: {signal.min():.4f} to {signal.max():.4f} V")
        print()
    
    print("✓ Test complete!")
    bridge.stop_streaming()
