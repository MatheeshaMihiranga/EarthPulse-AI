"""
EarthPulse AI - Live Matplotlib Graph for ESP32-S3 Geophone
Real-time visualization of differential voltage from hardware sensor
Similar to Raspberry Pi Differential-Graph-Display.py but for ESP32-S3 on Windows
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.hardware_receiver import GeophoneReceiver, find_serial_port
import numpy as np
from collections import deque

class LiveGeophoneGraph:
    """Live matplotlib graph for ESP32-S3 geophone data"""
    
    def __init__(self, com_port: str = None, x_len: int = 500):
        """
        Initialize live graph
        
        Args:
            com_port: Serial port (None = auto-detect)
            x_len: Number of data points to display
        """
        # Auto-detect port if not specified
        if com_port is None:
            print("Auto-detecting ESP32...")
            com_port = find_serial_port()
            if com_port is None:
                print("\n❌ Could not auto-detect ESP32")
                com_port = input("Enter COM port (e.g., COM5): ")
        
        # Connect to hardware
        print(f"\nConnecting to {com_port}...")
        self.receiver = GeophoneReceiver(port=com_port)
        print("✓ Connected successfully!\n")
        
        # Graph settings
        self.x_len = x_len
        self.y_range = [-0.5, 0.5]  # Voltage range (adjust based on your signal)
        
        # Data buffer
        self.data_buffer = deque(maxlen=x_len)
        for _ in range(x_len):
            self.data_buffer.append(0.0)
        
        # Sample counter
        self.sample_count = 0
        
        # Setup plot
        self.fig = plt.figure(figsize=(12, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_ylim(self.y_range)
        self.ax.set_xlim([0, x_len])
        
        # Create line
        self.xs = list(range(0, x_len))
        self.line, = self.ax.plot(self.xs, list(self.data_buffer), 'b-', linewidth=1)
        
        # Add labels
        plt.title('ESP32-S3 Geophone Data (Differential Voltage)', fontsize=16, fontweight='bold')
        plt.xlabel('Data Points (Sampling at ~1000 Hz)', fontsize=12)
        plt.ylabel('Differential Voltage (V)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add status text
        self.status_text = self.ax.text(0.02, 0.98, '', 
                                        transform=self.ax.transAxes,
                                        verticalalignment='top',
                                        fontfamily='monospace',
                                        fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def animate(self, frame):
        """Animation function called periodically"""
        try:
            # Read sample from ESP32
            sample = self.receiver.read_sample()
            
            if sample is not None:
                # Get differential voltage
                voltage = sample['voltage']
                
                # Add to buffer
                self.data_buffer.append(voltage)
                self.sample_count += 1
                
                # Update line
                self.line.set_ydata(list(self.data_buffer))
                
                # Update status text
                status = (f"Samples: {self.sample_count:,}\n"
                         f"Current: {voltage:+.4f} V\n"
                         f"Min: {min(self.data_buffer):+.4f} V\n"
                         f"Max: {max(self.data_buffer):+.4f} V\n"
                         f"Range: {max(self.data_buffer)-min(self.data_buffer):.4f} V")
                self.status_text.set_text(status)
                
                # Auto-adjust y-axis if signal goes out of range
                current_min = min(self.data_buffer)
                current_max = max(self.data_buffer)
                margin = 0.1  # 10% margin
                
                if current_min < self.y_range[0] or current_max > self.y_range[1]:
                    range_size = current_max - current_min
                    new_min = current_min - margin * range_size
                    new_max = current_max + margin * range_size
                    self.y_range = [new_min, new_max]
                    self.ax.set_ylim(self.y_range)
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Read error: {e}")
        
        return self.line, self.status_text
    
    def run(self):
        """Start the live graph"""
        print("=" * 70)
        print("🐘 EarthPulse AI - Live Geophone Graph")
        print("=" * 70)
        print(f"\n📡 Port: {self.receiver.ser.port}")
        print(f"🔬 Sampling: ~1000 Hz")
        print(f"📊 Display: {self.x_len} points ({self.x_len/1000:.1f} seconds)")
        print("\n� Tap the table near your geophone to see vibrations!")
        print("🔍 Y-axis will auto-scale to fit your signal")
        print("\nClose the graph window to stop\n")
        print("=" * 70 + "\n")
        
        # Start animation
        # interval=1 means update as fast as possible (will be limited by read speed)
        ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=1,
            blit=True,
            cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.receiver.close()
            print("✓ Disconnected")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Matplotlib Graph for ESP32-S3 Geophone')
    parser.add_argument('--port', type=str, default=None,
                       help='COM port (e.g., COM5). Auto-detects if not specified.')
    parser.add_argument('--points', type=int, default=500,
                       help='Number of data points to display (default: 500)')
    
    args = parser.parse_args()
    
    try:
        graph = LiveGeophoneGraph(com_port=args.port, x_len=args.points)
        graph.run()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
