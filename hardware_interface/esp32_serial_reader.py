"""
EarthPulse AI - ESP32 Serial Reader
Reads real-time seismic data from ESP32-S3 via USB
"""

import serial
import serial.tools.list_ports
import numpy as np
import time
import threading
import queue
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class ESP32Config:
    """ESP32 connection configuration"""
    port: str = "COM3"  # Windows default
    baudrate: int = 115200
    timeout: float = 1.0
    buffer_size: int = 1000  # Expected samples per packet


class ESP32SerialReader:
    """
    Reads seismic data from ESP32-S3 via USB Serial
    
    Data Protocol:
    - ESP32 sends: START,sample1,sample2,...,sampleN,END
    - Sampling rate: 1000 Hz
    - Buffer size: 1000 samples (1 second)
    """
    
    def __init__(self, config: Optional[ESP32Config] = None):
        """Initialize serial reader"""
        self.config = config or ESP32Config()
        self.serial_port: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_reading = False
        
        # Data buffers
        self.data_queue = queue.Queue(maxsize=10)
        self.latest_signal: Optional[np.ndarray] = None
        
        # Threading
        self.read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.packets_received = 0
        self.packets_dropped = 0
        self.last_packet_time = 0
        
    def list_available_ports(self):
        """List all available serial ports"""
        ports = serial.tools.list_ports.comports()
        available = []
        
        print("\n" + "="*60)
        print("Available Serial Ports:")
        print("="*60)
        
        if not ports:
            print("No serial ports found!")
            return []
        
        for i, port in enumerate(ports, 1):
            print(f"{i}. {port.device}")
            print(f"   Description: {port.description}")
            print(f"   Manufacturer: {port.manufacturer}")
            
            # Highlight ESP32 devices
            if "USB" in port.description or "CH340" in port.description or \
               "CP210" in port.description or "Serial" in port.description:
                print(f"   *** Likely ESP32 Device ***")
            print()
            
            available.append(port.device)
        
        return available
    
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to ESP32 device
        
        Args:
            port: Serial port (e.g., 'COM3' or '/dev/ttyUSB0')
                 If None, uses config.port
        
        Returns:
            True if connected successfully
        """
        if port:
            self.config.port = port
        
        try:
            print(f"\nConnecting to ESP32 on {self.config.port}...")
            
            self.serial_port = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Wait for device to stabilize
            time.sleep(2)
            
            # Clear any stale data
            self.serial_port.reset_input_buffer()
            
            self.is_connected = True
            print(f"✓ Connected to {self.config.port}")
            
            # Try to read device info
            self._send_command("STATUS")
            time.sleep(0.5)
            self._print_device_info()
            
            return True
            
        except serial.SerialException as e:
            print(f"✗ Failed to connect: {e}")
            print(f"\nTroubleshooting:")
            print(f"  1. Check USB cable is connected")
            print(f"  2. Verify ESP32 is powered on")
            print(f"  3. Try a different USB port")
            print(f"  4. Check if port is correct: {self.config.port}")
            print(f"  5. Close any other programs using the serial port")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        self.stop_reading()
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print(f"\n✓ Disconnected from {self.config.port}")
        
        self.is_connected = False
    
    def start_reading(self, callback: Optional[Callable] = None):
        """
        Start reading data in background thread
        
        Args:
            callback: Optional function to call with each data packet
                     Signature: callback(signal_data: np.ndarray)
        """
        if not self.is_connected:
            print("✗ Not connected to ESP32!")
            return
        
        if self.is_reading:
            print("Already reading data")
            return
        
        self.is_reading = True
        self._stop_event.clear()
        
        self.read_thread = threading.Thread(
            target=self._read_loop,
            args=(callback,),
            daemon=True
        )
        self.read_thread.start()
        
        print("✓ Started reading data from ESP32")
    
    def stop_reading(self):
        """Stop reading data"""
        if not self.is_reading:
            return
        
        self.is_reading = False
        self._stop_event.set()
        
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        
        print("✓ Stopped reading data")
    
    def get_latest_signal(self) -> Optional[np.ndarray]:
        """Get the most recent signal data"""
        return self.latest_signal
    
    def get_signal_blocking(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Wait for and return next signal packet (blocking)
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            Signal array or None if timeout
        """
        try:
            signal = self.data_queue.get(timeout=timeout)
            return signal
        except queue.Empty:
            print(f"✗ Timeout waiting for data ({timeout}s)")
            return None
    
    def _read_loop(self, callback: Optional[Callable]):
        """Background thread for reading serial data"""
        print("Reading thread started...")
        
        while self.is_reading and not self._stop_event.is_set():
            try:
                # Read line from serial
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                
                if not line:
                    continue
                
                # Handle single-line format FIRST: START,packet_num,val1,val2,...,valN,END
                if "START," in line:  # This is the actual format from ESP32
                    signal = self._parse_single_line_packet(line)
                    
                    if signal is not None and len(signal) > 0:
                        # Store latest signal
                        self.latest_signal = signal
                        
                        # Add to queue (non-blocking)
                        try:
                            self.data_queue.put(signal, block=False)
                        except queue.Full:
                            self.packets_dropped += 1
                        
                        # Update statistics
                        self.packets_received += 1
                        self.last_packet_time = time.time()
                        
                        # Call user callback
                        if callback:
                            callback(signal)
                
                # Parse multi-line data packet (legacy format)
                elif line.startswith("START"):
                    signal = self._parse_data_packet()
                    
                    if signal is not None and len(signal) > 0:
                        # Store latest signal
                        self.latest_signal = signal
                        
                        # Add to queue (non-blocking)
                        try:
                            self.data_queue.put(signal, block=False)
                        except queue.Full:
                            self.packets_dropped += 1
                        
                        # Update statistics
                        self.packets_received += 1
                        self.last_packet_time = time.time()
                        
                        # Call user callback
                        if callback:
                            callback(signal)
                
            except Exception as e:
                if self.is_reading:  # Only print if we're supposed to be reading
                    print(f"Read error: {e}")
                time.sleep(0.1)
        
        print("Reading thread stopped")
    
    def _parse_single_line_packet(self, line: str) -> Optional[np.ndarray]:
        """
        Parse single-line data packet from ESP32
        Format: START,packet_num,val1,val2,...,valN,END or variations
        Can handle partial lines (line may be truncated if END not present)
        """
        try:
            # Handle case where we have START but need to read until END
            if "START," in line and "END" not in line:
                # We have a partial line, keep reading until we find END
                data_parts = [line]
                while True:
                    next_line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    data_parts.append(next_line)
                    if "END" in next_line or not next_line:
                        break
                    if len(data_parts) > 10:  # Safety limit
                        break
                line = ''.join(data_parts)
            
            # Extract data between START and END
            if "START," in line:
                start_idx = line.find("START,") + 6  # After "START,"
            else:
                return None
            
            if ",END" in line:
                end_idx = line.find(",END")
            elif "END" in line:
                end_idx = line.find("END")
            else:
                # No END marker, take rest of line
                end_idx = len(line)
            
            # Extract comma-separated values
            data_str = line[start_idx:end_idx]
            values = data_str.split(',')
            
            # Convert to floats, skip first value (packet number) and any non-numeric values
            samples = []
            for i, val in enumerate(values):
                if not val or not val.strip():
                    continue
                try:
                    # Skip the first value (packet number)
                    if i == 0:
                        continue
                    samples.append(float(val))
                except ValueError:
                    continue
            
            if len(samples) > 0:
                signal = np.array(samples, dtype=np.float32)
                return signal
            
            return None
            
        except Exception as e:
            print(f"Parse error (single-line): {e}")
            return None
    
    def _parse_data_packet(self) -> Optional[np.ndarray]:
        """
        Parse data packet from ESP32
        Format: START,packet_num,val1,val2,...,valN,END or START\nval1,val2,...\nEND
        """
        try:
            samples = []
            first_line = True
            
            # Read samples until END marker
            while True:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                
                if line == "END" or line.endswith(",END"):
                    # Handle END on same line as data
                    if line.endswith(",END"):
                        line = line[:-4]  # Remove ",END"
                        if line:
                            try:
                                values = [float(x) for x in line.split(',') if x]
                                # Skip first value (packet number) if it's the first line
                                if first_line and len(values) > 0:
                                    values = values[1:]  # Skip packet number
                                samples.extend(values)
                            except ValueError:
                                pass
                    break
                
                # Parse comma-separated values
                try:
                    values = [float(x) for x in line.split(',') if x]
                    # Skip first value (packet number) if it's the first line
                    if first_line and len(values) > 0:
                        # First value is usually the packet number, skip it
                        values = values[1:]
                    samples.extend(values)
                    first_line = False
                except ValueError:
                    continue
                
                # Safety check - prevent infinite loop
                if len(samples) > self.config.buffer_size * 2:
                    print("⚠ Packet too large, discarding")
                    return None
            
            # Convert to numpy array
            if len(samples) > 0:
                signal = np.array(samples, dtype=np.float32)
                return signal
            
            return None
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def _send_command(self, command: str):
        """Send command to ESP32"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write(f"{command}\n".encode())
    
    def _print_device_info(self):
        """Print device status information"""
        if not self.serial_port or not self.serial_port.is_open:
            return
        
        # Read any available status output
        time.sleep(0.2)
        while self.serial_port.in_waiting:
            line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)
    
    def get_statistics(self) -> dict:
        """Get reader statistics"""
        return {
            'connected': self.is_connected,
            'reading': self.is_reading,
            'packets_received': self.packets_received,
            'packets_dropped': self.packets_dropped,
            'drop_rate': self.packets_dropped / max(self.packets_received, 1),
            'last_packet_time': self.last_packet_time,
            'queue_size': self.data_queue.qsize()
        }
    
    def print_statistics(self):
        """Print reader statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("ESP32 Reader Statistics")
        print("="*60)
        print(f"Connected:        {stats['connected']}")
        print(f"Reading:          {stats['reading']}")
        print(f"Packets Received: {stats['packets_received']}")
        print(f"Packets Dropped:  {stats['packets_dropped']}")
        print(f"Drop Rate:        {stats['drop_rate']*100:.2f}%")
        print(f"Queue Size:       {stats['queue_size']}")
        
        if stats['last_packet_time'] > 0:
            elapsed = time.time() - stats['last_packet_time']
            print(f"Last Packet:      {elapsed:.1f}s ago")
        print("="*60)


def demo():
    """Demo usage of ESP32SerialReader"""
    print("="*60)
    print("EarthPulse AI - ESP32 Serial Reader Demo")
    print("="*60)
    
    # Create reader
    reader = ESP32SerialReader()
    
    # List available ports
    ports = reader.list_available_ports()
    
    if not ports:
        print("\n✗ No serial ports found!")
        print("Please connect your ESP32-S3 device via USB")
        return
    
    # Auto-detect or ask user
    port = ports[0] if len(ports) == 1 else input("\nEnter port (e.g., COM3): ").strip()
    
    # Connect
    if not reader.connect(port):
        return
    
    # Callback to print data
    def on_data(signal):
        print(f"\n✓ Received {len(signal)} samples")
        print(f"  Range: [{signal.min():.4f}, {signal.max():.4f}] V")
        print(f"  Mean:  {signal.mean():.4f} V")
        print(f"  RMS:   {np.sqrt(np.mean(signal**2)):.4f} V")
    
    # Start reading
    reader.start_reading(callback=on_data)
    
    print("\nReading data from ESP32...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
            
            # Print stats every 10 packets
            if reader.packets_received % 10 == 0 and reader.packets_received > 0:
                reader.print_statistics()
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        reader.disconnect()
        print("\n✓ Demo complete")


if __name__ == "__main__":
    demo()
