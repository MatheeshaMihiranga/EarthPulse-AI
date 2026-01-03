"""
EarthPulse AI - Virtual IoT Device Simulator
Simulates complete geophone + ESP32 + soil sensor + LoRa system
"""

import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, Optional, Callable
from dataclasses import dataclass, asdict
import threading
import queue


@dataclass
class SensorReadings:
    """Complete sensor readings from virtual device"""
    timestamp: str
    seismic_data: np.ndarray  # Raw geophone ADC values
    soil_moisture: float  # Percentage
    soil_temperature: float  # Celsius
    battery_voltage: float  # Volts
    device_id: str
    sample_rate: int


@dataclass
class DeviceConfig:
    """Virtual device configuration"""
    device_id: str = "EARTHPULSE_001"
    geophone_sensitivity: float = 28.0  # V/(m/s)
    adc_bits: int = 12  # ESP32 ADC resolution
    adc_vref: float = 3.3  # Reference voltage
    sample_rate: int = 1000  # Hz
    buffer_size: int = 1000  # samples
    soil_moisture_pin: int = 34  # ADC pin
    lora_frequency: int = 433  # MHz
    lora_tx_power: int = 20  # dBm
    battery_low_threshold: float = 3.5  # Volts


class VirtualGeophone:
    """Simulates geophone sensor output"""
    
    def __init__(self, sensitivity: float = 28.0):
        """
        Initialize virtual geophone
        
        Args:
            sensitivity: Sensor sensitivity in V/(m/s)
        """
        self.sensitivity = sensitivity
        self.baseline_noise = 0.001  # V (1mV noise floor)
        
    def convert_vibration_to_voltage(self, 
                                    vibration_velocity: np.ndarray) -> np.ndarray:
        """
        Convert ground vibration velocity to voltage output
        
        Args:
            vibration_velocity: Ground velocity in m/s
            
        Returns:
            Voltage output
        """
        # Linear conversion with sensitivity
        voltage = vibration_velocity * self.sensitivity
        
        # Add sensor noise
        noise = np.random.normal(0, self.baseline_noise, len(voltage))
        voltage += noise
        
        return voltage


class VirtualESP32ADC:
    """Simulates ESP32 ADC with realistic characteristics"""
    
    def __init__(self, bits: int = 12, vref: float = 3.3):
        """
        Initialize virtual ADC
        
        Args:
            bits: ADC resolution (12-bit for ESP32)
            vref: Reference voltage
        """
        self.bits = bits
        self.vref = vref
        self.max_value = 2 ** bits - 1
        self.lsb = vref / self.max_value
        
        # ESP32 ADC nonlinearity characteristics
        self.nonlinearity = 0.02  # 2% nonlinearity
        
    def voltage_to_digital(self, voltage: np.ndarray) -> np.ndarray:
        """
        Convert analog voltage to digital ADC value
        
        Args:
            voltage: Input voltage
            
        Returns:
            Digital ADC values (0 to max_value)
        """
        # Clip to ADC range
        voltage_clipped = np.clip(voltage, 0, self.vref)
        
        # Ideal conversion
        digital = (voltage_clipped / self.vref) * self.max_value
        
        # Add ESP32 ADC nonlinearity (slight compression at extremes)
        digital = digital + self.nonlinearity * digital * (1 - digital / self.max_value)
        
        # Quantization
        digital = np.round(digital).astype(np.int16)
        
        # Add quantization noise
        digital = digital + np.random.randint(-1, 2, len(digital))
        digital = np.clip(digital, 0, self.max_value)
        
        return digital
    
    def digital_to_voltage(self, digital: np.ndarray) -> np.ndarray:
        """Convert digital ADC value back to voltage"""
        return (digital / self.max_value) * self.vref


class VirtualSoilMoistureSensor:
    """Simulates capacitive soil moisture sensor"""
    
    def __init__(self, adc: VirtualESP32ADC):
        """
        Initialize soil moisture sensor
        
        Args:
            adc: Virtual ADC for reading
        """
        self.adc = adc
        # Calibration values (wet = low ADC, dry = high ADC)
        self.adc_water = 1000  # Fully submerged
        self.adc_air = 3000  # In air
        
    def read_moisture(self, true_moisture: float) -> float:
        """
        Read soil moisture percentage
        
        Args:
            true_moisture: Actual soil moisture (0-100%)
            
        Returns:
            Measured moisture with sensor noise
        """
        # Convert moisture to ADC value (inverse relationship)
        adc_value = self.adc_air - (true_moisture / 100.0) * (self.adc_air - self.adc_water)
        
        # Add sensor noise
        adc_value += np.random.normal(0, 50)
        adc_value = np.clip(adc_value, self.adc_water, self.adc_air)
        
        # Convert back to percentage
        moisture = (1.0 - (adc_value - self.adc_water) / 
                   (self.adc_air - self.adc_water)) * 100.0
        
        return np.clip(moisture, 0, 100)
    
    def read_temperature(self, ambient_temp: float = 25.0) -> float:
        """Read soil temperature"""
        # Add sensor noise
        return ambient_temp + np.random.normal(0, 0.5)


class VirtualLoRa:
    """Simulates LoRa wireless communication module"""
    
    def __init__(self, frequency: int = 433, tx_power: int = 20):
        """
        Initialize LoRa module
        
        Args:
            frequency: Frequency in MHz
            tx_power: Transmit power in dBm
        """
        self.frequency = frequency
        self.tx_power = tx_power
        self.is_initialized = False
        self.transmission_log = []
        
    def initialize(self) -> bool:
        """Initialize LoRa module"""
        time.sleep(0.1)  # Simulate initialization delay
        self.is_initialized = True
        return True
    
    def transmit_alert(self, alert_data: Dict) -> bool:
        """
        Transmit alert packet
        
        Args:
            alert_data: Alert information
            
        Returns:
            Success status
        """
        if not self.is_initialized:
            return False
        
        # Simulate transmission time (depends on spreading factor)
        time.sleep(0.05)  # ~50ms for small packet
        
        # Log transmission
        transmission = {
            'timestamp': datetime.now().isoformat(),
            'data': alert_data,
            'frequency': self.frequency,
            'tx_power': self.tx_power
        }
        self.transmission_log.append(transmission)
        
        # 95% success rate (simulate occasional packet loss)
        return np.random.random() > 0.05


class VirtualIoTDevice:
    """
    Complete virtual IoT device
    Simulates: Geophone + ESP32 + Soil Sensor + LoRa
    """
    
    def __init__(self, config: Optional[DeviceConfig] = None):
        """Initialize virtual device"""
        self.config = config or DeviceConfig()
        
        # Initialize components
        self.geophone = VirtualGeophone(self.config.geophone_sensitivity)
        self.adc = VirtualESP32ADC(self.config.adc_bits, self.config.adc_vref)
        self.soil_sensor = VirtualSoilMoistureSensor(self.adc)
        self.lora = VirtualLoRa(self.config.lora_frequency, 
                               self.config.lora_tx_power)
        
        # Device state
        self.battery_voltage = 4.2  # Full battery
        self.is_running = False
        self.data_queue = queue.Queue()
        
        # Environmental simulation
        self.current_soil_moisture = 20.0  # %
        self.current_temperature = 25.0  # C
        
    def initialize(self) -> bool:
        """Initialize all device components"""
        print(f"[{self.config.device_id}] Initializing device...")
        
        # Initialize LoRa
        if not self.lora.initialize():
            print("LoRa initialization failed!")
            return False
        
        print(f"[{self.config.device_id}] Device initialized successfully")
        return True
    
    def set_environment(self, soil_moisture: float, temperature: float = 25.0):
        """Set environmental conditions"""
        self.current_soil_moisture = soil_moisture
        self.current_temperature = temperature
    
    def acquire_seismic_data(self, 
                            vibration_signal: np.ndarray) -> np.ndarray:
        """
        Simulate complete signal acquisition pipeline
        
        Args:
            vibration_signal: Ground vibration velocity in m/s
            
        Returns:
            Digital ADC values
        """
        # Step 1: Geophone converts vibration to voltage
        voltage = self.geophone.convert_vibration_to_voltage(vibration_signal)
        
        # Step 2: ESP32 ADC digitizes voltage
        digital = self.adc.voltage_to_digital(voltage)
        
        return digital
    
    def read_sensors(self, vibration_signal: np.ndarray) -> SensorReadings:
        """
        Read all sensors and return complete data packet
        
        Args:
            vibration_signal: Current vibration signal
            
        Returns:
            Complete sensor readings
        """
        # Acquire seismic data
        seismic_data = self.acquire_seismic_data(vibration_signal)
        
        # Read soil sensors
        soil_moisture = self.soil_sensor.read_moisture(self.current_soil_moisture)
        soil_temp = self.soil_sensor.read_temperature(self.current_temperature)
        
        # Simulate battery discharge
        self.battery_voltage -= 0.0001  # Slow discharge
        self.battery_voltage = max(self.battery_voltage, 3.0)
        
        readings = SensorReadings(
            timestamp=datetime.now().isoformat(),
            seismic_data=seismic_data,
            soil_moisture=soil_moisture,
            soil_temperature=soil_temp,
            battery_voltage=self.battery_voltage,
            device_id=self.config.device_id,
            sample_rate=self.config.sample_rate
        )
        
        return readings
    
    def send_alert(self, alert_type: str, confidence: float, 
                   details: Optional[Dict] = None) -> bool:
        """
        Send alert via LoRa
        
        Args:
            alert_type: Type of alert ("elephant_detected", etc.)
            confidence: Detection confidence (0-1)
            details: Additional information
            
        Returns:
            Success status
        """
        alert_data = {
            'device_id': self.config.device_id,
            'alert_type': alert_type,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'battery': self.battery_voltage,
            'soil_moisture': self.current_soil_moisture,
            'details': details or {}
        }
        
        success = self.lora.transmit_alert(alert_data)
        
        if success:
            print(f"[{self.config.device_id}] ALERT SENT: {alert_type} "
                  f"(confidence: {confidence:.2f})")
        else:
            print(f"[{self.config.device_id}] Alert transmission failed")
        
        return success
    
    def stream_data(self, 
                   signal_generator: Callable,
                   duration: float = 60.0,
                   callback: Optional[Callable] = None):
        """
        Stream data in real-time simulation
        
        Args:
            signal_generator: Function that generates vibration signals
            duration: Total duration in seconds
            callback: Optional callback for each data packet
        """
        self.is_running = True
        start_time = time.time()
        
        print(f"[{self.config.device_id}] Starting data stream for {duration}s")
        
        while self.is_running and (time.time() - start_time) < duration:
            # Generate vibration signal for this time window
            window_duration = self.config.buffer_size / self.config.sample_rate
            vibration = signal_generator(window_duration)
            
            # Read sensors
            readings = self.read_sensors(vibration)
            
            # Put in queue
            self.data_queue.put(readings)
            
            # Call callback if provided
            if callback:
                callback(readings)
            
            # Check battery
            if self.battery_voltage < self.config.battery_low_threshold:
                print(f"[{self.config.device_id}] WARNING: Low battery!")
            
            # Sleep to maintain sample rate
            time.sleep(window_duration)
        
        self.is_running = False
        print(f"[{self.config.device_id}] Data stream ended")
    
    def stop_streaming(self):
        """Stop data streaming"""
        self.is_running = False
    
    def get_status(self) -> Dict:
        """Get device status"""
        return {
            'device_id': self.config.device_id,
            'battery_voltage': self.battery_voltage,
            'soil_moisture': self.current_soil_moisture,
            'temperature': self.current_temperature,
            'is_running': self.is_running,
            'lora_initialized': self.lora.is_initialized,
            'queue_size': self.data_queue.qsize()
        }


if __name__ == "__main__":
    # Demo: Virtual device operation
    print("EarthPulse AI - Virtual IoT Device Demo")
    print("=" * 60)
    
    # Create device
    device = VirtualIoTDevice()
    
    # Initialize
    device.initialize()
    
    # Set environmental conditions
    device.set_environment(soil_moisture=22.0, temperature=26.0)
    
    print("\nDevice Status:")
    print("-" * 60)
    status = device.get_status()
    for key, value in status.items():
        print(f"  {key:20s}: {value}")
    
    # Simulate reading with synthetic signal
    print("\nAcquiring seismic data...")
    
    # Generate test vibration signal
    fs = device.config.sample_rate
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Simulated elephant footfall
    vibration = np.zeros_like(t)
    step_time = 0.5
    mask = (t >= step_time) & (t < step_time + 0.3)
    t_step = t[mask] - step_time
    # Ground velocity in m/s
    vibration[mask] = 0.001 * np.sin(2 * np.pi * 10 * t_step) * np.exp(-5 * t_step)
    
    # Read sensors
    readings = device.read_sensors(vibration)
    
    print(f"\nSensor Readings:")
    print(f"  Timestamp: {readings.timestamp}")
    print(f"  Seismic data: {len(readings.seismic_data)} samples")
    print(f"  ADC range: {np.min(readings.seismic_data)} to {np.max(readings.seismic_data)}")
    print(f"  Soil moisture: {readings.soil_moisture:.1f}%")
    print(f"  Soil temperature: {readings.soil_temperature:.1f}°C")
    print(f"  Battery: {readings.battery_voltage:.2f}V")
    
    # Test alert transmission
    print("\nTesting alert transmission...")
    device.send_alert(
        alert_type="elephant_detected",
        confidence=0.92,
        details={"distance_estimate": 50, "num_individuals": 1}
    )
    
    print("\n" + "=" * 60)
    print("Virtual device operational!")
    print(f"LoRa transmissions: {len(device.lora.transmission_log)}")
