"""
Test ESP32 Serial Connection
Quick test to verify your hardware is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware_interface.esp32_serial_reader import ESP32SerialReader, ESP32Config
import numpy as np
import time


def test_connection():
    """Test basic serial connection"""
    print("="*70)
    print("🔧 EarthPulse AI - Hardware Connection Test")
    print("="*70)
    print()
    
    # Create reader
    reader = ESP32SerialReader()
    
    # List ports
    print("Step 1: Finding serial ports...")
    ports = reader.list_available_ports()
    
    if not ports:
        print("\n❌ FAILED: No serial ports found")
        print("\nTroubleshooting:")
        print("  1. Connect ESP32-S3 via USB-C cable")
        print("  2. Install USB drivers (CH340 or CP210x)")
        print("  3. Check Device Manager (Windows) for COM ports")
        return False
    
    print("\n✅ PASSED: Found serial ports")
    
    # Select port
    if len(ports) == 1:
        port = ports[0]
        print(f"\nUsing port: {port}")
    else:
        port = input("\nEnter port (e.g., COM3): ").strip()
    
    # Test connection
    print("\nStep 2: Connecting to ESP32...")
    if not reader.connect(port):
        print("\n❌ FAILED: Could not connect")
        print("\nTroubleshooting:")
        print("  1. Check USB cable is connected")
        print("  2. Verify ESP32 firmware is uploaded")
        print("  3. Try pressing RST button on ESP32")
        print("  4. Close Arduino IDE Serial Monitor if open")
        return False
    
    print("✅ PASSED: Connected to ESP32")
    
    # Test data reception
    print("\nStep 3: Testing data reception...")
    print("Waiting for data packets (10 seconds)...")
    
    # Clear any buffered data from STATUS command
    time.sleep(0.5)
    if reader.serial_port:
        reader.serial_port.reset_input_buffer()
    
    packets_received = 0
    start_time = time.time()
    
    def count_packets(signal):
        nonlocal packets_received
        packets_received += 1
        print(f"  Packet {packets_received}: {len(signal)} samples, "
              f"RMS={np.sqrt(np.mean(signal**2)):.4f}V")
    
    reader.start_reading(callback=count_packets)
    
    # Wait for packets
    time.sleep(10)
    
    reader.stop_reading()
    reader.disconnect()
    
    # Evaluate results
    print()
    if packets_received == 0:
        print("❌ FAILED: No data received")
        print("\nTroubleshooting:")
        print("  1. Check ADS1115 wiring:")
        print("     • SDA → GPIO 8")
        print("     • SCL → GPIO 9")
        print("     • VDD → 3.3V")
        print("     • GND → GND")
        print("     • ADDR → GND")
        print("  2. Open Arduino Serial Monitor and check for errors")
        print("  3. Type 'STATUS' in Serial Monitor to verify system")
        return False
    
    elif packets_received < 8:
        print(f"⚠️  WARNING: Only {packets_received} packets received (expected ~10)")
        print("System may be slow or dropping packets")
        print("\nSuggestions:")
        print("  1. Try different USB port")
        print("  2. Check USB cable quality")
        print("  3. Close other programs to free system resources")
        return True
    
    else:
        print(f"✅ PASSED: Received {packets_received} packets")
        print("\n" + "="*70)
        print("🎉 SUCCESS! Your hardware is working correctly!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Mount geophone on ground (bury 10-15cm in soil)")
        print("  2. Run detection:")
        print("     python hardware_interface/realtime_detection_hardware.py --port", port)
        print()
        return True


def test_signal_quality():
    """Test signal quality with physical vibration"""
    print("\n" + "="*70)
    print("🧪 Signal Quality Test")
    print("="*70)
    print("\nThis test will measure signal levels with and without vibration.")
    print()
    
    # Create reader
    reader = ESP32SerialReader()
    
    # Get port
    ports = reader.list_available_ports()
    if not ports:
        print("❌ No serial ports found")
        return
    
    port = ports[0] if len(ports) == 1 else input("Enter port: ").strip()
    
    if not reader.connect(port):
        print("❌ Connection failed")
        return
    
    reader.start_reading()
    
    try:
        # Baseline measurement
        print("Step 1: Measuring baseline (do NOT touch anything)...")
        print("Collecting 5 samples...")
        
        baseline_rms = []
        for i in range(5):
            signal = reader.get_signal_blocking(timeout=5.0)
            if signal is not None:
                rms = np.sqrt(np.mean(signal**2))
                baseline_rms.append(rms)
                print(f"  Sample {i+1}: RMS = {rms:.6f} V")
        
        if not baseline_rms:
            print("❌ Failed to get baseline samples")
            return
        
        avg_baseline = np.mean(baseline_rms)
        print(f"\n✓ Average baseline: {avg_baseline:.6f} V")
        
        # Vibration test
        print("\nStep 2: Vibration test")
        print("⚠️  TAP the ground near the geophone NOW!")
        print("(Collecting 5 samples...)")
        
        vibration_rms = []
        for i in range(5):
            signal = reader.get_signal_blocking(timeout=5.0)
            if signal is not None:
                rms = np.sqrt(np.mean(signal**2))
                vibration_rms.append(rms)
                print(f"  Sample {i+1}: RMS = {rms:.6f} V")
        
        if not vibration_rms:
            print("❌ Failed to get vibration samples")
            return
        
        max_vibration = max(vibration_rms)
        print(f"\n✓ Maximum vibration: {max_vibration:.6f} V")
        
        # Analysis
        print("\n" + "="*70)
        print("📊 Signal Quality Analysis")
        print("="*70)
        
        ratio = max_vibration / avg_baseline if avg_baseline > 0 else 0
        
        print(f"Baseline RMS:     {avg_baseline:.6f} V")
        print(f"Vibration RMS:    {max_vibration:.6f} V")
        print(f"Signal-to-Noise:  {ratio:.2f}x")
        print()
        
        if ratio > 5:
            print("✅ EXCELLENT: Strong signal response")
            print("   Your geophone is working perfectly!")
        elif ratio > 2:
            print("✅ GOOD: Clear signal detection")
            print("   System should work well for elephant detection")
        elif ratio > 1.2:
            print("⚠️  FAIR: Weak signal response")
            print("   Consider:")
            print("   • Increase ADS1115 gain (GAIN_FOUR or GAIN_EIGHT)")
            print("   • Improve geophone coupling to ground")
            print("   • Check signal conditioning resistors")
        else:
            print("❌ POOR: No clear signal response")
            print("   Troubleshooting needed:")
            print("   • Verify geophone is connected properly")
            print("   • Check 1kΩ resistors are correct")
            print("   • Test geophone with multimeter (should show voltage when tapped)")
        
        print("="*70)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    
    finally:
        reader.stop_reading()
        reader.disconnect()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quality":
        test_signal_quality()
    else:
        success = test_connection()
        
        if success:
            response = input("\nRun signal quality test? (y/n): ").strip().lower()
            if response == 'y':
                test_signal_quality()
