"""
EarthPulse AI - Real-Time Detection with Hardware
Connects to ESP32-S3 and runs detection on real geophone data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from hardware_interface.esp32_serial_reader import ESP32SerialReader, ESP32Config
from edge_firmware_simulated.detection_system import ElephantDetectionSystem
from colorama import init, Fore, Back, Style
import argparse

init(autoreset=True)  # Initialize colorama for Windows


def print_banner():
    """Print application banner"""
    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + "🐘 EarthPulse AI - Real-Time Hardware Detection")
    print("="*70)
    print(Fore.YELLOW + "Connecting to ESP32-S3 geophone sensor...")
    print()


def print_detection_result(result: dict, signal_stats: dict):
    """Print formatted detection result"""
    print("\n" + "─"*70)
    
    # Signal statistics
    print(Fore.CYAN + "📊 Signal Statistics:")
    print(f"   Samples:    {signal_stats['samples']}")
    print(f"   RMS:        {signal_stats['rms']:.4f} V")
    print(f"   Peak-Peak:  {signal_stats['peak_to_peak']:.4f} V")
    print(f"   Mean:       {signal_stats['mean']:.4f} V")
    
    # Detection result
    if result['detected']:
        print("\n" + Fore.GREEN + Style.BRIGHT + "🐘 ELEPHANT DETECTED!")
        print(Fore.GREEN + f"   Confidence: {result['confidence']*100:.1f}%")
        
        # Direction information
        if 'direction' in result:
            direction = result['direction']
            status_icon = "⬆️" if direction['approaching'] else "⬇️"
            status_color = Fore.RED if direction['approaching'] else Fore.YELLOW
            
            print(f"\n{Fore.MAGENTA}📍 Movement Direction:")
            print(f"   Status:     {status_color}{status_icon} {direction['status']}")
            print(f"   Direction:  {direction['cardinal']}")
            print(f"   Distance:   {direction['distance']:.1f} m")
            print(f"   Velocity:   {direction['velocity']:.2f} m/s")
            print(f"   Confidence: {direction['confidence']*100:.1f}%")
        
        # Behavior information
        if 'behavior' in result:
            behavior = result['behavior']
            behavior_emoji = {
                'walking': '🚶',
                'running': '🏃',
                'feeding': '🌿',
                'standing': '🧍',
                'bathing': '💦',
                'unknown': '❓'
            }
            emoji = behavior_emoji.get(behavior['type'], '❓')
            
            print(f"\n{Fore.CYAN}🐘 Behavior Analysis:")
            print(f"   Activity:   {emoji} {behavior['type'].title()}")
            print(f"   Gait Speed: {behavior['gait_speed']:.2f} m/s")
            print(f"   Activity:   {behavior['activity_level'].title()}")
            print(f"   Weight Est: {behavior['estimated_weight']:.0f} kg")
            print(f"   Confidence: {behavior['confidence']*100:.1f}%")
        
        # Alert
        print("\n" + Back.RED + Fore.WHITE + Style.BRIGHT + " ⚠️  ALERT: Elephant in vicinity - Take precautions! ")
        
    else:
        print("\n" + Fore.YELLOW + "○ No elephant detected")
        print(Fore.YELLOW + f"   Confidence: {result['confidence']*100:.1f}%")
        print(Fore.YELLOW + f"   Classification: {result['classification']}")
    
    print("─"*70)


def main():
    """Main application loop"""
    parser = argparse.ArgumentParser(description='EarthPulse AI Hardware Detection')
    parser.add_argument('--port', type=str, default=None, 
                       help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=115200, 
                       help='Baud rate (default: 115200)')
    parser.add_argument('--model', type=str, default='./models/lstm_model.h5',
                       help='Path to trained model')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous monitoring mode')
    parser.add_argument('--log', type=str, default=None,
                       help='Log file path (optional)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Initialize hardware reader
    config = ESP32Config(
        port=args.port or "COM3",
        baudrate=args.baud
    )
    reader = ESP32SerialReader(config)
    
    # List available ports if needed
    if args.port is None:
        print(Fore.YELLOW + "No port specified, auto-detecting...")
        ports = reader.list_available_ports()
        
        if not ports:
            print(Fore.RED + "\n✗ No serial ports found!")
            print("Please connect your ESP32-S3 device via USB")
            return
        
        # Use first port if only one available
        if len(ports) == 1:
            config.port = ports[0]
            print(Fore.GREEN + f"\n✓ Auto-selected port: {config.port}")
        else:
            # Ask user to select
            print(f"\nEnter port number (1-{len(ports)}) or full port name:")
            user_input = input("> ").strip()
            
            try:
                idx = int(user_input) - 1
                if 0 <= idx < len(ports):
                    config.port = ports[idx]
                else:
                    print(Fore.RED + "Invalid selection!")
                    return
            except ValueError:
                config.port = user_input
    
    # Connect to ESP32
    if not reader.connect(config.port):
        print(Fore.RED + "\n✗ Failed to connect to ESP32!")
        print("Please check:")
        print("  1. USB cable is connected")
        print("  2. ESP32 firmware is uploaded")
        print("  3. Correct port is selected")
        print("  4. No other program is using the port")
        return
    
    # Initialize detection system
    print(f"\n{Fore.CYAN}Loading detection model...")
    try:
        detector = ElephantDetectionSystem(model_path=args.model)
        print(Fore.GREEN + f"✓ Model loaded: {args.model}")
    except Exception as e:
        print(Fore.RED + f"✗ Failed to load model: {e}")
        reader.disconnect()
        return
    
    # Setup logging if requested
    log_file = None
    if args.log:
        try:
            log_file = open(args.log, 'a')
            log_file.write(f"\n{'='*70}\n")
            log_file.write(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*70}\n")
            print(Fore.GREEN + f"✓ Logging to: {args.log}")
        except Exception as e:
            print(Fore.YELLOW + f"⚠ Failed to open log file: {e}")
    
    # Start reading
    print(f"\n{Fore.GREEN}✓ System ready!")
    print(Fore.CYAN + "\n" + "="*70)
    print("Starting real-time detection...")
    print("="*70)
    
    if args.continuous:
        print(Fore.YELLOW + "Mode: Continuous monitoring")
        print("Press Ctrl+C to stop")
    else:
        print(Fore.YELLOW + "Mode: Single detection")
    
    print()
    
    detection_count = 0
    packet_count = 0
    
    try:
        reader.start_reading()
        
        while True:
            # Get signal from hardware
            signal = reader.get_signal_blocking(timeout=5.0)
            
            if signal is None:
                print(Fore.RED + "⚠ Timeout waiting for data - check ESP32 connection")
                continue
            
            packet_count += 1
            
            # Calculate signal statistics
            signal_stats = {
                'samples': len(signal),
                'rms': np.sqrt(np.mean(signal**2)),
                'peak_to_peak': np.max(signal) - np.min(signal),
                'mean': np.mean(signal),
                'std': np.std(signal),
                'min': np.min(signal),
                'max': np.max(signal)
            }
            
            # Run detection
            timestamp = time.time()
            result = detector.process_signal(signal, timestamp)
            
            # Print result
            print_detection_result(result, signal_stats)
            
            # Log if enabled
            if log_file:
                log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                log_file.write(f"Detected: {result['detected']}\n")
                log_file.write(f"Confidence: {result['confidence']:.4f}\n")
                log_file.write(f"RMS: {signal_stats['rms']:.6f} V\n")
                
                if result['detected']:
                    detection_count += 1
                    log_file.write(f"Direction: {result.get('direction', {}).get('status', 'N/A')}\n")
                    log_file.write(f"Behavior: {result.get('behavior', {}).get('type', 'N/A')}\n")
                
                log_file.flush()
            
            # Count detections
            if result['detected']:
                detection_count += 1
            
            # Exit if not continuous mode
            if not args.continuous:
                break
            
            # Brief pause
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Interrupted by user")
    
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\n{Fore.CYAN}Shutting down...")
        reader.stop_reading()
        reader.disconnect()
        
        if log_file:
            log_file.write(f"\n{'='*70}\n")
            log_file.write(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Total packets: {packet_count}\n")
            log_file.write(f"Detections: {detection_count}\n")
            log_file.write(f"{'='*70}\n")
            log_file.close()
            print(Fore.GREEN + f"✓ Log saved to: {args.log}")
        
        # Print statistics
        print(f"\n{Fore.CYAN}{'='*70}")
        print("Session Statistics")
        print("="*70)
        print(f"Packets processed: {packet_count}")
        print(f"Elephants detected: {detection_count}")
        if packet_count > 0:
            print(f"Detection rate: {detection_count/packet_count*100:.1f}%")
        print("="*70)
        
        print(f"\n{Fore.GREEN}✓ System shutdown complete")


if __name__ == "__main__":
    main()
