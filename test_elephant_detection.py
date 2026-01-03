"""
Test script to validate elephant detection performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from synthetic_generator.seismic_signal_generator import SeismicSignalGenerator
from edge_firmware_simulated.detection_system import ElephantDetectionSystem


def test_elephant_detection():
    """Test elephant detection across various conditions"""
    
    print("=" * 70)
    print("EARTHPULSE AI - ELEPHANT DETECTION VALIDATION TEST")
    print("=" * 70)
    print()
    
    # Initialize system
    generator = SeismicSignalGenerator()
    system = ElephantDetectionSystem()
    
    # Test scenarios
    test_cases = [
        # (soil_moisture, distance, elephant_weight, num_tests)
        (15, 30, 4000, "Optimal conditions - Dry soil, close range"),
        (25, 50, 4500, "Moderate conditions - Medium moisture"),
        (35, 80, 3500, "Challenging - High moisture, far distance"),
        (10, 25, 5000, "Very dry soil - Large elephant, close"),
        (30, 60, 3000, "High moisture - Small elephant, medium range"),
    ]
    
    results = []
    
    for soil_moisture, distance, weight, description in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {description}")
        print(f"Conditions: Soil={soil_moisture}%, Distance={distance}m, Weight={weight}kg")
        print(f"{'='*70}")
        
        # Set soil conditions
        generator.set_soil_conditions(moisture=soil_moisture)
        
        detections = []
        confidences = []
        
        # Run multiple tests (need 2 frames for confirmation)
        for i in range(5):
            # Generate elephant signal
            signal, metadata = generator.generate_elephant_footfall(
                duration=1.0,
                num_steps=3,
                distance_m=distance,
                elephant_weight_kg=weight
            )
            
            # Process through detection system
            result = system.process_signal(signal, soil_moisture)
            
            print(f"\nFrame {i+1}:")
            print(f"  Predicted: {result['class_name']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Detected: {result['detected']}")
            
            if 'all_predictions' in result:
                print(f"  Top 3 predictions:")
                sorted_preds = sorted(result['all_predictions'].items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
                for cls, prob in sorted_preds:
                    print(f"    {cls}: {prob:.2%}")
            
            if result.get('detected'):
                detections.append(True)
                confidences.append(result['confidence'])
            elif result.get('reason') == 'awaiting_confirmation':
                print(f"  Status: Awaiting confirmation ({result.get('frames_confirmed', 0)}/2 frames)")
                confidences.append(result['confidence'])
        
        # Calculate success rate
        success_rate = len(detections) / 5
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        results.append({
            'description': description,
            'soil_moisture': soil_moisture,
            'distance': distance,
            'weight': weight,
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'detections': len(detections)
        })
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {len(detections)}/5 confirmed detections")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Confidence: {avg_confidence:.2%}")
        print(f"{'='*70}")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY - ELEPHANT DETECTION PERFORMANCE")
    print("=" * 70)
    print()
    
    for r in results:
        print(f"{r['description']}")
        print(f"  Soil: {r['soil_moisture']}% | Distance: {r['distance']}m | Weight: {r['weight']}kg")
        print(f"  ✓ Detections: {r['detections']}/5 ({r['success_rate']:.0%})")
        print(f"  ✓ Avg Confidence: {r['avg_confidence']:.1%}")
        print()
    
    # Overall statistics
    overall_success = np.mean([r['success_rate'] for r in results])
    overall_confidence = np.mean([r['avg_confidence'] for r in results])
    
    print("=" * 70)
    print(f"OVERALL PERFORMANCE:")
    print(f"  Detection Rate: {overall_success:.1%}")
    print(f"  Average Confidence: {overall_confidence:.1%}")
    print("=" * 70)
    print()
    
    # System stats
    stats = system.get_statistics()
    print(f"System Statistics:")
    print(f"  Total Predictions: {stats['total_predictions']}")
    print(f"  Elephant Detections: {stats['elephant_detections']}")
    print(f"  Confirmed Elephants: {stats['confirmed_elephants']}")
    print(f"  Confirmation Rate: {stats.get('confirmation_rate', 0):.1%}")
    print(f"  False Positives Filtered: {stats['false_positives_suppressed']}")
    print()
    
    return results


def test_false_positive_rejection():
    """Test that non-elephant signals are properly rejected"""
    
    print("\n" + "=" * 70)
    print("FALSE POSITIVE TEST - Non-Elephant Signals")
    print("=" * 70)
    print()
    
    generator = SeismicSignalGenerator()
    system = ElephantDetectionSystem()
    
    # Test non-elephant signals
    test_signals = [
        ('human_footsteps', generator.generate_human_footsteps),
        ('cattle_movement', generator.generate_cattle_movement),
        ('wind_vibration', generator.generate_wind_vibration),
        ('vehicle_passing', generator.generate_vehicle_passing),
        ('background_noise', generator.generate_background_noise),
    ]
    
    false_positives = 0
    total_tests = 0
    
    for signal_name, signal_func in test_signals:
        print(f"\nTesting: {signal_name.replace('_', ' ').title()}")
        
        elephant_detections = 0
        for i in range(3):
            signal, _ = signal_func(duration=1.0)
            result = system.process_signal(signal, soil_moisture=20.0)
            
            total_tests += 1
            
            if result['class_name'] == 'elephant_footfall' and result['detected']:
                elephant_detections += 1
                false_positives += 1
                print(f"  ⚠️  Frame {i+1}: FALSE POSITIVE - Detected as elephant!")
            else:
                print(f"  ✓ Frame {i+1}: Correctly identified as {result['class_name']}")
        
        if elephant_detections == 0:
            print(f"  ✓ PASS: No false elephant detections")
        else:
            print(f"  ✗ FAIL: {elephant_detections}/3 false detections")
    
    print(f"\n{'='*70}")
    print(f"FALSE POSITIVE RATE: {false_positives}/{total_tests} ({false_positives/total_tests:.1%})")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Run elephant detection tests
    results = test_elephant_detection()
    
    # Run false positive tests
    test_false_positive_rejection()
    
    print("\n✅ Testing complete!")
