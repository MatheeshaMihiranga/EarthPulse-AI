/*
 * EarthPulse AI - ESP32-S3 Geophone Firmware
 * 
 * Reads seismic vibrations from SM-24 geophone via ADS1115 ADC
 * Sends data to Python detection system via serial USB
 * 
 * Hardware:
 * - ESP32-S3 (any variant)
 * - ADS1115 16-bit ADC
 * - SM-24 Geophone 10Hz
 * - 3× 1kΩ resistors (voltage divider)
 * 
 * Connections:
 * - GPIO8 (SDA) -> ADS1115 SDA
 * - GPIO9 (SCL) -> ADS1115 SCL
 * - 3.3V -> ADS1115 VDD
 * - GND -> ADS1115 GND
 * - ADS1115 A0 -> Geophone+ (via resistor network)
 * - ADS1115 A1 -> Geophone- (via resistor network)
 * 
 * Author: EarthPulse AI Team
 * License: MIT
 */

#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// I2C pins (ESP32-S3 specific)
const int I2C_SDA = 8;   // GPIO8 -> ADS1115 SDA
const int I2C_SCL = 9;   // GPIO9 -> ADS1115 SCL

// Sampling configuration
const int SAMPLE_RATE = 100;  // Hz (100 samples per second - reduced for cleaner signal)
const int INTERVAL_MS = 1000 / SAMPLE_RATE;  // Milliseconds between samples

// Moving average filter to reduce noise
const int FILTER_SIZE = 5;
float filterBuffer[FILTER_SIZE] = {0};
int filterIndex = 0;

// ADS1115 configuration
Adafruit_ADS1115 ads;  // 16-bit ADC

// Timing
unsigned long lastSample = 0;
unsigned long sampleCount = 0;

// LED indicator (optional - built-in LED)
const int LED_PIN = LED_BUILTIN;
unsigned long lastBlink = 0;
bool ledState = false;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  delay(1000);  // Wait for serial to stabilize
  
  // Print header
  Serial.println();
  Serial.println("======================================");
  Serial.println("EarthPulse AI - Geophone Sensor Node");
  Serial.println("======================================");
  Serial.println();
  
  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // LED on during init
  
  // Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  Serial.print("I2C initialized on GPIO");
  Serial.print(I2C_SDA);
  Serial.print(" (SDA) and GPIO");
  Serial.print(I2C_SCL);
  Serial.println(" (SCL)");
  
  // Initialize ADS1115
  Serial.print("Initializing ADS1115... ");
  if (!ads.begin(0x48)) {  // I2C address 0x48 (ADDR to GND)
    Serial.println("FAILED!");
    Serial.println();
    Serial.println("ERROR: ADS1115 not found!");
    Serial.println();
    Serial.println("Troubleshooting:");
    Serial.println("  1. Check power connections:");
    Serial.println("     - VDD -> 3.3V (red wire)");
    Serial.println("     - GND -> GND (black wire)");
    Serial.println("  2. Check I2C connections:");
    Serial.println("     - SDA -> GPIO8 (green wire)");
    Serial.println("     - SCL -> GPIO9 (yellow wire)");
    Serial.println("  3. Check address pin:");
    Serial.println("     - ADDR -> GND (gray wire)");
    Serial.println("  4. Verify breadboard connections");
    Serial.println("  5. Check wire continuity with multimeter");
    Serial.println();
    
    // Blink LED rapidly to indicate error
    while (1) {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
  }
  
  Serial.println("SUCCESS!");
  
  // Configure ADS1115 for LOW NOISE operation
  // Gain: GAIN_SIXTEEN = ±0.256V range (highest precision, lowest noise)
  // This is perfect for geophone signals which are typically <100mV
  ads.setGain(GAIN_SIXTEEN);
  Serial.println("Gain: +/-0.256V (GAIN_SIXTEEN - Low Noise Mode)");
  
  // Data rate: 128 SPS (slower = more averaging = cleaner signal)
  ads.setDataRate(RATE_ADS1115_128SPS);
  Serial.println("Data rate: 128 SPS (High Precision Mode)");
  
  Serial.println();
  Serial.print("I2C address: 0x");
  Serial.println(0x48, HEX);
  
  Serial.println();
  Serial.println("System Configuration:");
  Serial.print("  Sample rate: ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.print("  Interval: ");
  Serial.print(INTERVAL_MS);
  Serial.println(" ms");
  Serial.println();
  
  // Calibration info
  Serial.println("Voltage Conversion:");
  Serial.println("  1 bit = 0.0078125 mV (16-bit, +/-0.256V range)");
  Serial.println("  Expected baseline: ~0V (differential measurement)");
  Serial.println("  Noise floor: <1 mV");
  Serial.println();
  
  digitalWrite(LED_PIN, LOW);  // LED off when ready
  
  Serial.println("======================================");
  Serial.println("System Ready!");
  Serial.println("======================================");
  Serial.println();
  Serial.println("Listening for seismic vibrations...");
  Serial.println("Tap table to test geophone response.");
  Serial.println();
  
  // Initial timestamp
  lastSample = millis();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  unsigned long currentTime = millis();
  
  // Maintain precise sampling rate
  if (currentTime - lastSample >= INTERVAL_MS) {
    lastSample = currentTime;
    
    // Read DIFFERENTIAL (A0-A1) - this is key for noise reduction!
    int16_t diff = ads.readADC_Differential_0_1();
    
    // Convert to voltage
    float voltageDiff = ads.computeVolts(diff);
    
    // Apply moving average filter to reduce noise
    filterBuffer[filterIndex] = voltageDiff;
    filterIndex = (filterIndex + 1) % FILTER_SIZE;
    
    float filteredVoltage = 0;
    for (int i = 0; i < FILTER_SIZE; i++) {
      filteredVoltage += filterBuffer[i];
    }
    filteredVoltage /= FILTER_SIZE;
    
    // Send filtered data in JSON format
    Serial.print("{");
    Serial.print("\"timestamp\":");
    Serial.print(currentTime);
    Serial.print(",\"voltage\":");
    Serial.print(filteredVoltage, 6);  // Filtered differential voltage
    Serial.print(",\"raw\":");
    Serial.print(voltageDiff, 6);      // Raw differential voltage
    Serial.print(",\"adc\":");
    Serial.print(diff);
    Serial.print(",\"count\":");
    Serial.print(sampleCount);
    Serial.println("}");
    
    sampleCount++;
    
    // Blink LED every second (heartbeat)
    if (currentTime - lastBlink >= 1000) {
      lastBlink = currentTime;
      ledState = !ledState;
      digitalWrite(LED_PIN, ledState);
    }
  }
}

// ============================================================================
// ALTERNATIVE: Human-Readable Output (Uncomment to use)
// ============================================================================

/*
void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastSample >= INTERVAL_MS) {
    lastSample = currentTime;
    
    int16_t adc0 = ads.readADC_SingleEnded(0);
    int16_t adc1 = ads.readADC_SingleEnded(1);
    int16_t diff = adc0 - adc1;
    
    float voltage0 = ads.computeVolts(adc0);
    float voltage1 = ads.computeVolts(adc1);
    float voltageDiff = ads.computeVolts(diff);
    
    // Human-readable format
    Serial.print("Time: ");
    Serial.print(currentTime);
    Serial.print(" ms | ");
    
    Serial.print("A0: ");
    Serial.print(voltage0, 4);
    Serial.print(" V | ");
    
    Serial.print("A1: ");
    Serial.print(voltage1, 4);
    Serial.print(" V | ");
    
    Serial.print("Diff: ");
    Serial.print(voltageDiff, 4);
    Serial.print(" V | ");
    
    Serial.print("ADC: ");
    Serial.print(adc0);
    Serial.print(" - ");
    Serial.print(adc1);
    Serial.print(" = ");
    Serial.println(diff);
    
    sampleCount++;
  }
}
*/

// ============================================================================
// NOTES
// ============================================================================

/*
 * Expected Values:
 * ----------------
 * Baseline (no vibration):
 *   A0: ~1.65V (midpoint of 0-3.3V divider)
 *   A1: ~1.65V
 *   Diff: ~0V
 *   ADC: ~10,800 (midpoint of 16-bit: 32767/2)
 * 
 * Small vibration (human footstep 5m away):
 *   Diff: ±0.05V
 *   ADC change: ±300
 * 
 * Medium vibration (stomp 2m away):
 *   Diff: ±0.2V
 *   ADC change: ±1,200
 * 
 * Large vibration (elephant 20m away):
 *   Diff: ±0.5V to 1.0V
 *   ADC change: ±3,000+
 * 
 * Troubleshooting:
 * ----------------
 * 1. Voltage always 0V or 3.3V:
 *    - Check resistor network connections
 *    - Verify R1, R2, R3 in series
 *    - Check geophone wires at correct nodes
 * 
 * 2. No voltage change on tap:
 *    - Test geophone with multimeter (should be ~400 ohms)
 *    - Check geophone wire connections
 *    - Verify ADS1115 A0/A1 to correct nodes
 * 
 * 3. Very noisy signal:
 *    - Check all ground connections
 *    - Verify power rail bridges (BB1 <-> BB2)
 *    - Keep geophone wires away from power wires
 * 
 * 4. "ADS1115 not found" error:
 *    - Most common: SDA and SCL swapped
 *    - Check green wire: GPIO8 -> SDA (not SCL!)
 *    - Check yellow wire: GPIO9 -> SCL (not SDA!)
 *    - Verify ADDR pin to GND
 *    - Check power connections
 * 
 * Serial Output Format:
 * ---------------------
 * JSON format makes parsing easy in Python:
 * {"timestamp":1234,"a0":1.650000,"a1":1.650000,"diff":0.000000,"adc0":10800,"adc1":10800,"count":0}
 * 
 * Python parsing example:
 *   import json
 *   data = json.loads(line)
 *   voltage = data['diff']
 *   timestamp = data['timestamp']
 * 
 * Performance:
 * ------------
 * - Sampling rate: 1000 Hz (1 sample per millisecond)
 * - ADS1115 max rate: 860 SPS (samples per second)
 * - Actual rate: ~850-900 Hz due to I2C overhead
 * - Resolution: 16-bit (65,536 levels)
 * - Precision: ~0.19 mV per bit
 * - Latency: ~1-2 ms per sample
 * 
 * Power Consumption:
 * ------------------
 * - ESP32-S3: ~100-200 mA (active)
 * - ADS1115: ~150 µA (continuous mode)
 * - Total: ~100-200 mA @ 3.3V
 * - USB power sufficient for testing
 * - Use 5V 1A supply for field deployment
 * 
 * Field Deployment:
 * -----------------
 * 1. Test in lab first (tap table)
 * 2. Deploy near elephant paths (20-50m distance)
 * 3. Bury geophone 10-20cm underground
 * 4. Protect electronics from moisture
 * 5. Use long USB cable or battery power
 * 6. Run Python script on laptop/Raspberry Pi
 * 7. Monitor for 24-48 hours
 * 8. Analyze detection patterns
 */
