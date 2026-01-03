/**
 * EarthPulse AI - ESP32-S3 Seismic Sensor Firmware
 * 
 * Hardware Configuration:
 * - ESP32-S3 Dev Board
 * - ADS1115 16-bit ADC (I2C)
 * - Geophone/Seismic Sensor connected to ADS1115
 * - 1kΩ resistors for signal conditioning
 * 
 * Pin Connections:
 * - ADS1115 SDA -> GPIO 8 (ESP32-S3 I2C SDA)
 * - ADS1115 SCL -> GPIO 9 (ESP32-S3 I2C SCL)
 * - ADS1115 VDD -> 3.3V
 * - ADS1115 GND -> GND
 * - Geophone -> ADS1115 A0 (differential input)
 * 
 * Communication: USB Serial @ 115200 baud
 */

#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// I2C Configuration for ESP32-S3
#define SDA_PIN 8
#define SCL_PIN 9

// ADS1115 Configuration
Adafruit_ADS1115 ads;  // 16-bit ADC

// Sampling Configuration
#define SAMPLING_RATE 1000      // 1000 Hz (1ms interval)
#define BUFFER_SIZE 1000        // 1 second of data
#define SEND_INTERVAL 1000      // Send data every 1 second (ms)

// Gain Configuration (adjust based on your sensor output)
// GAIN_TWOTHIRDS  +/- 6.144V  1 bit = 0.1875mV
// GAIN_ONE        +/- 4.096V  1 bit = 0.125mV
// GAIN_TWO        +/- 2.048V  1 bit = 0.0625mV (default)
// GAIN_FOUR       +/- 1.024V  1 bit = 0.03125mV
// GAIN_EIGHT      +/- 0.512V  1 bit = 0.015625mV
// GAIN_SIXTEEN    +/- 0.256V  1 bit = 0.0078125mV
#define ADS_GAIN GAIN_TWO

// Sensor Channel (A0-A3 or differential pairs)
#define SENSOR_CHANNEL 0  // Single-ended A0

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

float signalBuffer[BUFFER_SIZE];
int bufferIndex = 0;
unsigned long lastSampleTime = 0;
unsigned long lastSendTime = 0;
bool adsConnected = false;

// Statistics
float minValue = 0;
float maxValue = 0;
float sumValue = 0;
int sampleCount = 0;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize USB Serial
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  delay(1000);  // Give time for serial to stabilize
  
  Serial.println("====================================");
  Serial.println("EarthPulse AI - ESP32-S3 Firmware");
  Serial.println("====================================");
  Serial.println();
  
  // Initialize I2C with custom pins for ESP32-S3
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);  // 400kHz I2C speed
  
  Serial.print("I2C initialized (SDA: GPIO");
  Serial.print(SDA_PIN);
  Serial.print(", SCL: GPIO");
  Serial.print(SCL_PIN);
  Serial.println(")");
  
  // Initialize ADS1115
  Serial.print("Initializing ADS1115... ");
  
  if (!ads.begin()) {
    Serial.println("FAILED!");
    Serial.println("ERROR: ADS1115 not found. Check wiring:");
    Serial.println("  - VDD to 3.3V");
    Serial.println("  - GND to GND");
    Serial.println("  - SDA to GPIO 8");
    Serial.println("  - SCL to GPIO 9");
    Serial.println();
    Serial.println("Running in TEST MODE (simulated data)");
    adsConnected = false;
  } else {
    Serial.println("SUCCESS!");
    adsConnected = true;
    
    // Configure ADS1115
    ads.setGain(ADS_GAIN);
    ads.setDataRate(RATE_ADS1115_860SPS);  // Max sampling rate
    
    Serial.print("Gain set to: ");
    switch(ADS_GAIN) {
      case GAIN_TWOTHIRDS: Serial.println("+/- 6.144V"); break;
      case GAIN_ONE:       Serial.println("+/- 4.096V"); break;
      case GAIN_TWO:       Serial.println("+/- 2.048V"); break;
      case GAIN_FOUR:      Serial.println("+/- 1.024V"); break;
      case GAIN_EIGHT:     Serial.println("+/- 0.512V"); break;
      case GAIN_SIXTEEN:   Serial.println("+/- 0.256V"); break;
    }
    
    Serial.print("Reading from channel: A");
    Serial.println(SENSOR_CHANNEL);
  }
  
  Serial.println();
  Serial.print("Sampling rate: ");
  Serial.print(SAMPLING_RATE);
  Serial.println(" Hz");
  
  Serial.print("Buffer size: ");
  Serial.print(BUFFER_SIZE);
  Serial.println(" samples");
  
  Serial.println();
  Serial.println("System ready!");
  Serial.println("Sending data format: START,sample1,sample2,...,sampleN,END");
  Serial.println("====================================");
  Serial.println();
  
  lastSampleTime = micros();
  lastSendTime = millis();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  unsigned long currentMicros = micros();
  unsigned long currentMillis = millis();
  
  // Sample at fixed rate (1000 Hz = 1000 microseconds)
  if (currentMicros - lastSampleTime >= 1000) {
    lastSampleTime = currentMicros;
    
    float voltage = 0.0;
    
    if (adsConnected) {
      // Read from ADS1115
      int16_t adc_value = ads.readADC_SingleEnded(SENSOR_CHANNEL);
      voltage = ads.computeVolts(adc_value);
    } else {
      // Test mode - generate simulated seismic-like signal
      float t = millis() / 1000.0;
      // Low frequency rumble (0.5-2 Hz) + higher frequency vibration
      voltage = 0.1 * sin(2 * PI * 1.0 * t) +           // 1 Hz base
                0.05 * sin(2 * PI * 3.5 * t) +          // 3.5 Hz component
                0.02 * sin(2 * PI * 12.0 * t) +         // 12 Hz vibration
                0.01 * (random(-100, 100) / 100.0);     // Noise
    }
    
    // Store in buffer
    signalBuffer[bufferIndex] = voltage;
    bufferIndex++;
    
    // Update statistics
    if (sampleCount == 0 || voltage < minValue) minValue = voltage;
    if (sampleCount == 0 || voltage > maxValue) maxValue = voltage;
    sumValue += voltage;
    sampleCount++;
    
    // Send data packet when buffer is full
    if (bufferIndex >= BUFFER_SIZE) {
      sendDataPacket();
      
      // Reset buffer and statistics
      bufferIndex = 0;
      minValue = 0;
      maxValue = 0;
      sumValue = 0;
      sampleCount = 0;
    }
  }
  
  // Handle serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    handleCommand(command);
  }
}

// ============================================================================
// DATA TRANSMISSION
// ============================================================================

void sendDataPacket() {
  // Send data in CSV format for easy parsing
  Serial.print("START,");
  
  // Send timestamp
  Serial.print(millis());
  Serial.print(",");
  
  // Send all samples
  for (int i = 0; i < BUFFER_SIZE; i++) {
    Serial.print(signalBuffer[i], 6);  // 6 decimal places
    if (i < BUFFER_SIZE - 1) {
      Serial.print(",");
    }
  }
  
  Serial.println(",END");
  
  // Send statistics on separate line (for debugging)
  Serial.print("STATS,");
  Serial.print("min:");
  Serial.print(minValue, 6);
  Serial.print(",max:");
  Serial.print(maxValue, 6);
  Serial.print(",avg:");
  Serial.print(sumValue / sampleCount, 6);
  Serial.print(",rms:");
  
  // Calculate RMS
  float rms = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    rms += signalBuffer[i] * signalBuffer[i];
  }
  rms = sqrt(rms / BUFFER_SIZE);
  Serial.println(rms, 6);
}

// ============================================================================
// COMMAND HANDLING
// ============================================================================

void handleCommand(String cmd) {
  cmd.toUpperCase();
  
  if (cmd == "STATUS") {
    // Send system status
    Serial.println("STATUS:");
    Serial.print("  ADS1115: ");
    Serial.println(adsConnected ? "Connected" : "Disconnected (Test Mode)");
    Serial.print("  Sampling Rate: ");
    Serial.print(SAMPLING_RATE);
    Serial.println(" Hz");
    Serial.print("  Buffer Size: ");
    Serial.println(BUFFER_SIZE);
    Serial.print("  Channel: A");
    Serial.println(SENSOR_CHANNEL);
    Serial.print("  Uptime: ");
    Serial.print(millis() / 1000);
    Serial.println(" seconds");
    
  } else if (cmd == "RESET") {
    // Reset statistics
    bufferIndex = 0;
    minValue = 0;
    maxValue = 0;
    sumValue = 0;
    sampleCount = 0;
    Serial.println("OK: Reset complete");
    
  } else if (cmd == "SCAN") {
    // Scan I2C bus
    Serial.println("Scanning I2C bus...");
    byte error, address;
    int nDevices = 0;
    
    for (address = 1; address < 127; address++) {
      Wire.beginTransmission(address);
      error = Wire.endTransmission();
      
      if (error == 0) {
        Serial.print("  Device found at 0x");
        if (address < 16) Serial.print("0");
        Serial.println(address, HEX);
        nDevices++;
      }
    }
    
    if (nDevices == 0) {
      Serial.println("  No I2C devices found");
    } else {
      Serial.print("  Found ");
      Serial.print(nDevices);
      Serial.println(" device(s)");
    }
    
  } else if (cmd == "HELP") {
    // Show available commands
    Serial.println("Available Commands:");
    Serial.println("  STATUS - Show system status");
    Serial.println("  RESET  - Reset buffer and statistics");
    Serial.println("  SCAN   - Scan I2C bus for devices");
    Serial.println("  HELP   - Show this help message");
    
  } else {
    Serial.print("ERROR: Unknown command '");
    Serial.print(cmd);
    Serial.println("'");
    Serial.println("Type HELP for available commands");
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Calculate RMS value
float calculateRMS(float* buffer, int size) {
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += buffer[i] * buffer[i];
  }
  return sqrt(sum / size);
}
