# Hardware Specification & Acquisition Protocol

**Hardware**: Maxim Integrated MAX30102 Sensor & Espressif ESP32-WROOM-32 Microcontroller  
**Interface**: Fast I2C (400 kHz) & High-Speed UART (921,600 baud)  

---

## 1. Electrical Wiring & Pinout

Connect the MAX30102 sensor module to the ESP32 development board using the following pin mapping:

| MAX30102 Pin | ESP32 Pin | Function / Logic Level | Notes |
|---|---|---|---|
| **VIN** | **3.3V** | 3.3V DC Power | Do not connect to 5V; MAX30102 logic is 3.3V. |
| **GND** | **GND** | Ground Plane | Common system ground. |
| **SDA** | **GPIO 21** | I2C Serial Data | Standard ESP32 default SDA pin. |
| **SCL** | **GPIO 22** | I2C Serial Clock | Standard ESP32 default SCL pin. |
| **INT** | *Not Connected* | Interrupt Line | Optional hardware interrupt (polled FIFO used). |

> **Hardware Note on Pull-Up Resistors**: Most MAX30102 breakout boards (e.g. CJMCU-30102) include onboard $4.7\text{ k}\Omega$ pull-up resistors to 3.3V on the SDA and SCL lines. If using a bare IC or custom PCB, verify that external pull-up resistors are installed.

---

## 2. Sensor Configuration Parameters

The ESP32 firmware initializes the MAX30102 with the following register parameters:

```c
// MAX30102 Configuration Settings
#define SAMPLING_RATE_HZ     100      // 100 Hz output data rate
#define LED_PULSE_WIDTH_US   411      // 411 microseconds (18-bit ADC resolution)
#define IR_LED_CURRENT_MA    7.6      // Optimal optical penetration for fingertip
#define RED_LED_CURRENT_MA   7.6      // Balanced red optical emission
#define SAMPLE_AVERAGING     1        // No onboard decimation (raw fidelity)
#define ADC_FULL_SCALE_NA    4096     // 4096 nA full scale range
```

---

## 3. Serial Communication Protocol

The ESP32 streams ASCII frames over USB serial. Each line represents a discrete sample frame terminated by a newline (`\n`):

### Frame Format
```text
<sample_index>,<expected_timestamp_ms>,<host_timestamp_ms>,<ir_adc>,<red_adc>
```

### Example Telemetry Stream
```text
1042,10420,10421,138245,142100
1043,10430,10431,138290,142140
1044,10440,10440,138330,142195
1045,10450,10451,138380,142240
```

### Parameters
- **Baud Rate**: `921600` (recommended for low-latency multi-channel streaming) or `115200`.
- **Data Bits**: 8
- **Parity**: None
- **Stop Bits**: 1
- **Flow Control**: None

---

## 4. Calibration & Reference Protocol

For validating cuffless model predictions against clinical references during physical recording sessions:
1. **Sensor Placement**: Secure the MAX30102 sensor to the left index or middle finger using an elastic medical strap (avoid excessive tightening to prevent venous occlusion).
2. **Rest Period**: The volunteer must remain seated at rest for at least 5 minutes prior to data acquisition.
3. **Reference Cuff Timing**: Trigger an oscillometric arm cuff measurement (e.g. Omron HEM-7120) on the contralateral (right) arm while recording continuous optical pulse data.
4. **Timestamp Synchronization**: Record the exact timestamp when the cuff finishes deflation and displays SBP/DBP. The Streamlit pairing engine associates the reference reading with the corresponding 60-second optical sequence window.
