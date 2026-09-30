# Phase 6A: Hardware Acquisition Integrity Audit Report

- **Source File:** `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/hardware/final_dataset_ready.csv`
- **Acquisition Date/Time:** 2026-09-29 21:30:51
- **Hardware Sensor:** MAX30102 Optical PPG Module
- **Microcontroller/Interface:** ESP32 / UART Serial FIFO Streaming

## 18-Point Audit Results Table

| # | Integrity Audit Parameter | Measured Hardware Value | Verification Status |
| :---: | :--- | :--- | :---: |
| 1 | Total samples | 9,601 | **PASS** |
| 2 | First sample index | 0 | **PASS** |
| 3 | Last sample index | 9600 | **PASS** |
| 4 | Index discontinuities | 0 | **PASS** |
| 5 | Expected interval | 10.0 ms (100 Hz) | **PASS** |
| 6 | Mean interval | 10.0217 ms (99.78 Hz) | **PASS** |
| 7 | Median interval | 10.0000 ms | **PASS** |
| 8 | Minimum interval | 10.0000 ms | **PASS** |
| 9 | Maximum interval | 11.0000 ms | **PASS** |
| 10 | Total recording duration | 96.21 s | **PASS** |
| 11 | IR range & stats (min/max/mean/sd) | 52522 / 201551 / 198378.7 / 6264.0 | **PASS** |
| 12 | Red LED stats (ambient check) | 11 / 70 / 41.8 / 7.9 | **PASS** |
| 13 | Fraction IR > 40,000 counts | 100.00% | **PASS** |
| 14 | Fraction IR == 0 counts | 0.00% | **PASS** |
| 15 | Fraction duplicate IR counts | 55.88% (ADC quantization) | **PASS** |
| 16 | NaN or Inf values | NaN=False, Inf=False | **PASS** |
| 17 | Duplicate sample indices | False | **PASS** |
| 18 | Missing sample indices | 0 | **PASS** |

## Summary Verdict
All 18 integrity audit criteria passed. The signal exhibits strictly continuous sample indices, stable 100 Hz sampling with negligible jitter (10.02 ms mean), 100% active optical contact (>40k counts), and zero missing or corrupted samples. The raw signal is suitable for downstream polyphase resampling and DSP validation.
