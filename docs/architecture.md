# System Architecture & Technical Specification

**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Implementation**: ESP32 Microcontroller, PyTorch Deep Neural Networks, Streamlit Scientific UI  

---

## 1. End-to-End System Block Diagram

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         1. PHYSICAL ACQUISITION                             │
 │  MAX30102 Optoelectronic Sensor ──[I2C @ 400 kHz]──> ESP32 Microcontroller  │
 │  100 Hz Optical Pulse Sampling (IR & Red) ──[UART @ 921600 Baud]──> Host PC │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ (10-sample raw chunks @ 100 Hz)
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                          2. CAUSAL STREAMING DSP                            │
 │  ┌────────────────────────────────────────────────────────────────────────┐ │
 │  │ Stateful Rational Polyphase Resampler (100 Hz -> 125 Hz, L=5, M=4)     │ │
 │  │ Causal 3rd-Order Butterworth SOS Bandpass Filter (0.5 – 8.0 Hz)        │ │
 │  │ Backward Finite Differences: VPG (1st) & APG (2nd Derivative)          │ │
 │  │ Quality Control Gate: Flatline, Saturation, & Morphology Verification  │ │
 │  └────────────────────────────────────────────────────────────────────────┘ │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ (3 x 1250 Window Tensors @ 10s)
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         3. DEEP INFERENCE PIPELINE                          │
 │  ┌────────────────────────────────────────────────────────────────────────┐ │
 │  │ Phase 4A Spatial 1D-CNN Encoder (Input: [3, 1250] PPG+VPG+APG)         │ │
 │  │ 4 Conv Blocks (Conv1D + BatchNorm + LeakyReLU + MaxPool) ──> 64-dim    │ │
 │  └───────────────────────────────────┬────────────────────────────────────┘ │
 │                                      │ (Sequence of 6 Embeddings: 60s)      │
 │  ┌───────────────────────────────────▼────────────────────────────────────┐ │
 │  │ Phase 4B Temporal Sequence GRU (Input: [6, 64] Embeddings)             │ │
 │  │ 2-Layer Gated Recurrent Unit (Hidden Dim: 64, Dropout: 0.2)            │ │
 │  │ Output Head: Uncalibrated SBP & DBP continuous estimates               │ │
 │  └───────────────────────────────────┬────────────────────────────────────┘ │
 │                                      │ (Raw SBP & DBP)                      │
 │  ┌───────────────────────────────────▼────────────────────────────────────┐ │
 │  │ Phase 5C Extreme-Aware Recalibrator & Conformal Predictor              │ │
 │  │ Monotonic Piecewise Isotonic Regression (Compensates Regress-to-Mean)  │ │
 │  │ Distribution-Free 95% Conformal Prediction Intervals [Lower, Upper]    │ │
 │  └────────────────────────────────────────────────────────────────────────┘ │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    4. PHASE 7 MULTI-DOMAIN RELIABILITY                      │
 │  ┌────────────────────────────────────────────────────────────────────────┐ │
 │  │ Evaluates 4 Orthogonal Domains:                                        │ │
 │  │ 1. Signal Quality Gate (PTP, Clipping, Heart Rate)                     │ │
 │  │ 2. Epistemic Model Uncertainty (MC Dropout Variance: 30 passes)        │ │
 │  │ 3. Conformal Coverage Width (Calibrated Prediction Interval)           │ │
 │  │ 4. Rolling Temporal Stability (3-Window Backward Volatility & Jumps)   │ │
 │  │ Output State: [ TRUST | REVIEW | ABSTAIN ]                             │ │
 │  └────────────────────────────────────────────────────────────────────────┘ │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                       5. PRESENTATION & AUDIT LAYER                         │
 │  ┌────────────────────────────────────────────────────────────────────────┐ │
 │  │ Streamlit Scientific Dashboard (`code/phase6c_app/app.py`)             │ │
 │  │ - Mode 1: Guided Demonstration (8-Step Academic Walkthrough)           │ │
 │  │ - Mode 2: Live Experiment (Real-Time Serial, Replay, & CSV Upload)     │ │
 │  │ LLM Explanation Layer (Groq LLaMA-3.3 70B) under Authority Isolation   │ │
 │  └────────────────────────────────────────────────────────────────────────┘ │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Layer Specifications

### Sensor Configuration (MAX30102)
- **Interface**: I2C Bus running in Fast Mode ($400\text{ kHz}$).
- **Optical Channels**: Dual Photodiodes reading Infrared ($880\text{ nm}$) and Red ($660\text{ nm}$).
- **ADC Resolution**: 18-bit Delta-Sigma Analog-to-Digital Converter ($0\text{ to }262,143\text{ counts}$).
- **Sample Rate**: Configured for $100\text{ samples/second}$.
- **LED Pulse Width**: $411\text{ }\mu\text{s}$.
- **Sample Averaging**: 1 sample per FIFO entry (no on-chip decimation).

### Firmware Architecture (ESP32)
- **Framework**: Arduino C++ running on FreeRTOS.
- **Acquisition Core**: Core 0 services the I2C interrupt and reads sensor FIFO registers into a circular DMA buffer.
- **Serial Core**: Core 1 batches samples into 10-sample frames and streams structured ASCII records over USB UART at **$921,600\text{ baud}$**:
  ```text
  <sample_index>,<expected_timestamp_ms>,<host_timestamp_ms>,<ir_adc>,<red_adc>\n
  ```

---

## 3. DSP & Mathematical Filter Specifications

### Polyphase Rational Resampler
Because the MAX30102 acquisition clock operates at nominal $100\text{ Hz}$ while the frozen MIMIC-II neural model weights expect $125\text{ Hz}$, the pipeline includes a stateful polyphase rational resampler:
- **Up-sample factor ($L$)**: 5
- **Down-sample factor ($M$)**: 4
- **Ratio**: $125 / 100 = 1.25$
- **State Buffer**: Maintains polyphase filter memory across incoming serial chunks to guarantee continuity without edge ringing.

### Causal 3rd-Order Butterworth SOS Filter
- **Type**: 3rd-order Butterworth bandpass in Second-Order Sections (SOS) representation.
- **Passband**: $0.5\text{ Hz} \le f \le 8.0\text{ Hz}$ (passes heart rate fundamental frequencies from $30$ to $480\text{ BPM}$ and primary dicrotic notches while rejecting baseline wander and high-frequency sensor noise).
- **Causality Guarantee**: Employs strictly backward-facing difference equations (`scipy.signal.sosfilt`). No forward-backward filtering (`filtfilt`) is used anywhere in runtime code.

---

## 4. Model Architecture & Checkpoint Invariants

### Spatial 1D-CNN (Phase 4A)
- **Input Dimensions**: $[B, 3, 1250]$ (Channels: PPG, VPG, APG; Length: 10s @ 125 Hz).
- **Layers**:
  - Block 1: Conv1d(3, 16, kernel=7, stride=2, padding=3) + BatchNorm1d + LeakyReLU(0.1) + MaxPool1d(2)
  - Block 2: Conv1d(16, 32, kernel=5, stride=2, padding=2) + BatchNorm1d + LeakyReLU(0.1) + MaxPool1d(2)
  - Block 3: Conv1d(32, 64, kernel=5, stride=2, padding=2) + BatchNorm1d + LeakyReLU(0.1) + MaxPool1d(2)
  - Block 4: Conv1d(64, 64, kernel=3, stride=1, padding=1) + BatchNorm1d + LeakyReLU(0.1) + AdaptiveAvgPool1d(1)
- **Parameter Count**: $146,978$ parameters.
- **Frozen SHA-256 Hash**: `2c6c5478e5ec0c666cd2d5c43d7d2561c74ad25170f10075f24d2fa6610863ee`.

### Temporal Sequence GRU (Phase 4B)
- **Input Dimensions**: $[B, 6, 64]$ (6 consecutive 10s window embeddings representing 60 seconds).
- **Layers**: 2-layer Gated Recurrent Unit (hidden_dim=64, dropout=0.2, batch_first=True) + Linear(64, 32) + LeakyReLU(0.1) + Linear(32, 2).
- **Outputs**: $[\hat{\text{SBP}}, \hat{\text{DBP}}]$.
- **Parameter Count**: $27,106$ parameters.
- **Frozen SHA-256 Hash**: `26ecb0abf690bd067c5a083f80487e8f686c5c66ba49b9b84c7895fbaf8dcd81`.

---

## 5. Authority Isolation & Software Safety Architecture

To prevent generative hallucinations from impacting medical telemetry:
1. **Model Authority**: Blood pressure point predictions and conformal bounds are computed strictly by deterministic PyTorch tensors and scikit-learn isotonic regressors.
2. **Reliability Engine Authority**: Operational states (`TRUST`, `REVIEW`, `ABSTAIN`) are determined strictly by deterministic thresholds tuned on validation data.
3. **LLM Explainer Subservience**: The Large Language Model receives a stripped, read-only JSON payload containing summary metrics. It has zero access to raw sensor arrays, zero ability to edit prediction values, and any malformed response immediately triggers a deterministic local rule-based fallback.
