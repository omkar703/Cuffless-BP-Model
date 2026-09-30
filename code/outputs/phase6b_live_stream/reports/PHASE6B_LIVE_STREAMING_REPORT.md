# PHASE 6B — REAL-TIME MAX30102 STREAMING PIPELINE REPORT

**Project**: Calibration-Free Cuffless Blood-Pressure Estimation using Photoplethysmography Only  
**Pipeline Phase**: Phase 6B — Real-Time Streaming Architecture & Live Execution Engine  
**Execution Environment**: Local System (CPU & NVIDIA GeForce GTX 1650 Ti)  
**Execution Mode**: Strictly Inference-Only (Zero Retraining / Zero Parameter Updates)  
**Timestamp**: 2026-09-29 22:31:34  

---

## 1. Executive Summary & Objective

Phase 6B transitions the cuffless blood-pressure estimation pipeline from static offline batch analysis into a **real-time, host-side streaming execution engine**. It connects live sensor data arriving over serial to the frozen research deep neural network (Phase 4A CNN + Phase 4B Temporal GRU), proving that:
1. An incoming stream of optical PPG samples can be resampled from $\approx 100\text{ Hz}$ to $125\text{ Hz}$ incrementally with zero boundary distortion.
2. Causal filtering (0.5–8.0 Hz) and backward finite differences ($VPG, APG$) run statefully without requiring future samples.
3. 10-second windows and 60-second rolling sequences update causally as time advances.
4. The streaming architecture reproduces Phase 6A offline results bit-for-bit (maximum difference $< 10^{-4}\text{ mmHg}$).

> [!IMPORTANT]
> **Scientific Scope**: Phase 6B validates real-time streaming DSP and neural model execution. It does **NOT** validate BP estimation accuracy because no simultaneous reference cuff or catheter measurements were present. True hardware accuracy validation is reserved for Phase 6C.

---

## 2. Streaming Architecture Overview

```
                      MAX30102 Optical Sensor
                                 │
                                 ▼ (Raw IR stream @ ~100 Hz, 10 ms interval)
               ESP32 Serial Stream (921,600 baud, UART)
                                 │
                                 ▼
                     Sample Integrity Validation
                 (Index continuity, jitter, range)
                                 │
                                 ▼
             Stateful Rational Resampler (100 -> 125 Hz)
                   (up=5, down=4, persistent FIR)
                                 │
                                 ▼
               Stateful Causal Bandpass (0.5–8.0 Hz)
                (3rd-order Butterworth SOS, state zi)
                                 │
                                 ▼
                    Causal Backward Derivatives
                 (VPG = dPPG/dt, APG = dVPG/dt)
                                 │
                                 ▼
                10-Second Window Accumulation Buffer
                       (1,250 samples @ 125 Hz)
                                 │
                                 ▼
                   Per-Window Z-Score Normalization
                        (PPG, VPG, APG channels)
                                 │
                                 ▼
                 Engineering Quality Gating (QC)
                     (PASS / WARN / REJECT)
                                 │
                                 ▼
                   Frozen Phase 4A 1D CNN Encoder
                     (64-dim Latent Embeddings)
                                 │
                                 ▼
                  Rolling 6-Window History Buffer
                        (60 seconds context)
                                 │
                                 ▼
                    Frozen Phase 4B Causal GRU
                                 │
                                 ▼
                   Model Output: [SBP, DBP]
                                 │
                                 ▼
            Optional Phase 5C Post-Hoc Conformal Bounds
                                 │
                                 ▼
                    Live Display & CSV Logging
```

---

## 3. Streaming Resampler Verification (Chunk-Invariance)

A critical hazard in streaming signal processing is restarting polyphase filter state on arbitrary incoming chunks, which injects high-frequency boundary transients. The `StatefulRationalResampler` preserves historical input samples and polyphase phase offsets across chunk boundaries.

### Resampler Unit Test Results Across Chunk Sizes:
| Chunk Size | Output Samples | Expected Samples | Maximum Absolute Error | Pearson Correlation | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 sample** | 1,250 | 1,250 | **0.00e+00** | **1.00000000** | **PASS** |
| **7 samples** | 1,250 | 1,250 | **0.00e+00** | **1.00000000** | **PASS** |
| **16 samples** | 1,250 | 1,250 | **0.00e+00** | **1.00000000** | **PASS** |
| **32 samples** | 1,250 | 1,250 | **0.00e+00** | **1.00000000** | **PASS** |
| **100 samples** | 1,250 | 1,250 | **0.00e+00** | **1.00000000** | **PASS** |
| **137 samples** | 1,250 | 1,250 | **0.00e+00** | **1.00000000** | **PASS** |

The streaming resampler is mathematically bit-exact with `scipy.signal.resample_poly` while operating purely incrementally.

---

## 4. Stateful Causal Filter & Causal Derivatives

### 4.1 Stateful SOS Bandpass Filter (0.5–8.0 Hz)
- Implemented via `scipy.signal.sosfilt` with persistent state vector $z_i$.
- Tested across chunk sizes [1, 7, 16, 32, 100, 137]: Max error vs batch = **0.00e+00** (**PASS**).

### 4.2 Causal Backward Derivatives
- $VPG[n] = (PPG[n] - PPG[n-1]) / \Delta t$
- $APG[n] = (VPG[n] - VPG[n-1]) / \Delta t$ where $\Delta t = 1/125\text{ s}$.
- Maintains previous PPG and VPG sample values across streaming chunk boundaries: Max error vs batch = **0.00e+00** (**PASS**).

---

## 5. Startup State Machine & Warm-up Timeline

The pipeline tracks explicit state transitions:
1. `RESAMPLER_WARMUP` (0–0.25 s, 25 samples): FIR memory populating.
2. `FILTER_WARMUP` (0.25–2.0 s, 250 samples): IIR initial state settling.
3. `BUFFER_FILLING` (2.0–10.0 s): Accumulating the first 10-second window (1,250 samples).
4. `HISTORY_FILLING` (10.0–50.0 s): Windows 1 through 5 populating the 6-window history.
5. `READY` (>= 60.0 s): Full 60-second temporal history available. First prediction emitted at t = 60 s, followed by periodic updates every 10 seconds.

---

## 6. Streaming Replay vs Phase 6A Consistency

Replaying the 96.2-second hardware recording in 10-sample streaming chunks produced 4 consecutive 60-second sequence predictions:

| Sequence | Span | Phase 6A SBP (mmHg) | Streaming SBP (mmHg) | Difference | Phase 6A DBP (mmHg) | Streaming DBP (mmHg) | Difference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seq_00` | 60s | 143.99 | 143.99 | -0.0000 | 83.60 | 83.60 | +0.0000 |
| `seq_01` | 70s | 142.25 | 142.25 | -0.0000 | 83.53 | 83.53 | +0.0000 |
| `seq_02` | 80s | 143.86 | 143.86 | +0.0000 | 85.19 | 85.19 | +0.0000 |
| `seq_03` | 90s | 133.16 | 133.16 | +0.0000 | 80.86 | 80.86 | -0.0000 |

*Maximum discrepancy*: **0.000015 mmHg SBP** and **0.000008 mmHg DBP** (< 10^{-4} mmHg), confirming that the streaming architecture is completely consistent with the offline deployment proxy.

---

## 7. Real-Time Latency & Host Computation Headroom

Measured over 960 streaming chunk iterations (10 samples = 100 ms interval budget):
- **Mean Processing Time per Chunk**: **0.2180 ms**
- **95th Percentile Latency**: **0.2574 ms**
- **Peak Chunk Latency**: **3.1569 ms** (during 10s window neural inference)
- **Time Available per 10s Window**: 10,000 ms
- **Total Window Processing Latency**: < 4.2 ms
- **Headroom Factor**: > 459x faster than required real-time speed.

The host PC easily runs real-time streaming DSP and PyTorch inference without dropping samples or lagging behind the sensor clock.

---

## 8. Next Hardware Experiment (Phase 6C Specification)

To move from streaming pipeline validation to clinical/engineering BP estimation accuracy validation:
1. **Device**: MAX30102 + ESP32 streaming at 921,600 baud.
2. **Reference**: Certified oscillometric arm cuff (e.g. Omron HEM-7120 / HEM-7361T).
3. **Synchronized Logging**: Host script records cuff measurement trigger timestamp, inflation start/stop, and manual entry of measured reference SBP and DBP.
4. **Target Metrics**: Evaluation of Model SBP vs Reference SBP and Model DBP vs Reference DBP across multiple subjects and postures to determine genuine hardware MAE, RMSE, and correlation.
