# PHASE 6B REPLAY VALIDATION REPORT: NEW MAX30102 HARDWARE CAPTURE

**Project**: Calibration-Free Cuffless Blood-Pressure Estimation using Photoplethysmography Only  
**Validation Target**: Phase 6B Streaming Replay Pipeline on Fresh Physical MAX30102 Acquisition  
**Execution Timestamp**: 2026-09-29 23:23:36  
**Execution Mode**: Strictly Inference-Only (Zero Retraining / Zero Model Updates)  
**Host Environment**: Linux / Python 3.10.21 / PyTorch 2.14.0+cu130 / Intel Host CPU  

---

## 1. Objective

The primary objective of this experiment is to validate the end-to-end execution of the **Phase 6B real-time streaming pipeline** on an independent, newly acquired physical MAX30102 hardware capture. Specifically:
1. Ingest the newly acquired physical hardware CSV via an explicit compatibility adapter without modifying or deleting raw samples.
2. Verify hardware timing stability, sample index continuity, and sensor acquisition frequency.
3. Stream the raw optical IR signal through the **Stateful Rational Resampler** ($100\text{ Hz} \to 125\text{ Hz}$), **Stateful Causal Bandpass Filter** ($0.5–8.0\text{ Hz}$), and **Causal Backward Derivatives** ($VPG, APG$).
4. Partition the stream into $10\text{-second}$ non-overlapping windows ($1,250\text{ samples}$) and evaluate window-level engineering quality metrics.
5. Feed accepted windows into the frozen **Phase 4A 1D CNN Encoder** and update the rolling $60\text{-second}$ context buffer.
6. Generate causal $SBP$ and $DBP$ predictions using the frozen **Phase 4B Causal GRU**, mapped through **Phase 5C post-hoc conformal calibration**.
7. Audit host-side computational latency and real-time processing headroom.

> [!IMPORTANT]
> **No Clinical BP Accuracy Validation Claim**: This physical recording does not contain simultaneously measured reference arm-cuff or invasive arterial line blood-pressure labels. **This report validates engineering pipeline execution, signal conditioning, and model execution feasibility only. It does NOT claim BP accuracy, clinical equivalence, or cuff replacement capability.**

---

## 2. Input Dataset

- **Raw Acquisition File**: `final_dataset_ready(1).csv`
- **Source Location**: `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/hardware/new report hardware`
- **Derived Replay Input File**: `new_hardware_replay_input.csv` (saved in `input/`)
- **Total Physical Samples**: 8,370 rows
- **Input Channels**: `sample_index`, `expected_timestamp_ms`, `host_timestamp_ms`, `ir`, `red`
- **Model Channel Used**: **IR PPG Only** (Red channel preserved strictly for ambient/optical diagnostic logging).

### Compatibility Adapter Execution:
To interface with the streaming replay pipeline without altering the original hardware acquisition:
1. The original CSV was read and preserved completely unchanged.
2. Original sample order and all 8,370 rows were preserved bit-for-bit without deletion or threshold-based sample filtering.
3. `expected_timestamp_ms` was mapped from relative sample index: `sample_index - first_sample_index` ($0\text{ to } 83,690\text{ ms}$), respecting the observed $10\text{-unit}$ increment.
4. `host_timestamp_ms` was mapped from the acquisition timing column.

---

## 3. Hardware Timing Verification

A comprehensive hardware acquisition audit was executed on the derived replay file:

| Audit Parameter | Quantitative Metric | Engineering Tolerance | Status |
| :--- | :---: | :---: | :---: |
| **Row Count** | 8,370 samples | Exact match | **PASS** |
| **Sample Index Span** | 218470 to 302160 | Continuous step = 10 | **PASS** |
| **Sample Index Discontinuities** | **0** | 0 discontinuities | **PASS** |
| **Duplicate / Reverse Steps** | **0** | 0 duplicates | **PASS** |
| **Host Mean Interval** | **10.023 ms** | $10.0 \pm 0.5\text{ ms}$ | **PASS** |
| **Host Median Interval** | **10.000 ms** | $10.0\text{ ms}$ nominal | **PASS** |
| **Host Interval Range** | **[9, 11] ms** | Strict $[9, 11]\text{ ms}$ bounds | **PASS** |
| **Intervals Outside [9, 11] ms** | **0** | 0 outliers | **PASS** |
| **Total Recording Duration** | **83.880 s** | Nominal $83.88\text{ s}$ | **PASS** |
| **Effective Acquisition Rate** | **99.773 Hz** | $99.77\text{ Hz} \approx 100\text{ Hz}$ | **PASS** |
| **IR NaN / Inf Samples** | **0** | 0 NaN/Inf | **PASS** |
| **IR Zero Samples** | **0** | 0 zeros | **PASS** |
| **Red Ambient Signal Mean** | **41.59 counts** | $< 1,000\text{ counts}$ (ambient inactive) | **PASS** |

The hardware sensor maintained steady FIFO timing with zero dropped packets and tight host-read intervals strictly bounded within 9–11 ms.

---

## 4. Replay Preprocessing

The replay simulation streamed incoming raw IR samples in **10-sample chunks ($100\text{ ms}$ budget)** through the identical causal DSP architecture validated in Phase 6B:
1. **Rational Resampling ($100 \to 125\text{ Hz}$)**:
   - Stateful polyphase FIR filter ($up=5, down=4$, 101 Kaiser taps).
   - Preserves historical samples across chunk boundaries, ensuring $0.00\text{e}+00$ boundary distortion.
   - Total resampled samples: **10,463 samples** at $125\text{ Hz}$ ($83.70\text{ s}$).
2. **Causal Bandpass Filtering ($0.5–8.0\text{ Hz}$)**:
   - 3rd-order Butterworth filter implemented in Second-Order Sections (SOS).
   - Persistent delay vector $z_i$ updated incrementally without future lookahead.
3. **Causal Backward Derivatives**:
   - $VPG[n] = (PPG[n] - PPG[n-1]) / \Delta t$
   - $APG[n] = (VPG[n] - VPG[n-1]) / \Delta t$ where $\Delta t = 1/125\text{ s}$.
   - Computed causally on window-accumulated, z-scored normalized PPG to prevent numerical scaling drift.

---

## 5. Windowing & Engineering Quality Gating

At $125\text{ Hz}$, a $10\text{-second}$ model window requires exactly **1,250 samples**.
- **Total Complete 10s Windows Produced**: **8 windows** ($80.0\text{ seconds}$ coverage).
- **Residual Unwindowed Samples**: 463 samples ($3.7\text{ s}$, discarded safely as trailing partial window).

### Window-by-Window Quality Breakdown:

| Window ID | Time Span | QC Status | Heart Rate | Detected Peaks | Peak-to-Peak | Drift Delta | Quality Gating Diagnostic |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `win_00` | 0–10s | **WARN** | 82.4 bpm | 4 | 228593 | 169923 | High baseline wander/drift (169923 counts) |
| `win_01` | 10–20s | **PASS** | 76.5 bpm | 10 | 70736 | 1903 | Valid pulsatile morphology within physiological bounds |
| `win_02` | 20–30s | **PASS** | 111.9 bpm | 17 | 4022 | 1750 | Valid pulsatile morphology within physiological bounds |
| `win_03` | 30–40s | **PASS** | 69.4 bpm | 12 | 5501 | 1673 | Valid pulsatile morphology within physiological bounds |
| `win_04` | 40–50s | **PASS** | 91.5 bpm | 14 | 2798 | 281 | Valid pulsatile morphology within physiological bounds |
| `win_05` | 50–60s | **PASS** | 78.9 bpm | 12 | 2895 | 803 | Valid pulsatile morphology within physiological bounds |
| `win_06` | 60–70s | **PASS** | 92.6 bpm | 13 | 3201 | 1099 | Valid pulsatile morphology within physiological bounds |
| `win_07` | 70–80s | **PASS** | 80.2 bpm | 13 | 2877 | 1468 | Valid pulsatile morphology within physiological bounds |


### Quality Analysis:
- **Window 0 (`0–10s`)**: Flagged as **`WARN`** due to baseline drift during the initial finger placement transition at $t = 3.46\text{ s}$ (IR jumped from ambient ~1,000 counts to physiological ~170,000 counts). Crucially, the window was **NOT rejected** because pulsatile cycles were detected and no clipping occurred.
- **Windows 1 through 7 (`10–80s`)**: All **7 windows PASSED** with clean pulsatile waveforms, healthy physiological heart rates ($69.4–111.9\text{ bpm}$), and minimal baseline wander ($< 2,000\text{ counts}$).

---

## 6. Temporal Sequence Construction

The Phase 4B causal GRU requires a rolling sequence of **6 contiguous 10-second windows** ($60.0\text{ seconds}$ context):
- **Windows required before first prediction**: 6 complete windows ($t = 60.0\text{ s}$).
- **Total Valid 6-Window Sequences Formed**: **3 sequences**.

```
Timeline (s):   0    10   20   30   40   50   60   70   80
Windows:       |--W0--|--W1--|--W2--|--W3--|--W4--|--W5--|--W6--|--W7--|
Sequence 0:    [=========== Context (0 to 60s) ==========] -> Predict @ 60s
Sequence 1:         [=========== Context (10 to 70s) =========] -> Predict @ 70s
Sequence 2:              [=========== Context (20 to 80s) =========] -> Predict @ 80s
```

---

## 7. Frozen Model Configuration

All deep learning and statistical models operated in strictly frozen evaluation mode:

| Component | Architecture / Source | Parameters | Trainable | Checkpoint Path |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 4A Encoder** | 4-Stage 1D CNN + Dense Head | 146,978 | **0** | `code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt` |
| **Phase 4B Temporal Model** | 1-Layer Unidirectional Causal GRU (hidden=64) | 27,106 | **0** | `code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt` |
| **Total Neural Parameters** | Single Model Ensemble | **174,084** | **0** | **Strictly Frozen** |
| **Phase 5C Recalibration** | Isotonic Regression Models | N/A | 0 | `code/outputs/phase5c_extreme_aware/mappings/isotonic_[sbp,dbp].pkl` |
| **Phase 5C Uncertainty** | Regime-Binned 95% Conformal | N/A | 0 | `code/outputs/phase5c_extreme_aware/calibration/conformal_quantiles_by_bin.json` |

---

## 8. Prediction Results

The streaming pipeline generated 3 successive predictions across the rolling 60-second context windows:

| Sequence ID | Context Span | Target Window | Quality | Raw SBP | Calibrated SBP | 95% Conformal SBP | Raw DBP | Calibrated DBP | 95% Conformal DBP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seq_00` | 0–60s | `hw_win_05` | **WARN** | 150.66 | 151.43 | [115.9–182.8] | 84.30 | 87.95 | [73.7–109.0] |
| `seq_01` | 10–70s | `hw_win_06` | **PASS** | 148.74 | 149.16 | [113.6–180.5] | 85.34 | 88.25 | [74.0–109.3] |
| `seq_02` | 20–80s | `hw_win_07` | **PASS** | 145.38 | 143.19 | [107.6–174.5] | 84.51 | 87.95 | [73.7–109.0] |


### Prediction Summary Statistics:
- **Systolic Blood Pressure (SBP)**:
  - *Raw Model Output*: Mean = **148.26 mmHg** | Median = **148.74 mmHg** | Range = **[145.38, 150.66] mmHg** | Std = **2.18 mmHg**
  - *Calibrated Output*: Mean = **147.92 mmHg** | Median = **149.16 mmHg** | Range = **[143.19, 151.43] mmHg** | Std = **3.48 mmHg**
- **Diastolic Blood Pressure (DBP)**:
  - *Raw Model Output*: Mean = **84.72 mmHg** | Median = **84.51 mmHg** | Range = **[84.30, 85.34] mmHg** | Std = **0.45 mmHg**
  - *Calibrated Output*: Mean = **88.05 mmHg** | Median = **87.95 mmHg** | Range = **[87.95, 88.25] mmHg** | Std = **0.14 mmHg**

---

## 9. Signal and Quality Diagnostics

All 8 requested diagnostic figures were produced and verified:
1. **Raw IR Optical Trace** (`01_raw_ir_diagnostic.png`): Demonstrates the ambient-to-contact transition at $t = 3.46\text{ s}$ followed by steady physiological PPG.
2. **Resampled PPG Stream** (`02_resampled_ppg.png`): Confirms continuous 125 Hz polyphase reconstruction with zero edge spikes.
3. **Causal Filtered PPG** (`03_causal_filtered_ppg.png`): Shows baseline wander elimination and clean systolic waveforms.
4. **Causal VPG** (`04_causal_vpg.png`): First derivative exhibiting clear systolic upstroke velocity peaks.
5. **Causal APG** (`05_causal_apg.png`): Second derivative showing distinct $a, b, c, d, e$ wave acceleration complexes.
6. **Representative Model Input Windows** (`06_representative_model_input_windows.png`): 3-channel normalized tensor $[PPG, VPG, APG]$ for Window 5.
7. **SBP Prediction Trajectory** (`07_sbp_prediction_trajectory.png`): Trajectory of calibrated SBP with 95% conformal uncertainty bounds.
8. **DBP Prediction Trajectory** (`08_dbp_prediction_trajectory.png`): Trajectory of calibrated DBP with 95% conformal uncertainty bounds.

---

## 10. Host-Side Processing Benchmark

Streaming performance was measured across 837 chunk iterations (10 samples @ 100 Hz = $100.0\text{ ms}$ budget per chunk):
- **Total Replay Time**: **0.200 seconds** for an $83.88\text{-second}$ recording.
- **Mean Chunk Processing Latency**: **0.2380 ms** (Utilization: **0.24%** of processing budget).
- **95th Percentile Latency**: **0.3164 ms**.
- **Peak Chunk Latency** (includes full 10s window extraction + CNN + GRU forward passes): **4.5124 ms**. Peak chunk latency remained well below the 100 ms real-time input-chunk budget.
- **Host Execution Headroom**: **420.2x faster than required real-time speed** (based on mean chunk latency).

This replay demonstrates substantial host-side processing headroom relative to the 100 ms input-chunk budget, with no observed buffer overflow during the replay.

---

## 11. Comparison with Previous Hardware Captures

| Parameter | Previous Capture (`final_dataset_ready.csv`) | New Capture (`final_dataset_ready(1).csv`) | Note / Distinction |
| :--- | :---: | :---: | :--- |
| **Duration** | 96.01 s | 83.88 s | Both $> 60\text{ s}$ context requirement |
| **Raw Sample Count** | 9,601 | 8,370 | Nominal 100 Hz acquisition |
| **Sample Index Discontinuities** | 0 | 0 | Flawless FIFO transport in both |
| **10s Windows Produced** | 9 windows | 8 windows | Consistent 1,250 sample windows |
| **Window Quality** | 100% PASS | 87.5% PASS, 12.5% WARN | Initial placement in window 0 |
| **6-Window Sequences** | 4 sequences | 3 sequences | Expected due to duration difference |
| **Pipeline Completion** | **100% PASS** | **100% PASS** | Zero pipeline crashes or exceptions |
| **Mean Raw SBP Output** | 141.06 mmHg | 148.26 mmHg | Model outputs reflect different recordings |
| **Mean Raw DBP Output** | 83.29 mmHg | 84.72 mmHg | Stable diastolic predictions |
| **Host Headroom** | 458x real-time | 420.2x real-time | Sub-millisecond latency sustained |

---

## 12. Validation Limitations & Engineering Conclusion

### Critical Distinctions:
1. **Hardware Capture Validity**: The MAX30102 acquisition setup successfully captured optical PPG at $99.77\text{ Hz}$ with zero index discontinuities and tight timing ($9–11\text{ ms}$).
2. **Software Pipeline Execution**: The streaming preprocessing, windowing, and frozen model inference completed without execution errors or NaN/Inf outputs. One initial window was flagged WARN by the engineering quality gate because of sensor-contact settling and baseline drift.
3. **Model Prediction Output**: The frozen Phase 4A CNN and Phase 4B GRU produced stable predictions across all 3 temporal sequences.
4. **Blood-Pressure Accuracy**: **NO BP ACCURACY IS ESTABLISHED BY THIS EXPERIMENT.** Because reference cuff measurements were not simultaneously recorded, neither MAE, RMSE, nor clinical validity can be reported.

### Definitive Conclusion:
The fresh physical MAX30102 recording successfully passed the frozen Phase 6B hardware-to-model replay pipeline. This validates engineering execution of the hardware acquisition, streaming preprocessing, windowing, temporal inference, and host-side replay path. It does not establish blood-pressure accuracy or clinical validity. The project is ready for Phase 6C synchronized reference-cuff validation.
