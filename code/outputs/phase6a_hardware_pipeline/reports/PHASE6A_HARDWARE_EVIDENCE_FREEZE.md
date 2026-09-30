# PHASE 6A EVIDENCE FREEZE: HARDWARE-TO-MODEL PIPELINE VALIDATION

**Project**: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only  
**Execution Timestamp**: 2026-09-29 21:30:56  
**Mode**: Inference-Only Hardware Pipeline Compatibility Validation  
**Zero-Retraining Assertion**: Phase 6A was strictly inference-only. Zero neural models were trained or updated.  

---

## 1. Hardware Input & Ingestion Specification
- **Hardware Source File**: `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/hardware/final_dataset_ready.csv`
- **Source Size**: 9,601 samples (96.21 seconds)
- **Raw Channels Available**: `sample_index`, `timestamp_ms`, `ir`, `red`
- **Active Channel Used**: `ir` (Optical Photoplethysmogram)
- **Diagnostic / Excluded Channel**: `red` (Ambient room light verification only, NEVER fed to model)
- **Sampling Interval**: Mean = 10.0217 ms | Median = 10.0000 ms | Range = [10.0, 11.0] ms
- **Effective Acquisition Rate**: 99.78 Hz (nominal 100 Hz)
- **Index Continuity**: Exactly 0 index discontinuities detected across all 9,601 samples.
- **Sample Dropping**: ZERO internal samples deleted. No amplitude thresholding (`IR >= 50,000`) applied to raw timebase.

---

## 2. Resampling & DSP Specification
- **Resampling Method**: Polyphase filtering via `scipy.signal.resample_poly(up=5, down=4)` with Kaiser anti-aliasing FIR.
- **Resampled Frequency**: Exactly 125.0 Hz (12,002 samples, 96.016 s).
- **Filter Parameters**: 3rd-order Butterworth bandpass (0.5–8.0 Hz, Nyquist = 62.5 Hz).
- **Path A (Research Reference)**: `scipy.signal.filtfilt` (zero-phase forward-backward) + central `np.gradient`.
- **Path B (Causal Proxy)**: `scipy.signal.sosfilt` with state vector `zi` (forward-only, zero future access) + backward difference.
- **Normalization**: Per-window z-score $z = (x - \mu) / (\sigma + 10^{-8})$.

---

## 3. Modeling Windows & Sequences
- **Window Size**: 10.0 seconds (1,250 samples at 125 Hz).
- **Extracted Windows**: 9 complete non-overlapping windows.
- **Window Quality Classification**: PASS = 8, WARN = 1, REJECT = 0.
- **Sequence Length**: 6 consecutive windows (60.0 seconds history).
- **Causal Sequences**: 4 valid sequences (Seq 0: $0–60\text{s}$, Seq 1: $10–70\text{s}$, Seq 2: $20–80\text{s}$, Seq 3: $30–90\text{s}$).

---

## 4. Frozen Neural Model State
- **Phase 4A CNN Checkpoint**: `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt`
  - Backbone parameters: 146,978 (FROZEN: 0 trainable)
- **Phase 4B Temporal GRU Checkpoint**: `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt`
  - Recurrent parameters: 27,106 (FROZEN: 0 trainable)
- **Total Parameters**: 174,084 (Trainable: **0**)

---

## 5. Model Outputs on Hardware-Recorded PPG
*(Note: These outputs are research model inferences, NOT validated clinical blood pressure measurements.)*

### Primary Non-Overlapping Sequences:
| Sequence ID | Timestamp (s) | Target Window | Ref SBP (mmHg) | Causal SBP (mmHg) | Ref DBP (mmHg) | Causal DBP (mmHg) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| seq_00 | 60.0 | hw_win_05 | 140.92 | 143.99 | 78.68 | 83.60 |
| seq_01 | 70.0 | hw_win_06 | 140.91 | 142.25 | 79.53 | 83.53 |
| seq_02 | 80.0 | hw_win_07 | 139.11 | 143.86 | 79.60 | 85.19 |
| seq_03 | 90.0 | hw_win_08 | 142.12 | 133.16 | 78.80 | 80.86 |

---

## 6. Preprocessing Stability Metrics
- **SBP MAD Difference**: 4.53 mmHg (Relative: 3.21%)
- **DBP MAD Difference**: 4.14 mmHg (Relative: 5.23%)
- **Pearson Correlation (Ref vs Causal)**: SBP $r = -0.766$, DBP $r = 0.637$

---

## 7. Explicit Clinical Caveat
> [!CAUTION]
> **No Hardware BP Accuracy Measured**: The current MAX30102 hardware recording does not contain simultaneous reference blood-pressure measurements. Therefore, no BP estimation accuracy (MAE, RMSE, or AAMI/BHS compliance) is claimed or reported in Phase 6A.
