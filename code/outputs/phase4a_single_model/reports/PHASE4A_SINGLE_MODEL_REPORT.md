# PHASE 4A — Single-Model Neural Baseline Report

**Architecture:** PPG + VPG + APG 1D CNN  
**Mode:** Calibration-Free Cuffless Blood Pressure Estimation  
**Execution Environment:** Local System (NVIDIA GeForce GTX 1650 Ti)  
**Timestamp:** 2026-09-18 13:28:44  

---

## 1. Main Research Objective
Can a compact 3-channel neural representation consisting of PPG, VPG, and APG estimate SBP and DBP from a single 10-second PPG window under the existing calibration-free, record-level leakage-controlled protocol?

## 2. Dataset and Protocol
- **Dataset Source:** PhysioNet MIMIC-II / Kachuee Blood Pressure Dataset (12 MAT parts)
- **Record-level Partition:** 8,400 train records, 1,800 validation records, 1,800 test records
- **Eligible Windows:** Train: 183,517, Val: 39,461, Test: 38,361
- **Sampling Frequency:** Fs = 125 Hz (1,250 samples per 10-second non-overlapping window)

## 3. Strict Sensor Rules & Leakage Controls
- Channel 0 (PPG) was used strictly as model input.
- Channel 1 (ABP) was used strictly as reference ground truth target.
- Channel 2 (ECG) was strictly forbidden and never accessed.
- Record IDs between Train, Validation, and Test sets were strictly pairwise disjoint (0 record leakage).

## 4. Preprocessing & Input Definitions
- 3rd-order zero-phase Butterworth bandpass filter (0.5 - 8.0 Hz)
- Velocity Plethysmogram (VPG): 1st time derivative
- Acceleration Plethysmogram (APG): 2nd time derivative
- Per-window z-score normalization across all 3 channels
- Input tensor shape: [3, 1250], float32

## 5. Neural Architecture & Hyperparameters
- Architecture: 4 ConvBlocks (Conv1D + BN + ReLU + MaxPool/AdaptiveAvgPool) -> FC(128 -> 64) -> Dual Linear Heads (SBP, DBP)
- Total Parameters: 146,978 (trainable: 146,978)
- Loss: Huber loss (delta=5.0) summed across SBP and DBP
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Batch size: 128
- Early stopping patience: 8 epochs

## 6. Benchmark Performance Comparison (Frozen Test Set)

| Model / Architecture | Split | SBP MAE (mmHg) | DBP MAE (mmHg) | Combined MAE (mmHg) | SBP RMSE | DBP RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 3A Classical Baseline (HistGB)** | Test (Frozen) | 13.93 | 7.05 | 10.49 | 18.27 | 9.42 |
| **Phase 4A Neural Baseline (PPG+VPG+APG)** | Val | 11.29 | 5.62 | 8.46 | 15.54 | 8.28 |
| **Phase 4A Neural Baseline (PPG+VPG+APG)** | Test (Frozen) | 11.04 | 5.79 | 8.41 | 14.99 | 8.52 |

## 7. Clinical BP Range Stratified Errors (Descriptive)

### SBP Ranges:
| target   | range   |   sample_count |     mae |    rmse |       bias |   error_sd |
|:---------|:--------|---------------:|--------:|--------:|-----------:|-----------:|
| SBP      | <90     |            696 | 21.7465 | 26.0885 |  21.4653   |    14.8273 |
| SBP      | 90-119  |          12679 | 11.0873 | 15.0249 |   9.10616  |    11.951  |
| SBP      | 120-139 |          12759 |  7.9152 | 10.3534 |   0.702706 |    10.3295 |
| SBP      | 140-159 |           8512 | 10.9936 | 14.3036 |  -7.84783  |    11.9584 |
| SBP      | >=160   |           3715 | 19.6785 | 24.194  | -18.1653   |    15.9803 |

### DBP Ranges:
| target   | range   |   sample_count |      mae |     rmse |      bias |   error_sd |
|:---------|:--------|---------------:|---------:|---------:|----------:|-----------:|
| DBP      | <60     |          11647 |  5.24205 |  6.80191 |   4.57494 |    5.03348 |
| DBP      | 60-79   |          21716 |  4.30098 |  5.67566 |  -1.11338 |    5.56538 |
| DBP      | 80-89   |           3216 |  8.72206 | 11.0609  |  -8.17197 |    7.454   |
| DBP      | 90-99   |           1150 | 17.7381  | 20.4009  | -17.4119  |   10.6312  |
| DBP      | >=100   |            632 | 30.1425  | 32.5845  | -30.125   |   12.419   |

## 8. Record-Level Performance (Record-Independent Evaluation)
- SBP Record MAE: Mean = 11.44, Median = 8.10, SD = 9.37 mmHg
- DBP Record MAE: Mean = 6.19, Median = 4.17, SD = 5.92 mmHg
- Combined Record MAE: Mean = 8.81, Median = 6.73, SD = 6.42 mmHg

## 9. Hardware Deployment Note
Future deployment on MAX30102 (100 Hz) utilizes polyphase anti-aliasing resampling:
`ppg_125hz = resample_poly(ppg_100hz, up=5, down=4)`
Research training was conducted strictly at 125 Hz.

## 10. Scientific Limitations & Next Research Step
- Evaluated on ICU patient records; performance on ambulatory healthy subjects remains unverified.
- Phase 4A establishes the compact single-window neural baseline.
- Next Step (Phase 4B): Introduce temporal sequence context (GRU/LSTM/Transformer) over successive windows.
