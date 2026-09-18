# Phase 1: Dataset Quality Control & Validation Audit Report

## 1. Executive Dataset Summary

- **Dataset Source**: PhysioNet MIMIC-II Waveform Database (Curated by Kachuee et al.)
- **Acquisition Sampling Rate**: 125 Hz
- **Total Records Analyzed**: 12,000
- **Structurally Intact Records**: 12,000 (100.0%)
- **Valid Paired Records (PPG + ABP)**: 11,995 (100.0%)
- **Valid PPG-Only Records**: 5 (0.0%)
- **Valid ABP-Only Records**: 0 (0.0%)
- **Completely Rejected Records**: 0 (0.0%)

> [!NOTE]
> **Decoupled Quality Gates**: Records with corrupted arterial blood pressure (catheter disconnects, clamping, line damping) are **not discarded from PPG modeling**. Their PPG signals remain preserved and labeled as `valid_ppg_only`.

---

## 2. Raw Dataset Duration & Length Statistics

| Metric | Duration (Seconds) | Duration (Minutes) | Samples Count |
| :--- | :---: | :---: | :---: |
| **Minimum** | 8.0 s | 0.13 min | 1,000 |
| **Maximum** | 592.0 s | 9.87 min | 74,000 |
| **Mean** | 222.5 s | 3.71 min | 27,808 |
| **Median** | 160.0 s | 2.67 min | 20,000 |
| **Standard Deviation** | 196.6 s | 3.28 min | — |

---

## 3. PPG Signal Characteristics & Diagnostics

- **Valid PPG Count**: 12,000 (100.0%)
- **Peak-to-Peak (PTP) Amplitude Mean**: 2.585 V
- **Peak-to-Peak (PTP) Amplitude Median**: 2.694 V
- **PTP Amplitude 5th – 95th Percentile**: [0.580 V, 4.002 V]
- **Signal Standard Deviation (Mean)**: 0.5067

### Main Quality Control Findings for PPG:
1. **Flatline Dropouts**: Sensor detachment manifests as $\sigma < 10^{-5}$ and zero variance.
2. **Clipping / Saturation**: Certain sensors hit ADC rails where $>5\%$ of samples clamp to the extrema.
3. **Amplitude Variation**: Absolute optical amplitude varies heavily by subject tissue and probe gain; relative contour features are required rather than absolute thresholding.

---

## 4. ABP Reference Signal Characteristics & Beat Diagnostics

- **Valid ABP Reference Count**: 11,995 (100.0%)
- **Mean SBP (Beat-by-Beat Peak)**: 127.2 ± 21.9 mmHg
- **Median SBP**: 125.5 mmHg (Range: 69.7 – 196.4 mmHg)
- **Mean DBP (Beat-by-Beat Trough)**: 65.7 ± 11.1 mmHg
- **Median DBP**: 63.2 mmHg (Range: 50.5 – 134.5 mmHg)
- **Mean Arterial Pressure (MAP)**: 87.4 ± 13.6 mmHg
- **Estimated Heart Rate**: 87.6 ± 15.7 bpm
- **Average Valid Beat Percentage**: 99.7%

---

## 5. Signal Rejection Reasons Breakdown

### PPG Rejection Causes:
| Rejection Reason | Occurrences |
| :--- | :---: |

### ABP Rejection Causes:
| Rejection Reason | Occurrences |
| :--- | :---: |
| `DBP_OUT_OF_RANGE_P05_149.0_MMHG` | 1 |
| `LOW_PULSE_PRESSURE_8.5_MMHG` | 1 |
| `DBP_OUT_OF_RANGE_P05_141.8_MMHG` | 1 |
| `DBP_OUT_OF_RANGE_P05_155.5_MMHG` | 1 |
| `LOW_PULSE_PRESSURE_6.1_MMHG` | 1 |

---

## 6. Generated Visualizations & Artifacts

All diagnostic figures have been generated and saved under `code/outputs/figures/`:
1. `01_record_duration_distribution.png`: Histogram and KDE of recording durations.
2. `02_ppg_amplitude_distribution.png`: Distribution of PPG peak-to-peak amplitudes.
3. `03_abp_continuous_distribution.png`: Distribution of ABP continuous integration.
4. `04_sbp_distribution.png`: Distribution of beat-by-beat SBP ground truth.
5. `05_dbp_distribution.png`: Distribution of beat-by-beat DBP ground truth.
6. `06_quality_status_breakdown.png`: Bar chart of decoupled quality categories.
7. `07_rejection_reasons_breakdown.png`: Pareto charts of anomaly causes.

---

## 7. Strategic Decisions for Phase 2 Modeling

1. **Strictly Calibration-Free**: Avoid the calibrated delta crutch identified in earlier work.
2. **Pure PPG Pipeline**: Discard all ECG channels and PTT dependencies.
3. **Record-Level Partitioning**: All training/validation/test splits in Phase 2 must use the persistent `record_id` from `dataset_qc_manifest.csv` to ensure zero patient leakage.
4. **Offline Filter Transition**: The 0.5–8.0 Hz zero-phase filter applied here serves as the ground-truth offline standard. Phase 2 must design an equivalent causal streaming filter suitable for ESP32 deployment.
