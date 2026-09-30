# PHASE 6C: REFERENCE-CUFF BLOOD PRESSURE VALIDATION REPORT

**Project**: Calibration-Free Cuffless Blood-Pressure Estimation using Photoplethysmography Only  
**Session Identifier**: `02`  
**Execution Timestamp**: `2026-09-30T11:55:08.887892+00:00`  
**Source Data**: `uploaded.zip` (`ZIP`)  
**Validation Status**: **PHYSICAL VALIDATION COMPLETE**  

---

## 1. Executive Summary & Protocol Overview

Phase 6C provides the direct empirical comparison between frozen neural model estimates and physical reference oscillometric cuff measurements. The analysis strictly enforces the **Zero-Retraining Invariant**: all neural weights (174,084 parameters) remain 100% frozen, with zero fine-tuning, no adaptive domain shifting, and no recalibration against test data.

- **Total Raw PPG Samples**: `18,722`
- **Replay Duration**: `0.399 s` (Headroom: `472.6x` vs 100 ms chunk budget)
- **10-Second Windows Formed**: `18` (18 PASS, 0 WARN, 0 REJECT)
- **60-Second Sequences Evaluated**: `13`
- **Reference BP Events Ingested**: `1`
- **Deterministically Matched Pairs**: `1` (`1` meeting full quality inclusion criteria)

---

## 2. Hardware Acquisition Integrity Audit

**Audit Verdict**: **`WARN`**

| Audit Metric | Observed Value | Nominal Threshold / Expectation | Status |
|---|---|---|---|
| Row Count | 18,722 | > 1,000 samples | PASS |
| Sample Index Step (Modal) | 1 | 10 (10 ms @ 100 Hz) | WARN |
| Discontinuities vs Modal Step | 0 | 0 | PASS |
| Duplicate / Reverse Indices | 0 | 0 | PASS |
| Effective Acquisition Rate | 99.783 Hz | 99.0 – 101.0 Hz | PASS |
| Host Mean Interval | 10.022 ms | ~10.0 ms | PASS |
| Jitter Outside 9–11 ms | 3680 (19.66%) | 0 | WARN |
| IR NaN or Inf Samples | 0 | 0 | PASS |
| IR Amplitude Range | [163435, 184476] | [0, 262,143] (18-bit ADC) | PASS |
| Red Channel Mean | 41.63 counts | Ambient / Baseline | INFO |

> [!NOTE]
> **Audit Observations:**
> - 3680 host intervals (19.7%) outside 9–11 ms
> - Red channel mean is low (41.6 counts); indicates ambient/single-wavelength mode

---

## 3. Frozen Pipeline Execution & Quality Gating

The hardware stream is fed through the exact frozen Phase 6B architecture:
1. **Polyphase Rational Resampler**: $100\text{ Hz} \to 125\text{ Hz}$ ($P=5, Q=4$).
2. **Causal Butterworth SOS Bandpass**: $0.5\text{–}8.0\text{ Hz}$ (3rd order, stateful `sosfilt`, no future access).
3. **Causal Backward Differences**: First derivative (VPG) and second derivative (APG) via $\Delta t = 8\text{ ms}$.
4. **Window Normalization**: 10-second non-overlapping frames ($1,250$ samples), per-window z-score normalization.
5. **Feature Encoder**: Frozen Phase 4A 1D CNN (`best_model_ppg_vpg_apg.pt`, $146,978$ frozen parameters) producing 64-dimensional temporal embeddings.
6. **Temporal Context Model**: Frozen Phase 4B 1-Layer Unidirectional Causal GRU (`best_temporal_gru.pt`, $27,106$ frozen parameters) over rolling 6-window contexts ($60\text{ s}$).
7. **Extreme-Aware Calibration**: Frozen Phase 5C isotonic recalibration and bin-specific 95% conformal intervals.

### Window Quality Audit Summary
- Total 10-Second Windows: `18`
- `PASS` Windows: `18`
- `WARN` Windows: `0`
- `REJECT` Windows: `0`

---

## 4. Deterministic Reference-BP Pairing

Pairing Rule Applied: **`NEAREST_PREDICTION_TIMESTAMP`**  
Maximum Temporal Tolerance: **`±15.0 seconds`**  

For each `REFERENCE_BP_RESULT` event recorded by the reference device, the engine finds the completed causal model prediction whose target timestamp is temporally closest to the reference measurement. If the time difference exceeds the predefined tolerance, the event is marked `UNMATCHED`. No post-hoc cherry-picking or manual window selection is permitted.

### Matched Pairs Audit Table

| Event ID | Ref SBP/DBP | Pred SBP (Raw / Cal) | Pred DBP (Raw / Cal) | $\Delta t$ (s) | Sequence QC | Status | Reason |
|---|---|---|---|---|---|---|---|
| 2 | 120.0 / 80.0 | 137.5 / 136.9 | 84.6 / 87.9 | -3.43 | PASS | INCLUDED | NONE |

---

## 5. Statistical Validation Metrics

Calculated on **`1`** valid matched pairs meeting complete inclusion criteria. RAW and CALIBRATED metrics are reported independently.

| Metric | SBP (Raw) | SBP (Calibrated) | DBP (Raw) | DBP (Calibrated) | Clinical / Benchmark Standard |
|---|---|---|---|---|---|
| **Sample Size (N)** | 1 | 1 | 1 | 1 | — |
| **Mean Absolute Error (MAE)** | 17.54 mmHg | **16.87 mmHg** | 4.62 mmHg | **7.95 mmHg** | Lower is better |
| **Root Mean Squared Error (RMSE)** | 17.54 mmHg | 16.87 mmHg | 4.62 mmHg | 7.95 mmHg | Lower is better |
| **Mean Error (Bias)** | +17.54 mmHg | +16.87 mmHg | +4.62 mmHg | +7.95 mmHg | AAMI: $\le \pm 5.0$ mmHg |
| **Standard Deviation of Error** | 0.00 mmHg | 0.00 mmHg | 0.00 mmHg | 0.00 mmHg | AAMI: $\le 8.0$ mmHg |
| **Pearson Correlation ($r$)** | nan | nan | nan | nan | $p = nan$ |
| **Spearman Rank Correlation ($\rho$)** | nan | nan | nan | nan | Rank preservation |
| **Error $\le 5$ mmHg (BHS Grade)** | 0.0% | 0.0% | 100.0% | 0.0% | BHS: $\ge 60\%$ Grade A |
| **Error $\le 10$ mmHg (BHS Grade)** | 0.0% | 0.0% | 100.0% | 100.0% | BHS: $\ge 85\%$ Grade A |
| **AAMI Compliance Criteria** | NOT MET | **NOT MET** | NOT MET | **NOT MET** | Bias $\le 5$, SD $\le 8$ mmHg |

### Absolute Error Quantiles (Calibrated)
- **SBP**: Median (p50) = `16.87 mmHg`, p75 = `16.87 mmHg`, p95 = `16.87 mmHg`, Max = `16.87 mmHg`
- **DBP**: Median (p50) = `7.95 mmHg`, p75 = `7.95 mmHg`, p95 = `7.95 mmHg`, Max = `7.95 mmHg`

---

## 6. Stratified Subgroup Analysis

### SBP Range Stratification
- **<120 mmHg (Normotensive)**: Insufficient samples (N < 2) (N=0)
- **120-139 mmHg (Elevated/Stage 1)**: Insufficient samples (N < 2) (N=1)
- **>=140 mmHg (Stage 2/Hypertensive)**: Insufficient samples (N < 2) (N=0)

### DBP Range Stratification
- **<60 mmHg (Low)**: Insufficient samples (N < 2) (N=0)
- **60-79 mmHg (Normal)**: Insufficient samples (N < 2) (N=0)
- **>=80 mmHg (Elevated/Stage 1+)**: Insufficient samples (N < 2) (N=1)

---

## 7. Diagnostic Figures & Visualizations

The following diagnostic figures were generated during the execution and saved to the `figures/` directory:

- **Raw Waveform & Event Overlay**: `figures/01_raw_ir_bp_timeline.png`
- **Causal 3-Channel Traces (PPG, VPG, APG)**: `figures/02_causal_derivatives.png`
- **Prediction Timeline & Conformal Bands**: `figures/03_prediction_conformal_timeline.png`
- **Predicted vs Reference Scatter (SBP & DBP)**: `figures/04_scatter_predicted_vs_reference.png`
- **Error Distribution Histograms**: `figures/05_error_distributions.png`
- **Bland-Altman Agreement Plots**: `figures/06_bland_altman_agreement.png`

---

## 8. Limitations & Scientific Caveats

1. **Exploratory Sample Size**: A single session or small cohort cannot establish population-level clinical validity under ISO 81060-2 or AAMI/ESH protocols (which require $\ge 85$ human subjects).
2. **Sensor Attachment & Motion**: The MAX30102 reflective photoplethysmogram is susceptible to finger motion and contact pressure variation. While causal quality gating flags motion artifacts, severe motion will interrupt continuous temporal inference.
3. **Zero Retraining Invariant**: Models operate in strictly calibration-free inference mode. No subject-specific tuning or baseline offsets were adapted.
4. **Non-Medical Designation**: This software and report are strictly for engineering research and algorithmic validation. They do not constitute a medical device.

---

## 9. Reproducibility & Environment Statement

- **Operating System**: `Linux-7.0.0-31-generic-x86_64-with-glibc2.43`
- **Python Version**: `3.10.21`
- **PyTorch Version**: `2.14.0+cu130`
- **Phase 4A CNN Checkpoint**: `best_model_ppg_vpg_apg.pt` (`SHA256: 2c6c5478e5ec0c66...`)
- **Phase 4B GRU Checkpoint**: `best_temporal_gru.pt` (`SHA256: 26ecb0abf690bd06...`)
- **Total Neural Parameters**: `174,084` (**0 Trainable**)
- **Phase 5C Extreme Calibration**: `isotonic_sbp.pkl`, `isotonic_dbp.pkl`, `conformal_quantiles_by_bin.json`

```text
RESEARCH APPLICATION ONLY — NOT A MEDICAL DEVICE. This software executes offline validation of frozen neural research models on physical PPG recordings. No neural network parameters are trained, fine-tuned, or updated. Predictions do not establish medical accuracy or clinical validity and must not be used for diagnosis or treatment.
```