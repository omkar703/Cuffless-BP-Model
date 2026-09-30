# PHASE 6C: REFERENCE-CUFF BLOOD PRESSURE VALIDATION REPORT

**Project**: Calibration-Free Cuffless Blood-Pressure Estimation using Photoplethysmography Only  
**Session Identifier**: `03`  
**Execution Timestamp**: `2026-09-30T11:55:11.948729+00:00`  
**Source Data**: `uploaded.zip` (`ZIP`)  
**Validation Status**: **PHYSICAL VALIDATION COMPLETE**  

---

## 1. Executive Summary & Protocol Overview

Phase 6C provides the direct empirical comparison between frozen neural model estimates and physical reference oscillometric cuff measurements. The analysis strictly enforces the **Zero-Retraining Invariant**: all neural weights (174,084 parameters) remain 100% frozen, with zero fine-tuning, no adaptive domain shifting, and no recalibration against test data.

- **Total Raw PPG Samples**: `16,328`
- **Replay Duration**: `0.349 s` (Headroom: `470.4x` vs 100 ms chunk budget)
- **10-Second Windows Formed**: `16` (15 PASS, 1 WARN, 0 REJECT)
- **60-Second Sequences Evaluated**: `11`
- **Reference BP Events Ingested**: `2`
- **Deterministically Matched Pairs**: `2` (`2` meeting full quality inclusion criteria)

---

## 2. Hardware Acquisition Integrity Audit

**Audit Verdict**: **`WARN`**

| Audit Metric | Observed Value | Nominal Threshold / Expectation | Status |
|---|---|---|---|
| Row Count | 16,328 | > 1,000 samples | PASS |
| Sample Index Step (Modal) | 1 | 10 (10 ms @ 100 Hz) | WARN |
| Discontinuities vs Modal Step | 0 | 0 | PASS |
| Duplicate / Reverse Indices | 0 | 0 | PASS |
| Effective Acquisition Rate | 99.780 Hz | 99.0 – 101.0 Hz | PASS |
| Host Mean Interval | 10.022 ms | ~10.0 ms | PASS |
| Jitter Outside 9–11 ms | 3254 (19.93%) | 0 | WARN |
| IR NaN or Inf Samples | 0 | 0 | PASS |
| IR Amplitude Range | [779, 238315] | [0, 262,143] (18-bit ADC) | PASS |
| Red Channel Mean | 41.74 counts | Ambient / Baseline | INFO |

> [!NOTE]
> **Audit Observations:**
> - 3254 host intervals (19.9%) outside 9–11 ms
> - Red channel mean is low (41.7 counts); indicates ambient/single-wavelength mode

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
- Total 10-Second Windows: `16`
- `PASS` Windows: `15`
- `WARN` Windows: `1`
- `REJECT` Windows: `0`

---

## 4. Deterministic Reference-BP Pairing

Pairing Rule Applied: **`NEAREST_PREDICTION_TIMESTAMP`**  
Maximum Temporal Tolerance: **`±15.0 seconds`**  

For each `REFERENCE_BP_RESULT` event recorded by the reference device, the engine finds the completed causal model prediction whose target timestamp is temporally closest to the reference measurement. If the time difference exceeds the predefined tolerance, the event is marked `UNMATCHED`. No post-hoc cherry-picking or manual window selection is permitted.

### Matched Pairs Audit Table

| Event ID | Ref SBP/DBP | Pred SBP (Raw / Cal) | Pred DBP (Raw / Cal) | $\Delta t$ (s) | Sequence QC | Status | Reason |
|---|---|---|---|---|---|---|---|
| 2 | 120.0 / 80.0 | 145.7 / 143.9 | 80.9 / 81.0 | -3.15 | PASS | INCLUDED | NONE |
| 5 | 120.0 / 80.0 | 141.5 / 140.3 | 78.1 / 79.6 | -3.71 | PASS | INCLUDED | NONE |

---

## 5. Statistical Validation Metrics

Calculated on **`2`** valid matched pairs meeting complete inclusion criteria. RAW and CALIBRATED metrics are reported independently.

| Metric | SBP (Raw) | SBP (Calibrated) | DBP (Raw) | DBP (Calibrated) | Clinical / Benchmark Standard |
|---|---|---|---|---|---|
| **Sample Size (N)** | 2 | 2 | 2 | 2 | — |
| **Mean Absolute Error (MAE)** | 23.60 mmHg | **22.09 mmHg** | 1.42 mmHg | **0.69 mmHg** | Lower is better |
| **Root Mean Squared Error (RMSE)** | 23.70 mmHg | 22.16 mmHg | 1.51 mmHg | 0.76 mmHg | Lower is better |
| **Mean Error (Bias)** | +23.60 mmHg | +22.09 mmHg | -0.52 mmHg | +0.32 mmHg | AAMI: $\le \pm 5.0$ mmHg |
| **Standard Deviation of Error** | 2.98 mmHg | 2.54 mmHg | 2.00 mmHg | 0.97 mmHg | AAMI: $\le 8.0$ mmHg |
| **Pearson Correlation ($r$)** | nan | nan | nan | nan | $p = nan$ |
| **Spearman Rank Correlation ($\rho$)** | nan | nan | nan | nan | Rank preservation |
| **Error $\le 5$ mmHg (BHS Grade)** | 0.0% | 0.0% | 100.0% | 100.0% | BHS: $\ge 60\%$ Grade A |
| **Error $\le 10$ mmHg (BHS Grade)** | 0.0% | 0.0% | 100.0% | 100.0% | BHS: $\ge 85\%$ Grade A |
| **AAMI Compliance Criteria** | NOT MET | **NOT MET** | MET | **MET** | Bias $\le 5$, SD $\le 8$ mmHg |

### Absolute Error Quantiles (Calibrated)
- **SBP**: Median (p50) = `22.09 mmHg`, p75 = `22.98 mmHg`, p95 = `23.70 mmHg`, Max = `23.88 mmHg`
- **DBP**: Median (p50) = `0.69 mmHg`, p75 = `0.85 mmHg`, p95 = `0.97 mmHg`, Max = `1.00 mmHg`

---

## 6. Stratified Subgroup Analysis

### SBP Range Stratification
- **<120 mmHg (Normotensive)**: Insufficient samples (N < 2) (N=0)
- **120-139 mmHg (Elevated/Stage 1)** (N=2): Calibrated MAE = `22.09 mmHg`, Bias = `+22.09 mmHg`
- **>=140 mmHg (Stage 2/Hypertensive)**: Insufficient samples (N < 2) (N=0)

### DBP Range Stratification
- **<60 mmHg (Low)**: Insufficient samples (N < 2) (N=0)
- **60-79 mmHg (Normal)**: Insufficient samples (N < 2) (N=0)
- **>=80 mmHg (Elevated/Stage 1+)** (N=2): Calibrated MAE = `0.69 mmHg`, Bias = `+0.32 mmHg`

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