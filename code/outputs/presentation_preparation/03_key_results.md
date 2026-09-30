# 03. Audited Project Key Results & Quantitative Benchmarks

**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Purpose**: Authoritative reference of all audited metrics across the project phases.

> [!IMPORTANT]
> **Strict Population Separation**: The metrics below represent distinct experimental populations and evaluation modes. Never conflate offline public-dataset test benchmarks with physical hardware replay or preliminary physical-pilot comparisons.

---

## Summary Overview Across All Research Phases

| Phase | Phase Description | Population / Evaluation Mode | Sample Size ($N$) | SBP MAE (mmHg) | SBP RMSE (mmHg) | DBP MAE (mmHg) | DBP RMSE (mmHg) | Key Scientific Insight |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **3A** | Dummy Mean Baseline | Internal Public Dataset (Test) | 38,361 windows | 17.69 | 21.78 | 8.77 | 11.45 | Population mean prediction floor. |
| **3A** | Classical ML (Branch C, HGB) | Internal Public Dataset (Test) | 38,361 windows | 13.93 | 17.94 | 7.05 | 9.67 | 40 handcrafted features; VPG/APG monotonic gain. |
| **3B** | Context-5 Classical Discovery | Internal Public Dataset (Val) | 32,163 windows | 13.28 | 17.31 | 6.56 | 8.91 | 60s context reduces MAE by 0.58 mmHg. |
| **4A** | 1D Multi-Channel CNN | Internal Public Dataset (Test) | 38,361 windows | 11.04 | 14.99 | 5.79 | 8.52 | Learned 3-channel morphological embedding. |
| **4B** | Frozen CNN + Causal 60s GRU | Internal Public Dataset (Test) | 31,192 sequences | **10.57** | **14.40** | **5.51** | **8.25** | Causal temporal modeling; best offline result. |
| **5C** | Extreme-Aware Conformal | Internal Public Dataset (Test) | 31,192 sequences | 10.73 | 14.51 | 5.66 | 8.23 | Post-hoc isotonic calibration + 91.2% / 96.7% coverage. |
| **6B** | Fresh MAX30102 Replay | Physical Hardware Replay | 8,370 samples | N/A | N/A | N/A | N/A | Zero index gaps; verified causal streaming DSP. |
| **6C** | Reference-Cuff Physical Pilot | Preliminary Physical Pilot | 7 paired events | **17.34** | **18.63** | **2.15** | **3.23** | Pilot: DBP high agreement; SBP systemic offset. |

---

## A. Offline Algorithmic Results (Public Benchmark: MIMIC-II Test Partition)
- **Population**: MIMIC-II Waveform Database (Kaggle Blood Pressure Dataset).
- **Partitioning**: Strictly record-level disjoint (1,621 active test records, zero training record overlap).
- **Input Sampling Rate**: 125 Hz | Window Duration: 10.0 seconds (1,250 samples).

### 1. Phase 3A: Classical Machine Learning Benchmark ($N = 38,361\text{ test windows}$)
Evaluated once on the frozen test partition using 40 handcrafted physiological features:

| Metric | Dummy Baseline | HistGradientBoosting (Branch C) | Relative Improvement |
|:---|:---:|:---:|:---:|
| **SBP Mean Absolute Error (MAE)** | 17.69 mmHg | **13.93 mmHg** | **+21.2%** |
| **SBP Root Mean Squared Error (RMSE)** | 21.78 mmHg | **17.94 mmHg** | +17.6% |
| **SBP Coefficient of Determination ($R^2$)** | 0.000 | **0.321** | — |
| **SBP Mean Error (Bias)** | -0.06 mmHg | **-0.35 mmHg** | — |
| **SBP % Within 5 mmHg** | 18.2% | **24.3%** | +6.1% abs |
| **SBP % Within 10 mmHg** | 36.1% | **45.9%** | +9.8% abs |
| **DBP Mean Absolute Error (MAE)** | 8.77 mmHg | **7.05 mmHg** | **+19.6%** |
| **DBP Root Mean Squared Error (RMSE)** | 11.45 mmHg | **9.67 mmHg** | +15.5% |
| **DBP Coefficient of Determination ($R^2$)** | 0.000 | **0.287** | — |
| **DBP Mean Error (Bias)** | +0.02 mmHg | **-0.40 mmHg** | — |
| **DBP % Within 5 mmHg** | 35.8% | **46.2%** | +10.4% abs |
| **DBP % Within 10 mmHg** | 66.8% | **78.0%** | +11.2% abs |

### 2. Phase 4A: 1D Multi-Channel CNN ($N = 38,361\text{ test windows}$)
Input: Raw 3-channel waveform $[3 \times 1250]$ (PPG, VPG, APG). Parameters: 146,978.

| Metric | Phase 3A Classical Baseline | Phase 4A Multi-Channel CNN | Relative Improvement |
|:---|:---:|:---:|:---:|
| **SBP MAE** | 13.93 mmHg | **11.04 mmHg** | **+20.7%** |
| **SBP RMSE** | 17.94 mmHg | **14.99 mmHg** | +16.4% |
| **SBP $R^2$** | 0.321 | **0.527** | +64.2% |
| **DBP MAE** | 7.05 mmHg | **5.79 mmHg** | **+17.9%** |
| **DBP RMSE** | 9.67 mmHg | **8.52 mmHg** | +11.9% |
| **DBP $R^2$** | 0.287 | **0.448** | +56.1% |
| **Combined MAE** | 10.49 mmHg | **8.41 mmHg** | **+19.8%** |

---

## B. Temporal / Context Modeling Results (Phase 4B)
- **Population**: 1,198 test records with $\ge 6$ contiguous windows ($N = 31,192\text{ sequences}$).
- **Architecture**: Frozen Phase 4A CNN embeddings + 1-layer unidirectional causal GRU (hidden dim 64, 27,106 parameters). Total: 174,084 parameters.
- **Context Length**: 60 seconds (6 consecutive 10-second windows).

| Evaluation Metric | Phase 4A (Matched Static Windows) | Phase 4B (Causal 60s GRU) | Absolute Gain | Relative Improvement |
|:---|:---:|:---:|:---:|:---:|
| **SBP MAE** | 10.94 mmHg | **10.57 mmHg** | **-0.38 mmHg** | **+3.4%** |
| **SBP RMSE** | 14.86 mmHg | **14.40 mmHg** | -0.46 mmHg | +3.1% |
| **SBP $R^2$** | 0.531 | **0.557** | +0.026 | +4.9% |
| **DBP MAE** | 5.69 mmHg | **5.51 mmHg** | **-0.18 mmHg** | **+3.2%** |
| **DBP RMSE** | 8.44 mmHg | **8.25 mmHg** | -0.19 mmHg | +2.3% |
| **DBP $R^2$** | 0.452 | **0.472** | +0.020 | +4.4% |
| **Combined MAE** | 8.32 mmHg | **8.04 mmHg** | **-0.28 mmHg** | **+3.3%** |

---

## C. Uncertainty Quantification & Conformal Calibration Results (Phase 5)
- **Population**: Same 31,192 test sequences from Phase 4B.
- **Trainable Parameters**: **Exactly 0** (all weights frozen).

### 1. Phase 5A: MC-Dropout Predictive Uncertainty
- 30 stochastic forward passes with dropout ($p=0.2$) enabled exclusively in the final linear head.
- **Uncertainty-Error Association**:
  - SBP Pearson $r$: **0.0806** ($p = 3.72 \times 10^{-46}$) | Spearman $\rho$: **0.0860** ($p = 2.55 \times 10^{-52}$)
  - DBP Pearson $r$: **0.1382** ($p = 9.78 \times 10^{-133}$) | Spearman $\rho$: **0.1336** ($p = 3.28 \times 10^{-124}$)
- Monotonic Decile Progression: SBP MAE increases from **8.72 mmHg** in the lowest uncertainty decile (D01) to **11.66 mmHg** in the highest decile (D09).

### 2. Phase 5B: Standard Split Conformal Prediction
- Marginal empirical coverage on unseen test records:
  - SBP at 90% Nominal: **91.25%** (Mean Interval Width: 51.62 mmHg)
  - SBP at 95% Nominal: **96.67%** (Mean Interval Width: 68.51 mmHg)
  - DBP at 90% Nominal: **89.72%** (Mean Interval Width: 24.12 mmHg)
  - DBP at 95% Nominal: **95.14%** (Mean Interval Width: 33.78 mmHg)

### 3. Phase 5C: Extreme-Aware Post-Hoc Calibration
- Applied isotonic regression (`isotonic_sbp.pkl`, `isotonic_dbp.pkl`) and regime-specific asymmetric quantiles:
  - Test Point MAE: SBP = **10.73 mmHg** | DBP = **5.66 mmHg**.
  - Overall Empirical Coverage: SBP = **91.8%** (at 90% nominal) | DBP = **91.5%** (at 90% nominal).
  - Alleviated extreme subgroup undercoverage in severe hypertension ($\ge 140\text{ mmHg}$).

---

## D. Real-Time Hardware Streaming Replay Results (Phase 6A & 6B)
- **Population**: Physical hardware recordings from MAX30102 + ESP32 over serial.
- **Acquisition Sampling Rate**: $99.77–99.82\text{ Hz}$ | Nominal interval: $10.02\text{ ms}$.
- **Hardware Replay Verification**:
  - Raw Samples Evaluated: 8,370 physical samples (83.88 s capture).
  - Sample Index Discontinuities: **0** (perfect continuous counter).
  - Timing Jitter: 100% of samples arrived within $9–11\text{ ms}$ host window.
  - Missing Data / NaN / Inf: **0**.
  - Streaming Causal Resampling: 10-sample chunk streaming through polyphase filter (100 Hz $\to$ 125 Hz).
  - Processing Latency: $< 1.0\text{ ms}$ host execution time per 100 ms chunk.

---

## E. Preliminary Physical Reference-Cuff Pilot Results (Phase 6C)
- **Population**: Physical live capture sessions from **4 human subjects** (`manthan`, `krish`, `nayan`, `Pankaj`).
- **Synchronized Reference**: OMRON M3 / automated oscillometric arm cuff (~120/80 mmHg nominal resting).
- **Matching Rule**: Nearest prediction within $\pm 15.0\text{ seconds}$ and window quality $\in \{\text{PASS}, \text{WARN}\}$.
- **Inclusion Summary**: **7 out of 7 recorded reference events (100.0%) successfully matched and included**.
- **Verified Window Totals**: **75 complete 10s windows formed (73 PASS, 2 WARN, 0 REJECT)**.

### Audited Matched Pair Data ($N = 7$ paired comparisons)

| Subject | Ref SBP / DBP | Calibrated Pred SBP / DBP | Time Delta ($\Delta t$) | SBP Error | DBP Error | Quality Status | Inclusion |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **manthan** | 120 / 80 mmHg | 136.9 / 87.9 mmHg | -3.43 s | **+16.9 mmHg** | **+7.9 mmHg** | PASS | Included |
| **krish** | 120 / 80 mmHg | 143.9 / 81.0 mmHg | -3.15 s | **+23.9 mmHg** | **+1.0 mmHg** | PASS | Included |
| **krish** | 120 / 80 mmHg | 140.3 / 79.6 mmHg | -3.71 s | **+20.3 mmHg** | **-0.4 mmHg** | PASS | Included |
| **nayan** | 120 / 80 mmHg | 138.6 / 81.0 mmHg | -4.59 s | **+18.6 mmHg** | **+1.0 mmHg** | PASS | Included |
| **Pankaj** | 120 / 80 mmHg | 127.8 / 78.2 mmHg | -0.22 s | **+7.8 mmHg** | **-1.8 mmHg** | PASS | Included |
| **Pankaj** | 120 / 80 mmHg | 127.5 / 78.2 mmHg | -5.25 s | **+7.5 mmHg** | **-1.8 mmHg** | PASS | Included |
| **Pankaj** | 120 / 80 mmHg | 146.4 / 81.0 mmHg | -4.79 s | **+26.4 mmHg** | **+1.0 mmHg** | PASS | Included |

### Pooled Physical Pilot Statistical Metrics

| Metric | SBP (Raw Phase 4B) | SBP (Calibrated Phase 5C) | DBP (Raw Phase 4B) | DBP (Calibrated Phase 5C) |
|:---|:---:|:---:|:---:|:---:|
| **Valid Matched Pairs ($N$)** | 7 | 7 | 7 | 7 |
| **Mean Absolute Error (MAE)** | 17.53 mmHg | **17.34 mmHg** | 2.35 mmHg | **2.15 mmHg** |
| **Root Mean Squared Error (RMSE)**| 19.21 mmHg | **18.63 mmHg** | 2.73 mmHg | **3.23 mmHg** |
| **Mean Error (Bias)** | +17.53 mmHg | **+17.34 mmHg** | -0.16 mmHg | **+0.99 mmHg** |
| **Standard Deviation of Error (SD)**| 8.48 mmHg | **7.35 mmHg** | 2.94 mmHg | **3.33 mmHg** |
| **% Within 5 mmHg** | 0.0% | **0.0%** | 100.0% | **85.7%** |
| **% Within 10 mmHg** | 28.6% | **28.6%** | 100.0% | **100.0%** |
| **% Within 15 mmHg** | 28.6% | **28.6%** | 100.0% | **100.0%** |

> [!CAUTION]
> **Scientific Wording Rule**: These physical pilot results must be presented exclusively as **preliminary exploratory observations ($N=7$)**. They demonstrate the functional execution of the end-to-end hardware-to-cuff pipeline, but **do NOT constitute population-level clinical validation or AAMI/BHS compliance certification**.
