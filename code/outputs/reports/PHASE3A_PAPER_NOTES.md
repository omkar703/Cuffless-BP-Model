# Phase 3A: Publication Notes — Calibration-Free Cuffless BP Estimation from PPG

## 1. Research Question

Can handcrafted morphological, derivative, and spectral features extracted from a single-channel 10-second photoplethysmogram (PPG) accurately predict continuous systolic blood pressure (SBP) and diastolic blood pressure (DBP) in a strictly **calibration-free**, **subject-independent** setting? Furthermore, what are the exact physiological and mathematical failure modes of classical machine-learning models under cross-record distribution shift?

---

## 2. Dataset & Cohort Characterization

- **Source**: PhysioNet MIMIC-II Waveform Database (via Kachuee et al. Kaggle Blood Pressure Dataset).
- **Sampling Frequency**: $f_s = 125\text{ Hz}$ ($T_s = 8\text{ ms}$).
- **Total Validated Records**: 12,000 continuous multi-parameter recordings (PPG Channel 0, ABP Channel 1, ECG Channel 2).
- **Windowing Protocol**: 10.0-second non-overlapping windows (1,250 samples/window).
- **Modeling-Eligible Windows**: 261,339 supervised windows meeting strict dual-channel quality criteria (PPG valid and ABP valid with detectable, physiological cardiac cycles: SBP $\in [50, 240]\text{ mmHg}$, DBP $\in [30, 140]\text{ mmHg}$, pulse pressure $\ge 15\text{ mmHg}$).
- **Partitioning**:
  - **Train**: 8,400 records (70.0%), 183,517 eligible windows (70.2%).
  - **Validation**: 1,800 records (15.0%), 39,461 eligible windows (15.1%).
  - **Test**: 1,800 records (15.0%), 38,361 eligible windows (14.7%).

---

## 3. Leakage-Control Protocol

1. **Partition Granularity**: Split strictly at the `record_id` level. Zero windows from the same subject/record exist across training, validation, or test sets:
   $$\text{Records}_{\text{train}} \cap \text{Records}_{\text{val}} = \emptyset, \quad \text{Records}_{\text{train}} \cap \text{Records}_{\text{test}} = \emptyset, \quad \text{Records}_{\text{val}} \cap \text{Records}_{\text{test}} = \emptyset$$
2. **Preprocessing Fitting**:
   All imputers (`SimpleImputer(strategy='median')`) and normalizers (`StandardScaler`) are fit exclusively on the training partition ($X_{\text{train}}$).
3. **Target & Channel Isolation**:
   - Channel 0 (PPG) is the sole input source.
   - Channel 2 (ECG) is strictly excluded (ensuring pure cuffless PPG operation without pulse transit time dependency).
   - Ground truth targets (SBP, DBP, MAP) are computed from invasive ABP beats and strictly excluded from feature space. Automated assertions guarantee zero forbidden substrings (`abp`, `ecg`, `sbp`, `dbp`, `map`, `ptt`, `pat`, `calibration`).
4. **Calibration-Free Operation**:
   No subject calibration offsets, initial baseline blood pressure values, or delta-BP predictions are utilized.

---

## 4. PPG Representation Branches

To resolve the physiological scale vs. morphological invariance trade-off, three explicit representation branches were engineered:
1. **Branch A (Amplitude-Preserving)**: Filtered PPG ($0.5–8.0\text{ Hz}$ Butterworth filtfilt, 14 features). Retains raw optical attenuation amplitudes, peak-to-peak voltages, and pulsatile dispersion.
2. **Branch B (Normalized Morphology)**: Per-window Z-score normalized PPG ($26\text{ features}$). Standardizes amplitude to zero mean and unit variance, isolating pulse geometry (rise/decay times, slopes, area), pulse rate/timing, VPG (first derivative), APG (second derivative), and spectral entropy.
3. **Branch C (Combined Representation)**: Fusion of Branch A and Branch B ($40\text{ features}$), directly evaluating whether optical volume scale and geometric pulse morphology provide synergistic predictive value.

---

## 5. Classical Baselines

Five classical regression architectures were trained separately for SBP and DBP:
- **Model 0 (Dummy Baseline)**: Mean target regressor ($\hat{y} = \bar{y}_{\text{train}}$), serving as the empirical null hypothesis.
- **Model 1 (Linear Regression)**: Ordinary least squares.
- **Model 2 (Ridge Regression)**: $L_2$-regularized linear regression with validation-selected penalty ($\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]$).
- **Model 3 (Random Forest)**: 100 trees, maximum tree depth 12.
- **Model 4 (Histogram Gradient Boosting)**: Gradient boosted regression trees (maximum 150 iterations, maximum tree depth 8).

---

## 6. Feature-Group Cumulative Ablation

Ablations were systematically evaluated on identical splits:
- **Ablation 0**: Dummy Baseline (0 features)
- **Ablation 1**: Group A — Basic Waveform Statistics (11 features)
- **Ablation 2**: Groups A + B — Basic Statistics + Pulse Timing / Heart Rate (17 features)
- **Ablation 3**: Groups A + B + C — Added Pulse Morphology (Rise/Decay, Upstroke Slopes) (24 features)
- **Ablation 4**: Groups A + B + C + D — Added Velocity PPG (VPG) (29 features)
- **Ablation 5**: Groups A + B + C + D + E — Added Acceleration PPG (APG) (33 features)
- **Ablation 6**: Groups A + B + C + D + E + F — Added Spectral Entropy and Band Power (36 features)

---

## 7. Error Analysis Framework

1. **BP-Range Stratification**: Evaluates performance across clinical stages:
   - SBP: Hypotension ($<90$), Normotension ($90–119$), Pre-hypertension ($120–139$), Stage 1 ($140–159$), Stage 2 ($\ge 160\text{ mmHg}$).
   - DBP: Low ($<60$), Normal ($60–79$), Pre-hypertension ($80–89$), Stage 1 ($90–99$), Stage 2 ($\ge 100\text{ mmHg}$).
2. **Dual Weighting Analysis**:
   - Window-weighted: Equal weighting per 10-second window.
   - Record-weighted: Aggregates mean/median error per individual subject recording to measure genuine cross-subject generalization.
3. **Clustered Error Correlation**:
   Evaluates correlation between absolute errors and physiological parameters (heart rate, amplitude variation, baseline BP) with 95% confidence intervals derived from record-clustered bootstrap.

---

## 8. Main Findings (Empirical Summary)

1. **Information Content in Single-Channel PPG**:
   Handcrafted PPG features explain a meaningful portion of DBP variance ($R^2 \approx 0.60–0.68$), reducing MAE significantly below population mean. However, SBP prediction exhibits substantially higher variance and persistent bias at hypertensive extremes.
2. **Nonlinear Superiority**:
   Ensemble tree models (Random Forest and Histogram Gradient Boosting) consistently outperform linear and Ridge regression, capturing essential nonlinear interactions between pulse amplitude, upstroke velocity, and heart rate.
3. **Morphology and APG Contribution**:
   Feature group ablation demonstrates that pulse morphology (rise time, decay time) and second-derivative acceleration features (APG $b/a$ ratio) provide the largest marginal error reductions over basic waveform statistics.
4. **Regression-to-the-Mean at Blood Pressure Extremes**:
   Residual analysis reveals severe systematic underestimation in hypertensive ranges (SBP $\ge 140\text{ mmHg}$) and overestimation in hypotensive ranges (SBP $< 90\text{ mmHg}$).

---

## 9. Observed Limitations

1. **Absence of Calibrated Arterial Compliance Scale**:
   Without subject-specific calibration (e.g., vessel diameter, baseline stiffness), identical optical pulse contours from different subjects correspond to drastically different absolute pressures.
2. **Window Independence Assumption**:
   Evaluating 10-second windows independently discards vital long-term physiological context (slow autonomic drift, vasomotor oscillations).
3. **Extreme Blood Pressure Scarcity**:
   Hypotensive ($<90\text{ mmHg}$, $\approx 1.9\%$) and severe hypertensive ($\ge 160\text{ mmHg}$, $\approx 9.8\%$) ranges are heavily underrepresented relative to normotensive windows, biasing objective functions toward population means.

---

## 10. Research Hypothesis

> **Central Hypothesis**:
> The primary bottleneck of calibration-free cuffless BP estimation from PPG is not simple non-linearity, but **unresolved subject-level vascular compliance ambiguity** and **lack of multi-scale temporal context**. An architecture that jointly learns fine-grained morphological wave dynamics alongside multi-window temporal evolution will achieve substantially better generalization across hypertensive and normotensive subjects than static classical baselines.

---

## 11. Candidate Future Contribution

> [!IMPORTANT]
> **HYPOTHESIS — NOT YET VALIDATED**
> A dual-branch deep spatio-temporal architecture comprising:
> 1. A multi-scale 1D dilated convolutional neural network (CNN) or multi-resolution Wavelet block to capture local systolic/diastolic morphology, dicrotic notch characteristics, and derivative dynamics directly from raw PPG waveforms without handcrafted feature loss.
> 2. A recurrent or temporal attention mechanism (Bi-LSTM / Temporal Convolutional Network / Transformer encoder) operating across contiguous multi-window sequences to model patient-specific autonomic cardiovascular dynamics and vascular compliance drift over time.
> 
> *Validation and benchmarking of this proposed contribution will be conducted in subsequent research phases (Phase 3B).*
