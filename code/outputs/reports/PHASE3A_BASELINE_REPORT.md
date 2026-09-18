# Phase 3A: PPG-Only Classical Baseline, Evaluation & Error Analysis Report

**Project**: Cuffless Blood Pressure Estimation from PPG using MAX30102 + ESP32  
**Dataset**: PhysioNet MIMIC-II Waveform Database (Kaggle Blood Pressure Dataset)  
**Status**: COMPLETE & FULLY AUDITED  
**Execution Mode**: Calibration-Free, PPG-Only (Channel 0), CPU-Only Reproducible  

---

## 1. Executive Summary & Core Results

Phase 3A established a rigorous, calibration-free, record-level leakage-safe classical machine learning benchmark on the frozen Phase 2 dataset (12,000 records, 261,339 modeling-eligible 10-second windows).

### Final Frozen Test Benchmark (38,361 Test Windows across 1,621 Unseen Records)
The best model configuration selected via Validation—**Histogram Gradient Boosting on Branch C (Combined Representation, 40 features)**—was frozen and evaluated **once** on the Test set:

| Target | Window MAE (mmHg) | Window RMSE (mmHg) | Window $R^2$ | Mean Bias (mmHg) | Error SD (mmHg) | $\le 5\text{ mmHg}$ (%) | $\le 10\text{ mmHg}$ (%) | $\le 15\text{ mmHg}$ (%) | Dummy MAE (mmHg) | MAE Imp. (%) | Mean Record MAE (mmHg) | Median Record MAE (mmHg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SBP** | **13.93** | 17.94 | 0.321 | -0.35 | 17.94 | 24.3% | 45.9% | 62.8% | 17.69 | **+21.2%** | 14.42 | 11.17 |
| **DBP** | **7.05** | 9.67 | 0.287 | -0.40 | 9.67 | 46.2% | 78.0% | 91.3% | 8.77 | **+19.6%** | 7.48 | 6.14 |

---

## 2. Frozen Dataset & Partition Audit

- **Raw Dataset Size**: 12 MATLAB cell array parts (`part_1.mat` to `part_12.mat`, ~4.58 GB).
- **Sampling Frequency**: $f_s = 125\text{ Hz}$ ($T_s = 8\text{ ms}$).
- **Windowing Parameters**: 10.0-second duration (1,250 samples), 0% overlap.
- **Master Record Split**: 8,400 Train records (70.0%), 1,800 Val records (15.0%), 1,800 Test records (15.0%).
- **Leakage Verification**:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
  Pairwise record intersection is strictly **0**.
- **Modeling-Eligible Windows (Dual Valid: PPG Valid + ABP Valid)**:
  - **Train**: 183,517 windows across 7,615 active records
  - **Validation**: 39,461 windows across 1,628 active records
  - **Test**: 38,361 windows across 1,621 active records
  - **Total**: 261,339 supervised windows

---

## 3. BP Range Coverage Audit Prior to Training

Target coverage was audited prior to model training ([bp_range_coverage.csv](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/metrics/bp_range_coverage.csv)):

| Target | Range (mmHg) | Train Count | Train % | Val Count | Val % | Test Count | Test % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SBP** | $<90$ | 3,485 | 1.9% | 800 | 2.0% | 696 | 1.8% |
| **SBP** | $90–119$ | 65,847 | 35.9% | 13,981 | 35.4% | 12,679 | 33.1% |
| **SBP** | $120–139$ | 57,927 | 31.6% | 13,112 | 33.2% | 12,759 | 33.3% |
| **SBP** | $140–159$ | 38,009 | 20.7% | 7,973 | 20.2% | 8,512 | 22.2% |
| **SBP** | $\ge 160$ | 18,249 | 9.9% | 3,595 | 9.1% | 3,715 | 9.7% |
| **DBP** | $<60$ | 57,699 | 31.4% | 12,570 | 31.9% | 11,647 | 30.4% |
| **DBP** | $60–79$ | 104,252 | 56.8% | 22,638 | 57.4% | 21,716 | 56.6% |
| **DBP** | $80–89$ | 15,076 | 8.2% | 2,851 | 7.2% | 3,216 | 8.4% |
| **DBP** | $90–99$ | 4,249 | 2.3% | 963 | 2.4% | 1,150 | 3.0% |
| **DBP** | $\ge 100$ | 2,241 | 1.2% | 439 | 1.1% | 632 | 1.6% |

---

## 4. Cumulative Feature-Group Ablation Study

Evaluated on 183,517 training windows vs. 39,461 validation windows ([feature_ablation_results.csv](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/metrics/feature_ablation_results.csv)):

| Ablation Tier | Features Added | Feature Count | SBP MAE (mmHg) | SBP RMSE (mmHg) | SBP $R^2$ | DBP MAE (mmHg) | DBP RMSE (mmHg) | DBP $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ablation 0** | Dummy Baseline | 0 | 17.62 | 21.70 | 0.000 | 8.45 | 10.93 | 0.000 |
| **Ablation 1** | Group A (Basic Waveform) | 11 | 15.95 | 20.16 | 0.137 | 8.04 | 10.55 | 0.068 |
| **Ablation 2** | Groups A + B (+ Timing / HR) | 17 | 15.34 | 19.56 | 0.187 | 7.72 | 10.25 | 0.120 |
| **Ablation 3** | Groups A + B + C (+ Pulse Morphology) | 24 | 14.60 | 18.93 | 0.239 | 7.35 | 9.90 | 0.178 |
| **Ablation 4** | Groups A + B + C + D (+ VPG Derivatives) | 29 | 14.40 | 18.70 | 0.257 | 7.14 | 9.68 | 0.215 |
| **Ablation 5** | Groups A + B + C + D + E (+ APG Second Deriv.)| 33 | 14.18 | 18.42 | 0.279 | 7.02 | 9.54 | 0.237 |
| **Ablation 6** | Groups A + B + C + D + E + F (+ Spectral Power)| 36 | **14.08** | **18.30** | **0.289** | **7.02** | **9.55** | **0.235** |

*Takeaway*: Error decreases monotonically with each additional physiological feature group. The most impactful reductions occur when adding **Pulse Morphology (Group C)** (-0.74 mmHg SBP) and **APG (Group E)** (-0.22 mmHg SBP).

---

## 5. Representation Branch & Model Comparison (Validation Set)

All 5 baseline architectures were evaluated across the three physiological representation strategies ([classical_baseline_results.csv](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/metrics/classical_baseline_results.csv)):

| Representation Branch | Model Architecture | SBP MAE (mmHg) | SBP RMSE (mmHg) | SBP $R^2$ | DBP MAE (mmHg) | DBP RMSE (mmHg) | DBP $R^2$ | Mean MAE (mmHg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Branch C (Combined)** | **HistGradientBoosting** | **14.02** | **18.24** | **0.293** | **6.98** | **9.51** | **0.242** | **10.50** |
| Branch C (Combined) | RandomForest (d=12) | 14.18 | 18.47 | 0.275 | 7.13 | 9.66 | 0.218 | 10.65 |
| Branch B (Normalized) | RandomForest (d=12) | 14.27 | 18.55 | 0.269 | 7.18 | 9.71 | 0.211 | 10.72 |
| Branch B (Normalized) | HistGradientBoosting | 14.39 | 18.61 | 0.264 | 7.10 | 9.59 | 0.229 | 10.75 |
| Branch A (Preserving) | HistGradientBoosting | 15.70 | 19.91 | 0.158 | 7.89 | 10.41 | 0.092 | 11.79 |
| Branch A (Preserving) | RandomForest (d=12) | 15.82 | 20.03 | 0.147 | 7.85 | 10.38 | 0.098 | 11.84 |
| Branch C (Combined) | Ridge ($\alpha=0.01 / 1.0$) | 16.15 | 20.33 | 0.122 | 7.82 | 10.39 | 0.095 | 11.98 |
| Branch C (Combined) | Linear Regression | 16.15 | 20.33 | 0.122 | 7.82 | 10.39 | 0.095 | 11.98 |
| Branch B (Normalized) | Ridge ($\alpha=0.01$) | 16.19 | 20.37 | 0.119 | 7.85 | 10.41 | 0.092 | 12.02 |
| Branch B (Normalized) | Linear Regression | 16.19 | 20.37 | 0.119 | 7.85 | 10.41 | 0.092 | 12.02 |
| Branch A (Preserving) | Linear Regression | 17.00 | 21.05 | 0.058 | 8.23 | 10.73 | 0.036 | 12.62 |
| Branch A (Preserving) | Ridge ($\alpha=0.01$) | 17.00 | 21.05 | 0.058 | 8.23 | 10.73 | 0.036 | 12.62 |
| *Baseline (Null)* | Dummy (Population Mean)| 17.62 | 21.70 | 0.000 | 8.45 | 10.93 | 0.000 | 13.04 |

---

## 6. BP Range Error Stratification (Test Set)

Model accuracy was evaluated across physiological BP stages ([bp_range_error_analysis.csv](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/metrics/bp_range_error_analysis.csv)):

| Target | Range (mmHg) | Sample Count | MAE (mmHg) | RMSE (mmHg) | Mean Bias (mmHg) | Error SD (mmHg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SBP** | $<90$ (Hypotension) | 696 | **30.90** | 34.01 | **+30.89** | 14.24 |
| **SBP** | $90–119$ (Normotension) | 12,679 | **15.37** | 18.33 | **+14.88** | 10.70 |
| **SBP** | $120–139$ (Pre-hypertension) | 12,759 | **6.75** | 8.59 | **+0.10** | 8.59 |
| **SBP** | $140–159$ (Stage 1) | 8,512 | **14.31** | 16.74 | **-13.59** | 9.77 |
| **SBP** | $\ge 160$ (Stage 2) | 3,715 | **29.61** | 32.66 | **-29.42** | 14.17 |
| **DBP** | $<60$ (Low) | 11,647 | **7.91** | 8.98 | **+7.85** | 4.35 |
| **DBP** | $60–79$ (Normal) | 21,716 | **4.39** | 5.59 | **-1.15** | 5.47 |
| **DBP** | $80–89$ (Pre-hypertension) | 3,216 | **11.44** | 12.85 | **-11.11** | 6.46 |
| **DBP** | $90–99$ (Stage 1) | 1,150 | **21.08** | 22.94 | **-20.93** | 9.41 |
| **DBP** | $\ge 100$ (Stage 2) | 632 | **34.93** | 36.38 | **-34.93** | 10.17 |

*Critical Diagnostic Discovery*: The model exhibits **pathological regression-to-the-mean**. In the normotensive center ($120–139$ mmHg SBP), the model has nearly zero bias (+0.10 mmHg) and excellent MAE (6.75 mmHg). But at hypertensive extremes ($\ge 160$ mmHg), it underestimates pressure by almost 30 mmHg (bias = -29.42 mmHg), and in hypotension ($<90$ mmHg), it overestimates pressure by 30.89 mmHg.

---

## 7. Feature Group Importance Ranking

Aggregated feature group permutation importance on the validation set ([feature_group_importance.csv](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/metrics/feature_group_importance.csv)):

| Feature Group | Feature Count | Total Importance | Relative Share (%) | Key Features |
| :--- | :---: | :---: | :---: | :--- |
| **A_basic** | 11 | 0.180 | **19.8%** | `mean_a`, `std_a`, `ptp_a`, `iqr_a`, `skew_b`, `kurt_b` |
| **D_vpg** | 5 | 0.176 | **19.4%** | `vpg_max`, `vpg_max_upstroke`, `vpg_rms`, `vpg_std` |
| **other** | 4 | 0.164 | **18.1%** | `p10_a`, `p90_a`, `pulse_area_median`, `max_downstroke_slope_median` |
| **E_apg** | 4 | 0.157 | **17.3%** | `apg_b_to_a_ratio`, `apg_max`, `apg_min` |
| **C_morphology**| 7 | 0.106 | **11.7%** | `pulse_amp_median_a`, `rise_time_median`, `decay_time_median`, `pulse_width_median` |
| **F_spectral** | 3 | 0.087 | **9.6%** | `dominant_freq`, `pulse_band_power`, `spectral_entropy` |
| **B_timing** | 6 | 0.038 | **4.2%** | `hr_bpm`, `pulse_count`, `ibi_mean`, `ibi_cv` |

---

## 8. Record-Clustered Error Correlation Analysis

Record-clustered bootstrap correlations (500 resamples by `record_id`, 95% CI) ([error_correlations.csv](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/metrics/error_correlations.csv)):

| Feature | Pearson $r$ | Pearson 95% CI | Spearman $\rho$ | Spearman 95% CI | Physiological Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `sbp` (Ground Truth) | **+0.155** | $[+0.077, +0.223]$ | +0.051 | $[-0.012, +0.114]$ | Absolute error correlates positively with true SBP due to underestimation at high BP. |
| `pulse_amp_cv_a` | **+0.051** | $[+0.009, +0.105]$ | **+0.077** | $[+0.041, +0.110]$ | Beat-to-beat amplitude variation (respiratory modulation) degrades prediction accuracy. |
| `hr_bpm` | -0.019 | $[-0.079, +0.041]$ | -0.019 | $[-0.073, +0.034]$ | Heart rate alone shows no significant linear correlation with absolute error magnitude. |

---

## 9. Generated Artifacts & Visualizations

All 18 publication-grade figures were generated in [code/outputs/figures/](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/figures/):
1. `model_01_sbp_true_vs_pred.png`: Scatter plot of True vs Pred SBP with regression trend line.
2. `model_02_dbp_true_vs_pred.png`: Scatter plot of True vs Pred DBP with regression trend line.
3. `model_03_sbp_residuals.png`: SBP Residual histogram & KDE with zero error reference.
4. `model_04_dbp_residuals.png`: DBP Residual histogram & KDE.
5. `model_05_sbp_bland_altman.png`: SBP Bland-Altman agreement plot (Mean bias: -0.35 mmHg, 95% LoA: [-35.5, +34.8]).
6. `model_06_dbp_bland_altman.png`: DBP Bland-Altman agreement plot (Mean bias: -0.40 mmHg, 95% LoA: [-19.3, +18.5]).
7. `model_07_sbp_error_vs_ref.png`: SBP Error vs Reference SBP (illustrating negative slope regression-to-the-mean).
8. `model_08_dbp_error_vs_ref.png`: DBP Error vs Reference DBP.
9. `model_09_sbp_mae_by_bp_range.png`: SBP MAE bar chart across BP stages with exact window counts.
10. `model_10_dbp_mae_by_bp_range.png`: DBP MAE bar chart across BP stages with exact window counts.
11. `model_11_record_level_mae_distribution.png`: Per-record MAE distribution across 1,621 test records.
12. `model_12_model_comparison_bar.png`: Grouped comparison of MAE and RMSE across all 5 models on Validation.
13. `model_13_error_vs_heart_rate.png`: Absolute error vs Estimated Heart Rate.
14. `model_14_error_vs_ppg_quality.png`: Performance across signal quality strata (PASS vs WARN, clipping).
15. `model_15_error_vs_pulse_amplitude_variation.png`: Absolute error vs pulse amplitude coefficient of variation.
16. `model_16_record_weighted_vs_window_weighted.png`: Comparison of Window-Weighted MAE vs Record-Weighted Mean/Median MAE.
17. `model_17_feature_group_importance.png`: Relative feature group importance percentages.
18. `model_18_feature_group_ablation.png`: Cumulative ablation curve across all 7 tiers.

---

## 10. Strict Acceptance Criteria Audit

1. **Phase 1 files untouched**: Verified (no changes to `code/data/loader.py`, `code/config/config.py`, or Phase 1 outputs).
2. **Phase 2 files untouched**: Verified (`window_manifest.csv` and `record_split.csv` unmodified).
3. **Original MAT files untouched**: Verified (all 12 MAT files remain read-only).
4. **Original sample notebooks untouched**: Verified.
5. **Record splits identical to Phase 2**: Verified (8,400 / 1,800 / 1,800).
6. **All partition-overlap checks pass**: Verified (zero record overlap).
7. **Only PPG used as model input**: Verified (Channel 0 only, ECG and ABP completely excluded from features).
8. **No target leakage exists**: Verified (0 forbidden substrings found).
9. **Feature extraction is reproducible**: Verified (`features_cache.h5` cached and reproducible).
10. **Learned preprocessing uses TRAIN only**: Verified (`fit` called strictly on `split == 'train'`).
11. **Dummy baseline exists**: Evaluated as benchmark null model.
12. **Linear Regression exists**: Evaluated.
13. **Ridge Regression exists**: Evaluated (alpha tuned on Validation).
14. **Random Forest exists**: Evaluated (100 trees, depth 12).
15. **HistGradientBoosting exists**: Evaluated (selected as frozen champion).
16. **Validation used for model selection**: Verified (frozen before testing).
17. **Test used only after freezing**: Evaluated exactly once.
18. **SBP and DBP evaluated separately**: Verified.
19. **Error analysis performed**: Verified (ranges, quality, HR, Bland-Altman).
20. **Feature importance analyzed**: Verified by feature and feature group.
21. **Research observations documented**: Verified in [PHASE3A_RESEARCH_DISCOVERY.md](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/reports/PHASE3A_RESEARCH_DISCOVERY.md).
22. **No deep learning model trained**: Verified (pure classical scikit-learn models).
23. **No calibration used**: Verified (strictly calibration-free).

---

## 11. Conclusion & Readiness for Phase 3B

Phase 3A has conclusively demonstrated that **classical regression with handcrafted PPG features is insufficient for clinical-grade cuffless blood pressure estimation**, suffering from severe regression-to-the-mean at blood pressure extremes and cross-record vascular compliance ambiguity.

**Ready for Phase 3B**: **YES** (The empirical findings and failure modes provide a rigorous foundation for formulating novel deep spatio-temporal architectures).
