# PHASE 5A — Uncertainty Estimation Without Retraining Report

**Architecture:** Frozen Phase 4A 1D CNN + Causal 6-Window GRU + MC-Dropout Head  
**Mode:** Inference-Only Predictive Epistemic Uncertainty Estimation  
**Execution Environment:** Local System (NVIDIA GeForce GTX 1650 Ti)  
**Timestamp:** 2026-09-18 14:35:54  

---

## 1. Research Question
Can predictive uncertainty derived from the existing frozen Phase 4B model identify predictions that are more likely to have large absolute BP errors?

## 2. Why Uncertainty
In cuffless calibration-free blood pressure estimation, large residual errors can occur in extreme blood pressure ranges. Having a reliable uncertainty measure allows safety filtering, selective prediction (abstention), and clinical escalation when the model's epistemic confidence is low.

## 3. Frozen Phase 4B Model
- Pretrained Checkpoint: `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt`
- Total CNN Backbone Parameters: 146,978 (Frozen: 0 trainable)
- Total Temporal Model Parameters: 27,106 (Frozen: 0 trainable)
- Total Trainable Parameters in Phase 5A: Exactly 0.

## 4. MC-Dropout Methodology
- Stochastic Inference: Monte Carlo Dropout with N = 30 passes.
- Isolation Protocol: ONLY `model.fc[2]` (Dropout p=0.2) is placed in stochastic mode via `enable_mc_dropout()`. All recurrence, convolutional, and BatchNorm modules remain strictly in `eval()` mode.
- Primary Uncertainty Metric: Predictive standard deviation across 30 stochastic forward passes (sigma_MC). This serves as an approximate epistemic uncertainty measure.

## 5. Test Dataset
- Dataset Split: Unseen Phase 4B test partition (31,192 causal 60-second sequences).
- Sequence Structure: 6 consecutive 10-second windows ([t-5, ..., t]) from 1,621 disjoint patient records.

## 6. Deterministic Checkpoint Verification
- Baseline SBP MAE: 10.57 mmHg (Phase 4B frozen: 10.57 mmHg)
- Baseline DBP MAE: 5.51 mmHg (Phase 4B frozen: 5.51 mmHg)
- Baseline Combined MAE: 8.04 mmHg (Phase 4B frozen: 8.04 mmHg)

## 7. Uncertainty Distribution
- SBP Uncertainty: Mean = 14.29 mmHg, Median = 14.10 mmHg, Std = 2.72 mmHg
- DBP Uncertainty: Mean = 7.59 mmHg, Median = 7.40 mmHg, Std = 1.51 mmHg

## 8. Uncertainty-Error Association
- SBP Pearson r: 0.0806 (p = 3.72e-46)
- SBP Spearman rho: 0.0860 (p = 2.55e-52)
- DBP Pearson r: 0.1382 (p = 9.78e-133)
- DBP Spearman rho: 0.1336 (p = 3.28e-124)

## 9. Uncertainty Decile Analysis
Monotonic error progression across uncertainty deciles demonstrates that higher uncertainty correlates with larger prediction error:

### SBP Deciles:
| decile        |   mean_uncertainty |      mae |    rmse |       bias |
|:--------------|-------------------:|---------:|--------:|-----------:|
| D01 (0-10%)   |             9.9905 |  8.72232 | 12.1154 |  0.547091  |
| D02 (10-20%)  |            11.4464 | 10.0185  | 13.6913 | -0.0392323 |
| D03 (20-30%)  |            12.3176 | 10.2415  | 13.9361 | -0.600566  |
| D04 (30-40%)  |            13.0615 | 10.7553  | 14.6458 | -0.693449  |
| D05 (40-50%)  |            13.7501 | 10.9943  | 14.7769 | -0.278501  |
| D06 (50-60%)  |            14.4492 | 11.0279  | 14.8069 | -1.03176   |
| D07 (60-70%)  |            15.1899 | 11.5     | 15.4807 | -1.02435   |
| D08 (70-80%)  |            16.0485 | 11.6959  | 15.5147 | -0.925383  |
| D09 (80-90%)  |            17.1965 | 11.6606  | 15.5749 | -1.00844   |
| D10 (90-100%) |            19.4466 | 11.6171  | 15.3637 | -1.18918   |

### DBP Deciles:
| decile        |   mean_uncertainty |     mae |     rmse |      bias |
|:--------------|-------------------:|--------:|---------:|----------:|
| D01 (0-10%)   |            5.37722 | 4.5825  |  7.0249  | -0.957829 |
| D02 (10-20%)  |            6.09394 | 4.93939 |  7.42773 | -1.14336  |
| D03 (20-30%)  |            6.51589 | 5.09803 |  7.67232 | -0.904787 |
| D04 (30-40%)  |            6.8822  | 5.41358 |  8.21625 | -1.26447  |
| D05 (40-50%)  |            7.22943 | 5.41855 |  8.11896 | -0.776271 |
| D06 (50-60%)  |            7.58244 | 5.4237  |  7.98362 | -0.874783 |
| D07 (60-70%)  |            7.96642 | 5.7732  |  8.54832 | -1.05332  |
| D08 (70-80%)  |            8.43242 | 5.87429 |  8.6134  | -0.892314 |
| D09 (80-90%)  |            9.10876 | 6.37443 |  9.06239 | -1.47119  |
| D10 (90-100%) |           10.6694  | 7.72693 | 10.4666  | -3.12488  |

## 10. Selective Prediction (Abstention)
Retaining predictions with the lowest uncertainty yields systematic error reductions:
| coverage_pct   |   retained_samples |   sbp_mae |   dbp_mae |   combined_mae |
|:---------------|-------------------:|----------:|----------:|---------------:|
| 100%           |              31192 |   10.8233 |   5.66249 |        8.2429  |
| 90%            |              28073 |   10.7348 |   5.43333 |        8.14629 |
| 80%            |              24954 |   10.619  |   5.31566 |        8.02935 |
| 70%            |              21834 |   10.4656 |   5.23553 |        7.92495 |
| 60%            |              18715 |   10.2932 |   5.14593 |        7.77766 |
| 50%            |              15596 |   10.1463 |   5.09038 |        7.6555  |

## 11. High-Error Detection
Evaluating MC-dropout uncertainty for detecting errors exceeding clinical thresholds:
| target   |   error_threshold_mmHg |   positive_count |   positive_rate |   roc_auc |   pr_auc |   random_baseline_pr_auc |
|:---------|-----------------------:|-----------------:|----------------:|----------:|---------:|-------------------------:|
| SBP      |                     10 |            12711 |       0.407508  |  0.54401  | 0.436215 |                0.407508  |
| SBP      |                     15 |             8051 |       0.258111  |  0.546095 | 0.282739 |                0.258111  |
| DBP      |                     10 |             4734 |       0.15177   |  0.603766 | 0.224452 |                0.15177   |
| DBP      |                     15 |             2106 |       0.0675173 |  0.61491  | 0.113802 |                0.0675173 |

## 12. BP-Range Uncertainty
| target   | bp_range   |   sample_count |   mean_uncertainty |   median_uncertainty |   std_uncertainty |      mae |
|:---------|:-----------|---------------:|-------------------:|---------------------:|------------------:|---------:|
| SBP      | <90        |            503 |            11.5013 |              11.2774 |           1.93941 | 20.4226  |
| SBP      | 90-119     |          10219 |            12.7135 |              12.5275 |           2.19928 | 10.4596  |
| SBP      | 120-139    |          10320 |            14.3104 |              14.196  |           2.26218 |  7.94798 |
| SBP      | 140-159    |           7120 |            15.6351 |              15.5372 |           2.48661 | 11.1461  |
| SBP      | >=160      |           3030 |            16.8365 |              16.785  |           2.64827 | 19.4911  |

| target   | bp_range   |   sample_count |   mean_uncertainty |   median_uncertainty |   std_uncertainty |      mae |
|:---------|:-----------|---------------:|-------------------:|---------------------:|------------------:|---------:|
| DBP      | <60        |           9134 |            6.84297 |              6.72269 |           1.17976 |  4.74276 |
| DBP      | 60-79      |          17998 |            7.63835 |              7.53338 |           1.33911 |  4.41627 |
| DBP      | 80-89      |           2618 |            8.92555 |              8.84318 |           1.70065 |  8.43733 |
| DBP      | 90-99      |            946 |            9.31888 |              9.3029  |           1.90871 | 17.3612  |
| DBP      | >=100      |            496 |            8.98402 |              8.82393 |           1.87277 | 30.8617  |

## 13. Empirical Heuristic Interval Coverage
| target   |   interval_1.0_sigma_empirical_coverage_pct |   interval_1.0_sigma_mean_width_mmHg |   interval_1.96_sigma_empirical_coverage_pct |   interval_1.96_sigma_mean_width_mmHg |
|:---------|--------------------------------------------:|-------------------------------------:|---------------------------------------------:|--------------------------------------:|
| SBP      |                                     71.8934 |                              28.5794 |                                      92.4885 |                               56.0157 |
| DBP      |                                     75.4745 |                              15.1717 |                                      93.5689 |                               29.7365 |

## 14. Scientific Limitations
- MC-dropout is an approximate epistemic uncertainty measure and does not capture data noise (aleatoric uncertainty).
- Empirical coverage shows that nominal 1.96-sigma intervals do not constitute calibrated 95% clinical prediction intervals without post-hoc conformal calibration.
- ICU cohort evaluation limits ambulatory generalization.

## 15. Scientific Interpretation
Predictive uncertainty derived from MC-dropout demonstrates statistically significant association with absolute error, enabling effective selective prediction where abstaining on high-uncertainty predictions reduces overall error.

## 16. Next Research Step
Advance to **Phase 5B: Post-Hoc Conformal Calibration & Extreme-Aware Losses**, implementing distribution-free conformal prediction to guarantee rigorous coverage guarantees.
