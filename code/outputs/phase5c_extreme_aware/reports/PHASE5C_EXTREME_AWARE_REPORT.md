# PHASE 5C — Extreme-BP-Aware Post-Hoc Conformal Calibration Report

**Architecture:** Frozen Phase 4A 1D CNN + Causal 6-Window GRU + Post-Hoc Isotonic Calibration + Asymmetric Split Conformal Prediction  
**Mode:** Post-Hoc Inference-Only  
**Execution System:** Local Linux (NVIDIA GeForce GTX 1650 Ti)  
**Timestamp:** 2026-09-18 15:41:31  

---

## 1. Research Question
Does a post-hoc calibration procedure combining monotonic point-prediction calibration (Isotonic Regression) and prediction-regime-adaptive asymmetric split conformal prediction improve empirical coverage in extreme blood pressure regimes without producing impractically wide intervals everywhere?

## 2. Motivation from Phase 4B
Phase 4B established a strong temporal neural point baseline (SBP MAE = 10.57 mmHg, DBP MAE = 5.51 mmHg), but exhibited systematic regression-to-the-mean shrinkage where higher blood pressures are under-predicted and lower blood pressures are over-predicted.

## 3. Motivation from Phase 5A
Phase 5A demonstrated that MC-dropout predictive uncertainty correlates positively with absolute error ($p < 10^{-45}$), but the correlation is modest ($r \approx 0.08$ for SBP, $r \approx 0.14$ for DBP), indicating that epistemic dispersion alone does not fully explain directional bias.

## 4. Motivation from Phase 5B
Phase 5B confirmed that standard and uncertainty-scaled split conformal prediction satisfy marginal coverage ($\ge 90\%$ and $\ge 95\%$) over the full test distribution. However, Phase 5B exposed severe subgroup under-coverage in extreme ranges (e.g., SBP $\ge 160$ coverage dropped to 85.25%, DBP $\ge 100$ dropped to 9.48% under Method A).

## 5. Frozen Model Integrity & Zero Retraining Rule
- **Source Checkpoint:** `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt`
- **Backbone CNN Parameters:** 146,978 (Frozen: 0 trainable)
- **Temporal GRU Parameters:** 27,106 (Frozen: 0 trainable)
- **Total Trainable Parameters:** Exactly 0 (`requires_grad = False` across all layers). No optimizer was instantiated, and no neural weights were modified.

## 6. Strict Data Separation & Test Record Discrepancy Audit
- **Calibration Subset (607 records, 16,298 sequences):** Used exclusively to fit isotonic point calibration mappings.
- **Audit Subset (608 records, 15,865 sequences):** Used exclusively as the conformal calibration population to compute asymmetric nonconformity scores and conformal quantiles.
- **Test Partition (1,198 records, 31,192 sequences):** Strictly untouched until final evaluation.
- **Disjointness:** Zero patient record overlap verified across calibration, audit, and test sets.
- **Record Discrepancy Reconciliation:** Phase 4B window embeddings contained 1,621 active eligible records. However, 423 records contained $< 6$ contiguous windows and could not form 60-s sequences. Exactly 1,198 records contained $\ge 6$ windows, contributing all 31,192 complete test sequences. Both counts are verified and documented.

## 7. Post-Hoc Isotonic Calibration (Component A)
Fitted on the 16,298 calibration samples:
- `mappings/isotonic_sbp.pkl`
- `mappings/isotonic_dbp.pkl`

On the Audit set:
| target   | model               |      mae |     rmse |      bias |       r2 |
|:---------|:--------------------|---------:|---------:|----------:|---------:|
| SBP      | Raw Phase 4B        | 10.6698  | 14.5676  | -0.220982 | 0.528672 |
| SBP      | Phase 5C Calibrated | 10.7463  | 14.6143  | -0.165949 | 0.525644 |
| DBP      | Raw Phase 4B        |  5.05858 |  7.54181 | -0.802693 | 0.491145 |
| DBP      | Phase 5C Calibrated |  5.15815 |  7.50498 |  0.218607 | 0.496102 |

On the Test set:
| target   | model               |      mae |     rmse |      bias |       r2 |
|:---------|:--------------------|---------:|---------:|----------:|---------:|
| SBP      | Raw Phase 4B        | 10.5662  | 14.3959  | -0.631435 | 0.557363 |
| SBP      | Phase 5C Calibrated | 10.7312  | 14.5139  | -0.560938 | 0.550077 |
| DBP      | Raw Phase 4B        |  5.51458 |  8.2485  | -1.25041  | 0.472153 |
| DBP      | Phase 5C Calibrated |  5.6636  |  8.23206 | -0.207019 | 0.474256 |

## 8. Prediction-Regime Bins & Deterministic Merges (Component B)
Candidate bins were assessed on the Audit population:
- SBP: `<120` (4,289), `120-139` (7,661), `140-159` (3,423), `>=160` (492).
  Since `>=160` had $< 500$ samples, it was deterministically merged with adjacent `140-159` to form **`>=140`** (3,915 samples).
- DBP: `<60` (2,317), `60-79` (12,517), `80-89` (883), `>=90` (148).
  Since `>=90` had $< 500$ samples, it was deterministically merged with adjacent `80-89` to form **`>=80`** (1,031 samples).

## 9. Asymmetric Conformal Quantiles
Estimated on the Audit population with symmetric tail allocation (alpha_tail = 0.025 for 95%, 0.05 for 90%):
- **SBP Bins (95%):**
  - `<120`: $q_{lower} = 24.25$, $q_{upper} = 31.36$ (Width: 55.61 mmHg)
  - `120-139`: $q_{lower} = 29.88$, $q_{upper} = 32.00$ (Width: 61.88 mmHg)
  - `>=140`: $q_{lower} = 35.54$, $q_{upper} = 31.33$ (Width: 66.87 mmHg)
- **DBP Bins (95%):**
  - `<60`: $q_{lower} = 6.38$, $q_{upper} = 10.06$ (Width: 16.44 mmHg)
  - `60-79`: $q_{lower} = 13.51$, $q_{upper} = 18.97$ (Width: 32.47 mmHg)
  - `>=80`: $q_{lower} = 14.23$, $q_{upper} = 21.06$ (Width: 35.29 mmHg)

## 10. Conformal Calibration / Audit Population Results
| target   |   nominal |   empirical_coverage |   mean_width_mmHg |   median_width_mmHg |
|:---------|----------:|---------------------:|------------------:|--------------------:|
| SBP      |        90 |              90.0536 |           49.2891 |             49.8267 |
| DBP      |        90 |              90.0662 |           21.7632 |             22.9483 |
| SBP      |        95 |              95.0583 |           61.418  |             61.8819 |
| DBP      |        95 |              95.0709 |           30.3153 |             32.4734 |

## 11. Frozen-Test Set Results (31,192 Sequences)
| method                           | target   |   nominal_coverage |   empirical_coverage |   signed_coverage_error |   absolute_coverage_error |   mean_width_mmHg |   median_width_mmHg |   p90_width_mmHg |   lower_miss_rate_pct |   upper_miss_rate_pct |   efficiency_ratio |
|:---------------------------------|:---------|-------------------:|---------------------:|------------------------:|--------------------------:|------------------:|--------------------:|-----------------:|----------------------:|----------------------:|-------------------:|
| Phase 5C Extreme-Aware Conformal | SBP      |                 90 |              89.8628 |             -0.137215   |                0.137215   |           49.3018 |             49.8267 |          57.274  |               4.48192 |               5.6553  |           0.548634 |
| Phase 5C Extreme-Aware Conformal | SBP      |                 95 |              95.0019 |              0.00192357 |                0.00192357 |           61.4244 |             61.8819 |          66.8739 |               2.16722 |               2.83085 |           0.64656  |
| Phase 5C Extreme-Aware Conformal | DBP      |                 90 |              87.0351 |             -2.96486    |                2.96486    |           21.8545 |             22.9483 |          22.9483 |               6.07528 |               6.88959 |           0.2511   |
| Phase 5C Extreme-Aware Conformal | DBP      |                 95 |              93.2226 |             -1.77738    |                1.77738    |           30.4257 |             32.4734 |          32.4734 |               3.25404 |               3.52334 |           0.326376 |

## 12. Extreme-Target Subgroup Focus Comparison (Nominal 95%)
| extreme_subgroup              |   sample_count |   method_a_cov_95 |   method_b_cov_95 |   phase5c_cov_95 |   phase5c_mean_width_95 |
|:------------------------------|---------------:|------------------:|------------------:|-----------------:|------------------------:|
| SBP < 90 mmHg (Hypotension)   |            503 |          89.2644  |           81.3121 |          64.2147 |                 56.4095 |
| SBP >= 160 mmHg (Stage 2 HTN) |           3030 |          85.2475  |           87.3927 |          79.67   |                 65.9401 |
| DBP >= 90 mmHg (Hypertension) |           1442 |          37.9334  |           50.4854 |          45.0069 |                 33.4021 |
| DBP >= 100 mmHg (Stage 2 HTN) |            496 |           9.47581 |           22.7823 |          23.5887 |                 33.6865 |

## 13. Comparison with Phase 5B Baselines
- **Overall SBP 95% Coverage:** Method A = 96.67% (Width: 68.51 mmHg), Method B = 96.29% (Width: 68.01 mmHg), Phase 5C = 95.00% (Mean Width: 61.42 mmHg).
- **Overall DBP 95% Coverage:** Method A = 95.14% (Width: 33.78 mmHg), Method B = 95.48% (Width: 34.17 mmHg), Phase 5C = 93.22% (Mean Width: 30.43 mmHg).
- **Extreme SBP $\ge 160$ Coverage:** Method A = 85.25%, Method B = 87.39%, Phase 5C = 79.67%.
- **Extreme DBP $\ge 100$ Coverage:** Method A = 9.48%, Method B = 22.78%, Phase 5C = 23.59%.

## 14. Interval Width vs Error Association
| target   | method              |   nominal_coverage |   pearson_r |    pearson_p |   spearman_rho |   spearman_p |
|:---------|:--------------------|-------------------:|------------:|-------------:|---------------:|-------------:|
| SBP      | Phase 5C Asymmetric |                 95 |    0.13169  | 1.1183e-120  |       0.125791 | 3.3719e-110  |
| DBP      | Phase 5C Asymmetric |                 95 |    0.129551 | 8.03872e-117 |       0.13737  | 3.07697e-131 |

## 15. Point-Prediction Effect
Post-hoc isotonic calibration provides modest bias adjustment on the calibration distribution, but because it is a monotonic scalar transformation, it cannot fundamentally recover features or variance lost during neural compression. Point prediction error metrics remain broadly comparable to the frozen Phase 4B baseline.

## 16. Scientific Limitations
- Prediction-regime-adaptive split conformal prediction provides marginal coverage under exchangeability; it does not provide formal conditional coverage across all target strata.
- Regimes are defined on the predicted BP, not the true BP, because true BP is unavailable at test time.
- Extreme hypertensive ranges remain challenging due to point estimator shrinkage.
- ICU data dynamics may differ from ambulatory populations.

## 17. Interpretation & Deployment
In wearable firmware, post-hoc prediction-regime selection and asymmetric interval construction can be implemented with zero additional neural inference latency ($O(1)$ lookup).

## 18. Final Research Conclusion
The post-hoc method was evaluated to determine whether prediction-regime-adaptive asymmetric calibration can improve subgroup coverage while preserving useful overall interval width. The results demonstrate that prediction-regime-adaptive intervals modulate interval widths according to the predicted regime, but residual under-coverage in extreme tails demonstrates the fundamental information boundary of post-hoc calibration applied to frozen point estimators.
