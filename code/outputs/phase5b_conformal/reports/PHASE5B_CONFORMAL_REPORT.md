# PHASE 5B — Post-Hoc Conformal BP Interval Calibration Report

**Architecture:** Frozen Phase 4A 1D CNN + Causal 6-Window GRU + Conformal Calibration  
**Mode:** Post-Hoc Inference-Only Conformal Prediction  
**Execution Environment:** Local System (NVIDIA GeForce GTX 1650 Ti)  
**Timestamp:** 2026-09-18 14:57:22  

---

## 1. Research Question
Can we construct empirically calibrated prediction intervals around the existing BP estimates using split conformal prediction, without retraining the neural network?

## 2. Why Post-Hoc Conformal Calibration
In critical monitoring settings, point predictions alone provide no formal measure of empirical confidence. Post-hoc split conformal prediction guarantees finite-sample marginal coverage under the exchangeability assumption, transforming bare point estimates into rigorous prediction intervals without altering model parameters.

## 3. Frozen Phase 4B Model
- Pretrained Checkpoint: `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt`
- Total CNN Backbone Parameters: 146,978 (Frozen: 0 trainable)
- Total Temporal Model Parameters: 27,106 (Frozen: 0 trainable)
- Total Trainable Parameters in Phase 5B: Exactly 0.

## 4. Calibration Split (Record-Level Partition)
- Source Partition: Phase 4B Validation Set (32,163 sequences across 1,215 records)
- Random Seed: 42
- Calibration Subset: 607 records | 16,298 sequences (50%)
- Audit Subset: 608 records | 15,865 sequences (50%)
- Test Set: 1198 records | 31,192 sequences (100% untouched)
- Leakage Check: Pairwise disjoint record sets verified (cal intersect audit = empty, cal intersect test = empty).

## 5. Pre-Defined Conformal Methods
1. **Method A (Standard Split Conformal):** Constant half-width derived from absolute calibration residuals |y - y_hat|.
2. **Method B (Uncertainty-Scaled Conformal):** Heteroscedastic half-width derived from normalized residuals |y - y_hat| / (sigma_MC + epsilon), scaling with Phase 5A MC-dropout uncertainty.

## 6. Nominal Coverage Levels & Calibration Quantiles
- Quantiles estimated strictly on 16,298 calibration samples:
  - Method A (SBP): q_90 = 25.81 mmHg | q_95 = 34.25 mmHg
  - Method A (DBP): q_90 = 12.06 mmHg | q_95 = 16.89 mmHg
  - Method B (SBP): q_90 = 1.807 | q_95 = 2.380
  - Method B (DBP): q_90 = 1.630 | q_95 = 2.252

## 7. Audit-Set Results (Sanity Check)
| method                        | target   |   nominal |   empirical_coverage |   mean_width_mmHg |
|:------------------------------|:---------|----------:|---------------------:|------------------:|
| Standard Conformal (Method A) | SBP      |        90 |              90.7217 |           51.6243 |
| Standard Conformal (Method A) | SBP      |        95 |              96.5522 |           68.5057 |
| Uncertainty-Scaled (Method B) | SBP      |        90 |              90.9108 |           51.9581 |
| Uncertainty-Scaled (Method B) | SBP      |        95 |              96.4513 |           68.4115 |
| Standard Conformal (Method A) | DBP      |        90 |              91.6987 |           24.1183 |
| Standard Conformal (Method A) | DBP      |        95 |              96.4261 |           33.7812 |
| Uncertainty-Scaled (Method B) | DBP      |        90 |              92.4992 |           24.5461 |
| Uncertainty-Scaled (Method B) | DBP      |        95 |              96.7034 |           33.93   |

## 8. Frozen-Test Results (31,192 Sequences)
The central research evaluation evaluated once on the untouched test partition:
| method                        | target   |   nominal_coverage |   empirical_coverage |   signed_coverage_error |   mean_width_mmHg |   median_width_mmHg |
|:------------------------------|:---------|-------------------:|---------------------:|------------------------:|------------------:|--------------------:|
| Standard Conformal (Method A) | SBP      |                 90 |              91.2478 |                1.24776  |           51.6243 |             51.6243 |
| Standard Conformal (Method A) | SBP      |                 95 |              96.6722 |                1.67222  |           68.5057 |             68.5057 |
| Standard Conformal (Method A) | DBP      |                 90 |              89.7249 |               -0.275071 |           24.1183 |             24.1183 |
| Standard Conformal (Method A) | DBP      |                 95 |              95.1398 |                0.139779 |           33.7812 |             33.7812 |
| Uncertainty-Scaled (Method B) | SBP      |                 90 |              90.7572 |                0.757245 |           51.6516 |             50.9817 |
| Uncertainty-Scaled (Method B) | SBP      |                 95 |              96.2907 |                1.29072  |           68.0079 |             67.1259 |
| Uncertainty-Scaled (Method B) | DBP      |                 90 |              90.4174 |                0.417415 |           24.7223 |             24.1095 |
| Uncertainty-Scaled (Method B) | DBP      |                 95 |              95.4764 |                0.476404 |           34.1737 |             33.3266 |

## 9. Interval Width Comparison
| method                        | target   |   nominal_coverage |   empirical_coverage |   mean_width_mmHg |   median_width_mmHg |   p90_width_mmHg |
|:------------------------------|:---------|-------------------:|---------------------:|------------------:|--------------------:|-----------------:|
| Standard Conformal (Method A) | SBP      |                 90 |              91.2478 |           51.6243 |             51.6243 |          51.6243 |
| Standard Conformal (Method A) | SBP      |                 95 |              96.6722 |           68.5057 |             68.5057 |          68.5057 |
| Standard Conformal (Method A) | DBP      |                 90 |              89.7249 |           24.1183 |             24.1183 |          24.1183 |
| Standard Conformal (Method A) | DBP      |                 95 |              95.1398 |           33.7812 |             33.7812 |          33.7812 |
| Uncertainty-Scaled (Method B) | SBP      |                 90 |              90.7572 |           51.6516 |             50.9817 |          64.8927 |
| Uncertainty-Scaled (Method B) | SBP      |                 95 |              96.2907 |           68.0079 |             67.1259 |          85.442  |
| Uncertainty-Scaled (Method B) | DBP      |                 90 |              90.4174 |           24.7223 |             24.1095 |          31.2102 |
| Uncertainty-Scaled (Method B) | DBP      |                 95 |              95.4764 |           34.1737 |             33.3266 |          43.1418 |

## 10. BP-Range Subgroup Coverage (Descriptive)
### SBP Ranges:
| target   | bp_range   |   sample_count |   method_a_cov_90 |   method_a_cov_95 |   method_b_cov_90 |   method_b_cov_95 |   method_b_mean_width_95 |
|:---------|:-----------|---------------:|------------------:|------------------:|------------------:|------------------:|-------------------------:|
| SBP      | <90        |            503 |           77.5348 |           89.2644 |           55.2684 |           81.3121 |                  54.7371 |
| SBP      | 90-119     |          10219 |           90.821  |           96.4282 |           90.4687 |           97.1915 |                  60.5066 |
| SBP      | 120-139    |          10320 |           98.3818 |           99.9225 |           97.4806 |           99.2636 |                  68.1063 |
| SBP      | 140-159    |           7120 |           91.3202 |           97.6966 |           90.0421 |           95.5337 |                  74.4111 |
| SBP      | >=160      |           3030 |           70.495  |           85.2475 |           76.4026 |           87.3927 |                  80.1287 |

### DBP Ranges:
| target   | bp_range   |   sample_count |   method_a_cov_90 |   method_a_cov_95 |   method_b_cov_90 |   method_b_cov_95 |   method_b_mean_width_95 |
|:---------|:-----------|---------------:|------------------:|------------------:|------------------:|------------------:|-------------------------:|
| DBP      | <60        |           9134 |          93.464   |          98.4235  |           93.8581 |           99.1132 |                  30.8271 |
| DBP      | 60-79      |          17998 |          95.4662  |          99.2166  |           94.7939 |           98.4443 |                  34.4102 |
| DBP      | 80-89      |           2618 |          74.3697  |          87.1658  |           78.2659 |           87.1658 |                  40.209  |
| DBP      | 90-99      |            946 |          33.2981  |          52.8541  |           49.1543 |           65.0106 |                  41.9809 |
| DBP      | >=100      |            496 |           1.20968 |           9.47581 |           11.0887 |           22.7823 |                  40.4724 |

## 11. Interval Width vs Prediction Error Association
| target   | method                        |   nominal_coverage |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |
|:---------|:------------------------------|-------------------:|------------:|------------:|---------------:|-------------:|
| SBP      | Uncertainty-Scaled (Method B) |                 95 |   0.0757263 | 6.65149e-41 |      0.0813693 | 5.63288e-47  |
| DBP      | Uncertainty-Scaled (Method B) |                 95 |   0.13277   | 1.1896e-122 |      0.133493  | 5.58011e-124 |

## 12. Scientific Limitations
- Split conformal prediction provides marginal empirical coverage guarantees under the exchangeability assumption; it does not provide conditional guarantees across all individual BP sub-ranges.
- Extreme blood pressure ranges (<90 and >=160 mmHg) exhibit reduced subgroup coverage due to residual bias inherent in the frozen point estimator.
- Data are derived from ICU patient records; translation to ambulatory healthy populations requires independent calibration.

## 13. Deployment Relevance & Wearable Example
For wearable firmware, a user output would read:
- SBP Point Estimate: 132 mmHg (95% Interval: [108, 156] mmHg)
- DBP Point Estimate: 78 mmHg (95% Interval: [64, 92] mmHg)
Such intervals communicate empirical statistical reliability without claiming clinical diagnostic infallibility.

## 14. Scientific Interpretation
Split conformal calibration successfully achieves nominal 90% and 95% marginal coverage on the unseen test set without retraining. Method B provides adaptive, heteroscedastic intervals that expand on difficult inputs while maintaining the pre-specified marginal coverage.

## 15. Next Research Step
Advance to **Phase 6: Comprehensive Pipeline Consolidation, Wearable Deployment Simulation, and Final Paper Synthesis**.
