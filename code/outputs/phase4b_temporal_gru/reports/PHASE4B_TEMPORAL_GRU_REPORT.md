# PHASE 4B — Temporal Context Extension Report

**Architecture:** Frozen Phase 4A 1D CNN + Causal 6-Window GRU  
**Mode:** Calibration-Free Cuffless Blood Pressure Estimation  
**Execution Environment:** Local System (NVIDIA GeForce GTX 1650 Ti)  
**Timestamp:** 2026-09-18 14:13:56  

---

## 1. Research Question
Does adding causal temporal context from the previous 60 seconds (6 consecutive 10-second windows: $[t-5, t-4, t-3, t-2, t-1, t]$) improve blood-pressure estimation beyond the frozen Phase 4A single-window CNN representation?

## 2. Why Temporal Context
Blood pressure exhibits physiological autocorrelation driven by vascular tone, baroreflex buffering, and autonomic modulation. While a single 10-second PPG window captures immediate pulse wave velocity (PWV) and reflection wave indices, historical temporal trends across 60 seconds provide low-frequency physiological trajectories that single windows cannot perceive.

## 3. Relationship to Phase 3B Findings
Phase 3B demonstrated that among classical feature aggregations, Context-5 (60 seconds of sequential history) yielded the lowest MAE (SBP MAE = 13.28 mmHg, DBP MAE = 6.56 mmHg). Phase 4B tests whether this 60-second temporal window length benefits neural representations learned by the Phase 4A CNN.

## 4. Frozen Phase 4A Baseline Reference
- Full Test Set (38,361 windows): SBP MAE = 11.0368 mmHg, DBP MAE = 5.7859 mmHg, Combined MAE = 8.4114 mmHg
- Matched Test Subset (31,192 windows): SBP MAE = 10.9417 mmHg, DBP MAE = 5.6943 mmHg, Combined MAE = 8.3180 mmHg

## 5. Sequence Construction
- Sequence Length: 6 windows (6 x 10s = 60s)
- Sequence Integrity: All 6 windows belong to the identical `record_id` and are strictly consecutive in time.
- Prediction Target: Current window BP, SBP(t) and DBP(t).

## 6. Causality Guarantee
- Strictly unidirectional GRU (`bidirectional = False`).
- Temporal order: Oldest to newest (w[t-5] -> w[t-4] -> w[t-3] -> w[t-2] -> w[t-1] -> w[t]).
- Zero future window access.

## 7. Frozen CNN Encoder
- Pretrained model loaded from `code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt`.
- Total CNN Parameters: 146,978 (Trainable: 0).
- Latent Representation: 64-dimensional feature vector extracted immediately prior to the SBP/DBP linear heads.

## 8. GRU Architecture
- Causal GRU: `input_size = 64`, `hidden_size = 64`, `num_layers = 1`, `batch_first = True`.
- Dense Projection: `Linear(64 -> 32) -> ReLU -> Dropout(0.2)`.
- Dual Heads: `SBP Linear(32 -> 1)`, `DBP Linear(32 -> 1)`.
- Trainable Parameters: 27,106.

## 9. Training Configuration
- Loss: Huber loss (delta = 5.0)
- Optimizer: AdamW (lr = 1e-3, weight_decay = 1e-4)
- Batch Size: 256
- Max Epochs: 40 (Early stopping patience = 8)
- Selection Criterion: Lowest Validation Combined MAE

## 10. Sequence Retention Statistics
- Train: 149,500 valid sequences from 183,517 eligible windows (81.46%)
- Val:   32,163 valid sequences from 39,461 eligible windows (81.51%)
- Test:  31,192 valid sequences from 38,361 eligible windows (81.31%)

## 11. Validation Results
- Best Validation Epoch: 5
- Validation SBP MAE:  10.74 mmHg
- Validation DBP MAE:  5.33 mmHg
- Validation Comb MAE: 8.03 mmHg

## 12. Full Phase 4B Test Results (31,192 Sequences)
- Test SBP MAE:  10.57 mmHg (RMSE: 14.40, R2: 0.557, Bias: -0.63)
- Test DBP MAE:  5.51 mmHg (RMSE: 8.25, R2: 0.472, Bias: -1.25)
- Test Comb MAE: 8.04 mmHg

## 13. Primary Research Comparison: Matched Test Subset

| Model / Architecture | Split / Subset | Test Windows | SBP MAE (mmHg) | DBP MAE (mmHg) | Combined MAE (mmHg) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 4A Frozen CNN** | Full Test | 38,361 | 11.04 | 5.79 | 8.41 |
| **Phase 4A Frozen CNN** | Matched Test | 31,192 | 10.94 | 5.69 | 8.32 |
| **Phase 4B Frozen-CNN + Causal GRU** | Matched Test | 31,192 | 10.57 | 5.51 | 8.04 |

### Temporal Gain:
- SBP MAE Gain:      +0.38 mmHg
- DBP MAE Gain:      +0.18 mmHg
- Combined MAE Gain: +0.28 mmHg

## 14. Clinical BP Range Stratified Errors (Descriptive)

### SBP Ranges:
| target   | range   |   sample_count |      mae |     rmse |       bias |   error_sd |
|:---------|:--------|---------------:|---------:|---------:|-----------:|-----------:|
| SBP      | <90     |            503 | 20.5178  | 23.2278  |  20.5169   |   10.8898  |
| SBP      | 90-119  |          10219 | 10.219   | 14.1169  |   8.45642  |   11.3038  |
| SBP      | 120-139 |          10320 |  7.64762 |  9.98684 |  -0.106505 |    9.98628 |
| SBP      | 140-159 |           7120 | 10.8755  | 14.2812  |  -8.14013  |   11.7341  |
| SBP      | >=160   |           3030 | 19.2991  | 23.5118  | -18.9356   |   13.9373  |

### DBP Ranges:
| target   | range   |   sample_count |      mae |     rmse |      bias |   error_sd |
|:---------|:--------|---------------:|---------:|---------:|----------:|-----------:|
| DBP      | <60     |           9134 |  4.62601 |  6.18116 |   4.0254  |    4.69072 |
| DBP      | 60-79   |          17998 |  4.23684 |  5.61127 |  -1.26315 |    5.46725 |
| DBP      | 80-89   |           2618 |  8.27999 | 10.7143  |  -8.10385 |    7.00885 |
| DBP      | 90-99   |            946 | 17.4056  | 19.9378  | -17.4012  |    9.73211 |
| DBP      | >=100   |            496 | 30.9663  | 32.6879  | -30.9663  |   10.4684  |

## 15. Record-Level Performance (Record-Independent Evaluation)
- SBP Record MAE: Mean = 11.10, Median = 7.60, SD = 9.49 mmHg
- DBP Record MAE: Mean = 5.85, Median = 3.82, SD = 5.92 mmHg
- Combined Record MAE: Mean = 8.48, Median = 6.44, SD = 6.50 mmHg

## 16. Scientific Limitations
- Evaluated on ICU patient records; external generalization to ambulatory healthy cohorts requires separate empirical validation.
- Missing history at the onset of monitoring sessions (18.6% initial-window attrition).
- Single fixed context length (60 seconds) evaluated; multi-scale context remains an area for further investigation.

## 17. Scientific Interpretation
Temporal context from the preceding 60 seconds produces a measurable reduction in prediction error over the frozen CNN representation.

## 18. Next Research Step
Advance to **Phase 5: Model Calibration & Uncertainty Estimation**, exploring whether lightweight calibration or epistemic uncertainty bounds can resolve remaining extreme-BP residual errors.
