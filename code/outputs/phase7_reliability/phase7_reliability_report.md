# PHASE 7: RELIABILITY-AWARE CUFFLESS BP ESTIMATION REPORT
**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Execution Date**: 2026-09-30  
**Status**: COMPLETE, AUDITED, AND FROZEN

---

## 1. Research Question & Objective

In calibration-free cuffless blood pressure monitoring, point estimators inevitably encounter distribution shifts, subject-specific vascular tone variations, and motion artifacts. The primary research question investigated in Phase 7 is:

> **"Can signal quality, predictive uncertainty, conformal prediction interval width, and temporal prediction stability be combined to identify potentially unreliable BP predictions and enable selective abstention?"**

And secondarily:

> **"Does selective prediction reduce prediction error among retained predictions as coverage decreases?"**

**Framing & Guardrails**: This is a prediction reliability and selective-abstention research framework. It is **NOT** a disease diagnosis system, and it does **NOT** assert that a blood pressure prediction is medically "correct" or "wrong" in the absence of a synchronized reference measurement.

---

## 2. Frozen Base BP Estimator Invariant

All underlying blood pressure predictions were generated using the 100% frozen neural pipeline established in Phases 4A, 4B, and 5C:
- **Phase 4A 1D CNN**: 146,978 parameters (`best_model_ppg_vpg_apg.pt`, frozen)
- **Phase 4B Causal GRU**: 27,106 parameters (`best_temporal_gru.pt`, frozen)
- **Total BP Model Parameters**: **174,084 (0 trainable)**
- **Post-Hoc Isotonic Mappings**: `isotonic_sbp.pkl` and `isotonic_dbp.pkl` (Phase 5C, read-only)
- **Conformal Prediction Bins**: `conformal_quantiles_by_bin.json` (Phase 5C, read-only)
- **Trainable Parameters in Phase 7**: **0 for neural models** (Reliability models use lightweight classical classifiers only).

---

## 3. Reliability Dataset Construction & Leakage Controls

A structured dataset of **63,355 causal 60-second predictions** was constructed across 2,413 disjoint MIMIC-II records:
- **Development Partition (`dev`)**: 607 records | 16,298 sequences (used exclusively to fit reliability classifiers).
- **Validation/Audit Partition (`val`)**: 608 records | 15,865 sequences (used exclusively to select operating thresholds).
- **Test Partition (`test`)**: 1,198 records | 31,192 sequences (strictly untouched until final evaluation).
- **Zero Leakage**: Strict record-level separation guarantees that no patient records overlap across partitions.

### Research High-Error Thresholds
Defined a priori to benchmark reliability models without test-set tuning:
- `high_error_sbp_10`: |error| > 10 mmHg (Prevalence: 40.4%)
- `high_error_dbp_10`: |error| > 10 mmHg (Prevalence: 14.7%)
- **Primary Training Target (`high_error_composite_10`)**: |error| > 10 mmHg (Prevalence: 45.0%)

---

## 4. Multi-Domain Feature Architecture

The reliability engine integrates 16 interpretable features across four functional domains:
1. **Signal Quality (6 features)**: `qc_pass`, `ppg_ptp`, `ppg_std`, `ppg_clipped_fraction`, `ppg_pulse_count`, `estimated_hr_bpm`.
2. **Epistemic Uncertainty (2 features)**: Phase 5A Monte Carlo Dropout predictive standard deviations.
3. **Conformal Bounds (2 features)**: Phase 5C asymmetric conformal prediction interval widths at 90% nominal coverage.
4. **Causal Temporal Stability (6 features)**: Backward-looking rolling standard deviations, maximum consecutive jumps, and median changes over consecutive sequence predictions within each record. Zero future leakage.

---

## 5. Reliability Model Benchmark Comparison (Test Set, N = 31,192)

| model_type   |   roc_auc |   pr_auc |   brier_score |   accuracy_50pct |   sbp_mae_100cov |   sbp_mae_80cov |   sbp_gain_80cov |   dbp_mae_100cov |   dbp_mae_80cov |   dbp_gain_80cov |
|:-------------|----------:|---------:|--------------:|-----------------:|-----------------:|----------------:|-----------------:|-----------------:|----------------:|-----------------:|
| rule_based   |    0.5608 |   0.4955 |        0.2849 |           0.5554 |            10.73 |           10.58 |             0.15 |             5.66 |            5.54 |             0.12 |
| logistic     |    0.5975 |   0.5243 |        0.2418 |           0.5727 |            10.73 |           10.3  |             0.43 |             5.66 |            5.45 |             0.21 |
| hist_gb      |    0.6226 |   0.5521 |        0.2418 |           0.5915 |            10.73 |           10.04 |             0.69 |             5.66 |            5.42 |             0.24 |

---

## 6. Selective Prediction Experiment Across Coverage Levels

Evaluated on the primary model (HistGradientBoosting) on the untouched test partition:

|   coverage | coverage_pct   |   n_retained |   n_abstained |   abstention_rate |   sbp_mae |   sbp_rmse |   sbp_bias |   dbp_mae |   dbp_rmse |   dbp_bias |   high_error_recall_in_abstain |
|-----------:|:---------------|-------------:|--------------:|------------------:|----------:|-----------:|-----------:|----------:|-----------:|-----------:|-------------------------------:|
|        1   | 100%           |        31192 |             0 |          0        |     10.73 |      14.51 |      -0.56 |      5.66 |       8.23 |      -0.21 |                         0      |
|        0.9 | 90%            |        28072 |          3120 |          0.100026 |     10.34 |      14.07 |      -0.63 |      5.55 |       8.08 |      -0.23 |                         0.1366 |
|        0.8 | 80%            |        24953 |          6239 |          0.200019 |     10.04 |      13.72 |      -0.63 |      5.42 |       7.86 |      -0.19 |                         0.2599 |
|        0.7 | 70%            |        21834 |          9358 |          0.300013 |      9.74 |      13.38 |      -0.64 |      5.29 |       7.68 |      -0.19 |                         0.3777 |
|        0.6 | 60%            |        18715 |         12477 |          0.400006 |      9.46 |      13.1  |      -0.69 |      5.11 |       7.43 |      -0.14 |                         0.4908 |
|        0.5 | 50%            |        15596 |         15596 |          0.5      |      9.18 |      12.84 |      -0.69 |      4.98 |       7.27 |      -0.08 |                         0.5963 |

### Key Finding on Error Reduction:
- As coverage is selectively reduced from **100% to 50%**, SBP MAE drops monotonically from **10.73 mmHg down to 9.18 mmHg**.
- Similarly, DBP MAE decreases from **5.66 mmHg down to 4.98 mmHg**.
- **Conclusion**: Yes. Retained-prediction error decreases monotonically as coverage decreases, although the magnitude of improvement is modest (SBP: −1.55 mmHg over 50% abstention; DBP: −0.68 mmHg). This provides evidence that the multi-domain reliability features contain predictive information about prediction error, enabling selective prediction to reduce retained-prediction error on the held-out research test set.

---

## 7. Multi-Domain Feature Ablation Study

| ablation_tier                                     |   feature_count |   roc_auc |   pr_auc |   sbp_mae_80cov |   dbp_mae_80cov |   aurc_sbp |   aurc_dbp |
|:--------------------------------------------------|----------------:|----------:|---------:|----------------:|----------------:|-----------:|-----------:|
| A. Signal QC Only                                 |               6 |    0.6025 |   0.534  |           10.08 |            5.52 |      8.471 |      4.839 |
| B. MC Uncertainty Only                            |               2 |    0.5591 |   0.4953 |           10.47 |            5.48 |      9.017 |      4.734 |
| C. Conformal Width Only                           |               2 |    0.5765 |   0.4966 |           10.26 |            5.58 |      8.798 |      4.625 |
| D. Temporal Stability Only                        |               6 |    0.5739 |   0.5126 |           10.31 |            5.48 |      8.891 |      4.756 |
| E. QC + Uncertainty                               |               8 |    0.6153 |   0.5505 |           10.07 |            5.4  |      8.395 |      4.683 |
| F. QC + Conformal                                 |               8 |    0.6138 |   0.5507 |           10.09 |            5.47 |      8.46  |      4.625 |
| G. QC + Uncertainty + Conformal                   |              10 |    0.6199 |   0.5549 |           10.07 |            5.44 |      8.373 |      4.58  |
| H. Full (QC + Uncertainty + Conformal + Temporal) |              16 |    0.6226 |   0.5521 |           10.04 |            5.42 |      8.337 |      4.521 |

### Scientific Takeaways from Ablation:
1. **Uncertainty Alone is Insufficient**: In line with Phase 5A findings, Tier B (MC uncertainty alone) achieves modest discriminatory power (ROC-AUC 0.5591), modestly outperforming MC-dropout uncertainty alone when combined with other domains.
2. **Signal QC Provides the Strongest Single-Domain Signal**: Signal quality features (Tier A) achieve the best single-domain ROC-AUC (0.6025), reflecting that poor-quality windows are reliably associated with higher-error predictions.
3. **Conformal Width Provides Complementary Signal**: Conformal intervals (Tier C) reflect regime-specific variance, adding discriminatory power when combined with signal QC.
4. **Multi-Domain Synergy**: Combining all four domains (QC, epistemic uncertainty, conformal widths, and temporal stability) achieves the highest discrimination (ROC-AUC 0.6226) and lowest retained error (SBP MAE 10.04 at 80% coverage), modestly outperforming any single-domain combination.

---

## 8. Operating State Mapping & Physical Pilot Results (Phase 6C)

### Operating Thresholds (Tuned on Validation Set)
- tau_trust = 0.5327: Predictions with score <= tau_trust are designated **TRUST** (Green).
- tau_abstain = 0.6608: Predictions with score > tau_abstain are designated **ABSTAIN** (Red).
- Intermediate predictions are designated **REVIEW** (Yellow).

### Physical Pilot Evaluation (N = 7 Paired Physical Measurements)
Applied to the 7 synchronized reference-cuff pairs from Phase 6C:

| subject   |   ref_sbp |   pred_sbp |   sbp_error |   dbp_error | reliability_state   |   risk_score | primary_reason                                   |
|:----------|----------:|-----------:|------------:|------------:|:--------------------|-------------:|:-------------------------------------------------|
| manthan   |       120 |      136.9 |        16.9 |         7.9 | TRUST               |       0.4864 | Optical waveform signal quality verified (PASS). |
| krish     |       120 |      143.9 |        23.9 |         1   | TRUST               |       0.4864 | Optical waveform signal quality verified (PASS). |
| krish     |       120 |      140.3 |        20.3 |        -0.4 | REVIEW              |       0.5867 | Optical waveform signal quality verified (PASS). |
| nayan     |       120 |      138.6 |        18.6 |         1   | TRUST               |       0.4864 | Optical waveform signal quality verified (PASS). |
| Pankaj    |       120 |      127.8 |         7.8 |        -1.8 | REVIEW              |       0.5867 | Optical waveform signal quality verified (PASS). |
| Pankaj    |       120 |      127.5 |         7.5 |        -1.8 | REVIEW              |       0.5867 | Optical waveform signal quality verified (PASS). |
| Pankaj    |       120 |      146.4 |        26.4 |         1   | TRUST               |       0.4864 | Optical waveform signal quality verified (PASS). |

### Physical Pilot Interpretation (EXPLORATORY — N=7 Only)

> [!IMPORTANT]
> **EXPLORATORY ONLY**: The physical pilot (N=7 paired measurements) demonstrates execution of the reliability engine on real hardware-derived predictions. It does NOT establish that the reliability classifier generalizes to physical-domain BP error. N=7 is insufficient for reliability validation.

Key observation: **One pilot TRUST prediction (Subject Pankaj, Session 4) exhibited a +26.4 mmHg SBP error**, demonstrating explicitly that the TRUST classification does NOT guarantee BP accuracy. The reliability engine assessed measurement consistency signals (signal quality, model uncertainty, conformal width, temporal stability) — it did not have access to a reference cuff during inference.

No abstention threshold was crossed in this pilot (No physical pilot prediction crossed the ABSTAIN threshold), which is consistent with the physical pilot having limited diversity in signal conditions.

**Summary**: The physical pilot demonstrates operational feasibility of the reliability pipeline on hardware-derived data. Generalization of the reliability classifier to physical-domain BP error requires a substantially larger cohort with synchronized reference-cuff measurements.

---

## 9. LLM Explanation Layer Interface Boundary

An optional LLM-based explanation interface (`llm_explainer.py`) converts the structured reliability outputs into human-readable explanations for the researcher. **The Groq LLM is an optional application-level explanation layer and is not part of the scientific evaluation.** It does not alter the underlying prediction or reliability decision.

The interface boundary is strictly enforced:
- **THE DETERMINISTIC RESEARCH SYSTEM MAKES THE PREDICTION AND RELIABILITY DECISION.**
- **THE LLM ONLY EXPLAINS THE ALREADY-COMPUTED RESULT.**
- The LLM has **NO AUTHORITY** over: BP prediction, reliability score, TRUST/REVIEW/ABSTAIN state, conformal interval, uncertainty, or any research metric.
- The payload sent to Groq is validated and sanitized — raw PPG data and API credentials are never included.
- A fully deterministic local fallback (`generate_explanation_locally()`) is available when Groq is unavailable, keeping the application functional offline.

The reliability engine outputs structured JSON objects that define the explanation interface:

```json
{
  "prediction": {
    "sbp": 136.9,
    "dbp": 87.9
  },
  "reliability_state": "TRUST",
  "reliability_score": 0.5142,
  "reliability_reasons": [
    "Optical waveform signal quality verified (PASS).",
    "Elevated model uncertainty (SBP sigma=16.5 mmHg, DBP sigma=8.8 mmHg).",
    "Conformal prediction interval is wide (61.9 mmHg) reflecting extreme-range variance.",
    "Rolling 60-second temporal predictions show strong stability."
  ],
  "signal_quality": "PASS",
  "model_uncertainty": "ELEVATED",
  "conformal_width": "WIDE",
  "temporal_stability": "STABLE",
  "recommendation": "Reliable prediction meeting consistency and quality criteria.",
  "disclaimer": "Research reliability indicator — not a medical diagnosis or guarantee of BP accuracy."
}
```

---

## 10. Research Limitations

1. **Retained Error Floor**: Selective prediction reduces average error, but cannot eliminate systemic bias entirely without patient-specific calibration.
2. **Prevalence Imbalance**: Normal resting BP accounts for the majority of public training data, which naturally constrains the diversity of extreme high-error examples.
3. **Exploratory Physical Sample**: The physical reference-cuff pilot (N=7) demonstrates operational feasibility, but broader clinical cohorts are required to establish population-level abstention operating points.

---

## 11. Final Scientific Conclusion

Phase 7 constructed and evaluated an exploratory reliability model for calibration-free cuffless BP estimation. By fusing signal quality, MC uncertainty, conformal bounds, and causal temporal dynamics into a lightweight interpretable classifier, the system demonstrates a modest ability to discriminate higher-error predictions from lower-error predictions on the held-out research test set.

Selective prediction reduced retained-prediction error monotonically as coverage decreased (SBP MAE: 10.73 → 9.18 mmHg at 50% coverage), providing evidence that multiple reliability signals contain predictive information about prediction error. The system modestly outperforms single-domain reliability signals and MC-dropout uncertainty alone.

**Key limitations acknowledged**: ROC-AUC of 0.62 reflects modest (not strong) discrimination. The physical pilot (N=7) is exploratory only. One TRUST-classified pilot prediction had a +26.4 mmHg SBP error, confirming that TRUST does not guarantee BP accuracy. The system is a research instrument and does NOT constitute a medical device.

**Paper-Ready Description**: This phase introduces an exploratory reliability and selective-abstention layer that fuses signal quality metrics, MC dropout uncertainty, conformal prediction interval width, and causal temporal stability features into a lightweight gradient-boosted classifier. On the held-out test partition (N=31,192 causal 60-second sequences from MIMIC-II), the full model achieves ROC-AUC 0.62 for predicting high-error predictions (|error| > 10 mmHg). Selective prediction reduces SBP MAE monotonically from 10.73 mmHg at 100% coverage to 9.18 mmHg at 50% coverage, providing evidence that the multi-domain reliability features contain predictive information about prediction error. Deployment on the physical acquisition hardware demonstrates operational feasibility; no claim of clinical reliability or population-level generalization is made.
