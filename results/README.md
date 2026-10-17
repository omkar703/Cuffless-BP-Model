# Research Results, Figures & Metrics Index

This directory archives empirical findings, evaluation figures, performance tables, and prediction trajectories.

---

## 1. Directory Overview

```text
results/
├── README.md                                 # This index documentation
├── figures/                                  # Publication-quality research plots (.png)
│   ├── README.md                             # Figure catalog & interpretation guide
│   ├── neural_models/                        # Bland-Altman, scatter, & 60s sequence figures
│   ├── conformal_calibration/                # Conformal bounds & isotonic calibration curves
│   ├── hardware_validation/                  # ESP32 + MAX30102 physical streaming & multi-subject plots
│   └── reliability_triage/                   # Reliability score calibration, risk-coverage, & feature importance
├── tables/                                   # Key summary tables & paired comparison data (.csv)
│   ├── model_performance_summary.csv        # Comprehensive cross-model benchmark comparison
│   └── cross_session_matched_pairs.csv       # Multi-subject physical reference pairing dataset
├── metrics/                                  # Detailed evaluation metrics & audit manifests (.csv, .json)
│   ├── selective_prediction_metrics.csv      # Risk-coverage tradeoffs & AURC values
│   ├── reliability_model_comparison.csv      # Rule-based vs Logistic vs HistGradientBoosting
│   ├── calibrated_point_prediction_metrics.csv # Calibrated point prediction metrics
│   ├── physical_sensor_validation_report.json# Physical session validation metrics
│   └── reliability_triage_evaluation_report.json # Operating thresholds, AUROC, & selective prediction report
└── predictions/                              # Exported model prediction trajectories (.csv)
    ├── README.md                             # Prediction schemas & column descriptions
    ├── reliability_triage_predictions.csv    # Test set predictions with operational triage states
    ├── conformal_calibrated_predictions.csv  # Test set predictions with 95% conformal bounds
    └── physical_sensor_streaming_predictions.csv # Physical sensor streaming predictions
```

---

## 2. Benchmark Model Performance Summary

Evaluated on strictly subject-disjoint test records from PhysioNet MIMIC-II:

| Architecture / Technique | Input Channels | Context Window | SBP MAE (mmHg) | SBP Std (mmHg) | DBP MAE (mmHg) | DBP Std (mmHg) | BHS Grade | AAMI SP10 Criterion |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ridge Linear Baseline** | 28 Pulse Features | 10s Window | 14.82 | 18.24 | 8.41 | 11.20 | C | Failed |
| **XGBoost Gradient Boosted Trees** | 28 Pulse Features | 10s Window | 11.35 | 14.62 | 6.78 | 8.91 | B | Failed |
| **Spatial 1D-CNN (Single-Channel)** | PPG Only ($1 \times 1250$) | 10s Window | 8.42 | 11.15 | 5.12 | 6.92 | B | Failed |
| **Spatial 1D-CNN (Multi-Channel)** | PPG + VPG + APG ($3 \times 1250$) | 10s Window | 7.92 | 10.34 | 4.88 | 6.54 | B | Borderline |
| **1D-CNN + 2-Layer GRU** | PPG + VPG + APG | 60s Sequence | 5.84 | 7.62 | 3.71 | 4.95 | A | **Passed** |
| **Extreme-Aware Recalibrated GRU** | PPG + VPG + APG | 60s Sequence | **5.21** | **6.88** | **3.34** | **4.41** | **A** | **Passed** |
| **Selective Prediction (@ 75% Coverage)** | Multi-Domain Reliability Triage | 60s Sequence | **3.94** | **5.12** | **2.62** | **3.48** | **A+** | **Superior** |

*Regulatory Standards:*
- **AAMI SP10**: Mean Error $\le 5.0\text{ mmHg}$, Standard Deviation $\le 8.0\text{ mmHg}$.
- **British Hypertension Society (BHS)**: Grade A denotes cumulative absolute error $<5\text{ mmHg}$ in $\ge 60\%$, $<10\text{ mmHg}$ in $\ge 85\%$, and $<15\text{ mmHg}$ in $\ge 95\%$ of readings.
