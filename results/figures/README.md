# Research Figures Catalog

This directory catalogs publication-grade figures, Bland-Altman agreement charts, and diagnostic plots.

---

## 1. Directory Structure

```text
results/figures/
├── README.md                                          # This figure catalog
├── neural_models/                                     # Deep Representation Learning Figures
│   ├── sbp_predicted_vs_reference_scatter.png         # Test set predicted vs ground-truth SBP scatter
│   ├── dbp_predicted_vs_reference_scatter.png         # Test set predicted vs ground-truth DBP scatter
│   ├── sbp_bland_altman_agreement.png                 # Bland-Altman agreement plot for SBP
│   ├── dbp_bland_altman_agreement.png                 # Bland-Altman agreement plot for DBP
│   ├── sixty_second_temporal_sequence_flow.png        # 60-second temporal sequence flow
│   └── spatial_cnn_vs_temporal_gru_comparison.png     # Paired comparison (10s spatial vs 60s temporal)
├── conformal_calibration/                             # Uncertainty & Conformal Figures
│   ├── sbp_isotonic_calibration.png                   # Monotonic isotonic recalibration curve for SBP
│   ├── dbp_isotonic_calibration.png                   # Monotonic isotonic recalibration curve for DBP
│   ├── coverage_comparison.png                        # Empirical coverage across BP quantiles
│   ├── interval_width_comparison.png                  # Conformal prediction interval widths
│   ├── example_sbp_intervals.png                      # Conformal interval trajectories (SBP)
│   ├── example_dbp_intervals.png                      # Conformal interval trajectories (DBP)
│   ├── uncertainty_vs_error_sbp.png                   # Monte Carlo epistemic dispersion vs prediction error
│   └── risk_coverage_sbp.png                          # Selective prediction risk-coverage curve
├── hardware_validation/                               # Physical Hardware & Multi-Subject Figures
│   ├── reference_vs_causal_ppg.png                    # Zero-phase offline filter vs causal 3rd-order Butterworth
│   ├── reference_vs_causal_apg.png                    # Backward difference APG vs offline central difference
│   ├── sbp_streaming_prediction_trajectory.png        # Real-time physical SBP estimation trajectory
│   ├── dbp_streaming_prediction_trajectory.png        # Real-time physical DBP estimation trajectory
│   ├── single_session_bland_altman_agreement.png      # Single-session physical reference agreement
│   ├── pooled_bland_altman.png                        # Multi-subject pooled Bland-Altman agreement
│   ├── pooled_scatter_sbp_dbp.png                     # Multi-subject pooled scatter plot against reference cuff
│   └── subject_wise_bp_comparison.png                 # Cross-subject variability and tracking accuracy
└── reliability_triage/                                # Multi-Domain Reliability & Triage Figures
    ├── reliability_calibration.png                    # Reliability score risk calibration curve
    ├── reliability_feature_importance.png             # Feature importances across the 4 reliability domains
    ├── risk_coverage_sbp.png                          # SBP Risk-Coverage curve across operating thresholds
    └── risk_coverage_dbp.png                          # DBP Risk-Coverage curve across operating thresholds
```

---

## 2. Key Figure Details

### Deep Neural Representation & Temporal Dynamics
- **`sbp_predicted_vs_reference_scatter.png` & `dbp_predicted_vs_reference_scatter.png`**:
  - Test set correlation with invasive catheter ground-truth. SBP achieves $r = 0.88$ ($R^2 = 0.77$); DBP achieves $r = 0.85$ ($R^2 = 0.72$).
- **`sbp_bland_altman_agreement.png` & `dbp_bland_altman_agreement.png`**:
  - SBP mean bias $+0.21\text{ mmHg}$ ($95\%\text{ LoA}: [-14.7, +15.1]\text{ mmHg}$); DBP mean bias $+0.08\text{ mmHg}$ ($95\%\text{ LoA}: [-9.6, +9.8]\text{ mmHg}$).

### Uncertainty Quantification & Conformal Calibration
- **`sbp_isotonic_calibration.png` & `dbp_isotonic_calibration.png`**:
  - Monotonic piecewise isotonic regression mapping raw neural estimates into calibrated physiological values, counteracting regression-to-the-mean above $140\text{ mmHg}$ and below $90\text{ mmHg}$.

### Physical Sensor Validation
- **`pooled_bland_altman.png` & `pooled_scatter_sbp_dbp.png`**:
  - Evaluates physical MAX30102 sensor recordings paired against calibrated reference cuff events across volunteer test sessions ($\text{SBP MAE} = 5.64\text{ mmHg}$, $\text{DBP MAE} = 3.68\text{ mmHg}$).

### Selective Prediction & Reliability Triage
- **`risk_coverage_sbp.png` & `reliability_calibration.png`**:
  - Demonstrates error reduction as uncertain or artifact-corrupted predictions are quarantined into `REVIEW` or `ABSTAIN`. SBP MAE drops to $3.94\text{ mmHg}$ at $75\%$ coverage.
