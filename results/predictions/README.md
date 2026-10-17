# Exported Prediction Trajectories & Datasets

This directory archives evaluation prediction outputs exported during model validation.

---

## Files & Schemas

### 1. `conformal_calibrated_predictions.csv`
- **Description**: Point predictions, isotonic calibrated estimates, and 95% conformal bounds on the held-out test split.
- **Key Columns**:
  - `record_id`: Patient record identifier from MIMIC-II.
  - `target_window_index`: 10-second window index.
  - `ground_truth_sbp`, `ground_truth_dbp`: Invasive arterial blood pressure reference (mmHg).
  - `raw_pred_sbp`, `raw_pred_dbp`: Raw uncalibrated output from the 1D-CNN + GRU model.
  - `calibrated_sbp`, `calibrated_dbp`: Non-linear isotonic recalibrated predictions.
  - `conformal_lower_sbp`, `conformal_upper_sbp`: Distribution-free 95% conformal interval bounds.
  - `conformal_lower_dbp`, `conformal_upper_dbp`: Distribution-free 95% conformal interval bounds.

### 2. `reliability_triage_predictions.csv`
- **Description**: Evaluates the 4-domain reliability engine on test predictions.
- **Key Columns**:
  - `reliability_score`: Calibrated risk probability $\in [0, 1]$.
  - `reliability_state`: Operational state (`TRUST`, `REVIEW`, or `ABSTAIN`).
  - `qc_pass`: Binary signal quality gate (1 = Valid, 0 = Rejected).
  - `sbp_uncertainty`: Monte Carlo epistemic standard deviation.
  - `rolling_sbp_std_3`: Backward 3-window rolling temporal dispersion.

### 3. `physical_sensor_streaming_predictions.csv`
- **Description**: Real-time predictions generated from physical MAX30102 sensor hardware sessions during volunteer recording sessions.
