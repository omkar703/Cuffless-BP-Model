# Phase 7: Prediction Reliability & Selective Abstention Engine
**Project**: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

This directory contains the complete research artifacts, datasets, models, and evaluations for Phase 7.

## Key Files:
- `reliability_dataset.csv`: 63,355 rows combining predictions, errors, signal QC, MC uncertainty, conformal intervals, and causal temporal features.
- `reliability_dataset_schema.json`: Complete data dictionary and domain metadata.
- `reliability_model_comparison.csv`: Comparative metrics across Rule-Based, Logistic Regression, and HistGradientBoosting classifiers.
- `selective_prediction_metrics.csv`: Retained MAE/RMSE across coverage levels (100% to 50%).
- `risk_coverage_sbp.png`: SBP risk-coverage curve.
- `risk_coverage_dbp.png`: DBP risk-coverage curve.
- `reliability_feature_importance.png`: Permutation feature importance ranking.
- `reliability_calibration.png`: Reliability calibration plot.
- `llm_interface_payload_sample.json`: Structured interface boundary for future LLM explanation layer.
- `phase7_reliability_report.md`: Comprehensive formal research report.
- `phase7_reliability_report.json`: Machine-readable results summary.

## Execution Command:
```bash
./.venv/bin/python code/scripts/run_phase7_reliability.py
```
