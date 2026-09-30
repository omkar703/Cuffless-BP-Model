# PHASE 5B EVIDENCE FREEZE: POST-HOC CONFORMAL CALIBRATION

- **Timestamp:** 2026-09-18 14:57:22
- **Phase 4B Checkpoint:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt
- **Phase 5A Uncertainty Source:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase5a_uncertainty/predictions/phase5a_uncertainty_predictions.csv
- **Calibration Records:** 607 (16,298 sequences)
- **Audit Records:** 608 (15,865 sequences)
- **Test Sequences:** 31,192 (Untouched)
- **Random Seed:** 42
- **Epsilon:** 1e-06
- **CNN Parameters Frozen:** 146,978
- **GRU Parameters Frozen:** 27,106
- **Trainable Parameters:** 0 (STRICT INFERENCE-ONLY)
- **Point Prediction Check:** SBP MAE = 10.5662 mmHg, DBP MAE = 5.5146 mmHg, Comb MAE = 8.0404 mmHg
- **Method A SBP 95% Empirical Coverage:** 96.67% (Mean Width: 68.51 mmHg)
- **Method B SBP 95% Empirical Coverage:** 96.29% (Mean Width: 68.01 mmHg)
- **Method A DBP 95% Empirical Coverage:** 95.14% (Mean Width: 33.78 mmHg)
- **Method B DBP 95% Empirical Coverage:** 95.48% (Mean Width: 34.17 mmHg)
- **Environment:** Python 3.10.21, PyTorch 2.14.0+cu130, CUDA 13.0

Phase 5B was post-hoc and inference-only. No neural network parameters were trained or updated.
