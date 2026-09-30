# PHASE 5A EVIDENCE FREEZE: UNCERTAINTY ESTIMATION WITHOUT RETRAINING

- **Timestamp:** 2026-09-18 14:35:54
- **Phase 4B Checkpoint:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt
- **Test Sequences:** 31,192 (from /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/sequences/test_sequences.npz)
- **MC Passes:** 30
- **Dropout Probability:** 0.2
- **CNN Parameters Frozen:** 146,978
- **GRU Parameters Frozen:** 27,106
- **Trainable Parameters:** 0 (STRICT INFERENCE-ONLY)
- **Deterministic Checkpoint Verification:** SBP MAE = 10.5662 mmHg, DBP MAE = 5.5146 mmHg, Comb MAE = 8.0404 mmHg
- **SBP Uncertainty-Error Correlation:** Pearson r = 0.0806, Spearman rho = 0.0860
- **DBP Uncertainty-Error Correlation:** Pearson r = 0.1382, Spearman rho = 0.1336
- **SBP 100% Coverage MAE:** 10.8233 mmHg
- **SBP 80% Coverage MAE:** 10.6190 mmHg
- **DBP 100% Coverage MAE:** 5.6625 mmHg
- **DBP 80% Coverage MAE:** 5.3157 mmHg
- **Environment:** Python 3.10.21, PyTorch 2.14.0+cu130, CUDA 13.0

Phase 5A was inference-only. No neural network parameters were trained or updated.
