# PHASE 4B EVIDENCE FREEZE: TEMPORAL CONTEXT EXTENSION

- **Timestamp:** 2026-09-18 14:13:56
- **Dataset Path:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/BloodPressureDataset
- **Manifest Path:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/windows/window_manifest.csv.gz
- **Phase 4A Checkpoint:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt
- **Original Eligible Windows:** Train = 183,517, Val = 39,461, Test = 38,361
- **Valid 60-sec Sequences:** Train = 149,500, Val = 32,163, Test = 31,192
- **Sequence Retention:** Train = 81.46%, Val = 81.51%, Test = 81.31%
- **Sampling Frequency:** Fs = 125 Hz
- **Window Length:** 10 seconds (1,250 samples)
- **Sequence Length:** 6 windows (60 seconds total history)
- **Ordering:** Strictly causal (oldest to newest: t-5 -> t)
- **CNN Parameters:** 146,978 (FROZEN: 0 trainable)
- **Trainable Temporal Parameters:** 27,106
- **GRU Architecture:** 1-layer unidirectional GRU (hidden=64) + FC(64->32) + Dual Linear Heads
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Batch Size:** 256
- **Random Seed:** 42
- **Best Validation Epoch:** 5
- **Best Validation Combined MAE:** 8.0324 mmHg
- **Phase 4B Matched Test SBP MAE:** 10.5662 mmHg
- **Phase 4B Matched Test DBP MAE:** 5.5146 mmHg
- **Phase 4B Matched Test Comb MAE:** 8.0404 mmHg
- **Phase 4A Matched Test SBP MAE:** 10.9417 mmHg
- **Phase 4A Matched Test DBP MAE:** 5.6943 mmHg
- **Phase 4A Matched Test Comb MAE:** 8.3180 mmHg
- **Temporal Gain (SBP):** +0.3755 mmHg
- **Temporal Gain (DBP):** +0.1797 mmHg
- **Temporal Gain (Combined):** +0.2776 mmHg
- **Checkpoint Path:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt
- **Environment:** Python 3.10.21, PyTorch 2.14.0+cu130, CUDA 13.0

Phase 4A CNN weights were frozen. Phase 4B trained only one temporal GRU regression model.
