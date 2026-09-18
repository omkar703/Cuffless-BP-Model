# PHASE 4A SINGLE-MODEL EVIDENCE FREEZE

Only one neural model was trained in Phase 4A:
PPG + VPG + APG 1D CNN.

- **Timestamp:** 2026-09-18 13:28:44
- **Dataset Path:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/BloodPressureDataset
- **Manifest Path:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/windows/window_manifest.csv.gz
- **Split Records:** Train = 7615, Val = 1628, Test = 1621
- **Eligible Windows:** Train = 183517, Val = 39461, Test = 38361
- **Sampling Rate:** Fs = 125 Hz (1,250 samples / 10s)
- **Input Channels:** 3 (PPG, VPG, APG)
- **Architecture:** PPGCNNBaseline (146,978 parameters)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Batch Size:** 128
- **Max Epochs:** 50 (Early stopping patience = 8)
- **Random Seed:** 42
- **Best Validation Epoch:** 48
- **Best Validation SBP MAE:** 11.2883 mmHg
- **Best Validation DBP MAE:** 5.6219 mmHg
- **Best Validation Combined MAE:** 8.4551 mmHg
- **Final Test SBP MAE:** 11.0368 mmHg (RMSE=14.9900, R2=0.5265)
- **Final Test DBP MAE:** 5.7859 mmHg (RMSE=8.5152, R2=0.4478)
- **Final Test Combined MAE:** 8.4114 mmHg
- **Checkpoint Path:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt
- **Environment:** Python 3.10.21, PyTorch 2.14.0+cu130, CUDA 13.0
