# PHASE 5C EVIDENCE FREEZE: EXTREME-BP-AWARE CONFORMAL CALIBRATION

- **Timestamp:** 2026-09-18 15:41:31
- **Phase 4B Checkpoint:** /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt
- **Calibration Records:** 607 (16,298 sequences)
- **Audit Records:** 608 (15,865 sequences)
- **Test Sequences:** 31,192 (Untouched)
- **Verified Test Contributing Records:** 1,198 (Records with $\ge 6$ windows)
- **Verified Test Window Embedding Records:** 1,621 (Single-window level)
- **Isotonic Config:** `increasing=True, out_of_bounds='clip'`
- **SBP Prediction Bins:** `<120`, `120-139`, `>=140` (Deterministic merge of sparse `>=160` bin)
- **DBP Prediction Bins:** `<60`, `60-79`, `>=80` (Deterministic merge of sparse `>=90` bin)
- **Nominal Coverage Levels:** 90% ($\alpha = 0.10$, $\alpha_{{tail}} = 0.05$), 95% ($\alpha = 0.05$, $\alpha_{{tail}} = 0.025$)
- **Conformal Quantile Rule:** Finite-sample order statistic $k = \lceil (n+1)(1 - \alpha_{{tail}}) \rceil$
- **Trainable Parameters:** 0 (STRICT INFERENCE-ONLY)
- **Deterministic Baseline Check:** SBP MAE = 10.5662 mmHg, DBP MAE = 5.5146 mmHg, Comb MAE = 8.0404 mmHg
- **Phase 5C SBP 95% Test Coverage:** 95.00% (Mean Width: 61.42 mmHg)
- **Phase 5C DBP 95% Test Coverage:** 93.22% (Mean Width: 30.43 mmHg)
- **Phase 5C SBP >= 160 Coverage:** 79.67%
- **Phase 5C DBP >= 100 Coverage:** 23.59%

Phase 5C was completely post-hoc and inference-only. No neural network parameters were trained or updated.
