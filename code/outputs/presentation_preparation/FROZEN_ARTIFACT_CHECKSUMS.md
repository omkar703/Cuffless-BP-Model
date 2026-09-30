# Frozen Artifact Verification & Integrity Checksums

**Verification Date**: 2026-09-30  
**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Purpose**: Formal cryptographic proof that presentation preparation, documentation updates, and consistency auditing did NOT alter any frozen neural network weights, calibration mappings, or historical experimental outputs.

---

## 1. Deep Learning Model Checkpoints (100% Frozen Invariant)

The trained PyTorch neural network checkpoints remain completely untouched. Both models operate in strict evaluation mode (`eval()`) with 0 trainable parameters:

| Checkpoint Path | Architecture | Parameter Count | Trainable Params | File Size (Bytes) | SHA-256 Cryptographic Checksum |
|:---|:---|:---:|:---:|:---:|:---|
| `code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt` | 1D Multi-Channel CNN (PPG+VPG+APG) | 146,978 | 0 | 2,665,005 | `2c6c5478e5ec0c666cd2d5c43d7d2561c74ad25170f10075f24d2fa6610863ee` |
| `code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt` | 1-Layer Causal Temporal GRU | 27,106 | 0 | 1,047,138 | `26ecb0abf690bd067c5a083f80487e8f686c5c66ba49b9b84c7895fbaf8dcd81` |

**Combined Neural Parameters**: $146,978 + 27,106 = \mathbf{174,084\text{ parameters}}$ (**0 trainable**).

---

## 2. Phase 5C Post-Hoc Calibration & Conformal Artifacts

The isotonic point-calibration regressors and asymmetric conformal calibration tables fitted during Phase 5C remain identical and read-only:

| Artifact Path | Purpose | File Size (Bytes) | SHA-256 Cryptographic Checksum |
|:---|:---|:---:|:---|
| `code/outputs/phase5c_extreme_aware/mappings/isotonic_sbp.pkl` | Isotonic SBP calibration mapping | 2,054 | `44d25b279fcde22992b4c323de1ed6e5fa4595ffdcc7a75523a236fdd356a6bd` |
| `code/outputs/phase5c_extreme_aware/mappings/isotonic_dbp.pkl` | Isotonic DBP calibration mapping | 1,926 | `5932346ce2c55188ca885cb489100e4fab1a4b9a6973b0ef4153a9ba42c67828` |
| `code/outputs/phase5c_extreme_aware/calibration/conformal_quantiles_by_bin.json` | Asymmetric conformal bounds by BP bin | 2,032 | `e0e50101cfa216d75afa447c07e1f1f9a1ab15cde3a4069718f7b33907c7f2d8` |
| `code/outputs/phase5c_extreme_aware/calibration/asymmetric_conformal_scores_sbp.npz`| Audit SBP nonconformity scores | 60,311 | `9aa0b16cfd8edac74fbc65198e7011ed12426b3341725b733e5bd9a7ea0d9e3d` |
| `code/outputs/phase5c_extreme_aware/calibration/asymmetric_conformal_scores_dbp.npz`| Audit DBP nonconformity scores | 59,787 | `06149bbbfd9b9d4cf1a8e66a64b96c44fed95664d721357f32c5c19b8a0a3a0d` |

---

## 3. Physical Pilot Evaluation Deliverables (Phase 6C)

The cross-session matched prediction-reference records generated from the 5 physical hardware sessions:

| Artifact Path | Content Description | Rows | File Size (Bytes) | SHA-256 Cryptographic Checksum |
|:---|:---|:---:|:---:|:---|
| `code/outputs/phase6c_samples_evaluation/cross_session_matched_pairs.csv` | Full-precision matched prediction-reference pairs ($N=7$) | 7 | 2,298 | `3a2f15deb5374e3fc99398555e8e87feb9e5a8de91ea1d6fd0ad20c137ecfe79` |
| `code/outputs/phase6c_samples_evaluation/phase6c_multi_sample_validation_report.json` | Complete machine-readable session audits and metrics | — | 11,670 | `3fc678149588e669399d2e7c7a177e87a3b504a9408e72856f7f0987a864a541` |

---

## 4. Verification Statement

I confirm that:
1. No neural model weights have been re-trained, fine-tuned, or updated.
2. No calibration curves or nonconformity scores have been refitted.
3. No historical metrics or raw hardware logs have been modified, fabricated, or deleted.
4. All presentation preparation files were generated strictly as documentation, organization, and presentation aids.
