# Phase 7 Frozen Artifact Checksum Verification

**Verification Timestamp**: 2026-09-30  
**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Purpose**: Cryptographic proof that Phase 7 reliability engine development did NOT alter any frozen neural models or Phase 5C calibration artifacts, and to record Phase 7 research artifact integrity.

---

## 1. Frozen Base Artifacts — SHA-256 (Must NEVER Change)

| Component | File Path | Status | Parameter Count | File Size (Bytes) | SHA-256 Cryptographic Hash |
|:---|:---|:---:|:---:|:---:|:---|
| **Phase 4A 1D CNN** | `code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt` | **FROZEN** | 146,978 (0 trainable) | 2,665,005 | `2c6c5478e5ec0c666cd2d5c43d7d2561c74ad25170f10075f24d2fa6610863ee` |
| **Phase 4B Causal GRU** | `code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt` | **FROZEN** | 27,106 (0 trainable) | 1,047,138 | `26ecb0abf690bd067c5a083f80487e8f686c5c66ba49b9b84c7895fbaf8dcd81` |
| **Phase 5C Isotonic SBP** | `code/outputs/phase5c_extreme_aware/mappings/isotonic_sbp.pkl` | **FROZEN** | Read-Only Mapping | 2,054 | `44d25b279fcde22992b4c323de1ed6e5fa4595ffdcc7a75523a236fdd356a6bd` |
| **Phase 5C Isotonic DBP** | `code/outputs/phase5c_extreme_aware/mappings/isotonic_dbp.pkl` | **FROZEN** | Read-Only Mapping | 1,926 | `5932346ce2c55188ca885cb489100e4fab1a4b9a6973b0ef4153a9ba42c67828` |
| **Phase 5C Conformal Bins** | `code/outputs/phase5c_extreme_aware/calibration/conformal_quantiles_by_bin.json` | **FROZEN** | Read-Only Quantiles | 2,032 | `e0e50101cfa216d75afa447c07e1f1f9a1ab15cde3a4069718f7b33907c7f2d8` |

---

## 2. Phase 7 Research Artifacts — SHA-256 (Frozen Research Results)

| Component | File Path | Status | Description | File Size (Bytes) | SHA-256 Cryptographic Hash |
|:---|:---|:---:|:---|:---:|:---|
| **Phase 7 Reliability Engine** | `code/outputs/phase7_reliability/reliability_engine_model.pkl` | **FROZEN** | Fitted HistGB; tau_trust=0.5327, tau_abstain=0.6608 | 264,453 | `fe8ecd804848abcbb1125167991a1a61a14efaf5c49e13260fa8e39505616c4a` |
| **Phase 7 Reliability Dataset** | `code/outputs/phase7_reliability/reliability_dataset.csv` | **FROZEN** | 63,355 rows × 50 cols; dev/val/test split | 34,631,178 | `6a1027f0391f0ef806d6692c68272ecb9c3323482cd0f10f06c22f4e6e8c79ca` |
| **Model Comparison CSV** | `code/outputs/phase7_reliability/reliability_model_comparison.csv` | **FROZEN** | Rule-based / Logistic / HistGB benchmark | — | `356561d23b51ab4628fc80fbcd7b4ce86b26b604a31ece1817f64f2b466a43ca` |
| **Selective Prediction CSV** | `code/outputs/phase7_reliability/selective_prediction_metrics.csv` | **FROZEN** | Coverage 100%→50% metrics table | — | `164ae08c1d76fb2502ddba34837959b570dee239cf7b0ba19157d645ca229f97` |
| **Test Predictions CSV** | `code/outputs/phase7_reliability/reliability_test_predictions.csv` | **FROZEN** | Per-prediction scores and states for test partition | 5,767,576 | `b6bb6682b2e736a8947211acc9cf3edead5e3e7e975f351a5162f1306321f291` |

---

## 3. Integrity Confirmation

1. **Zero Model Modifications**: No layers, weights, biases, or architectures of the Phase 4A CNN or Phase 4B GRU were modified or retrained.
2. **Zero Calibration Shift**: Phase 5C extreme-aware isotonic curves and conformal quantiles remain bit-for-bit identical to their frozen state.
3. **Additive Research Extension**: Phase 7 operates strictly downstream as an independent reliability and selective-abstention layer.
4. **Phase 7 Research Results Frozen**: The reliability engine model (`reliability_engine_model.pkl`) and research dataset (`reliability_dataset.csv`) are frozen — do not regenerate.

---

## 4. Verification Command

To re-verify all frozen base artifacts at any time:

```bash
sha256sum \
  code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt \
  code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt \
  code/outputs/phase5c_extreme_aware/mappings/isotonic_sbp.pkl \
  code/outputs/phase5c_extreme_aware/mappings/isotonic_dbp.pkl \
  code/outputs/phase5c_extreme_aware/calibration/conformal_quantiles_by_bin.json
```

To verify Phase 7 research artifacts:

```bash
sha256sum \
  code/outputs/phase7_reliability/reliability_engine_model.pkl \
  code/outputs/phase7_reliability/reliability_dataset.csv \
  code/outputs/phase7_reliability/reliability_model_comparison.csv \
  code/outputs/phase7_reliability/selective_prediction_metrics.csv
```
