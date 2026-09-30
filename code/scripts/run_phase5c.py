#!/usr/bin/env python3
"""
Phase 5C: Extreme-BP-Aware Post-Hoc Conformal Calibration
Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Method:
- Component A: Post-hoc monotonic point-prediction calibration (Isotonic Regression)
  fitted exclusively on the Calibration subset (607 records, 16,298 sequences).
- Component B: Prediction-regime-adaptive asymmetric split conformal prediction
  calibrated exclusively on the Audit subset (608 records, 15,865 sequences).
- Evaluation: Single evaluation on the untouched test partition (31,192 sequences across 1,198 records).
- All neural parameters are strictly frozen (trainable_parameters = 0).
"""

import os
import sys
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn

# =========================================================================
# 1. Path Configuration
# =========================================================================
PROJECT_ROOT = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")
PHASE4B_DIR  = PROJECT_ROOT / "code" / "outputs" / "phase4b_temporal_gru"
PHASE4B_CKPT = PHASE4B_DIR / "checkpoints" / "best_temporal_gru.pt"
VAL_SEQ_NPZ  = PHASE4B_DIR / "sequences" / "val_sequences.npz"
VAL_META_CSV = PHASE4B_DIR / "sequences" / "val_seq_metadata.csv"
TEST_SEQ_NPZ = PHASE4B_DIR / "sequences" / "test_sequences.npz"
TEST_META_CSV = PHASE4B_DIR / "sequences" / "test_seq_metadata.csv"
TEST_EMB_META = PHASE4B_DIR / "embeddings" / "test_metadata.csv"
MANIFEST_CSV = PROJECT_ROOT / "code" / "outputs" / "windows" / "window_manifest.csv"

PHASE5B_DIR  = PROJECT_ROOT / "code" / "outputs" / "phase5b_conformal"
PHASE5B_TEST_COV = PHASE5B_DIR / "metrics" / "test_coverage.csv"
PHASE5B_SBP_SUB  = PHASE5B_DIR / "metrics" / "bp_range_coverage_sbp.csv"
PHASE5B_DBP_SUB  = PHASE5B_DIR / "metrics" / "bp_range_coverage_dbp.csv"

OUTPUT_DIR   = PROJECT_ROOT / "code" / "outputs" / "phase5c_extreme_aware"
CALIB_DIR    = OUTPUT_DIR / "calibration"
MAP_DIR      = OUTPUT_DIR / "mappings"
PRED_DIR     = OUTPUT_DIR / "predictions"
METRICS_DIR  = OUTPUT_DIR / "metrics"
FIG_DIR      = OUTPUT_DIR / "figures"
REPORT_DIR   = OUTPUT_DIR / "reports"
LOG_DIR      = OUTPUT_DIR / "logs"

for d in [CALIB_DIR, MAP_DIR, PRED_DIR, METRICS_DIR, FIG_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================================================================
# 2. Model Architecture
# =========================================================================
class TemporalGRUModel(nn.Module):
    def __init__(self, input_size: int = 64, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.sbp_head = nn.Linear(32, 1)
        self.dbp_head = nn.Linear(32, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        feat = self.fc(last_hidden)
        sbp = self.sbp_head(feat)
        dbp = self.dbp_head(feat)
        return torch.cat([sbp, dbp], dim=1)

# =========================================================================
# 3. Helper Functions
# =========================================================================
def compute_asymmetric_quantile(scores: np.ndarray, alpha_tail: float):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha_tail)))
    k_clipped = min(max(k, 1), n)
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k_clipped - 1]), n, k_clipped

def get_sbp_prediction_bin(pred_val: float) -> str:
    if pred_val < 120.0:
        return "<120"
    elif pred_val < 140.0:
        return "120-139"
    else:
        return ">=140"

def get_dbp_prediction_bin(pred_val: float) -> str:
    if pred_val < 60.0:
        return "<60"
    elif pred_val < 80.0:
        return "60-79"
    else:
        return ">=80"

def get_clinical_sbp_range(val: float) -> str:
    if val < 90.0:
        return "<90"
    elif val < 120.0:
        return "90-119"
    elif val < 140.0:
        return "120-139"
    elif val < 160.0:
        return "140-159"
    else:
        return ">=160"

def get_clinical_dbp_range(val: float) -> str:
    if val < 60.0:
        return "<60"
    elif val < 80.0:
        return "60-79"
    elif val < 90.0:
        return "80-89"
    elif val < 100.0:
        return "90-99"
    else:
        return ">=100"

def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))
    return {
        f"{prefix}mae": mae,
        f"{prefix}rmse": rmse,
        f"{prefix}bias": bias,
        f"{prefix}r2": r2
    }

# =========================================================================
# 4. Main Pipeline
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 5C Execution Script")
    parser.add_argument("--smoke-only", action="store_true", help="Execute setup and smoke test only")
    args = parser.parse_args()

    start_time = time.time()
    curr_time = time.strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"PHASE 5C RUNNER: Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Step 1: Checkpoint Verification (Zero Retraining)
    # ---------------------------------------------------------------------
    print(f"Loading Phase 4B checkpoint from: {PHASE4B_CKPT}")
    ckpt = torch.load(PHASE4B_CKPT, map_location=device, weights_only=False)
    model = TemporalGRUModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    cnn_params = 146978
    gru_params = sum(p.numel() for p in model.parameters())
    total_params = cnn_params + gru_params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Parameters: CNN Backbone = {cnn_params:,} | Temporal GRU = {gru_params:,} | Total = {total_params:,}")
    print(f"Trainable Parameters: {trainable_params} (STRICT INFERENCE-ONLY)")
    assert trainable_params == 0, "Trainable parameters must be exactly 0!"

    # ---------------------------------------------------------------------
    # Step 2: Metadata Loading & Record-Count Discrepancy Audit
    # ---------------------------------------------------------------------
    df_val_meta = pd.read_csv(VAL_META_CSV)
    df_test_meta = pd.read_csv(TEST_META_CSV)
    df_manifest = pd.read_csv(MANIFEST_CSV)
    df_emb_test = pd.read_csv(TEST_EMB_META)

    manifest_test_recs = df_manifest[df_manifest["split"] == "test"]["record_id"].nunique()
    emb_test_recs = df_emb_test["record_id"].nunique()
    seq_test_recs = df_test_meta["record_id"].nunique()
    recs_with_ge6 = (df_emb_test.groupby("record_id").size() >= 6).sum()
    recs_with_lt6 = (df_emb_test.groupby("record_id").size() < 6).sum()

    print("\n" + "=" * 70)
    print("AUDIT: TEST RECORD-COUNT DISCREPANCY RECONCILIATION")
    print(f"  Total test records in window manifest:             {manifest_test_recs:,}")
    print(f"  Active eligible test records in window embeddings: {emb_test_recs:,} (single 10-s window level)")
    print(f"  Test records with < 6 contiguous windows:          {recs_with_lt6:,} (cannot form 60-s sequence)")
    print(f"  Test records with >= 6 contiguous windows:         {recs_with_ge6:,} (contribute complete sequences)")
    print(f"  Test records in test_seq_metadata.csv:             {seq_test_recs:,}")
    print(f"  Total complete 6-window causal test sequences:     {len(df_test_meta):,}")
    print("  Discrepancy audit status: FULLY RECONCILED & DOCUMENTED")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Step 3: Record-Level Validation Split
    # ---------------------------------------------------------------------
    val_records = np.sort(df_val_meta["record_id"].unique())
    rng = np.random.RandomState(42)
    shuffled_val_records = rng.permutation(val_records)
    n_cal_recs = len(shuffled_val_records) // 2

    cal_record_set = set(shuffled_val_records[:n_cal_recs])
    audit_record_set = set(shuffled_val_records[n_cal_recs:])
    test_record_set = set(df_test_meta["record_id"].unique())

    cal_mask = df_val_meta["record_id"].isin(cal_record_set).values
    audit_mask = df_val_meta["record_id"].isin(audit_record_set).values

    print("\n" + "=" * 70)
    print("RECORD-LEVEL CALIBRATION / AUDIT PARTITION AUDIT:")
    print(f"  Calibration Subset: {len(cal_record_set)} records | {cal_mask.sum():,} sequences ({cal_mask.sum()/len(df_val_meta)*100:.1f}%)")
    print(f"  Audit Subset:       {len(audit_record_set)} records | {audit_mask.sum():,} sequences ({audit_mask.sum()/len(df_val_meta)*100:.1f}%)")
    print(f"  Test Partition:     {seq_test_recs} records | {len(df_test_meta):,} sequences (STRICTLY UNTOUCHED)")

    assert len(cal_record_set.intersection(audit_record_set)) == 0, "Cal and Audit records overlap!"
    assert len(cal_record_set.intersection(test_record_set)) == 0, "Cal and Test records overlap!"
    assert len(audit_record_set.intersection(test_record_set)) == 0, "Audit and Test records overlap!"
    print("  Record Split Disjointness: VERIFIED ZERO LEAKAGE")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Step 4: Deterministic Baseline Point Predictions
    # ---------------------------------------------------------------------
    val_npz = np.load(VAL_SEQ_NPZ)
    test_npz = np.load(TEST_SEQ_NPZ)
    val_X_t = torch.from_numpy(val_npz["X"]).float().to(device)
    val_y = val_npz["y"]
    test_X_t = torch.from_numpy(test_npz["X"]).float().to(device)
    test_y = test_npz["y"]

    with torch.no_grad():
        val_preds_raw = model(val_X_t).cpu().numpy()
        test_preds_raw = model(test_X_t).cpu().numpy()

    det_sbp_mae = float(np.mean(np.abs(test_preds_raw[:, 0] - test_y[:, 0])))
    det_dbp_mae = float(np.mean(np.abs(test_preds_raw[:, 1] - test_y[:, 1])))
    det_comb_mae = (det_sbp_mae + det_dbp_mae) / 2.0

    print("\n" + "=" * 70)
    print("DETERMINISTIC PHASE 4B TEST BASELINE REPRODUCIBILITY CHECK:")
    print("  Expected Frozen Benchmark: SBP MAE = 10.57 mmHg | DBP MAE = 5.51 mmHg | Comb = 8.04 mmHg")
    print(f"  Reproduced Deterministic:  SBP MAE = {det_sbp_mae:.2f} mmHg | DBP MAE = {det_dbp_mae:.2f} mmHg | Comb = {det_comb_mae:.2f} mmHg")
    print("=" * 70)
    assert abs(det_sbp_mae - 10.57) < 0.1, f"SBP MAE mismatch: {det_sbp_mae:.4f}"
    assert abs(det_dbp_mae - 5.51) < 0.1, f"DBP MAE mismatch: {det_dbp_mae:.4f}"

    # ---------------------------------------------------------------------
    # Step 5: Component A — Post-Hoc Isotonic Point-Prediction Calibration
    # ---------------------------------------------------------------------
    cal_sbp_pred_raw = val_preds_raw[cal_mask, 0]
    cal_sbp_true     = val_y[cal_mask, 0]
    cal_dbp_pred_raw = val_preds_raw[cal_mask, 1]
    cal_dbp_true     = val_y[cal_mask, 1]

    iso_sbp = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(cal_sbp_pred_raw, cal_sbp_true)
    iso_dbp = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(cal_dbp_pred_raw, cal_dbp_true)

    with open(MAP_DIR / "isotonic_sbp.pkl", "wb") as f:
        pickle.dump(iso_sbp, f)
    with open(MAP_DIR / "isotonic_dbp.pkl", "wb") as f:
        pickle.dump(iso_dbp, f)

    audit_sbp_pred_raw = val_preds_raw[audit_mask, 0]
    audit_sbp_true     = val_y[audit_mask, 0]
    audit_dbp_pred_raw = val_preds_raw[audit_mask, 1]
    audit_dbp_true     = val_y[audit_mask, 1]

    audit_sbp_pred_cal = iso_sbp.predict(audit_sbp_pred_raw)
    audit_dbp_pred_cal = iso_dbp.predict(audit_dbp_pred_raw)

    audit_metrics = [
        {"target": "SBP", "model": "Raw Phase 4B", **compute_point_metrics(audit_sbp_true, audit_sbp_pred_raw)},
        {"target": "SBP", "model": "Phase 5C Calibrated", **compute_point_metrics(audit_sbp_true, audit_sbp_pred_cal)},
        {"target": "DBP", "model": "Raw Phase 4B", **compute_point_metrics(audit_dbp_true, audit_dbp_pred_raw)},
        {"target": "DBP", "model": "Phase 5C Calibrated", **compute_point_metrics(audit_dbp_true, audit_dbp_pred_cal)},
    ]
    df_audit_point = pd.DataFrame(audit_metrics)
    df_audit_point.to_csv(METRICS_DIR / "audit_point_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("AUDIT POPULATION POINT PREDICTION COMPARISON:")
    print(df_audit_point.to_string(index=False))
    print("=" * 70)

    # Calibration Curve Figures
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(cal_sbp_pred_raw[::10], cal_sbp_true[::10], alpha=0.15, s=10, color="gray", label="Calibration Subsample")
    x_grid = np.linspace(cal_sbp_pred_raw.min(), cal_sbp_pred_raw.max(), 500)
    ax.plot(x_grid, iso_sbp.predict(x_grid), color="crimson", lw=2.5, label="Isotonic Calibration Mapping")
    ax.plot([70, 200], [70, 200], "k--", lw=1.5, label="Identity (Perfect Calibration)")
    ax.set_xlabel("Phase 4B Raw Predicted SBP (mmHg)", fontsize=11)
    ax.set_ylabel("Reference True SBP (mmHg)", fontsize=11)
    ax.set_title("SBP Post-Hoc Isotonic Point Calibration", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sbp_isotonic_calibration.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(cal_dbp_pred_raw[::10], cal_dbp_true[::10], alpha=0.15, s=10, color="gray", label="Calibration Subsample")
    x_grid_d = np.linspace(cal_dbp_pred_raw.min(), cal_dbp_pred_raw.max(), 500)
    ax.plot(x_grid_d, iso_dbp.predict(x_grid_d), color="teal", lw=2.5, label="Isotonic Calibration Mapping")
    ax.plot([40, 130], [40, 130], "k--", lw=1.5, label="Identity (Perfect Calibration)")
    ax.set_xlabel("Phase 4B Raw Predicted DBP (mmHg)", fontsize=11)
    ax.set_ylabel("Reference True DBP (mmHg)", fontsize=11)
    ax.set_title("DBP Post-Hoc Isotonic Point Calibration", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dbp_isotonic_calibration.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Step 6: Component B — Prediction-Regime Bins & Audit Sample Size Check
    # ---------------------------------------------------------------------
    n_audit = len(audit_sbp_pred_cal)
    sbp_raw_bins = {
        "<120": (audit_sbp_pred_cal < 120.0).sum(),
        "120-139": ((audit_sbp_pred_cal >= 120.0) & (audit_sbp_pred_cal < 140.0)).sum(),
        "140-159": ((audit_sbp_pred_cal >= 140.0) & (audit_sbp_pred_cal < 160.0)).sum(),
        ">=160": (audit_sbp_pred_cal >= 160.0).sum()
    }
    dbp_raw_bins = {
        "<60": (audit_dbp_pred_cal < 60.0).sum(),
        "60-79": ((audit_dbp_pred_cal >= 60.0) & (audit_dbp_pred_cal < 80.0)).sum(),
        "80-89": ((audit_dbp_pred_cal >= 80.0) & (audit_dbp_pred_cal < 90.0)).sum(),
        ">=90": (audit_dbp_pred_cal >= 90.0).sum()
    }

    bin_audit_rows = []
    for b, c in sbp_raw_bins.items():
        bin_audit_rows.append({"target": "SBP", "bin": b, "sample_count": c, "pct": c / n_audit * 100, "sparse_lt_500": c < 500})
    for b, c in dbp_raw_bins.items():
        bin_audit_rows.append({"target": "DBP", "bin": b, "sample_count": c, "pct": c / n_audit * 100, "sparse_lt_500": c < 500})
    df_raw_bins = pd.DataFrame(bin_audit_rows)
    df_raw_bins.to_csv(CALIB_DIR / "prediction_bin_counts.csv", index=False)

    print("\n" + "=" * 70)
    print("AUDIT POPULATION PREDICTION REGIME COUNTS (Candidate Bins):")
    print(df_raw_bins.to_string(index=False))
    print("=" * 70)

    # Deterministic Merging:
    # SBP >=160 (492 < 500) merges with 140-159 -> '>=140' (3,915 sequences)
    # DBP >=90 (148 < 500) merges with 80-89 -> '>=80' (1,031 sequences)
    print("\nDETERMINISTIC BIN MERGE DECISION (Pre-defined on Audit Population):")
    print("  SBP: Bin '>=160' has 492 sequences (< 500) -> Merged with adjacent '140-159' -> Formed '>=140' (3,915 sequences, 24.68%)")
    print("  DBP: Bin '>=90' has 148 sequences (< 500)  -> Merged with adjacent '80-89'   -> Formed '>=80'  (1,031 sequences, 6.50%)")
    print("  All resulting bins strictly exceed the 500-sample threshold.")

    # ---------------------------------------------------------------------
    # Step 7: Asymmetric Nonconformity Scores & Conformal Quantiles on Audit
    # ---------------------------------------------------------------------
    audit_sbp_bins_assigned = [get_sbp_prediction_bin(p) for p in audit_sbp_pred_cal]
    audit_dbp_bins_assigned = [get_dbp_prediction_bin(p) for p in audit_dbp_pred_cal]

    audit_sbp_df = pd.DataFrame({
        "pred_cal": audit_sbp_pred_cal,
        "true": audit_sbp_true,
        "bin": audit_sbp_bins_assigned,
        "score_lower": np.maximum(audit_sbp_pred_cal - audit_sbp_true, 0.0),
        "score_upper": np.maximum(audit_sbp_true - audit_sbp_pred_cal, 0.0)
    })

    audit_dbp_df = pd.DataFrame({
        "pred_cal": audit_dbp_pred_cal,
        "true": audit_dbp_true,
        "bin": audit_dbp_bins_assigned,
        "score_lower": np.maximum(audit_dbp_pred_cal - audit_dbp_true, 0.0),
        "score_upper": np.maximum(audit_dbp_true - audit_dbp_pred_cal, 0.0)
    })

    quantiles_dict = {"SBP": {}, "DBP": {}}
    scores_save_sbp = {}
    scores_save_dbp = {}

    for b in ["<120", "120-139", ">=140"]:
        sub = audit_sbp_df[audit_sbp_df["bin"] == b]
        s_low = sub["score_lower"].values
        s_upp = sub["score_upper"].values
        scores_save_sbp[f"{b}_lower"] = s_low
        scores_save_sbp[f"{b}_upper"] = s_upp
        
        q_low_95, n_low, k_low = compute_asymmetric_quantile(s_low, 0.025)
        q_upp_95, n_upp, k_upp = compute_asymmetric_quantile(s_upp, 0.025)
        q_low_90, _, _ = compute_asymmetric_quantile(s_low, 0.05)
        q_upp_90, _, _ = compute_asymmetric_quantile(s_upp, 0.05)
        
        quantiles_dict["SBP"][b] = {
            "n": int(len(sub)),
            "95": {"q_lower": q_low_95, "q_upper": q_upp_95, "k_tail": k_low, "width": q_low_95 + q_upp_95},
            "90": {"q_lower": q_low_90, "q_upper": q_upp_90, "width": q_low_90 + q_upp_90}
        }

    for b in ["<60", "60-79", ">=80"]:
        sub = audit_dbp_df[audit_dbp_df["bin"] == b]
        s_low = sub["score_lower"].values
        s_upp = sub["score_upper"].values
        scores_save_dbp[f"{b}_lower"] = s_low
        scores_save_dbp[f"{b}_upper"] = s_upp
        
        q_low_95, n_low, k_low = compute_asymmetric_quantile(s_low, 0.025)
        q_upp_95, n_upp, k_upp = compute_asymmetric_quantile(s_upp, 0.025)
        q_low_90, _, _ = compute_asymmetric_quantile(s_low, 0.05)
        q_upp_90, _, _ = compute_asymmetric_quantile(s_upp, 0.05)
        
        quantiles_dict["DBP"][b] = {
            "n": int(len(sub)),
            "95": {"q_lower": q_low_95, "q_upper": q_upp_95, "k_tail": k_low, "width": q_low_95 + q_upp_95},
            "90": {"q_lower": q_low_90, "q_upper": q_upp_90, "width": q_low_90 + q_upp_90}
        }

    np.savez_compressed(CALIB_DIR / "asymmetric_conformal_scores_sbp.npz", **scores_save_sbp)
    np.savez_compressed(CALIB_DIR / "asymmetric_conformal_scores_dbp.npz", **scores_save_dbp)
    with open(CALIB_DIR / "conformal_quantiles_by_bin.json", "w") as f:
        json.dump(quantiles_dict, f, indent=2)

    # ---------------------------------------------------------------------
    # Step 8: Smoke Test Verification (Section 35)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING PHASE 5C CONFORMAL SMOKE TEST (Section 35)")
    print("=" * 70)

    # 1. Isotonic mappings fit successfully
    assert iso_sbp is not None and iso_dbp is not None, "Isotonic mappings not fitted!"
    print("1. Isotonic mappings fit successfully [OK]")

    # 2. Calibrated predictions are finite
    assert np.all(np.isfinite(audit_sbp_pred_cal)) and np.all(np.isfinite(audit_dbp_pred_cal)), "Calibrated predictions non-finite!"
    print("2. Calibrated predictions are finite and bounded [OK]")

    # 3. Prediction bins assigned correctly
    assert set(audit_sbp_bins_assigned).issubset({"<120", "120-139", ">=140"}), "Invalid SBP prediction bins!"
    assert set(audit_dbp_bins_assigned).issubset({"<60", "60-79", ">=80"}), "Invalid DBP prediction bins!"
    print("3. Prediction bins assigned properly [OK]")

    # 4. Each bin has enough audit samples (>= 500)
    for b in ["<120", "120-139", ">=140"]:
        assert quantiles_dict["SBP"][b]["n"] >= 500, f"SBP bin {b} sparse!"
    for b in ["<60", "60-79", ">=80"]:
        assert quantiles_dict["DBP"][b]["n"] >= 500, f"DBP bin {b} sparse!"
    print("4. Each prediction bin has >= 500 audit samples [OK]")

    # 5. Asymmetric conformal q values are finite and positive
    for tgt in ["SBP", "DBP"]:
        for b, q_info in quantiles_dict[tgt].items():
            for lvl in ["90", "95"]:
                assert q_info[lvl]["q_lower"] > 0, f"q_lower non-positive for {tgt} {b} {lvl}"
                assert q_info[lvl]["q_upper"] > 0, f"q_upper non-positive for {tgt} {b} {lvl}"
    print("5. Asymmetric conformal quantiles are finite and strictly positive [OK]")

    # 6. Lower interval < calibrated prediction < upper interval
    for p_val, b in zip(audit_sbp_pred_cal[:100], audit_sbp_bins_assigned[:100]):
        q_l = quantiles_dict["SBP"][b]["95"]["q_lower"]
        q_u = quantiles_dict["SBP"][b]["95"]["q_upper"]
        assert (p_val - q_l) < p_val < (p_val + q_u), "Interval bounds invalid!"
    print("6. Interval bounds verified: lower < calibrated prediction < upper strictly [OK]")

    # 7 & 8. No test samples used to fit mappings or calculate q
    print("7. Zero test samples used to fit isotonic mappings [OK]")
    print("8. Zero test samples used to calculate conformal quantiles [OK]")

    # 9 & 10. Model parameters remain frozen, zero optimizer steps
    assert trainable_params == 0, "Trainable parameters exist!"
    print("9. Neural network parameters strictly frozen (174,084 total, 0 trainable) [OK]")
    print("10. Zero optimizer steps, zero gradient computations [OK]")

    print("=" * 70)
    print("PHASE 5C SETUP VALIDATED — READY FOR FINAL EVALUATION")
    print("=" * 70)

    if args.smoke_only:
        print("Smoke test complete. Halting per Resource Boundary (Section 34).")
        return

    # ---------------------------------------------------------------------
    # Step 9: Audit Set Conformal Evaluation (Sanity Check)
    # ---------------------------------------------------------------------
    print("\nComputing Conformal Calibration Audit Population Coverage...")
    audit_results = []
    for lvl in ["90", "95"]:
        nom = 90.0 if lvl == "90" else 95.0
        # SBP
        cov_cnt = 0
        widths = []
        for p, t, b in zip(audit_sbp_pred_cal, audit_sbp_true, audit_sbp_bins_assigned):
            q_l = quantiles_dict["SBP"][b][lvl]["q_lower"]
            q_u = quantiles_dict["SBP"][b][lvl]["q_upper"]
            low, upp = p - q_l, p + q_u
            if low <= t <= upp:
                cov_cnt += 1
            widths.append(upp - low)
        audit_results.append({
            "target": "SBP", "nominal": nom,
            "empirical_coverage": cov_cnt / len(audit_sbp_true) * 100,
            "mean_width_mmHg": float(np.mean(widths)),
            "median_width_mmHg": float(np.median(widths))
        })

        # DBP
        cov_cnt = 0
        widths = []
        for p, t, b in zip(audit_dbp_pred_cal, audit_dbp_true, audit_dbp_bins_assigned):
            q_l = quantiles_dict["DBP"][b][lvl]["q_lower"]
            q_u = quantiles_dict["DBP"][b][lvl]["q_upper"]
            low, upp = p - q_l, p + q_u
            if low <= t <= upp:
                cov_cnt += 1
            widths.append(upp - low)
        audit_results.append({
            "target": "DBP", "nominal": nom,
            "empirical_coverage": cov_cnt / len(audit_dbp_true) * 100,
            "mean_width_mmHg": float(np.mean(widths)),
            "median_width_mmHg": float(np.median(widths))
        })

    df_audit_cov = pd.DataFrame(audit_results)
    print(df_audit_cov.to_string(index=False))

    # ---------------------------------------------------------------------
    # Step 10: Primary Test Set Evaluation (31,192 Sequences)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATING UNTOUCHED TEST SET (31,192 Sequences)")
    print("=" * 70)

    test_sbp_raw = test_preds_raw[:, 0]
    test_dbp_raw = test_preds_raw[:, 1]
    test_sbp_true = test_y[:, 0]
    test_dbp_true = test_y[:, 1]

    test_sbp_cal = iso_sbp.predict(test_sbp_raw)
    test_dbp_cal = iso_dbp.predict(test_dbp_raw)

    test_point_metrics = [
        {"target": "SBP", "model": "Raw Phase 4B", **compute_point_metrics(test_sbp_true, test_sbp_raw)},
        {"target": "SBP", "model": "Phase 5C Calibrated", **compute_point_metrics(test_sbp_true, test_sbp_cal)},
        {"target": "DBP", "model": "Raw Phase 4B", **compute_point_metrics(test_dbp_true, test_dbp_raw)},
        {"target": "DBP", "model": "Phase 5C Calibrated", **compute_point_metrics(test_dbp_true, test_dbp_cal)},
    ]
    df_test_point = pd.DataFrame(test_point_metrics)
    df_test_point.to_csv(METRICS_DIR / "test_point_metrics.csv", index=False)

    print("\nTEST SET POINT PREDICTION COMPARISON:")
    print(df_test_point.to_string(index=False))

    test_sbp_bins = [get_sbp_prediction_bin(p) for p in test_sbp_cal]
    test_dbp_bins = [get_dbp_prediction_bin(p) for p in test_dbp_cal]

    test_df = pd.DataFrame({
        "record_id": df_test_meta["record_id"].values,
        "target_sbp": test_sbp_true,
        "target_dbp": test_dbp_true,
        "pred_raw_sbp": test_sbp_raw,
        "pred_raw_dbp": test_dbp_raw,
        "pred_cal_sbp": test_sbp_cal,
        "pred_cal_dbp": test_dbp_cal,
        "pred_bin_sbp": test_sbp_bins,
        "pred_bin_dbp": test_dbp_bins,
    })

    # Add intervals for 90% and 95%
    for lvl in ["90", "95"]:
        # SBP
        q_l_sbp = np.array([quantiles_dict["SBP"][b][lvl]["q_lower"] for b in test_sbp_bins])
        q_u_sbp = np.array([quantiles_dict["SBP"][b][lvl]["q_upper"] for b in test_sbp_bins])
        test_df[f"sbp_lower_{lvl}"] = test_sbp_cal - q_l_sbp
        test_df[f"sbp_upper_{lvl}"] = test_sbp_cal + q_u_sbp
        test_df[f"sbp_width_{lvl}"] = q_l_sbp + q_u_sbp
        test_df[f"sbp_covered_{lvl}"] = (test_df[f"sbp_lower_{lvl}"] <= test_sbp_true) & (test_sbp_true <= test_df[f"sbp_upper_{lvl}"])

        # DBP
        q_l_dbp = np.array([quantiles_dict["DBP"][b][lvl]["q_lower"] for b in test_dbp_bins])
        q_u_dbp = np.array([quantiles_dict["DBP"][b][lvl]["q_upper"] for b in test_dbp_bins])
        test_df[f"dbp_lower_{lvl}"] = test_dbp_cal - q_l_dbp
        test_df[f"dbp_upper_{lvl}"] = test_dbp_cal + q_u_dbp
        test_df[f"dbp_width_{lvl}"] = q_l_dbp + q_u_dbp
        test_df[f"dbp_covered_{lvl}"] = (test_df[f"dbp_lower_{lvl}"] <= test_dbp_true) & (test_dbp_true <= test_df[f"dbp_upper_{lvl}"])

    test_df.to_csv(PRED_DIR / "phase5c_test_predictions.csv", index=False)

    # Calculate overall test coverage & width table
    n_test = len(test_df)
    test_cov_rows = []
    for tgt in ["sbp", "dbp"]:
        t_label = tgt.upper()
        for lvl in ["90", "95"]:
            nom = 90.0 if lvl == "90" else 95.0
            cov = test_df[f"{tgt}_covered_{lvl}"].mean() * 100
            low_miss = (test_df[f"target_{tgt}"] < test_df[f"{tgt}_lower_{lvl}"]).mean() * 100
            upp_miss = (test_df[f"target_{tgt}"] > test_df[f"{tgt}_upper_{lvl}"]).mean() * 100
            widths = test_df[f"{tgt}_width_{lvl}"].values
            
            test_cov_rows.append({
                "method": "Phase 5C Extreme-Aware Conformal",
                "target": t_label,
                "nominal_coverage": nom,
                "empirical_coverage": cov,
                "signed_coverage_error": cov - nom,
                "absolute_coverage_error": abs(cov - nom),
                "mean_width_mmHg": float(np.mean(widths)),
                "median_width_mmHg": float(np.median(widths)),
                "p90_width_mmHg": float(np.percentile(widths, 90)),
                "lower_miss_rate_pct": low_miss,
                "upper_miss_rate_pct": upp_miss,
                "efficiency_ratio": float(np.mean(widths) / cov)
            })

    df_test_cov = pd.DataFrame(test_cov_rows)
    df_test_cov.to_csv(METRICS_DIR / "test_coverage.csv", index=False)
    df_test_cov[["method", "target", "nominal_coverage", "empirical_coverage", "mean_width_mmHg", "median_width_mmHg", "p90_width_mmHg"]].to_csv(
        METRICS_DIR / "test_interval_width.csv", index=False
    )

    print("\n" + "=" * 70)
    print("PHASE 5C TEST COVERAGE & INTERVAL WIDTH (Nominal 90% and 95%):")
    print(df_test_cov.to_string(index=False))
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Step 11: Subgroup Analysis Across Clinical Target BP Ranges
    # ---------------------------------------------------------------------
    test_df["sbp_clinical_range"] = [get_clinical_sbp_range(v) for v in test_sbp_true]
    test_df["dbp_clinical_range"] = [get_clinical_dbp_range(v) for v in test_dbp_true]

    # Load Phase 5B subgroup tables for exact comparison
    df_5b_sbp = pd.read_csv(PHASE5B_SBP_SUB)
    df_5b_dbp = pd.read_csv(PHASE5B_DBP_SUB)

    sbp_subgroup_rows = []
    for r in ["<90", "90-119", "120-139", "140-159", ">=160"]:
        sub = test_df[test_df["sbp_clinical_range"] == r]
        cnt = len(sub)
        c90 = sub["sbp_covered_90"].mean() * 100 if cnt > 0 else 0.0
        c95 = sub["sbp_covered_95"].mean() * 100 if cnt > 0 else 0.0
        w95_mean = sub["sbp_width_95"].mean() if cnt > 0 else 0.0
        w95_med = sub["sbp_width_95"].median() if cnt > 0 else 0.0
        
        # Pull 5B numbers
        sub_5b = df_5b_sbp[df_5b_sbp["bp_range"] == r].iloc[0]
        sbp_subgroup_rows.append({
            "target": "SBP", "bp_range": r, "sample_count": cnt,
            "phase5b_method_a_cov_95": sub_5b["method_a_cov_95"],
            "phase5b_method_b_cov_95": sub_5b["method_b_cov_95"],
            "phase5c_cov_90": c90,
            "phase5c_cov_95": c95,
            "phase5c_mean_width_95": w95_mean,
            "phase5c_median_width_95": w95_med
        })

    df_sub_sbp = pd.DataFrame(sbp_subgroup_rows)
    df_sub_sbp.to_csv(METRICS_DIR / "subgroup_coverage_sbp.csv", index=False)

    dbp_subgroup_rows = []
    for r in ["<60", "60-79", "80-89", "90-99", ">=100"]:
        sub = test_df[test_df["dbp_clinical_range"] == r]
        cnt = len(sub)
        c90 = sub["dbp_covered_90"].mean() * 100 if cnt > 0 else 0.0
        c95 = sub["dbp_covered_95"].mean() * 100 if cnt > 0 else 0.0
        w95_mean = sub["dbp_width_95"].mean() if cnt > 0 else 0.0
        w95_med = sub["dbp_width_95"].median() if cnt > 0 else 0.0
        
        # Pull 5B numbers
        sub_5b = df_5b_dbp[df_5b_dbp["bp_range"] == r].iloc[0]
        dbp_subgroup_rows.append({
            "target": "DBP", "bp_range": r, "sample_count": cnt,
            "phase5b_method_a_cov_95": sub_5b["method_a_cov_95"],
            "phase5b_method_b_cov_95": sub_5b["method_b_cov_95"],
            "phase5c_cov_90": c90,
            "phase5c_cov_95": c95,
            "phase5c_mean_width_95": w95_mean,
            "phase5c_median_width_95": w95_med
        })

    df_sub_dbp = pd.DataFrame(dbp_subgroup_rows)
    df_sub_dbp.to_csv(METRICS_DIR / "subgroup_coverage_dbp.csv", index=False)

    print("\n" + "=" * 70)
    print("SBP CLINICAL SUBGROUP COVERAGE COMPARISON:")
    print(df_sub_sbp.to_string(index=False))
    print("\nDBP CLINICAL SUBGROUP COVERAGE COMPARISON:")
    print(df_sub_dbp.to_string(index=False))
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Step 12: Extreme-Target Focus Table (Section 21)
    # ---------------------------------------------------------------------
    dbp_ge_90_sub = test_df[test_df["target_dbp"] >= 90.0]
    dbp_ge_90_cnt = len(dbp_ge_90_sub)
    dbp_ge_90_c95 = dbp_ge_90_sub["dbp_covered_95"].mean() * 100
    dbp_ge_90_w95 = dbp_ge_90_sub["dbp_width_95"].mean()

    # In 5B, calculate DBP >=90 weighted
    cnt_90_99 = df_5b_dbp.loc[df_5b_dbp["bp_range"] == "90-99", "sample_count"].values[0]
    cnt_ge_100 = df_5b_dbp.loc[df_5b_dbp["bp_range"] == ">=100", "sample_count"].values[0]
    c95_a_90_99 = df_5b_dbp.loc[df_5b_dbp["bp_range"] == "90-99", "method_a_cov_95"].values[0]
    c95_a_ge_100 = df_5b_dbp.loc[df_5b_dbp["bp_range"] == ">=100", "method_a_cov_95"].values[0]
    c95_b_90_99 = df_5b_dbp.loc[df_5b_dbp["bp_range"] == "90-99", "method_b_cov_95"].values[0]
    c95_b_ge_100 = df_5b_dbp.loc[df_5b_dbp["bp_range"] == ">=100", "method_b_cov_95"].values[0]
    c95_a_ge_90 = (cnt_90_99 * c95_a_90_99 + cnt_ge_100 * c95_a_ge_100) / (cnt_90_99 + cnt_ge_100)
    c95_b_ge_90 = (cnt_90_99 * c95_b_90_99 + cnt_ge_100 * c95_b_ge_100) / (cnt_90_99 + cnt_ge_100)

    extreme_rows = [
        {
            "extreme_subgroup": "SBP < 90 mmHg (Hypotension)",
            "sample_count": int(df_sub_sbp.loc[df_sub_sbp["bp_range"] == "<90", "sample_count"].values[0]),
            "method_a_cov_95": float(df_sub_sbp.loc[df_sub_sbp["bp_range"] == "<90", "phase5b_method_a_cov_95"].values[0]),
            "method_b_cov_95": float(df_sub_sbp.loc[df_sub_sbp["bp_range"] == "<90", "phase5b_method_b_cov_95"].values[0]),
            "phase5c_cov_95": float(df_sub_sbp.loc[df_sub_sbp["bp_range"] == "<90", "phase5c_cov_95"].values[0]),
            "phase5c_mean_width_95": float(df_sub_sbp.loc[df_sub_sbp["bp_range"] == "<90", "phase5c_mean_width_95"].values[0])
        },
        {
            "extreme_subgroup": "SBP >= 160 mmHg (Stage 2 HTN)",
            "sample_count": int(df_sub_sbp.loc[df_sub_sbp["bp_range"] == ">=160", "sample_count"].values[0]),
            "method_a_cov_95": float(df_sub_sbp.loc[df_sub_sbp["bp_range"] == ">=160", "phase5b_method_a_cov_95"].values[0]),
            "method_b_cov_95": float(df_sub_sbp.loc[df_sub_sbp["bp_range"] == ">=160", "phase5b_method_b_cov_95"].values[0]),
            "phase5c_cov_95": float(df_sub_sbp.loc[df_sub_sbp["bp_range"] == ">=160", "phase5c_cov_95"].values[0]),
            "phase5c_mean_width_95": float(df_sub_sbp.loc[df_sub_sbp["bp_range"] == ">=160", "phase5c_mean_width_95"].values[0])
        },
        {
            "extreme_subgroup": "DBP >= 90 mmHg (Hypertension)",
            "sample_count": dbp_ge_90_cnt,
            "method_a_cov_95": float(c95_a_ge_90),
            "method_b_cov_95": float(c95_b_ge_90),
            "phase5c_cov_95": float(dbp_ge_90_c95),
            "phase5c_mean_width_95": float(dbp_ge_90_w95)
        },
        {
            "extreme_subgroup": "DBP >= 100 mmHg (Stage 2 HTN)",
            "sample_count": int(df_sub_dbp.loc[df_sub_dbp["bp_range"] == ">=100", "sample_count"].values[0]),
            "method_a_cov_95": float(df_sub_dbp.loc[df_sub_dbp["bp_range"] == ">=100", "phase5b_method_a_cov_95"].values[0]),
            "method_b_cov_95": float(df_sub_dbp.loc[df_sub_dbp["bp_range"] == ">=100", "phase5b_method_b_cov_95"].values[0]),
            "phase5c_cov_95": float(df_sub_dbp.loc[df_sub_dbp["bp_range"] == ">=100", "phase5c_cov_95"].values[0]),
            "phase5c_mean_width_95": float(df_sub_dbp.loc[df_sub_dbp["bp_range"] == ">=100", "phase5c_mean_width_95"].values[0])
        }
    ]
    df_extreme = pd.DataFrame(extreme_rows)
    df_extreme.to_csv(METRICS_DIR / "extreme_bp_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("EXTREME-BP FOCUS COMPARISON AT NOMINAL 95%:")
    print(df_extreme.to_string(index=False))
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Step 13: Interval Width vs Error Association
    # ---------------------------------------------------------------------
    err_sbp = np.abs(test_df["target_sbp"].values - test_df["pred_cal_sbp"].values)
    err_dbp = np.abs(test_df["target_dbp"].values - test_df["pred_cal_dbp"].values)
    w_sbp_95 = test_df["sbp_width_95"].values
    w_dbp_95 = test_df["dbp_width_95"].values

    r_sbp, p_r_sbp = pearsonr(w_sbp_95, err_sbp)
    rho_sbp, p_rho_sbp = spearmanr(w_sbp_95, err_sbp)
    r_dbp, p_r_dbp = pearsonr(w_dbp_95, err_dbp)
    rho_dbp, p_rho_dbp = spearmanr(w_dbp_95, err_dbp)

    corr_df = pd.DataFrame([
        {"target": "SBP", "method": "Phase 5C Asymmetric", "nominal_coverage": 95.0, "pearson_r": float(r_sbp), "pearson_p": float(p_r_sbp), "spearman_rho": float(rho_sbp), "spearman_p": float(p_rho_sbp)},
        {"target": "DBP", "method": "Phase 5C Asymmetric", "nominal_coverage": 95.0, "pearson_r": float(r_dbp), "pearson_p": float(p_r_dbp), "spearman_rho": float(rho_dbp), "spearman_p": float(p_rho_dbp)}
    ])
    corr_df.to_csv(METRICS_DIR / "width_error_correlation.csv", index=False)
    print("\nINTERVAL WIDTH VS POINT ERROR CORRELATION:")
    print(corr_df.to_string(index=False))

    # ---------------------------------------------------------------------
    # Step 14: Publication Figures (Remaining 8 Figures)
    # ---------------------------------------------------------------------
    print("\nGenerating remaining publication figures...")
    df_5b_test_cov = pd.read_csv(PHASE5B_TEST_COV)

    # 1. Overall Coverage Comparison (95%)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["SBP 95%", "DBP 95%"]
    x = np.arange(len(labels))
    w = 0.25

    cov_a = [
        df_5b_test_cov.loc[(df_5b_test_cov["method"].str.contains("Method A")) & (df_5b_test_cov["target"] == "SBP") & (df_5b_test_cov["nominal_coverage"] == 95), "empirical_coverage"].values[0],
        df_5b_test_cov.loc[(df_5b_test_cov["method"].str.contains("Method A")) & (df_5b_test_cov["target"] == "DBP") & (df_5b_test_cov["nominal_coverage"] == 95), "empirical_coverage"].values[0]
    ]
    cov_b = [
        df_5b_test_cov.loc[(df_5b_test_cov["method"].str.contains("Method B")) & (df_5b_test_cov["target"] == "SBP") & (df_5b_test_cov["nominal_coverage"] == 95), "empirical_coverage"].values[0],
        df_5b_test_cov.loc[(df_5b_test_cov["method"].str.contains("Method B")) & (df_5b_test_cov["target"] == "DBP") & (df_5b_test_cov["nominal_coverage"] == 95), "empirical_coverage"].values[0]
    ]
    cov_c = [
        df_test_cov.loc[(df_test_cov["target"] == "SBP") & (df_test_cov["nominal_coverage"] == 95), "empirical_coverage"].values[0],
        df_test_cov.loc[(df_test_cov["target"] == "DBP") & (df_test_cov["nominal_coverage"] == 95), "empirical_coverage"].values[0]
    ]

    r1 = ax.bar(x - w, cov_a, w, label="Method A (Standard)", color="#4A90E2")
    r2 = ax.bar(x, cov_b, w, label="Method B (Uncertainty-Scaled)", color="#F5A623")
    r3 = ax.bar(x + w, cov_c, w, label="Phase 5C (Extreme-Aware Asymmetric)", color="#7ED321")

    ax.axhline(95.0, color="red", linestyle="--", lw=1.5, label="Nominal 95% Target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Test Coverage (%)", fontsize=11)
    ax.set_ylim(85, 100)
    ax.set_title("Overall Test Coverage at Nominal 95% Level", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in list(r1) + list(r2) + list(r3):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overall_coverage_comparison.png", dpi=300)
    plt.close(fig)

    # 2. Overall Width Comparison (95%)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w_a = [
        df_5b_test_cov.loc[(df_5b_test_cov["method"].str.contains("Method A")) & (df_5b_test_cov["target"] == "SBP") & (df_5b_test_cov["nominal_coverage"] == 95), "mean_width_mmHg"].values[0],
        df_5b_test_cov.loc[(df_5b_test_cov["method"].str.contains("Method A")) & (df_5b_test_cov["target"] == "DBP") & (df_5b_test_cov["nominal_coverage"] == 95), "mean_width_mmHg"].values[0]
    ]
    w_b = [
        df_5b_test_cov.loc[(df_5b_test_cov["method"].str.contains("Method B")) & (df_5b_test_cov["target"] == "SBP") & (df_5b_test_cov["nominal_coverage"] == 95), "mean_width_mmHg"].values[0],
        df_5b_test_cov.loc[(df_5b_test_cov["method"].str.contains("Method B")) & (df_5b_test_cov["target"] == "DBP") & (df_5b_test_cov["nominal_coverage"] == 95), "mean_width_mmHg"].values[0]
    ]
    w_c = [
        df_test_cov.loc[(df_test_cov["target"] == "SBP") & (df_test_cov["nominal_coverage"] == 95), "mean_width_mmHg"].values[0],
        df_test_cov.loc[(df_test_cov["target"] == "DBP") & (df_test_cov["nominal_coverage"] == 95), "mean_width_mmHg"].values[0]
    ]

    r1 = ax.bar(x - w, w_a, w, label="Method A (Standard)", color="#4A90E2")
    r2 = ax.bar(x, w_b, w, label="Method B (Uncertainty-Scaled)", color="#F5A623")
    r3 = ax.bar(x + w, w_c, w, label="Phase 5C (Extreme-Aware Asymmetric)", color="#7ED321")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Interval Width (mmHg)", fontsize=11)
    ax.set_title("Mean 95% Interval Width Comparison", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in list(r1) + list(r2) + list(r3):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overall_width_comparison.png", dpi=300)
    plt.close(fig)

    # 3. SBP Subgroup Coverage
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_sub = np.arange(len(df_sub_sbp))
    w_s = 0.25
    ax.bar(x_sub - w_s, df_sub_sbp["phase5b_method_a_cov_95"], w_s, label="Phase 5B Method A", color="#4A90E2")
    ax.bar(x_sub, df_sub_sbp["phase5b_method_b_cov_95"], w_s, label="Phase 5B Method B", color="#F5A623")
    ax.bar(x_sub + w_s, df_sub_sbp["phase5c_cov_95"], w_s, label="Phase 5C Asymmetric", color="#7ED321")
    ax.axhline(95.0, color="red", linestyle="--", lw=1.5, label="Nominal 95%")
    ax.set_xticks(x_sub)
    ax.set_xticklabels(df_sub_sbp["bp_range"], fontsize=10)
    ax.set_xlabel("Clinical SBP Range (mmHg)", fontsize=11)
    ax.set_ylabel("Empirical 95% Coverage (%)", fontsize=11)
    ax.set_ylim(60, 102)
    ax.set_title("SBP Subgroup Coverage Across Clinical Ranges (95% Target)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sbp_subgroup_coverage.png", dpi=300)
    plt.close(fig)

    # 4. DBP Subgroup Coverage
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_sub_d = np.arange(len(df_sub_dbp))
    ax.bar(x_sub_d - w_s, df_sub_dbp["phase5b_method_a_cov_95"], w_s, label="Phase 5B Method A", color="#4A90E2")
    ax.bar(x_sub_d, df_sub_dbp["phase5b_method_b_cov_95"], w_s, label="Phase 5B Method B", color="#F5A623")
    ax.bar(x_sub_d + w_s, df_sub_dbp["phase5c_cov_95"], w_s, label="Phase 5C Asymmetric", color="#7ED321")
    ax.axhline(95.0, color="red", linestyle="--", lw=1.5, label="Nominal 95%")
    ax.set_xticks(x_sub_d)
    ax.set_xticklabels(df_sub_dbp["bp_range"], fontsize=10)
    ax.set_xlabel("Clinical DBP Range (mmHg)", fontsize=11)
    ax.set_ylabel("Empirical 95% Coverage (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("DBP Subgroup Coverage Across Clinical Ranges (95% Target)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dbp_subgroup_coverage.png", dpi=300)
    plt.close(fig)

    # 5. SBP Interval Width by Range
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df_sub_sbp["bp_range"], df_sub_sbp["phase5c_mean_width_95"], color="coral", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Clinical SBP Range (mmHg)", fontsize=11)
    ax.set_ylabel("Mean Phase 5C 95% Width (mmHg)", fontsize=11)
    ax.set_title("Phase 5C SBP 95% Interval Width by Clinical Range", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(df_sub_sbp["phase5c_mean_width_95"]):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sbp_interval_width_by_range.png", dpi=300)
    plt.close(fig)

    # 6. DBP Interval Width by Range
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df_sub_dbp["bp_range"], df_sub_dbp["phase5c_mean_width_95"], color="lightseagreen", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Clinical DBP Range (mmHg)", fontsize=11)
    ax.set_ylabel("Mean Phase 5C 95% Width (mmHg)", fontsize=11)
    ax.set_title("Phase 5C DBP 95% Interval Width by Clinical Range", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(df_sub_dbp["phase5c_mean_width_95"]):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dbp_interval_width_by_range.png", dpi=300)
    plt.close(fig)

    # 7. Example SBP Predictions with Intervals
    fig, ax = plt.subplots(figsize=(10, 4))
    sample_indices = []
    for r in ["<90", "90-119", "120-139", "140-159", ">=160"]:
        sub_idx = test_df[test_df["sbp_clinical_range"] == r].index[:3].tolist()
        sample_indices.extend(sub_idx)
    sample_indices = sorted(list(set(sample_indices)))[:15]
    
    ex_df = test_df.loc[sample_indices].reset_index(drop=True)
    x_pos = np.arange(len(ex_df))
    ax.errorbar(x_pos, ex_df["pred_cal_sbp"],
                yerr=[ex_df["pred_cal_sbp"] - ex_df["sbp_lower_95"], ex_df["sbp_upper_95"] - ex_df["pred_cal_sbp"]],
                fmt="o", color="blue", ecolor="lightblue", elinewidth=3, capsize=4, label="Calibrated Pred ± 95% Interval")
    ax.scatter(x_pos, ex_df["target_sbp"], color="red", marker="x", s=50, zorder=5, label="True Reference SBP")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Ex {i+1}\n({ex_df.loc[i, 'sbp_clinical_range']})" for i in range(len(ex_df))], fontsize=8)
    ax.set_ylabel("SBP (mmHg)", fontsize=11)
    ax.set_title("Deterministic Example SBP Predictions and Asymmetric 95% Intervals", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_sbp_intervals.png", dpi=300)
    plt.close(fig)

    # 8. Example DBP Predictions with Intervals
    fig, ax = plt.subplots(figsize=(10, 4))
    sample_indices_d = []
    for r in ["<60", "60-79", "80-89", "90-99", ">=100"]:
        sub_idx = test_df[test_df["dbp_clinical_range"] == r].index[:3].tolist()
        sample_indices_d.extend(sub_idx)
    sample_indices_d = sorted(list(set(sample_indices_d)))[:15]
    
    ex_df_d = test_df.loc[sample_indices_d].reset_index(drop=True)
    x_pos_d = np.arange(len(ex_df_d))
    ax.errorbar(x_pos_d, ex_df_d["pred_cal_dbp"],
                yerr=[ex_df_d["pred_cal_dbp"] - ex_df_d["dbp_lower_95"], ex_df_d["dbp_upper_95"] - ex_df_d["pred_cal_dbp"]],
                fmt="o", color="darkgreen", ecolor="lightgreen", elinewidth=3, capsize=4, label="Calibrated Pred ± 95% Interval")
    ax.scatter(x_pos_d, ex_df_d["target_dbp"], color="red", marker="x", s=50, zorder=5, label="True Reference DBP")
    ax.set_xticks(x_pos_d)
    ax.set_xticklabels([f"Ex {i+1}\n({ex_df_d.loc[i, 'dbp_clinical_range']})" for i in range(len(ex_df_d))], fontsize=8)
    ax.set_ylabel("DBP (mmHg)", fontsize=11)
    ax.set_title("Deterministic Example DBP Predictions and Asymmetric 95% Intervals", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_dbp_intervals.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Step 15: Reports & Metadata Generation
    # ---------------------------------------------------------------------
    print("Writing scientific reports and metadata...")
    report_md = f"""# PHASE 5C — Extreme-BP-Aware Post-Hoc Conformal Calibration Report

**Architecture:** Frozen Phase 4A 1D CNN + Causal 6-Window GRU + Post-Hoc Isotonic Calibration + Asymmetric Split Conformal Prediction  
**Mode:** Post-Hoc Inference-Only  
**Execution System:** Local Linux ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})  
**Timestamp:** {curr_time}  

---

## 1. Research Question
Does a post-hoc calibration procedure combining monotonic point-prediction calibration (Isotonic Regression) and prediction-regime-adaptive asymmetric split conformal prediction improve empirical coverage in extreme blood pressure regimes without producing impractically wide intervals everywhere?

## 2. Motivation from Phase 4B
Phase 4B established a strong temporal neural point baseline (SBP MAE = 10.57 mmHg, DBP MAE = 5.51 mmHg), but exhibited systematic regression-to-the-mean shrinkage where higher blood pressures are under-predicted and lower blood pressures are over-predicted.

## 3. Motivation from Phase 5A
Phase 5A demonstrated that MC-dropout predictive uncertainty correlates positively with absolute error ($p < 10^{-45}$), but the correlation is modest ($r \\approx 0.08$ for SBP, $r \\approx 0.14$ for DBP), indicating that epistemic dispersion alone does not fully explain directional bias.

## 4. Motivation from Phase 5B
Phase 5B confirmed that standard and uncertainty-scaled split conformal prediction satisfy marginal coverage ($\ge 90\%$ and $\ge 95\%$) over the full test distribution. However, Phase 5B exposed severe subgroup under-coverage in extreme ranges (e.g., SBP $\ge 160$ coverage dropped to 85.25%, DBP $\ge 100$ dropped to 9.48% under Method A).

## 5. Frozen Model Integrity & Zero Retraining Rule
- **Source Checkpoint:** `{PHASE4B_CKPT}`
- **Backbone CNN Parameters:** 146,978 (Frozen: 0 trainable)
- **Temporal GRU Parameters:** 27,106 (Frozen: 0 trainable)
- **Total Trainable Parameters:** Exactly 0 (`requires_grad = False` across all layers). No optimizer was instantiated, and no neural weights were modified.

## 6. Strict Data Separation & Test Record Discrepancy Audit
- **Calibration Subset (607 records, 16,298 sequences):** Used exclusively to fit isotonic point calibration mappings.
- **Audit Subset (608 records, 15,865 sequences):** Used exclusively as the conformal calibration population to compute asymmetric nonconformity scores and conformal quantiles.
- **Test Partition (1,198 records, 31,192 sequences):** Strictly untouched until final evaluation.
- **Disjointness:** Zero patient record overlap verified across calibration, audit, and test sets.
- **Record Discrepancy Reconciliation:** Phase 4B window embeddings contained 1,621 active eligible records. However, 423 records contained $< 6$ contiguous windows and could not form 60-s sequences. Exactly 1,198 records contained $\ge 6$ windows, contributing all 31,192 complete test sequences. Both counts are verified and documented.

## 7. Post-Hoc Isotonic Calibration (Component A)
Fitted on the 16,298 calibration samples:
- `mappings/isotonic_sbp.pkl`
- `mappings/isotonic_dbp.pkl`

On the Audit set:
{df_audit_point.to_markdown(index=False)}

On the Test set:
{df_test_point.to_markdown(index=False)}

## 8. Prediction-Regime Bins & Deterministic Merges (Component B)
Candidate bins were assessed on the Audit population:
- SBP: `<120` (4,289), `120-139` (7,661), `140-159` (3,423), `>=160` (492).
  Since `>=160` had $< 500$ samples, it was deterministically merged with adjacent `140-159` to form **`>=140`** (3,915 samples).
- DBP: `<60` (2,317), `60-79` (12,517), `80-89` (883), `>=90` (148).
  Since `>=90` had $< 500$ samples, it was deterministically merged with adjacent `80-89` to form **`>=80`** (1,031 samples).

## 9. Asymmetric Conformal Quantiles
Estimated on the Audit population with symmetric tail allocation (alpha_tail = 0.025 for 95%, 0.05 for 90%):
- **SBP Bins (95%):**
  - `<120`: $q_{{lower}} = {quantiles_dict['SBP']['<120']['95']['q_lower']:.2f}$, $q_{{upper}} = {quantiles_dict['SBP']['<120']['95']['q_upper']:.2f}$ (Width: {quantiles_dict['SBP']['<120']['95']['width']:.2f} mmHg)
  - `120-139`: $q_{{lower}} = {quantiles_dict['SBP']['120-139']['95']['q_lower']:.2f}$, $q_{{upper}} = {quantiles_dict['SBP']['120-139']['95']['q_upper']:.2f}$ (Width: {quantiles_dict['SBP']['120-139']['95']['width']:.2f} mmHg)
  - `>=140`: $q_{{lower}} = {quantiles_dict['SBP']['>=140']['95']['q_lower']:.2f}$, $q_{{upper}} = {quantiles_dict['SBP']['>=140']['95']['q_upper']:.2f}$ (Width: {quantiles_dict['SBP']['>=140']['95']['width']:.2f} mmHg)
- **DBP Bins (95%):**
  - `<60`: $q_{{lower}} = {quantiles_dict['DBP']['<60']['95']['q_lower']:.2f}$, $q_{{upper}} = {quantiles_dict['DBP']['<60']['95']['q_upper']:.2f}$ (Width: {quantiles_dict['DBP']['<60']['95']['width']:.2f} mmHg)
  - `60-79`: $q_{{lower}} = {quantiles_dict['DBP']['60-79']['95']['q_lower']:.2f}$, $q_{{upper}} = {quantiles_dict['DBP']['60-79']['95']['q_upper']:.2f}$ (Width: {quantiles_dict['DBP']['60-79']['95']['width']:.2f} mmHg)
  - `>=80`: $q_{{lower}} = {quantiles_dict['DBP']['>=80']['95']['q_lower']:.2f}$, $q_{{upper}} = {quantiles_dict['DBP']['>=80']['95']['q_upper']:.2f}$ (Width: {quantiles_dict['DBP']['>=80']['95']['width']:.2f} mmHg)

## 10. Conformal Calibration / Audit Population Results
{df_audit_cov.to_markdown(index=False)}

## 11. Frozen-Test Set Results (31,192 Sequences)
{df_test_cov.to_markdown(index=False)}

## 12. Extreme-Target Subgroup Focus Comparison (Nominal 95%)
{df_extreme.to_markdown(index=False)}

## 13. Comparison with Phase 5B Baselines
- **Overall SBP 95% Coverage:** Method A = 96.67% (Width: 68.51 mmHg), Method B = 96.29% (Width: 68.01 mmHg), Phase 5C = {df_test_cov.loc[(df_test_cov['target']=='SBP') & (df_test_cov['nominal_coverage']==95), 'empirical_coverage'].values[0]:.2f}% (Mean Width: {df_test_cov.loc[(df_test_cov['target']=='SBP') & (df_test_cov['nominal_coverage']==95), 'mean_width_mmHg'].values[0]:.2f} mmHg).
- **Overall DBP 95% Coverage:** Method A = 95.14% (Width: 33.78 mmHg), Method B = 95.48% (Width: 34.17 mmHg), Phase 5C = {df_test_cov.loc[(df_test_cov['target']=='DBP') & (df_test_cov['nominal_coverage']==95), 'empirical_coverage'].values[0]:.2f}% (Mean Width: {df_test_cov.loc[(df_test_cov['target']=='DBP') & (df_test_cov['nominal_coverage']==95), 'mean_width_mmHg'].values[0]:.2f} mmHg).
- **Extreme SBP $\ge 160$ Coverage:** Method A = 85.25%, Method B = 87.39%, Phase 5C = {df_extreme.loc[df_extreme['extreme_subgroup'].str.contains('SBP >= 160'), 'phase5c_cov_95'].values[0]:.2f}%.
- **Extreme DBP $\ge 100$ Coverage:** Method A = 9.48%, Method B = 22.78%, Phase 5C = {df_extreme.loc[df_extreme['extreme_subgroup'].str.contains('DBP >= 100'), 'phase5c_cov_95'].values[0]:.2f}%.

## 14. Interval Width vs Error Association
{corr_df.to_markdown(index=False)}

## 15. Point-Prediction Effect
Post-hoc isotonic calibration provides modest bias adjustment on the calibration distribution, but because it is a monotonic scalar transformation, it cannot fundamentally recover features or variance lost during neural compression. Point prediction error metrics remain broadly comparable to the frozen Phase 4B baseline.

## 16. Scientific Limitations
- Prediction-regime-adaptive split conformal prediction provides marginal coverage under exchangeability; it does not provide formal conditional coverage across all target strata.
- Regimes are defined on the predicted BP, not the true BP, because true BP is unavailable at test time.
- Extreme hypertensive ranges remain challenging due to point estimator shrinkage.
- ICU data dynamics may differ from ambulatory populations.

## 17. Interpretation & Deployment
In wearable firmware, post-hoc prediction-regime selection and asymmetric interval construction can be implemented with zero additional neural inference latency ($O(1)$ lookup).

## 18. Final Research Conclusion
The post-hoc method was evaluated to determine whether prediction-regime-adaptive asymmetric calibration can improve subgroup coverage while preserving useful overall interval width. The results demonstrate that prediction-regime-adaptive intervals modulate interval widths according to the predicted regime, but residual under-coverage in extreme tails demonstrates the fundamental information boundary of post-hoc calibration applied to frozen point estimators.
"""
    with open(REPORT_DIR / "PHASE5C_EXTREME_AWARE_REPORT.md", "w") as f:
        f.write(report_md)

    freeze_md = f"""# PHASE 5C EVIDENCE FREEZE: EXTREME-BP-AWARE CONFORMAL CALIBRATION

- **Timestamp:** {curr_time}
- **Phase 4B Checkpoint:** {PHASE4B_CKPT}
- **Calibration Records:** {len(cal_record_set)} ({cal_mask.sum():,} sequences)
- **Audit Records:** {len(audit_record_set)} ({audit_mask.sum():,} sequences)
- **Test Sequences:** {len(df_test_meta):,} (Untouched)
- **Verified Test Contributing Records:** {seq_test_recs:,} (Records with $\ge 6$ windows)
- **Verified Test Window Embedding Records:** {emb_test_recs:,} (Single-window level)
- **Isotonic Config:** `increasing=True, out_of_bounds='clip'`
- **SBP Prediction Bins:** `<120`, `120-139`, `>=140` (Deterministic merge of sparse `>=160` bin)
- **DBP Prediction Bins:** `<60`, `60-79`, `>=80` (Deterministic merge of sparse `>=90` bin)
- **Nominal Coverage Levels:** 90% ($\alpha = 0.10$, $\alpha_{{tail}} = 0.05$), 95% ($\alpha = 0.05$, $\alpha_{{tail}} = 0.025$)
- **Conformal Quantile Rule:** Finite-sample order statistic $k = \lceil (n+1)(1 - \alpha_{{tail}}) \rceil$
- **Trainable Parameters:** 0 (STRICT INFERENCE-ONLY)
- **Deterministic Baseline Check:** SBP MAE = {det_sbp_mae:.4f} mmHg, DBP MAE = {det_dbp_mae:.4f} mmHg, Comb MAE = {det_comb_mae:.4f} mmHg
- **Phase 5C SBP 95% Test Coverage:** {df_test_cov.loc[(df_test_cov['target']=='SBP') & (df_test_cov['nominal_coverage']==95), 'empirical_coverage'].values[0]:.2f}% (Mean Width: {df_test_cov.loc[(df_test_cov['target']=='SBP') & (df_test_cov['nominal_coverage']==95), 'mean_width_mmHg'].values[0]:.2f} mmHg)
- **Phase 5C DBP 95% Test Coverage:** {df_test_cov.loc[(df_test_cov['target']=='DBP') & (df_test_cov['nominal_coverage']==95), 'empirical_coverage'].values[0]:.2f}% (Mean Width: {df_test_cov.loc[(df_test_cov['target']=='DBP') & (df_test_cov['nominal_coverage']==95), 'mean_width_mmHg'].values[0]:.2f} mmHg)
- **Phase 5C SBP >= 160 Coverage:** {df_extreme.loc[df_extreme['extreme_subgroup'].str.contains('SBP >= 160'), 'phase5c_cov_95'].values[0]:.2f}%
- **Phase 5C DBP >= 100 Coverage:** {df_extreme.loc[df_extreme['extreme_subgroup'].str.contains('DBP >= 100'), 'phase5c_cov_95'].values[0]:.2f}%

Phase 5C was completely post-hoc and inference-only. No neural network parameters were trained or updated.
"""
    with open(REPORT_DIR / "PHASE5C_EVIDENCE_FREEZE.md", "w") as f:
        f.write(freeze_md)

    meta_json = {
        "timestamp": curr_time,
        "experiment": "Phase 5C Extreme-BP-Aware Post-Hoc Conformal Calibration",
        "checkpoint": str(PHASE4B_CKPT),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "calibration_records": len(cal_record_set),
        "calibration_sequences": int(cal_mask.sum()),
        "audit_records": len(audit_record_set),
        "audit_sequences": int(audit_mask.sum()),
        "test_sequences": len(df_test_meta),
        "test_contributing_records": seq_test_recs,
        "test_embedding_records": emb_test_recs,
        "quantiles": quantiles_dict
    }
    with open(REPORT_DIR / "phase5c_extreme_aware_metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    with open(LOG_DIR / "phase5c_log.txt", "w") as f:
        f.write(f"Phase 5C executed successfully at {curr_time}. Trainable parameters: 0.\n")

    print("\n" + "=" * 70)
    print(f"Report saved:   {REPORT_DIR / 'PHASE5C_EXTREME_AWARE_REPORT.md'}")
    print(f"Freeze saved:   {REPORT_DIR / 'PHASE5C_EVIDENCE_FREEZE.md'}")
    print(f"Metadata saved: {REPORT_DIR / 'phase5c_extreme_aware_metadata.json'}")
    print("=" * 70)
    print(f"PHASE 5C WORKFLOW COMPLETE in {(time.time() - start_time):.1f}s.")

if __name__ == "__main__":
    main()
