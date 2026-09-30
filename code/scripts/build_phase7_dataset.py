"""
Phase 7: Build Reliability Research Dataset
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Constructs a structured multi-domain reliability dataset where each row represents
ONE frozen 60-second causal sequence BP prediction.

Combines:
1. Predictions (Raw Phase 4B & Calibrated Phase 5C SBP/DBP)
2. Ground Truth Targets (Reference SBP/DBP)
3. Prediction Errors (Signed & Absolute)
4. Research High-Error Labels (Thresholds: 10 mmHg & 15 mmHg for SBP/DBP)
5. Signal Quality Indicators (QC Status, PTP, STD, Clipped Fraction, Pulse Count, HR)
6. Epistemic Uncertainty (Phase 5A MC-Dropout Predictive Standard Deviation)
7. Conformal Bounds (Phase 5C Extreme-Aware Asymmetric Quantiles & Widths)
8. Causal Temporal Stability (Rolling Mean, SD, Max Jump, Median Change, Trend Slopes)
9. Record-Level Split Partition (dev: 16,298 | val: 15,865 | test: 31,192)
"""

import sys
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Set paths
PROJECT_ROOT = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")
OUTPUT_DIR = PROJECT_ROOT / "code/outputs/phase7_reliability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_SEQ_NPZ = PROJECT_ROOT / "code/outputs/phase4b_temporal_gru/sequences/val_sequences.npz"
TEST_SEQ_NPZ = PROJECT_ROOT / "code/outputs/phase4b_temporal_gru/sequences/test_sequences.npz"
VAL_META_CSV = PROJECT_ROOT / "code/outputs/phase4b_temporal_gru/sequences/val_seq_metadata.csv"
TEST_META_CSV = PROJECT_ROOT / "code/outputs/phase4b_temporal_gru/sequences/test_seq_metadata.csv"

WINDOW_MANIFEST_CSV = PROJECT_ROOT / "code/outputs/windows/window_manifest.csv"
P4B_CKPT = PROJECT_ROOT / "code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt"
ISO_SBP_PKL = PROJECT_ROOT / "code/outputs/phase5c_extreme_aware/mappings/isotonic_sbp.pkl"
ISO_DBP_PKL = PROJECT_ROOT / "code/outputs/phase5c_extreme_aware/mappings/isotonic_dbp.pkl"
CONFORMAL_JSON = PROJECT_ROOT / "code/outputs/phase5c_extreme_aware/calibration/conformal_quantiles_by_bin.json"

CAL_U_NPY = PROJECT_ROOT / "code/outputs/phase5b_conformal/calibration/cal_uncertainties.npy"
AUDIT_U_NPY = PROJECT_ROOT / "code/outputs/phase5b_conformal/calibration/audit_uncertainties.npy"
TEST_P5A_CSV = PROJECT_ROOT / "code/outputs/phase5a_uncertainty/predictions/phase5a_uncertainty_predictions.csv"
TEST_P5C_CSV = PROJECT_ROOT / "code/outputs/phase5c_extreme_aware/predictions/phase5c_test_predictions.csv"


class TemporalGRUModel(nn.Module):
    def __init__(self, input_size: int = 64, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
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


def get_conformal_widths(sbp_preds: np.ndarray, dbp_preds: np.ndarray, quantiles: dict):
    n = len(sbp_preds)
    sbp_w90 = np.zeros(n, dtype=np.float32)
    sbp_w95 = np.zeros(n, dtype=np.float32)
    dbp_w90 = np.zeros(n, dtype=np.float32)
    dbp_w95 = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        s = sbp_preds[i]
        d = dbp_preds[i]
        
        bin_s = "<120" if s < 120.0 else ("120-139" if s < 140.0 else ">=140")
        bin_d = "<60" if d < 60.0 else ("60-79" if d < 80.0 else ">=80")
        
        sbp_w90[i] = quantiles["SBP"][bin_s]["90"]["width"]
        sbp_w95[i] = quantiles["SBP"][bin_s]["95"]["width"]
        dbp_w90[i] = quantiles["DBP"][bin_d]["90"]["width"]
        dbp_w95[i] = quantiles["DBP"][bin_d]["95"]["width"]
        
    return sbp_w90, sbp_w95, dbp_w90, dbp_w95


def compute_causal_temporal_features(df: pd.DataFrame, sbp_col: str = "pred_cal_sbp", dbp_col: str = "pred_cal_dbp") -> pd.DataFrame:
    """
    Computes backward-looking rolling statistics within each record.
    Strictly causal: row i only sees rows <= i within the same record_id.
    """
    df = df.sort_values(["record_id", "target_window_index"]).copy()
    n = len(df)
    
    rolling_sbp_mean_3 = np.zeros(n, dtype=np.float32)
    rolling_sbp_std_3 = np.zeros(n, dtype=np.float32)
    rolling_dbp_mean_3 = np.zeros(n, dtype=np.float32)
    rolling_dbp_std_3 = np.zeros(n, dtype=np.float32)
    rolling_sbp_std_6 = np.zeros(n, dtype=np.float32)
    rolling_dbp_std_6 = np.zeros(n, dtype=np.float32)
    max_abs_sbp_jump_3 = np.zeros(n, dtype=np.float32)
    max_abs_dbp_jump_3 = np.zeros(n, dtype=np.float32)
    median_abs_sbp_change_3 = np.zeros(n, dtype=np.float32)
    median_abs_dbp_change_3 = np.zeros(n, dtype=np.float32)
    sbp_trend_slope_3 = np.zeros(n, dtype=np.float32)
    dbp_trend_slope_3 = np.zeros(n, dtype=np.float32)
    
    rec_ids = df["record_id"].values
    sbp_vals = df[sbp_col].values
    dbp_vals = df[dbp_col].values
    
    cur_rec = None
    rec_start = 0
    
    for i in range(n):
        if rec_ids[i] != cur_rec:
            cur_rec = rec_ids[i]
            rec_start = i
            
        # Window 3 (up to past 3 steps)
        w3_start = max(rec_start, i - 2)
        s3 = sbp_vals[w3_start:i+1]
        d3 = dbp_vals[w3_start:i+1]
        
        rolling_sbp_mean_3[i] = np.mean(s3)
        rolling_dbp_mean_3[i] = np.mean(d3)
        
        if len(s3) > 1:
            rolling_sbp_std_3[i] = np.std(s3, ddof=1)
            rolling_dbp_std_3[i] = np.std(d3, ddof=1)
            diffs_s = np.abs(np.diff(s3))
            diffs_d = np.abs(np.diff(d3))
            max_abs_sbp_jump_3[i] = np.max(diffs_s)
            max_abs_dbp_jump_3[i] = np.max(diffs_d)
            median_abs_sbp_change_3[i] = np.median(diffs_s)
            median_abs_dbp_change_3[i] = np.median(diffs_d)
            sbp_trend_slope_3[i] = (s3[-1] - s3[0]) / (len(s3) - 1)
            dbp_trend_slope_3[i] = (d3[-1] - d3[0]) / (len(d3) - 1)
            
        # Window 6 (up to past 6 steps)
        w6_start = max(rec_start, i - 5)
        s6 = sbp_vals[w6_start:i+1]
        d6 = dbp_vals[w6_start:i+1]
        if len(s6) > 1:
            rolling_sbp_std_6[i] = np.std(s6, ddof=1)
            rolling_dbp_std_6[i] = np.std(d6, ddof=1)
            
    df["rolling_sbp_mean_3"] = rolling_sbp_mean_3
    df["rolling_sbp_std_3"] = rolling_sbp_std_3
    df["rolling_dbp_mean_3"] = rolling_dbp_mean_3
    df["rolling_dbp_std_3"] = rolling_dbp_std_3
    df["rolling_sbp_std_6"] = rolling_sbp_std_6
    df["rolling_dbp_std_6"] = rolling_dbp_std_6
    df["max_abs_sbp_jump_3"] = max_abs_sbp_jump_3
    df["max_abs_dbp_jump_3"] = max_abs_dbp_jump_3
    df["median_abs_sbp_change_3"] = median_abs_sbp_change_3
    df["median_abs_dbp_change_3"] = median_abs_dbp_change_3
    df["sbp_trend_slope_3"] = sbp_trend_slope_3
    df["dbp_trend_slope_3"] = dbp_trend_slope_3
    
    return df


def main():
    print("=" * 80)
    print("PHASE 7: CONSTRUCTING RELIABILITY RESEARCH DATASET")
    print("=" * 80)
    t_start = time.time()
    
    # 1. Load calibration files
    with open(ISO_SBP_PKL, "rb") as f:
        iso_sbp = pickle.load(f)
    with open(ISO_DBP_PKL, "rb") as f:
        iso_dbp = pickle.load(f)
    with open(CONFORMAL_JSON, "r") as f:
        conformal_quantiles = json.load(f)
        
    # 2. Load sequence metadata
    print("Loading sequence metadata...")
    df_val_meta = pd.read_csv(VAL_META_CSV)
    df_test_meta = pd.read_csv(TEST_META_CSV)
    
    # 3. Establish identical record-level split for val partition
    val_records = np.sort(df_val_meta["record_id"].unique())
    rng = np.random.RandomState(42)
    shuffled_val_records = rng.permutation(val_records)
    n_cal_recs = len(shuffled_val_records) // 2

    cal_record_set = set(shuffled_val_records[:n_cal_recs])
    audit_record_set = set(shuffled_val_records[n_cal_recs:])
    test_record_set = set(df_test_meta["record_id"].unique())

    cal_mask = df_val_meta["record_id"].isin(cal_record_set).values
    audit_mask = df_val_meta["record_id"].isin(audit_record_set).values

    print(f"  Dev/Calibration partition: {len(cal_record_set)} records | {cal_mask.sum():,} sequences")
    print(f"  Val/Audit partition:       {len(audit_record_set)} records | {audit_mask.sum():,} sequences")
    print(f"  Test partition:            {len(test_record_set)} records | {len(df_test_meta):,} sequences")

    # 4. Load or compute predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}...")
    model = TemporalGRUModel().to(device)
    ckpt = torch.load(P4B_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_npz = np.load(VAL_SEQ_NPZ)
    test_npz = np.load(TEST_SEQ_NPZ)

    with torch.no_grad():
        val_X_t = torch.from_numpy(val_npz["X"]).float().to(device)
        test_X_t = torch.from_numpy(test_npz["X"]).float().to(device)
        val_preds_raw = model(val_X_t).cpu().numpy()
        test_preds_raw = model(test_X_t).cpu().numpy()

    # Calibrate point predictions using frozen isotonic regressors
    val_cal_sbp = iso_sbp.predict(val_preds_raw[:, 0])
    val_cal_dbp = iso_dbp.predict(val_preds_raw[:, 1])
    test_cal_sbp = iso_sbp.predict(test_preds_raw[:, 0])
    test_cal_dbp = iso_dbp.predict(test_preds_raw[:, 1])

    # 5. Load uncertainties
    print("Loading uncertainty estimates...")
    cal_u = np.load(CAL_U_NPY)
    audit_u = np.load(AUDIT_U_NPY)
    val_u = np.zeros((len(df_val_meta), 2), dtype=np.float32)
    val_u[cal_mask] = cal_u
    val_u[audit_mask] = audit_u

    df_test_p5a = pd.read_csv(TEST_P5A_CSV)
    test_u_sbp = df_test_p5a["sbp_uncertainty"].values
    test_u_dbp = df_test_p5a["dbp_uncertainty"].values

    # 6. Conformal interval widths
    print("Computing conformal interval widths...")
    val_sbp_w90, val_sbp_w95, val_dbp_w90, val_dbp_w95 = get_conformal_widths(val_cal_sbp, val_cal_dbp, conformal_quantiles)
    test_sbp_w90, test_sbp_w95, test_dbp_w90, test_dbp_w95 = get_conformal_widths(test_cal_sbp, test_cal_dbp, conformal_quantiles)

    # 7. Build preliminary DataFrames
    df_val = df_val_meta.copy()
    df_val["split"] = np.where(cal_mask, "dev", "val")
    df_val["pred_raw_sbp"] = val_preds_raw[:, 0]
    df_val["pred_raw_dbp"] = val_preds_raw[:, 1]
    df_val["pred_cal_sbp"] = val_cal_sbp
    df_val["pred_cal_dbp"] = val_cal_dbp
    df_val["sbp_uncertainty"] = val_u[:, 0]
    df_val["dbp_uncertainty"] = val_u[:, 1]
    df_val["sbp_conformal_width_90"] = val_sbp_w90
    df_val["sbp_conformal_width_95"] = val_sbp_w95
    df_val["dbp_conformal_width_90"] = val_dbp_w90
    df_val["dbp_conformal_width_95"] = val_dbp_w95

    df_test = df_test_meta.copy()
    df_test["split"] = "test"
    df_test["pred_raw_sbp"] = test_preds_raw[:, 0]
    df_test["pred_raw_dbp"] = test_preds_raw[:, 1]
    df_test["pred_cal_sbp"] = test_cal_sbp
    df_test["pred_cal_dbp"] = test_cal_dbp
    df_test["sbp_uncertainty"] = test_u_sbp
    df_test["dbp_uncertainty"] = test_u_dbp
    df_test["sbp_conformal_width_90"] = test_sbp_w90
    df_test["sbp_conformal_width_95"] = test_sbp_w95
    df_test["dbp_conformal_width_90"] = test_dbp_w90
    df_test["dbp_conformal_width_95"] = test_dbp_w95

    df_all = pd.concat([df_val, df_test], ignore_index=True)
    print(f"Combined total sequences: {len(df_all):,}")

    # 8. Join signal quality metrics from window manifest
    print("Loading and joining window quality metrics...")
    df_win = pd.read_csv(WINDOW_MANIFEST_CSV, usecols=[
        "window_id", "ppg_quality_status", "ppg_ptp", "ppg_std", 
        "ppg_clipped_fraction", "ppg_pulse_count", "estimated_hr_bpm"
    ])
    df_win = df_win.rename(columns={"window_id": "target_window_id"})
    
    df_all = df_all.merge(df_win, on="target_window_id", how="left")
    df_all["qc_pass"] = (df_all["ppg_quality_status"] == "PASS").astype(int)

    # Impute missing QC with median if any (safety check)
    for col in ["ppg_ptp", "ppg_std", "ppg_clipped_fraction", "ppg_pulse_count", "estimated_hr_bpm"]:
        if df_all[col].isnull().any():
            median_val = df_all.loc[df_all["split"] == "dev", col].median()
            df_all[col] = df_all[col].fillna(median_val)

    # 9. Compute Causal Temporal Stability Features
    print("Computing causal temporal stability features...")
    df_all = compute_causal_temporal_features(df_all, sbp_col="pred_cal_sbp", dbp_col="pred_cal_dbp")

    # 10. Compute Ground Truth Target & Errors
    df_all["target_sbp"] = df_all["sbp_true"]
    df_all["target_dbp"] = df_all["dbp_true"]
    
    df_all["sbp_error_raw"] = df_all["pred_raw_sbp"] - df_all["target_sbp"]
    df_all["dbp_error_raw"] = df_all["pred_raw_dbp"] - df_all["target_dbp"]
    df_all["sbp_error_cal"] = df_all["pred_cal_sbp"] - df_all["target_sbp"]
    df_all["dbp_error_cal"] = df_all["pred_cal_dbp"] - df_all["target_dbp"]
    
    df_all["abs_sbp_error_raw"] = np.abs(df_all["sbp_error_raw"])
    df_all["abs_dbp_error_raw"] = np.abs(df_all["dbp_error_raw"])
    df_all["abs_sbp_error_cal"] = np.abs(df_all["sbp_error_cal"])
    df_all["abs_dbp_error_cal"] = np.abs(df_all["dbp_error_cal"])

    # 11. Pre-defined Research High-Error Labels
    df_all["high_error_sbp_10"] = (df_all["abs_sbp_error_cal"] > 10.0).astype(int)
    df_all["high_error_sbp_15"] = (df_all["abs_sbp_error_cal"] > 15.0).astype(int)
    df_all["high_error_dbp_10"] = (df_all["abs_dbp_error_cal"] > 10.0).astype(int)
    df_all["high_error_dbp_15"] = (df_all["abs_dbp_error_cal"] > 15.0).astype(int)
    df_all["high_error_composite_10"] = ((df_all["abs_sbp_error_cal"] > 10.0) | (df_all["abs_dbp_error_cal"] > 10.0)).astype(int)
    df_all["high_error_composite_15"] = ((df_all["abs_sbp_error_cal"] > 15.0) | (df_all["abs_dbp_error_cal"] > 10.0)).astype(int)

    # 12. Add prediction_id
    df_all["prediction_id"] = [f"pred_{i:06d}" for i in range(len(df_all))]

    # 13. Reorder and save dataset
    cols_order = [
        "prediction_id", "sequence_id", "record_id", "target_window_id", "target_window_index", "split",
        "target_sbp", "target_dbp",
        "pred_raw_sbp", "pred_raw_dbp", "pred_cal_sbp", "pred_cal_dbp",
        "sbp_error_raw", "dbp_error_raw", "sbp_error_cal", "dbp_error_cal",
        "abs_sbp_error_raw", "abs_dbp_error_raw", "abs_sbp_error_cal", "abs_dbp_error_cal",
        "high_error_sbp_10", "high_error_sbp_15", "high_error_dbp_10", "high_error_dbp_15",
        "high_error_composite_10", "high_error_composite_15",
        "ppg_quality_status", "qc_pass", "ppg_ptp", "ppg_std", "ppg_clipped_fraction", "ppg_pulse_count", "estimated_hr_bpm",
        "sbp_uncertainty", "dbp_uncertainty",
        "sbp_conformal_width_90", "sbp_conformal_width_95", "dbp_conformal_width_90", "dbp_conformal_width_95",
        "rolling_sbp_mean_3", "rolling_sbp_std_3", "rolling_dbp_mean_3", "rolling_dbp_std_3",
        "rolling_sbp_std_6", "rolling_dbp_std_6",
        "max_abs_sbp_jump_3", "max_abs_dbp_jump_3",
        "median_abs_sbp_change_3", "median_abs_dbp_change_3",
        "sbp_trend_slope_3", "dbp_trend_slope_3"
    ]
    
    df_out = df_all[cols_order].copy()
    out_csv = OUTPUT_DIR / "reliability_dataset.csv"
    print(f"Saving reliability dataset to: {out_csv}...")
    df_out.to_csv(out_csv, index=False)

    # 14. Save Schema JSON
    schema = {
        "dataset_name": "Phase 7 Reliability Research Dataset",
        "total_rows": len(df_out),
        "split_counts": df_out["split"].value_counts().to_dict(),
        "columns": {
            col: {
                "dtype": str(df_out[col].dtype),
                "domain": "Identification / Partition" if col in ["prediction_id", "sequence_id", "record_id", "target_window_id", "target_window_index", "split"]
                          else ("Ground Truth Targets" if col in ["target_sbp", "target_dbp"]
                          else ("Frozen Point Predictions" if "pred_" in col
                          else ("Prediction Errors" if "error" in col and "high" not in col
                          else ("Research High-Error Labels" if "high_error" in col
                          else ("Signal Quality Indicators" if col in ["ppg_quality_status", "qc_pass", "ppg_ptp", "ppg_std", "ppg_clipped_fraction", "ppg_pulse_count", "estimated_hr_bpm"]
                          else ("Epistemic Uncertainty" if "uncertainty" in col
                          else ("Conformal Bounds" if "conformal" in col
                          else "Causal Temporal Stability"))))))),
                "min": float(df_out[col].min()) if np.issubdtype(df_out[col].dtype, np.number) else None,
                "max": float(df_out[col].max()) if np.issubdtype(df_out[col].dtype, np.number) else None,
                "null_count": int(df_out[col].isnull().sum())
            } for col in df_out.columns
        }
    }
    schema_json = OUTPUT_DIR / "reliability_dataset_schema.json"
    with open(schema_json, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Saved schema JSON to: {schema_json}")

    print("\n" + "=" * 80)
    print("RELIABILITY DATASET AUDIT SUMMARY:")
    print(f"  Total records:           {df_out['record_id'].nunique():,}")
    print(f"  Total predictions:       {len(df_out):,}")
    print(f"  Partition split:         dev={cal_mask.sum():,} | val={audit_mask.sum():,} | test={len(df_test):,}")
    print(f"  SBP Test MAE (Cal):      {df_out.loc[df_out['split']=='test', 'abs_sbp_error_cal'].mean():.2f} mmHg")
    print(f"  DBP Test MAE (Cal):      {df_out.loc[df_out['split']=='test', 'abs_dbp_error_cal'].mean():.2f} mmHg")
    print(f"  High-Error Rate SBP>10:  {df_out.loc[df_out['split']=='test', 'high_error_sbp_10'].mean()*100:.1f}%")
    print(f"  High-Error Rate DBP>10:  {df_out.loc[df_out['split']=='test', 'high_error_dbp_10'].mean()*100:.1f}%")
    print(f"  Execution Time:          {time.time()-t_start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
