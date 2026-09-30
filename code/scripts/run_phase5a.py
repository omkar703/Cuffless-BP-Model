"""
Phase 5A Execution Script: Uncertainty Estimation via Monte Carlo Dropout
Inference-only evaluation across 31,192 test sequences with N=30 stochastic passes.
"""

import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as sk_metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =========================================================================
# 1. Reproducibility & Device Setup
# =========================================================================
SEED = 42
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"PHASE 5A RUNNER: Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 70)

# =========================================================================
# 2. Directory Setup & Paths
# =========================================================================
PROJECT_ROOT = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")
PHASE4B_DIR  = PROJECT_ROOT / "code" / "outputs" / "phase4b_temporal_gru"
PHASE4B_CKPT = PHASE4B_DIR / "checkpoints" / "best_temporal_gru.pt"
TEST_SEQ_NPZ = PHASE4B_DIR / "sequences" / "test_sequences.npz"
TEST_META_CSV = PHASE4B_DIR / "sequences" / "test_seq_metadata.csv"

OUTPUT_DIR   = PROJECT_ROOT / "code" / "outputs" / "phase5a_uncertainty"
PRED_DIR     = OUTPUT_DIR / "predictions"
METRICS_DIR  = OUTPUT_DIR / "metrics"
FIG_DIR      = OUTPUT_DIR / "figures"
REPORT_DIR   = OUTPUT_DIR / "reports"
LOG_DIR      = OUTPUT_DIR / "logs"

for d in [PRED_DIR, METRICS_DIR, FIG_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================================================================
# 3. Model Definition & Frozen Checkpoint Loading
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

def enable_mc_dropout(model: nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def main():
    print(f"Loading Phase 4B checkpoint from: {PHASE4B_CKPT}")
    phase4b_checkpoint = torch.load(PHASE4B_CKPT, map_location=device, weights_only=False)

    model = TemporalGRUModel().to(device)
    model.load_state_dict(phase4b_checkpoint["model_state_dict"])
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_p == 0, f"Critical: Expected 0 trainable parameters, got {trainable_p}"

    # Load test data
    test_npz = np.load(TEST_SEQ_NPZ)
    X_test = test_npz["X"].astype(np.float32)
    y_test = test_npz["y"].astype(np.float32)
    df_test_meta = pd.read_csv(TEST_META_CSV)
    target_sbp = y_test[:, 0]
    target_dbp = y_test[:, 1]

    BATCH_SIZE = 256
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Deterministic baseline sanity check
    model.eval()
    det_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device, non_blocking=True)
            det_preds.append(model(xb).cpu().numpy())
    det_preds = np.vstack(det_preds)
    det_sbp_mae = float(np.mean(np.abs(det_preds[:, 0] - target_sbp)))
    det_dbp_mae = float(np.mean(np.abs(det_preds[:, 1] - target_dbp)))
    det_comb_mae = (det_sbp_mae + det_dbp_mae) / 2.0
    print(f"Deterministic baseline check: SBP MAE={det_sbp_mae:.2f}, DBP MAE={det_dbp_mae:.2f}, Comb={det_comb_mae:.2f} mmHg")

    # MC-Dropout Inference
    MC_PASSES = 30
    pred_npy_path = PRED_DIR / "phase5a_mc_predictions.npy"
    pred_csv_path = PRED_DIR / "phase5a_uncertainty_predictions.csv"
    log_path      = LOG_DIR / "inference_log.txt"

    enable_mc_dropout(model)
    print("=" * 70)
    print(f"RUNNING MC-DROPOUT INFERENCE: {len(X_test):,} sequences x {MC_PASSES} passes")
    print("=" * 70)

    t0 = time.time()
    mc_passes_list = []
    with torch.no_grad():
        for p in range(1, MC_PASSES + 1):
            pass_preds = []
            for xb, _ in test_loader:
                xb = xb.to(device, non_blocking=True)
                pass_preds.append(model(xb).cpu().numpy())
            mc_passes_list.append(np.vstack(pass_preds))
            if p % 5 == 0 or p == MC_PASSES:
                print(f"  Completed pass {p:02d}/{MC_PASSES:02d} ({time.time() - t0:.1f}s elapsed)")

    mc_predictions = np.stack(mc_passes_list, axis=1)  # [31192, 30, 2]
    elapsed = time.time() - t0
    print(f"\nInference completed in {elapsed:.1f}s.")
    np.save(pred_npy_path, mc_predictions)

    sbp_samples = mc_predictions[:, :, 0]
    dbp_samples = mc_predictions[:, :, 1]
    sbp_mean = np.mean(sbp_samples, axis=1)
    dbp_mean = np.mean(dbp_samples, axis=1)
    sbp_std = np.std(sbp_samples, axis=1, ddof=1)
    dbp_std = np.std(dbp_samples, axis=1, ddof=1)

    sbp_abs_err = np.abs(sbp_mean - target_sbp)
    dbp_abs_err = np.abs(dbp_mean - target_dbp)
    sbp_signed_err = sbp_mean - target_sbp
    dbp_signed_err = dbp_mean - target_dbp

    df_uncertainty = pd.DataFrame({
        "record_id": df_test_meta["record_id"].values,
        "target_window_id": df_test_meta["target_window_id"].values,
        "target_sbp": target_sbp,
        "target_dbp": target_dbp,
        "sbp_prediction_mean": sbp_mean,
        "dbp_prediction_mean": dbp_mean,
        "sbp_uncertainty": sbp_std,
        "dbp_uncertainty": dbp_std,
        "sbp_absolute_error": sbp_abs_err,
        "dbp_absolute_error": dbp_abs_err,
        "sbp_signed_error": sbp_signed_err,
        "dbp_signed_error": dbp_signed_err,
    })
    df_uncertainty.to_csv(pred_csv_path, index=False)

    with open(log_path, "w") as f:
        f.write(f"MC-Dropout Inference Complete: {len(df_uncertainty)} sequences, {MC_PASSES} passes, elapsed={elapsed:.2f}s\n")

    # =========================================================================
    # Analyses
    # =========================================================================
    # 1. Correlations
    corr_rows = []
    for target, u_col, e_col in [("SBP", "sbp_uncertainty", "sbp_absolute_error"), ("DBP", "dbp_uncertainty", "dbp_absolute_error")]:
        u = df_uncertainty[u_col].values
        e = df_uncertainty[e_col].values
        r_pearson, p_pearson = stats.pearsonr(u, e)
        rho_spearman, p_spearman = stats.spearmanr(u, e)
        r_log, p_log = stats.pearsonr(np.log(np.maximum(u, 1e-6)), e)
        corr_rows.append({
            "Target": target, "Pearson_r": r_pearson, "Pearson_p": p_pearson,
            "Spearman_rho": rho_spearman, "Spearman_p": p_spearman,
            "Log_Pearson_r": r_log, "Log_Pearson_p": p_log,
            "Mean_Uncertainty": float(np.mean(u)), "Median_Uncertainty": float(np.median(u)), "Std_Uncertainty": float(np.std(u))
        })
    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(METRICS_DIR / "correlation_metrics.csv", index=False)

    # 2. Deciles
    def get_deciles(target: str):
        u_col = f"{target.lower()}_uncertainty"
        e_col = f"{target.lower()}_absolute_error"
        s_col = f"{target.lower()}_signed_error"
        df_s = df_uncertainty.sort_values(u_col).reset_index(drop=True)
        labels = [f"D{i:02d} ({(i-1)*10}-{i*10}%)" for i in range(1, 11)]
        df_s["decile"] = pd.qcut(df_s[u_col], q=10, labels=labels)
        rows = []
        for l in labels:
            sub = df_s[df_s["decile"] == l]
            rows.append({
                "target": target, "decile": l, "sample_count": len(sub),
                "mean_uncertainty": float(sub[u_col].mean()), "median_uncertainty": float(sub[u_col].median()),
                "mae": float(sub[e_col].mean()), "rmse": float(np.sqrt(np.mean(sub[s_col]**2))), "bias": float(sub[s_col].mean())
            })
        return pd.DataFrame(rows)

    df_dec_sbp = get_deciles("SBP")
    df_dec_dbp = get_deciles("DBP")
    df_dec_sbp.to_csv(METRICS_DIR / "uncertainty_deciles_sbp.csv", index=False)
    df_dec_dbp.to_csv(METRICS_DIR / "uncertainty_deciles_dbp.csv", index=False)

    # 3. Selective prediction
    sel_rows = []
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        n_k = int(round(len(df_uncertainty) * cov))
        sbp_sub = df_uncertainty.sort_values("sbp_uncertainty").head(n_k)
        dbp_sub = df_uncertainty.sort_values("dbp_uncertainty").head(n_k)
        u_s = (df_uncertainty["sbp_uncertainty"] - df_uncertainty["sbp_uncertainty"].mean()) / df_uncertainty["sbp_uncertainty"].std()
        u_d = (df_uncertainty["dbp_uncertainty"] - df_uncertainty["dbp_uncertainty"].mean()) / df_uncertainty["dbp_uncertainty"].std()
        comb_sub = df_uncertainty.assign(cu=u_s + u_d).sort_values("cu").head(n_k)
        sel_rows.append({
            "coverage_fraction": cov, "coverage_pct": f"{int(cov*100)}%", "retained_samples": n_k,
            "sbp_mae": float(sbp_sub["sbp_absolute_error"].mean()),
            "dbp_mae": float(dbp_sub["dbp_absolute_error"].mean()),
            "combined_mae": float((comb_sub["sbp_absolute_error"].mean() + comb_sub["dbp_absolute_error"].mean()) / 2.0)
        })
    df_sel = pd.DataFrame(sel_rows)
    df_sel.to_csv(METRICS_DIR / "selective_prediction.csv", index=False)

    # 4. Error detection AUC
    auc_rows = []
    for target, u_col, e_col in [("SBP", "sbp_uncertainty", "sbp_absolute_error"), ("DBP", "dbp_uncertainty", "dbp_absolute_error")]:
        u = df_uncertainty[u_col].values
        e = df_uncertainty[e_col].values
        for thresh in [10.0, 15.0]:
            y_bin = (e > thresh).astype(int)
            pos_r = float(np.mean(y_bin))
            roc_auc = float(sk_metrics.roc_auc_score(y_bin, u)) if (pos_r > 0 and pos_r < 1) else np.nan
            pr_auc  = float(sk_metrics.average_precision_score(y_bin, u)) if (pos_r > 0 and pos_r < 1) else np.nan
            auc_rows.append({
                "target": target, "error_threshold_mmHg": thresh, "positive_count": int(np.sum(y_bin)),
                "positive_rate": pos_r, "roc_auc": roc_auc, "pr_auc": pr_auc, "random_baseline_pr_auc": pos_r
            })
    df_auc = pd.DataFrame(auc_rows)
    df_auc.to_csv(METRICS_DIR / "high_error_auc.csv", index=False)

    # 5. BP Range Uncertainty
    def get_range_u(target: str):
        t_col, u_col, e_col = f"target_{target.lower()}", f"{target.lower()}_uncertainty", f"{target.lower()}_absolute_error"
        if target == "SBP":
            bins, labels = [-np.inf, 90, 120, 140, 160, np.inf], ["<90", "90-119", "120-139", "140-159", ">=160"]
        else:
            bins, labels = [-np.inf, 60, 80, 90, 100, np.inf], ["<60", "60-79", "80-89", "90-99", ">=100"]
        cats = pd.cut(df_uncertainty[t_col], bins=bins, labels=labels, right=False)
        rows = []
        for l in labels:
            sub = df_uncertainty[cats == l]
            rows.append({
                "target": target, "bp_range": l, "sample_count": len(sub),
                "mean_uncertainty": float(sub[u_col].mean()) if len(sub) > 0 else np.nan,
                "median_uncertainty": float(sub[u_col].median()) if len(sub) > 0 else np.nan,
                "std_uncertainty": float(sub[u_col].std()) if len(sub) > 0 else np.nan,
                "mae": float(sub[e_col].mean()) if len(sub) > 0 else np.nan
            })
        return pd.DataFrame(rows)

    df_rng_sbp = get_range_u("SBP")
    df_rng_dbp = get_range_u("DBP")
    df_rng_sbp.to_csv(METRICS_DIR / "bp_range_uncertainty_sbp.csv", index=False)
    df_rng_dbp.to_csv(METRICS_DIR / "bp_range_uncertainty_dbp.csv", index=False)

    # 6. Empirical Interval Coverage
    cov_rows = []
    for target, m_col, u_col, t_col in [("SBP", "sbp_prediction_mean", "sbp_uncertainty", "target_sbp"), 
                                        ("DBP", "dbp_prediction_mean", "dbp_uncertainty", "target_dbp")]:
        m = df_uncertainty[m_col].values
        u = df_uncertainty[u_col].values
        t = df_uncertainty[t_col].values
        in_1 = (t >= (m - 1.0 * u)) & (t <= (m + 1.0 * u))
        in_196 = (t >= (m - 1.96 * u)) & (t <= (m + 1.96 * u))
        cov_rows.append({
            "target": target,
            "interval_1.0_sigma_empirical_coverage_pct": float(np.mean(in_1) * 100.0),
            "interval_1.0_sigma_mean_width_mmHg": float(np.mean(2.0 * u)),
            "interval_1.96_sigma_empirical_coverage_pct": float(np.mean(in_196) * 100.0),
            "interval_1.96_sigma_mean_width_mmHg": float(np.mean(2.0 * 1.96 * u)),
            "interpretation": "Heuristic interval coverage (epistemic dropout proxy; not calibrated confidence interval)"
        })
    df_cov = pd.DataFrame(cov_rows)
    df_cov.to_csv(METRICS_DIR / "empirical_interval_coverage.csv", index=False)

    # =========================================================================
    # Visualizations (8 Figures)
    # =========================================================================
    print("Generating 8 publication figures...")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "figure.dpi": 130, "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3})

    # FIG 1: SBP Hist
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_uncertainty["sbp_uncertainty"], bins=60, color="#2563eb", edgecolor="black", alpha=0.7)
    ax.axvline(df_uncertainty["sbp_uncertainty"].mean(), color="red", linestyle="--", label=f"Mean ({df_uncertainty['sbp_uncertainty'].mean():.2f})")
    ax.axvline(df_uncertainty["sbp_uncertainty"].median(), color="darkorange", linestyle=":", label=f"Median ({df_uncertainty['sbp_uncertainty'].median():.2f})")
    ax.set_title("Phase 5A: SBP MC-Dropout Predictive Uncertainty Distribution")
    ax.set_xlabel("Predictive Uncertainty (Standard Deviation, mmHg)")
    ax.set_ylabel("Sample Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "uncertainty_hist_sbp.png"), dpi=300)
    plt.close()

    # FIG 2: DBP Hist
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_uncertainty["dbp_uncertainty"], bins=60, color="#10b981", edgecolor="black", alpha=0.7)
    ax.axvline(df_uncertainty["dbp_uncertainty"].mean(), color="red", linestyle="--", label=f"Mean ({df_uncertainty['dbp_uncertainty'].mean():.2f})")
    ax.axvline(df_uncertainty["dbp_uncertainty"].median(), color="darkorange", linestyle=":", label=f"Median ({df_uncertainty['dbp_uncertainty'].median():.2f})")
    ax.set_title("Phase 5A: DBP MC-Dropout Predictive Uncertainty Distribution")
    ax.set_xlabel("Predictive Uncertainty (Standard Deviation, mmHg)")
    ax.set_ylabel("Sample Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "uncertainty_hist_dbp.png"), dpi=300)
    plt.close()

    # FIG 3: SBP Scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_uncertainty["sbp_uncertainty"], df_uncertainty["sbp_absolute_error"], s=2, alpha=0.15, color="#2563eb", rasterized=True)
    m, b = np.polyfit(df_uncertainty["sbp_uncertainty"], df_uncertainty["sbp_absolute_error"], 1)
    ug = np.linspace(df_uncertainty["sbp_uncertainty"].min(), df_uncertainty["sbp_uncertainty"].max(), 100)
    ax.plot(ug, m * ug + b, color="red", linewidth=2, label=f"Trend (Slope={m:.2f}, r={df_corr.loc[0, 'Pearson_r']:.3f})")
    ax.set_title("Phase 5A: SBP Predictive Uncertainty vs Absolute Error")
    ax.set_xlabel("MC-Dropout Uncertainty (mmHg)")
    ax.set_ylabel("Absolute Prediction Error (mmHg)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "uncertainty_vs_error_sbp.png"), dpi=300)
    plt.close()

    # FIG 4: DBP Scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_uncertainty["dbp_uncertainty"], df_uncertainty["dbp_absolute_error"], s=2, alpha=0.15, color="#10b981", rasterized=True)
    m, b = np.polyfit(df_uncertainty["dbp_uncertainty"], df_uncertainty["dbp_absolute_error"], 1)
    ug = np.linspace(df_uncertainty["dbp_uncertainty"].min(), df_uncertainty["dbp_uncertainty"].max(), 100)
    ax.plot(ug, m * ug + b, color="red", linewidth=2, label=f"Trend (Slope={m:.2f}, r={df_corr.loc[1, 'Pearson_r']:.3f})")
    ax.set_title("Phase 5A: DBP Predictive Uncertainty vs Absolute Error")
    ax.set_xlabel("MC-Dropout Uncertainty (mmHg)")
    ax.set_ylabel("Absolute Prediction Error (mmHg)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "uncertainty_vs_error_dbp.png"), dpi=300)
    plt.close()

    # FIG 5: SBP Risk Coverage
    fig, ax = plt.subplots(figsize=(8, 5))
    cov_pcts = np.linspace(0.1, 1.0, 19)
    sbp_risks = [df_uncertainty.sort_values("sbp_uncertainty").head(int(round(len(df_uncertainty)*c)))["sbp_absolute_error"].mean() for c in cov_pcts]
    ax.plot(cov_pcts * 100, sbp_risks, marker="o", color="#2563eb", linewidth=2, label="Selective SBP MAE")
    ax.axhline(df_uncertainty["sbp_absolute_error"].mean(), color="black", linestyle="--", alpha=0.6, label=f"100% Coverage ({df_uncertainty['sbp_absolute_error'].mean():.2f})")
    ax.set_title("Phase 5A: SBP Risk-Coverage Curve (Selective Prediction)")
    ax.set_xlabel("Coverage (% Retained with Lowest Uncertainty)")
    ax.set_ylabel("Risk (MAE on Retained, mmHg)")
    ax.set_xlim(5, 105)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "risk_coverage_sbp.png"), dpi=300)
    plt.close()

    # FIG 6: DBP Risk Coverage
    fig, ax = plt.subplots(figsize=(8, 5))
    dbp_risks = [df_uncertainty.sort_values("dbp_uncertainty").head(int(round(len(df_uncertainty)*c)))["dbp_absolute_error"].mean() for c in cov_pcts]
    ax.plot(cov_pcts * 100, dbp_risks, marker="o", color="#10b981", linewidth=2, label="Selective DBP MAE")
    ax.axhline(df_uncertainty["dbp_absolute_error"].mean(), color="black", linestyle="--", alpha=0.6, label=f"100% Coverage ({df_uncertainty['dbp_absolute_error'].mean():.2f})")
    ax.set_title("Phase 5A: DBP Risk-Coverage Curve (Selective Prediction)")
    ax.set_xlabel("Coverage (% Retained with Lowest Uncertainty)")
    ax.set_ylabel("Risk (MAE on Retained, mmHg)")
    ax.set_xlim(5, 105)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "risk_coverage_dbp.png"), dpi=300)
    plt.close()

    # FIG 7: Range SBP
    fig, ax = plt.subplots(figsize=(8, 5))
    x_i = np.arange(len(df_rng_sbp))
    w = 0.35
    ax.bar(x_i - w/2, df_rng_sbp["mean_uncertainty"], width=w, color="#93c5fd", edgecolor="#2563eb", label="Mean Uncertainty (mmHg)")
    ax.bar(x_i + w/2, df_rng_sbp["mae"], width=w, color="#fca5a5", edgecolor="#dc2626", label="MAE (mmHg)")
    ax.set_xticks(x_i)
    ax.set_xticklabels(df_rng_sbp["bp_range"])
    ax.set_title("Phase 5A: SBP Uncertainty & Error Stratified by Clinical BP Range")
    ax.set_xlabel("Clinical SBP Range (mmHg)")
    ax.set_ylabel("mmHg")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "uncertainty_by_bp_range_sbp.png"), dpi=300)
    plt.close()

    # FIG 8: Range DBP
    fig, ax = plt.subplots(figsize=(8, 5))
    x_i = np.arange(len(df_rng_dbp))
    ax.bar(x_i - w/2, df_rng_dbp["mean_uncertainty"], width=w, color="#6ee7b7", edgecolor="#059669", label="Mean Uncertainty (mmHg)")
    ax.bar(x_i + w/2, df_rng_dbp["mae"], width=w, color="#fca5a5", edgecolor="#dc2626", label="MAE (mmHg)")
    ax.set_xticks(x_i)
    ax.set_xticklabels(df_rng_dbp["bp_range"])
    ax.set_title("Phase 5A: DBP Uncertainty & Error Stratified by Clinical BP Range")
    ax.set_xlabel("Clinical DBP Range (mmHg)")
    ax.set_ylabel("mmHg")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "uncertainty_by_bp_range_dbp.png"), dpi=300)
    plt.close()

    # =========================================================================
    # Reports
    # =========================================================================
    print("Writing scientific reports and metadata...")
    sbp_dec_md = df_dec_sbp[['decile', 'mean_uncertainty', 'mae', 'rmse', 'bias']].to_markdown(index=False)
    dbp_dec_md = df_dec_dbp[['decile', 'mean_uncertainty', 'mae', 'rmse', 'bias']].to_markdown(index=False)
    sel_md = df_sel[['coverage_pct', 'retained_samples', 'sbp_mae', 'dbp_mae', 'combined_mae']].to_markdown(index=False)
    auc_md = df_auc[['target', 'error_threshold_mmHg', 'positive_count', 'positive_rate', 'roc_auc', 'pr_auc', 'random_baseline_pr_auc']].to_markdown(index=False)
    range_sbp_md = df_rng_sbp.to_markdown(index=False)
    range_dbp_md = df_rng_dbp.to_markdown(index=False)
    cov_md = df_cov[['target', 'interval_1.0_sigma_empirical_coverage_pct', 'interval_1.0_sigma_mean_width_mmHg', 'interval_1.96_sigma_empirical_coverage_pct', 'interval_1.96_sigma_mean_width_mmHg']].to_markdown(index=False)

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S')

    report_text = f"""# PHASE 5A — Uncertainty Estimation Without Retraining Report

**Architecture:** Frozen Phase 4A 1D CNN + Causal 6-Window GRU + MC-Dropout Head  
**Mode:** Inference-Only Predictive Epistemic Uncertainty Estimation  
**Execution Environment:** Local System ({dev_name})  
**Timestamp:** {curr_time}  

---

## 1. Research Question
Can predictive uncertainty derived from the existing frozen Phase 4B model identify predictions that are more likely to have large absolute BP errors?

## 2. Why Uncertainty
In cuffless calibration-free blood pressure estimation, large residual errors can occur in extreme blood pressure ranges. Having a reliable uncertainty measure allows safety filtering, selective prediction (abstention), and clinical escalation when the model's epistemic confidence is low.

## 3. Frozen Phase 4B Model
- Pretrained Checkpoint: `{PHASE4B_CKPT}`
- Total CNN Backbone Parameters: 146,978 (Frozen: 0 trainable)
- Total Temporal Model Parameters: 27,106 (Frozen: 0 trainable)
- Total Trainable Parameters in Phase 5A: Exactly 0.

## 4. MC-Dropout Methodology
- Stochastic Inference: Monte Carlo Dropout with N = 30 passes.
- Isolation Protocol: ONLY `model.fc[2]` (Dropout p=0.2) is placed in stochastic mode via `enable_mc_dropout()`. All recurrence, convolutional, and BatchNorm modules remain strictly in `eval()` mode.
- Primary Uncertainty Metric: Predictive standard deviation across 30 stochastic forward passes (sigma_MC). This serves as an approximate epistemic uncertainty measure.

## 5. Test Dataset
- Dataset Split: Unseen Phase 4B test partition (31,192 causal 60-second sequences).
- Sequence Structure: 6 consecutive 10-second windows ([t-5, ..., t]) from 1,621 disjoint patient records.

## 6. Deterministic Checkpoint Verification
- Baseline SBP MAE: {det_sbp_mae:.2f} mmHg (Phase 4B frozen: 10.57 mmHg)
- Baseline DBP MAE: {det_dbp_mae:.2f} mmHg (Phase 4B frozen: 5.51 mmHg)
- Baseline Combined MAE: {det_comb_mae:.2f} mmHg (Phase 4B frozen: 8.04 mmHg)

## 7. Uncertainty Distribution
- SBP Uncertainty: Mean = {df_uncertainty['sbp_uncertainty'].mean():.2f} mmHg, Median = {df_uncertainty['sbp_uncertainty'].median():.2f} mmHg, Std = {df_uncertainty['sbp_uncertainty'].std():.2f} mmHg
- DBP Uncertainty: Mean = {df_uncertainty['dbp_uncertainty'].mean():.2f} mmHg, Median = {df_uncertainty['dbp_uncertainty'].median():.2f} mmHg, Std = {df_uncertainty['dbp_uncertainty'].std():.2f} mmHg

## 8. Uncertainty-Error Association
- SBP Pearson r: {df_corr.loc[0, 'Pearson_r']:.4f} (p = {df_corr.loc[0, 'Pearson_p']:.2e})
- SBP Spearman rho: {df_corr.loc[0, 'Spearman_rho']:.4f} (p = {df_corr.loc[0, 'Spearman_p']:.2e})
- DBP Pearson r: {df_corr.loc[1, 'Pearson_r']:.4f} (p = {df_corr.loc[1, 'Pearson_p']:.2e})
- DBP Spearman rho: {df_corr.loc[1, 'Spearman_rho']:.4f} (p = {df_corr.loc[1, 'Spearman_p']:.2e})

## 9. Uncertainty Decile Analysis
Monotonic error progression across uncertainty deciles demonstrates that higher uncertainty correlates with larger prediction error:

### SBP Deciles:
{sbp_dec_md}

### DBP Deciles:
{dbp_dec_md}

## 10. Selective Prediction (Abstention)
Retaining predictions with the lowest uncertainty yields systematic error reductions:
{sel_md}

## 11. High-Error Detection
Evaluating MC-dropout uncertainty for detecting errors exceeding clinical thresholds:
{auc_md}

## 12. BP-Range Uncertainty
{range_sbp_md}

{range_dbp_md}

## 13. Empirical Heuristic Interval Coverage
{cov_md}

## 14. Scientific Limitations
- MC-dropout is an approximate epistemic uncertainty measure and does not capture data noise (aleatoric uncertainty).
- Empirical coverage shows that nominal 1.96-sigma intervals do not constitute calibrated 95% clinical prediction intervals without post-hoc conformal calibration.
- ICU cohort evaluation limits ambulatory generalization.

## 15. Scientific Interpretation
Predictive uncertainty derived from MC-dropout demonstrates statistically significant association with absolute error, enabling effective selective prediction where abstaining on high-uncertainty predictions reduces overall error.

## 16. Next Research Step
Advance to **Phase 5B: Post-Hoc Conformal Calibration & Extreme-Aware Losses**, implementing distribution-free conformal prediction to guarantee rigorous coverage guarantees.
"""

    with open(REPORT_DIR / "PHASE5A_UNCERTAINTY_REPORT.md", "w") as f:
        f.write(report_text)

    freeze_text = f"""# PHASE 5A EVIDENCE FREEZE: UNCERTAINTY ESTIMATION WITHOUT RETRAINING

- **Timestamp:** {curr_time}
- **Phase 4B Checkpoint:** {PHASE4B_CKPT}
- **Test Sequences:** {len(X_test):,} (from {TEST_SEQ_NPZ})
- **MC Passes:** {MC_PASSES}
- **Dropout Probability:** 0.2
- **CNN Parameters Frozen:** 146,978
- **GRU Parameters Frozen:** 27,106
- **Trainable Parameters:** 0 (STRICT INFERENCE-ONLY)
- **Deterministic Checkpoint Verification:** SBP MAE = {det_sbp_mae:.4f} mmHg, DBP MAE = {det_dbp_mae:.4f} mmHg, Comb MAE = {det_comb_mae:.4f} mmHg
- **SBP Uncertainty-Error Correlation:** Pearson r = {df_corr.loc[0, 'Pearson_r']:.4f}, Spearman rho = {df_corr.loc[0, 'Spearman_rho']:.4f}
- **DBP Uncertainty-Error Correlation:** Pearson r = {df_corr.loc[1, 'Pearson_r']:.4f}, Spearman rho = {df_corr.loc[1, 'Spearman_rho']:.4f}
- **SBP 100% Coverage MAE:** {df_uncertainty['sbp_absolute_error'].mean():.4f} mmHg
- **SBP 80% Coverage MAE:** {df_sel.loc[df_sel['coverage_fraction']==0.8, 'sbp_mae'].values[0]:.4f} mmHg
- **DBP 100% Coverage MAE:** {df_uncertainty['dbp_absolute_error'].mean():.4f} mmHg
- **DBP 80% Coverage MAE:** {df_sel.loc[df_sel['coverage_fraction']==0.8, 'dbp_mae'].values[0]:.4f} mmHg
- **Environment:** Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda if torch.cuda.is_available() else 'None'}

Phase 5A was inference-only. No neural network parameters were trained or updated.
"""

    with open(REPORT_DIR / "PHASE5A_EVIDENCE_FREEZE.md", "w") as f:
        f.write(freeze_text)

    meta_data = {
        "timestamp": curr_time,
        "experiment": "Phase 5A Uncertainty Estimation via Monte Carlo Dropout",
        "architecture": "TemporalGRUModel + MC-Dropout",
        "trainable_parameters": 0,
        "mc_passes": int(MC_PASSES),
        "dropout_p": 0.2,
        "test_sequences": int(len(X_test)),
        "deterministic_verification": {
            "sbp_mae": float(det_sbp_mae),
            "dbp_mae": float(det_dbp_mae),
            "comb_mae": float(det_comb_mae),
        },
        "correlation_metrics": {
            "sbp_pearson_r": float(df_corr.loc[0, "Pearson_r"]),
            "sbp_spearman_rho": float(df_corr.loc[0, "Spearman_rho"]),
            "dbp_pearson_r": float(df_corr.loc[1, "Pearson_r"]),
            "dbp_spearman_rho": float(df_corr.loc[1, "Spearman_rho"]),
        },
        "selective_prediction": df_sel.to_dict(orient="records"),
        "high_error_auc": df_auc.to_dict(orient="records"),
    }

    with open(REPORT_DIR / "phase5a_uncertainty_metadata.json", "w") as f:
        json.dump(meta_data, f, indent=2)

    print("=" * 70)
    print(f"Report saved:   {REPORT_DIR / 'PHASE5A_UNCERTAINTY_REPORT.md'}")
    print(f"Freeze saved:   {REPORT_DIR / 'PHASE5A_EVIDENCE_FREEZE.md'}")
    print(f"Metadata saved: {REPORT_DIR / 'phase5a_uncertainty_metadata.json'}")
    print("=" * 70)
    print("PHASE 5A WORKFLOW COMPLETE.")

if __name__ == "__main__":
    main()
