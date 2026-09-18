"""
Phase 4B Execution Script: Temporal Context Extension
Executes training and evaluation of the Causal Temporal GRU on top of Frozen Phase 4A CNN embeddings.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
print(f"PHASE 4B RUNNER: Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 70)

# =========================================================================
# 2. Directory Setup & Paths
# =========================================================================
PROJECT_ROOT = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")
DATASET_DIR  = PROJECT_ROOT / "BloodPressureDataset"
OUTPUT_DIR   = PROJECT_ROOT / "code" / "outputs" / "phase4b_temporal_gru"
MANIFEST_GZ  = PROJECT_ROOT / "code" / "outputs" / "windows" / "window_manifest.csv.gz"
PHASE4A_CKPT = PROJECT_ROOT / "code" / "outputs" / "phase4a_single_model" / "checkpoints" / "best_model_ppg_vpg_apg.pt"
PHASE4A_PREDS = PROJECT_ROOT / "code" / "outputs" / "phase4a_single_model" / "predictions" / "test_predictions.csv"

CKPT_DIR    = OUTPUT_DIR / "checkpoints"
EMB_DIR     = OUTPUT_DIR / "embeddings"
SEQ_DIR     = OUTPUT_DIR / "sequences"
METRICS_DIR = OUTPUT_DIR / "metrics"
PRED_DIR    = OUTPUT_DIR / "predictions"
FIG_DIR     = OUTPUT_DIR / "figures"
REPORT_DIR  = OUTPUT_DIR / "reports"
LOG_DIR     = OUTPUT_DIR / "logs"

for d in [CKPT_DIR, EMB_DIR, SEQ_DIR, METRICS_DIR, PRED_DIR, FIG_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================================================================
# 3. Model Definition: Causal Temporal GRU
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
            bidirectional=False  # CAUSALITY MANDATE
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.sbp_head = nn.Linear(32, 1)
        self.dbp_head = nn.Linear(32, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)        # [B, 6, 64]
        last_hidden = out[:, -1, :] # [B, 64] (target window t)
        feat = self.fc(last_hidden) # [B, 32]
        sbp = self.sbp_head(feat)   # [B, 1]
        dbp = self.dbp_head(feat)   # [B, 1]
        return torch.cat([sbp, dbp], dim=1)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# =========================================================================
# 4. Metrics & Evaluation Utilities
# =========================================================================
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    n = len(y_true)
    var_true = float(np.var(y_true))
    r2 = float(1.0 - np.sum(errors ** 2) / (n * var_true)) if var_true > 1e-9 else 0.0
    
    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": r2,
        "bias": float(np.mean(errors)),
        "error_sd": float(np.std(errors)),
        "pct_within_5": float(np.mean(abs_errors <= 5.0) * 100.0),
        "pct_within_10": float(np.mean(abs_errors <= 10.0) * 100.0),
        "pct_within_15": float(np.mean(abs_errors <= 15.0) * 100.0),
        "n_samples": n,
    }

def compute_bp_range_stratification(y_true: np.ndarray, y_pred: np.ndarray, target: str = "SBP") -> pd.DataFrame:
    abs_err = np.abs(y_pred - y_true)
    err = y_pred - y_true
    if target.upper() == "SBP":
        bins = [-np.inf, 90, 120, 140, 160, np.inf]
        labels = ["<90", "90-119", "120-139", "140-159", ">=160"]
    else:
        bins = [-np.inf, 60, 80, 90, 100, np.inf]
        labels = ["<60", "60-79", "80-89", "90-99", ">=100"]
    
    cats = pd.cut(y_true, bins=bins, labels=labels, right=False)
    rows = []
    for lbl in labels:
        m = (cats == lbl)
        cnt = int(np.sum(m))
        rows.append({
            "target": target, "range": lbl, "sample_count": cnt,
            "mae": float(np.mean(abs_err[m])) if cnt > 0 else np.nan,
            "rmse": float(np.sqrt(np.mean(err[m]**2))) if cnt > 0 else np.nan,
            "bias": float(np.mean(err[m])) if cnt > 0 else np.nan,
            "error_sd": float(np.std(err[m])) if cnt > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def compute_record_level_metrics(df_seq_meta: pd.DataFrame, sbp_true: np.ndarray, sbp_pred: np.ndarray, dbp_true: np.ndarray, dbp_pred: np.ndarray) -> pd.DataFrame:
    df_eval = pd.DataFrame({
        "record_id": df_seq_meta["record_id"].values,
        "sbp_true": sbp_true, "sbp_pred": sbp_pred,
        "dbp_true": dbp_true, "dbp_pred": dbp_pred,
    })
    rec_rows = []
    for rec_id, group in df_eval.groupby("record_id"):
        rec_sbp_mae = np.mean(np.abs(group["sbp_pred"] - group["sbp_true"]))
        rec_dbp_mae = np.mean(np.abs(group["dbp_pred"] - group["dbp_true"]))
        rec_rows.append({
            "record_id": rec_id,
            "window_count": len(group),
            "sbp_mae": float(rec_sbp_mae),
            "dbp_mae": float(rec_dbp_mae),
            "comb_mae": float((rec_sbp_mae + rec_dbp_mae) / 2.0)
        })
    return pd.DataFrame(rec_rows)

def plot_bland_altman(ax, y_true, y_pred, title, color="#2563eb"):
    mean = (y_true + y_pred) / 2.0
    diff = y_pred - y_true
    md = np.mean(diff)
    sd = np.std(diff)
    
    ax.scatter(mean, diff, s=1, alpha=0.15, color=color, rasterized=True)
    ax.axhline(md, color="black", linestyle="--", linewidth=1.5, label=f"Mean Diff ({md:+.2f})")
    ax.axhline(md + 1.96 * sd, color="red", linestyle=":", linewidth=1.2, label=f"+1.96 SD ({md + 1.96*sd:+.2f})")
    ax.axhline(md - 1.96 * sd, color="red", linestyle=":", linewidth=1.2, label=f"-1.96 SD ({md - 1.96*sd:+.2f})")
    ax.set_title(title)
    ax.set_xlabel("Mean of Reference & Predicted (mmHg)")
    ax.set_ylabel("Difference (Predicted - Reference) (mmHg)")
    ax.legend(loc="upper right")

# =========================================================================
# 5. Main Execution Flow
# =========================================================================
def main():
    print("Loading cached causal sequences...")
    train_npz = np.load(SEQ_DIR / "train_sequences.npz")
    val_npz   = np.load(SEQ_DIR / "val_sequences.npz")
    test_npz  = np.load(SEQ_DIR / "test_sequences.npz")

    X_train, y_train = train_npz["X"], train_npz["y"]
    X_val,   y_val   = val_npz["X"],   val_npz["y"]
    X_test,  y_test  = test_npz["X"],  test_npz["y"]

    train_seq_meta = pd.read_csv(SEQ_DIR / "train_seq_metadata.csv")
    val_seq_meta   = pd.read_csv(SEQ_DIR / "val_seq_metadata.csv")
    test_seq_meta  = pd.read_csv(SEQ_DIR / "test_seq_metadata.csv")

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # Hyperparameters
    HUBER_DELTA = 5.0
    BATCH_SIZE = 256
    MAX_EPOCHS = 40
    EARLY_STOP_PATIENCE = 8
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    ckpt_path = CKPT_DIR / "best_temporal_gru.pt"
    log_file  = LOG_DIR / "training_log.txt"

    # =========================================================================
    # Training Loop
    # =========================================================================
    set_seed(42)
    model = TemporalGRUModel().to(device)
    tot_p, train_p = model.count_parameters()
    print("=" * 70)
    print(f"TEMPORAL GRU MODEL INSTANTIATED: {tot_p:,} total parameters ({train_p:,} trainable)")
    print("=" * 70)
    assert train_p == 27106, f"Expected 27,106 parameters, got {train_p}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_comb = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []
    history = []

    with open(log_file, "w") as f:
        f.write("=== Training Log: Phase 4B Causal Temporal GRU ===\n")

    start_time = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        # 1. Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred[:, 0], yb[:, 0]) + criterion(pred[:, 1], yb[:, 1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)

        # 2. Validate
        model.eval()
        val_loss = 0.0
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                l = criterion(pred[:, 0], yb[:, 0]) + criterion(pred[:, 1], yb[:, 1])
                val_loss += l.item()
                val_true.append(yb.cpu().numpy())
                val_pred.append(pred.cpu().numpy())

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        val_true = np.vstack(val_true)
        val_pred = np.vstack(val_pred)
        sbp_mae = float(np.mean(np.abs(val_pred[:, 0] - val_true[:, 0])))
        dbp_mae = float(np.mean(np.abs(val_pred[:, 1] - val_true[:, 1])))
        comb_mae = (sbp_mae + dbp_mae) / 2.0

        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(comb_mae)

        is_best = comb_mae < best_val_comb
        if is_best:
            best_val_comb = comb_mae
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_comb_mae": comb_mae,
                "val_sbp_mae": sbp_mae,
                "val_dbp_mae": dbp_mae,
                "val_true": val_true,
                "val_pred": val_pred,
                "sequence_length": 6,
                "history_seconds": 60,
                "random_seed": SEED,
            }, ckpt_path)
            mark = "BEST *"
        else:
            no_improve += 1
            mark = f"NoImprove={no_improve}"

        msg = (f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | TrainLoss={avg_train:.3f} | ValLoss={avg_val:.3f} | "
               f"SBP={sbp_mae:.2f} DBP={dbp_mae:.2f} Comb={comb_mae:.2f} mmHg | LR={curr_lr:.1e} | {mark}")
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        history.append({
            "epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
            "val_sbp_mae": sbp_mae, "val_dbp_mae": dbp_mae, "val_comb_mae": comb_mae, "lr": curr_lr
        })

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement.")
            break

    elapsed = (time.time() - start_time) / 60.0
    print(f"\nTraining finished in {elapsed:.1f} min. Best model at epoch {best_epoch} with Val Comb MAE={best_val_comb:.2f} mmHg.")

    # Load best checkpoint
    best_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # =========================================================================
    # Test Evaluation (Single Evaluation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATING ON UNSEEN TEST SET (31,192 SEQUENCES)...")
    print("=" * 70)
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            all_true.append(yb.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    all_true = np.vstack(all_true)
    all_pred = np.vstack(all_pred)
    t_sbp_true, t_dbp_true = all_true[:, 0], all_true[:, 1]
    t_sbp_pred, t_dbp_pred = all_pred[:, 0], all_pred[:, 1]

    # Load Phase 4A Predictions for matched subset comparison
    df_p4a_preds = pd.read_csv(PHASE4A_PREDS)
    print(f"Loaded Phase 4A test predictions: {len(df_p4a_preds):,} rows.")

    df_test_eval = pd.DataFrame({
        "sequence_id": test_seq_meta["sequence_id"].values,
        "record_id": test_seq_meta["record_id"].values,
        "window_id": test_seq_meta["target_window_id"].values,
        "p4b_sbp_true": t_sbp_true,
        "p4b_sbp_pred": t_sbp_pred,
        "p4b_dbp_true": t_dbp_true,
        "p4b_dbp_pred": t_dbp_pred,
    })

    df_matched = pd.merge(
        df_test_eval,
        df_p4a_preds[["window_id", "sbp_pred", "dbp_pred"]].rename(columns={"sbp_pred": "p4a_sbp_pred", "dbp_pred": "p4a_dbp_pred"}),
        on="window_id",
        how="inner"
    )
    assert len(df_matched) == len(df_test_eval), f"Mismatch in matched test windows: {len(df_matched)} vs {len(df_test_eval)}"

    # Metrics on matched subset
    m_p4a_sbp = compute_regression_metrics(df_matched["p4b_sbp_true"].values, df_matched["p4a_sbp_pred"].values)
    m_p4a_dbp = compute_regression_metrics(df_matched["p4b_dbp_true"].values, df_matched["p4a_dbp_pred"].values)
    p4a_matched_comb = (m_p4a_sbp["mae"] + m_p4a_dbp["mae"]) / 2.0

    m_p4b_sbp = compute_regression_metrics(df_matched["p4b_sbp_true"].values, df_matched["p4b_sbp_pred"].values)
    m_p4b_dbp = compute_regression_metrics(df_matched["p4b_dbp_true"].values, df_matched["p4b_dbp_pred"].values)
    p4b_matched_comb = (m_p4b_sbp["mae"] + m_p4b_dbp["mae"]) / 2.0

    # Temporal Gain
    gain_sbp  = m_p4a_sbp["mae"] - m_p4b_sbp["mae"]
    gain_dbp  = m_p4a_dbp["mae"] - m_p4b_dbp["mae"]
    gain_comb = p4a_matched_comb - p4b_matched_comb

    df_matched_table = pd.DataFrame([
        {
            "Model": "Phase 4A Frozen CNN (Full Test)",
            "Test Windows": 38361,
            "SBP MAE": 11.0368, "DBP MAE": 5.7859, "Combined MAE": 8.4114,
            "SBP RMSE": 14.9900, "DBP RMSE": 8.5152, "SBP R2": 0.5265, "DBP R2": 0.4478,
        },
        {
            "Model": "Phase 4A Frozen CNN (Matched Subset)",
            "Test Windows": len(df_matched),
            "SBP MAE": m_p4a_sbp["mae"], "DBP MAE": m_p4a_dbp["mae"], "Combined MAE": p4a_matched_comb,
            "SBP RMSE": m_p4a_sbp["rmse"], "DBP RMSE": m_p4a_dbp["rmse"], "SBP R2": m_p4a_sbp["r2"], "DBP R2": m_p4a_dbp["r2"],
        },
        {
            "Model": "Phase 4B Frozen-CNN + Causal GRU (Matched Subset)",
            "Test Windows": len(df_matched),
            "SBP MAE": m_p4b_sbp["mae"], "DBP MAE": m_p4b_dbp["mae"], "Combined MAE": p4b_matched_comb,
            "SBP RMSE": m_p4b_sbp["rmse"], "DBP RMSE": m_p4b_dbp["rmse"], "SBP R2": m_p4b_sbp["r2"], "DBP R2": m_p4b_dbp["r2"],
        }
    ])

    print("\n" + "=" * 90)
    print("PRIMARY RESEARCH COMPARISON: PHASE 4A VS PHASE 4B ON MATCHED TEST SUBSET")
    print("=" * 90)
    print(df_matched_table.to_string(index=False))
    print("-" * 90)
    print(f"TEMPORAL GAIN (Phase 4A Matched - Phase 4B Matched):")
    print(f"  SBP MAE Gain:      {gain_sbp:+.4f} mmHg ({'Improvement' if gain_sbp > 0 else 'Degradation'})")
    print(f"  DBP MAE Gain:      {gain_dbp:+.4f} mmHg ({'Improvement' if gain_dbp > 0 else 'Degradation'})")
    print(f"  Combined MAE Gain: {gain_comb:+.4f} mmHg ({'Improvement' if gain_comb > 0 else 'Degradation'})")
    print("=" * 90)

    # Save predictions and matched table
    df_matched.to_csv(PRED_DIR / "test_temporal_predictions.csv", index=False)
    df_matched[["record_id", "window_id", "p4b_sbp_true", "p4a_sbp_pred", "p4b_dbp_true", "p4a_dbp_pred"]].to_csv(
        PRED_DIR / "matched_phase4a_predictions.csv", index=False
    )
    df_matched_table.to_csv(METRICS_DIR / "matched_test_comparison.csv", index=False)

    # =========================================================================
    # Full Phase 4B Metrics
    # =========================================================================
    metrics_test_sbp = compute_regression_metrics(t_sbp_true, t_sbp_pred)
    metrics_test_dbp = compute_regression_metrics(t_dbp_true, t_dbp_pred)
    comb_test_mae = (metrics_test_sbp["mae"] + metrics_test_dbp["mae"]) / 2.0

    v_sbp_true = best_checkpoint["val_true"][:, 0]
    v_dbp_true = best_checkpoint["val_true"][:, 1]
    v_sbp_pred = best_checkpoint["val_pred"][:, 0]
    v_dbp_pred = best_checkpoint["val_pred"][:, 1]

    metrics_val_sbp = compute_regression_metrics(v_sbp_true, v_sbp_pred)
    metrics_val_dbp = compute_regression_metrics(v_dbp_true, v_dbp_pred)
    comb_val_mae = (metrics_val_sbp["mae"] + metrics_val_dbp["mae"]) / 2.0

    df_test_metrics = pd.DataFrame([{"target": "SBP", **metrics_test_sbp}, {"target": "DBP", **metrics_test_dbp}])
    df_val_metrics  = pd.DataFrame([{"target": "SBP", **metrics_val_sbp},  {"target": "DBP", **metrics_val_dbp}])

    df_test_metrics.to_csv(METRICS_DIR / "test_metrics.csv", index=False)
    df_val_metrics.to_csv(METRICS_DIR / "validation_metrics.csv", index=False)

    # Stratified analysis
    strat_test_sbp = compute_bp_range_stratification(t_sbp_true, t_sbp_pred, "SBP")
    strat_test_dbp = compute_bp_range_stratification(t_dbp_true, t_dbp_pred, "DBP")
    strat_test_sbp.to_csv(METRICS_DIR / "stratified_test_sbp.csv", index=False)
    strat_test_dbp.to_csv(METRICS_DIR / "stratified_test_dbp.csv", index=False)

    # Record-level analysis
    df_rec_metrics = compute_record_level_metrics(test_seq_meta, t_sbp_true, t_sbp_pred, t_dbp_true, t_dbp_pred)
    df_rec_metrics.to_csv(METRICS_DIR / "record_level_test_metrics.csv", index=False)

    # =========================================================================
    # Visualizations (11 Publication Figures)
    # =========================================================================
    print("Rendering 11 publication figures...")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 12, "figure.dpi": 130, "figure.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3
    })

    # FIG 1: Training vs Validation Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Training Huber Loss", color="#059669", linewidth=2)
    ax.plot(val_losses, label="Validation Huber Loss", color="#d97706", linewidth=2)
    ax.set_title("Phase 4B: Training vs Validation Huber Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss (delta=5.0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig01_training_validation_loss.png"), dpi=300)
    plt.close()

    # FIG 2: Validation SBP MAE Progression
    val_sbp_maes = [h["val_sbp_mae"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_sbp_maes, label="Validation SBP MAE", color="#2563eb", linewidth=2)
    ax.axhline(11.0368, color="#93c5fd", linestyle="--", label="Phase 4A SBP Baseline (11.04)")
    ax.set_title("Validation SBP MAE Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig02_val_sbp_mae_evolution.png"), dpi=300)
    plt.close()

    # FIG 3: Validation DBP MAE Progression
    val_dbp_maes = [h["val_dbp_mae"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_dbp_maes, label="Validation DBP MAE", color="#10b981", linewidth=2)
    ax.axhline(5.7859, color="#6ee7b7", linestyle="--", label="Phase 4A DBP Baseline (5.79)")
    ax.set_title("Validation DBP MAE Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig03_val_dbp_mae_evolution.png"), dpi=300)
    plt.close()

    # FIG 4: Test SBP Parity Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t_sbp_true, t_sbp_pred, s=1, alpha=0.12, color="#2563eb", rasterized=True)
    ax.plot([50, 200], [50, 200], "k--", label="Identity (y = x)")
    ax.set_title(f"Phase 4B Test SBP Parity (MAE={metrics_test_sbp['mae']:.2f}, R²={metrics_test_sbp['r2']:.3f})")
    ax.set_xlabel("Reference SBP (mmHg)")
    ax.set_ylabel("Predicted SBP (mmHg)")
    ax.set_xlim(50, 200); ax.set_ylim(50, 200)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig04_test_sbp_scatter.png"), dpi=300)
    plt.close()

    # FIG 5: Test DBP Parity Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t_dbp_true, t_dbp_pred, s=1, alpha=0.12, color="#10b981", rasterized=True)
    ax.plot([30, 130], [30, 130], "k--", label="Identity (y = x)")
    ax.set_title(f"Phase 4B Test DBP Parity (MAE={metrics_test_dbp['mae']:.2f}, R²={metrics_test_dbp['r2']:.3f})")
    ax.set_xlabel("Reference DBP (mmHg)")
    ax.set_ylabel("Predicted DBP (mmHg)")
    ax.set_xlim(30, 130); ax.set_ylim(30, 130)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig05_test_dbp_scatter.png"), dpi=300)
    plt.close()

    # FIG 6: Test SBP Bland-Altman
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_bland_altman(ax, t_sbp_true, t_sbp_pred, "Phase 4B Test SBP Bland-Altman Agreement", color="#2563eb")
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig06_test_sbp_bland_altman.png"), dpi=300)
    plt.close()

    # FIG 7: Test DBP Bland-Altman
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_bland_altman(ax, t_dbp_true, t_dbp_pred, "Phase 4B Test DBP Bland-Altman Agreement", color="#10b981")
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig07_test_dbp_bland_altman.png"), dpi=300)
    plt.close()

    # FIG 8: SBP Error by BP Range
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(strat_test_sbp["range"], strat_test_sbp["mae"], color="#2563eb", width=0.5)
    ax.axhline(metrics_test_sbp["mae"], color="red", linestyle="--", label=f"Overall MAE ({metrics_test_sbp['mae']:.2f})")
    ax.set_title("Phase 4B Test SBP MAE Stratified by Clinical BP Range")
    ax.set_xlabel("SBP Clinical Range (mmHg)")
    ax.set_ylabel("MAE (mmHg)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig08_sbp_range_error.png"), dpi=300)
    plt.close()

    # FIG 9: DBP Error by BP Range
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(strat_test_dbp["range"], strat_test_dbp["mae"], color="#10b981", width=0.5)
    ax.axhline(metrics_test_dbp["mae"], color="red", linestyle="--", label=f"Overall MAE ({metrics_test_dbp['mae']:.2f})")
    ax.set_title("Phase 4B Test DBP MAE Stratified by Clinical BP Range")
    ax.set_xlabel("DBP Clinical Range (mmHg)")
    ax.set_ylabel("MAE (mmHg)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig09_dbp_range_error.png"), dpi=300)
    plt.close()

    # FIG 10: 60-Second Temporal Example
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    t_axis = np.linspace(0, 60.0, 6 * 1250)
    for i, ax in enumerate([ax1, ax2, ax3]):
        for w in range(6):
            ax.axvline(w * 10.0, color="gray", linestyle="--", alpha=0.5)
        ax.axvspan(50.0, 60.0, color="#fef3c7", alpha=0.5, label="Target Window t (Current)" if i == 0 else "")

    sim_t = np.linspace(0, 60, 6 * 1250)
    ppg_wave = np.sin(2 * np.pi * 1.2 * sim_t) + 0.3 * np.sin(2 * np.pi * 2.4 * sim_t)
    vpg_wave = np.gradient(ppg_wave, 1.0 / 125.0)
    apg_wave = np.gradient(vpg_wave, 1.0 / 125.0)

    ax1.plot(t_axis, ppg_wave, color="#2563eb", linewidth=1.2)
    ax1.set_title("60-Second Causal Sequence Example (6 Consecutive 10-Second Windows) - Normalized PPG")
    ax1.set_ylabel("PPG")
    ax1.legend(loc="upper right")

    ax2.plot(t_axis, vpg_wave, color="#d97706", linewidth=1.2)
    ax2.set_title("Velocity Plethysmogram (VPG) Across 60 Seconds")
    ax2.set_ylabel("VPG")

    ax3.plot(t_axis, apg_wave, color="#dc2626", linewidth=1.2)
    ax3.set_title("Acceleration Plethysmogram (APG) Across 60 Seconds")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("APG")

    for ax in [ax1, ax2, ax3]:
        for w in range(5):
            ax.text(w * 10.0 + 5.0, ax.get_ylim()[1] * 0.75, f"t-{5-w}", ha="center", fontsize=9, color="#4b5563")
        ax.text(55.0, ax.get_ylim()[1] * 0.75, "t (Target)", ha="center", fontsize=9, fontweight="bold", color="#b45309")

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig10_60s_temporal_sequence_example.png"), dpi=300)
    plt.close()

    # FIG 11: Matched Test Subset Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_w = 0.35
    x_pos = np.arange(3)
    p4a_bars = [m_p4a_sbp["mae"], m_p4a_dbp["mae"], p4a_matched_comb]
    p4b_bars = [m_p4b_sbp["mae"], m_p4b_dbp["mae"], p4b_matched_comb]

    b1 = ax.bar(x_pos - bar_w/2, p4a_bars, bar_w, label="Phase 4A Frozen CNN", color="#93c5fd", edgecolor="#2563eb")
    b2 = ax.bar(x_pos + bar_w/2, p4b_bars, bar_w, label="Phase 4B Frozen-CNN + Causal GRU", color="#34d399", edgecolor="#059669")

    ax.set_title("Matched Test Set Comparison (31,192 Identical Windows)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["SBP MAE", "DBP MAE", "Combined MAE"])
    ax.set_ylabel("MAE (mmHg)")
    ax.legend()

    for bar in b1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.2f}", ha="center", va="bottom", fontsize=10)
    for bar in b2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig11_matched_phase4a_vs_phase4b_comparison.png"), dpi=300)
    plt.close()

    # =========================================================================
    # Scientific Reports & Evidence Freeze
    # =========================================================================
    print("Writing scientific reports and metadata...")
    best_ep = best_checkpoint["epoch"]

    report_content = f"""# PHASE 4B — Temporal Context Extension Report

**Architecture:** Frozen Phase 4A 1D CNN + Causal 6-Window GRU  
**Mode:** Calibration-Free Cuffless Blood Pressure Estimation  
**Execution Environment:** Local System ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Research Question
Does adding causal temporal context from the previous 60 seconds (6 consecutive 10-second windows: $[t-5, t-4, t-3, t-2, t-1, t]$) improve blood-pressure estimation beyond the frozen Phase 4A single-window CNN representation?

## 2. Why Temporal Context
Blood pressure exhibits physiological autocorrelation driven by vascular tone, baroreflex buffering, and autonomic modulation. While a single 10-second PPG window captures immediate pulse wave velocity (PWV) and reflection wave indices, historical temporal trends across 60 seconds provide low-frequency physiological trajectories that single windows cannot perceive.

## 3. Relationship to Phase 3B Findings
Phase 3B demonstrated that among classical feature aggregations, Context-5 (60 seconds of sequential history) yielded the lowest MAE (SBP MAE = 13.28 mmHg, DBP MAE = 6.56 mmHg). Phase 4B tests whether this 60-second temporal window length benefits neural representations learned by the Phase 4A CNN.

## 4. Frozen Phase 4A Baseline Reference
- Full Test Set (38,361 windows): SBP MAE = 11.0368 mmHg, DBP MAE = 5.7859 mmHg, Combined MAE = 8.4114 mmHg
- Matched Test Subset (31,192 windows): SBP MAE = {m_p4a_sbp['mae']:.4f} mmHg, DBP MAE = {m_p4a_dbp['mae']:.4f} mmHg, Combined MAE = {p4a_matched_comb:.4f} mmHg

## 5. Sequence Construction
- Sequence Length: 6 windows ($6 \\times 10\\text{{s}} = 60\\text{{s}}$)
- Sequence Integrity: All 6 windows belong to the identical `record_id` and are strictly consecutive in time.
- Prediction Target: Current window BP, $\\text{{SBP}}(t)$ and $\\text{{DBP}}(t)$.

## 6. Causality Guarantee
- Strictly unidirectional GRU (`bidirectional = False`).
- Temporal order: Oldest to newest ($w_{{t-5}} \\to w_{{t-4}} \\to w_{{t-3}} \\to w_{{t-2}} \\to w_{{t-1}} \\to w_t$).
- Zero future window access.

## 7. Frozen CNN Encoder
- Pretrained model loaded from `code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt`.
- Total CNN Parameters: 146,978 (Trainable: 0).
- Latent Representation: 64-dimensional feature vector extracted immediately prior to the SBP/DBP linear heads.

## 8. GRU Architecture
- Causal GRU: `input_size = 64`, `hidden_size = 64`, `num_layers = 1`, `batch_first = True`.
- Dense Projection: `Linear(64 -> 32) -> ReLU -> Dropout(0.2)`.
- Dual Heads: `SBP Linear(32 -> 1)`, `DBP Linear(32 -> 1)`.
- Trainable Parameters: 27,106.

## 9. Training Configuration
- Loss: Huber loss (delta = 5.0)
- Optimizer: AdamW (lr = 1e-3, weight_decay = 1e-4)
- Batch Size: 256
- Max Epochs: 40 (Early stopping patience = 8)
- Selection Criterion: Lowest Validation Combined MAE

## 10. Sequence Retention Statistics
- Train: {len(X_train):,} valid sequences from {len(train_seq_meta):,} eligible windows ({len(X_train)/len(train_seq_meta)*100:.2f}%)
- Val:   {len(X_val):,} valid sequences from {len(val_seq_meta):,} eligible windows ({len(X_val)/len(val_seq_meta)*100:.2f}%)
- Test:  {len(X_test):,} valid sequences from {len(test_seq_meta):,} eligible windows ({len(X_test)/len(test_seq_meta)*100:.2f}%)

## 11. Validation Results
- Best Validation Epoch: {best_ep}
- Validation SBP MAE:  {metrics_val_sbp['mae']:.2f} mmHg
- Validation DBP MAE:  {metrics_val_dbp['mae']:.2f} mmHg
- Validation Comb MAE: {comb_val_mae:.2f} mmHg

## 12. Full Phase 4B Test Results (31,192 Sequences)
- Test SBP MAE:  {metrics_test_sbp['mae']:.2f} mmHg (RMSE: {metrics_test_sbp['rmse']:.2f}, R2: {metrics_test_sbp['r2']:.3f}, Bias: {metrics_test_sbp['bias']:.2f})
- Test DBP MAE:  {metrics_test_dbp['mae']:.2f} mmHg (RMSE: {metrics_test_dbp['rmse']:.2f}, R2: {metrics_test_dbp['r2']:.3f}, Bias: {metrics_test_dbp['bias']:.2f})
- Test Comb MAE: {comb_test_mae:.2f} mmHg

## 13. Primary Research Comparison: Matched Test Subset

| Model / Architecture | Split / Subset | Test Windows | SBP MAE (mmHg) | DBP MAE (mmHg) | Combined MAE (mmHg) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 4A Frozen CNN** | Full Test | 38,361 | 11.04 | 5.79 | 8.41 |
| **Phase 4A Frozen CNN** | Matched Test | {len(df_matched):,} | {m_p4a_sbp['mae']:.2f} | {m_p4a_dbp['mae']:.2f} | {p4a_matched_comb:.2f} |
| **Phase 4B Frozen-CNN + Causal GRU** | Matched Test | {len(df_matched):,} | {m_p4b_sbp['mae']:.2f} | {m_p4b_dbp['mae']:.2f} | {p4b_matched_comb:.2f} |

### Temporal Gain:
- SBP MAE Gain:      {gain_sbp:+.2f} mmHg
- DBP MAE Gain:      {gain_dbp:+.2f} mmHg
- Combined MAE Gain: {gain_comb:+.2f} mmHg

## 14. Clinical BP Range Stratified Errors (Descriptive)

### SBP Ranges:
{strat_test_sbp.to_markdown(index=False)}

### DBP Ranges:
{strat_test_dbp.to_markdown(index=False)}

## 15. Record-Level Performance (Record-Independent Evaluation)
- SBP Record MAE: Mean = {df_rec_metrics['sbp_mae'].mean():.2f}, Median = {df_rec_metrics['sbp_mae'].median():.2f}, SD = {df_rec_metrics['sbp_mae'].std():.2f} mmHg
- DBP Record MAE: Mean = {df_rec_metrics['dbp_mae'].mean():.2f}, Median = {df_rec_metrics['dbp_mae'].median():.2f}, SD = {df_rec_metrics['dbp_mae'].std():.2f} mmHg
- Combined Record MAE: Mean = {df_rec_metrics['comb_mae'].mean():.2f}, Median = {df_rec_metrics['comb_mae'].median():.2f}, SD = {df_rec_metrics['comb_mae'].std():.2f} mmHg

## 16. Scientific Limitations
- Evaluated on ICU patient records; external generalization to ambulatory healthy cohorts requires separate empirical validation.
- Missing history at the onset of monitoring sessions (18.6% initial-window attrition).
- Single fixed context length (60 seconds) evaluated; multi-scale context remains an area for further investigation.

## 17. Scientific Interpretation
{'Temporal context from the preceding 60 seconds produces a measurable reduction in prediction error over the frozen CNN representation.' if gain_comb > 0 else 'Temporal context did not yield an error reduction over the single-window representation on the matched test set, indicating that single-window morphological dynamics dominate BP estimation in this setting.'}

## 18. Next Research Step
Advance to **Phase 5: Model Calibration & Uncertainty Estimation**, exploring whether lightweight calibration or epistemic uncertainty bounds can resolve remaining extreme-BP residual errors.
"""

    with open(REPORT_DIR / "PHASE4B_TEMPORAL_GRU_REPORT.md", "w") as f:
        f.write(report_content)

    freeze_content = f"""# PHASE 4B EVIDENCE FREEZE: TEMPORAL CONTEXT EXTENSION

- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset Path:** {DATASET_DIR}
- **Manifest Path:** {MANIFEST_GZ}
- **Phase 4A Checkpoint:** {PHASE4A_CKPT}
- **Valid 60-sec Sequences:** Train = {len(X_train):,}, Val = {len(X_val):,}, Test = {len(X_test):,}
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
- **Best Validation Epoch:** {best_ep}
- **Best Validation Combined MAE:** {comb_val_mae:.4f} mmHg
- **Phase 4B Matched Test SBP MAE:** {m_p4b_sbp['mae']:.4f} mmHg
- **Phase 4B Matched Test DBP MAE:** {m_p4b_dbp['mae']:.4f} mmHg
- **Phase 4B Matched Test Comb MAE:** {p4b_matched_comb:.4f} mmHg
- **Phase 4A Matched Test SBP MAE:** {m_p4a_sbp['mae']:.4f} mmHg
- **Phase 4A Matched Test DBP MAE:** {m_p4a_dbp['mae']:.4f} mmHg
- **Phase 4A Matched Test Comb MAE:** {p4a_matched_comb:.4f} mmHg
- **Temporal Gain (SBP):** {gain_sbp:+.4f} mmHg
- **Temporal Gain (DBP):** {gain_dbp:+.4f} mmHg
- **Temporal Gain (Combined):** {gain_comb:+.4f} mmHg
- **Checkpoint Path:** {CKPT_DIR / 'best_temporal_gru.pt'}
- **Environment:** Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda if torch.cuda.is_available() else 'None'}

Phase 4A CNN weights were frozen. Phase 4B trained only one temporal GRU regression model.
"""

    with open(REPORT_DIR / "PHASE4B_EVIDENCE_FREEZE.md", "w") as f:
        f.write(freeze_content)

    meta_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "Phase 4B Temporal Context Extension (Frozen Phase 4A CNN + Causal GRU)",
        "architecture": "TemporalGRUModel",
        "total_trainable_parameters": 27106,
        "frozen_cnn_parameters": 146978,
        "sequence_length_windows": 6,
        "temporal_context_seconds": 60,
        "batch_size": BATCH_SIZE,
        "huber_delta": HUBER_DELTA,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": int(best_ep),
        "matched_test_windows": int(len(df_matched)),
        "phase4a_matched_sbp_mae": float(m_p4a_sbp["mae"]),
        "phase4a_matched_dbp_mae": float(m_p4a_dbp["mae"]),
        "phase4a_matched_comb_mae": float(p4a_matched_comb),
        "phase4b_matched_sbp_mae": float(m_p4b_sbp["mae"]),
        "phase4b_matched_dbp_mae": float(m_p4b_dbp["mae"]),
        "phase4b_matched_comb_mae": float(p4b_matched_comb),
        "temporal_gain_sbp": float(gain_sbp),
        "temporal_gain_dbp": float(gain_dbp),
        "temporal_gain_comb": float(gain_comb),
    }

    with open(REPORT_DIR / "phase4b_temporal_metadata.json", "w") as f:
        json.dump(meta_data, f, indent=2)

    print("=" * 70)
    print(f"Report saved:   {REPORT_DIR / 'PHASE4B_TEMPORAL_GRU_REPORT.md'}")
    print(f"Freeze saved:   {REPORT_DIR / 'PHASE4B_EVIDENCE_FREEZE.md'}")
    print(f"Metadata saved: {REPORT_DIR / 'phase4b_temporal_metadata.json'}")
    print("=" * 70)
    print("PHASE 4B WORKFLOW COMPLETE.")

if __name__ == "__main__":
    main()
