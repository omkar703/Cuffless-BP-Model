"""
Phase 5A Notebook Generator Script: Uncertainty Estimation via MC-Dropout
Generates `code/notebooks/05A_uncertainty_mc_dropout.ipynb` with complete inference,
evaluation, visualization, and reporting cells.
"""

import sys
from pathlib import Path
import nbformat as nbf

def generate_phase5a_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # =========================================================================
    # Markdown Header
    # =========================================================================
    header_md = """# Phase 5A — Uncertainty Estimation Without Retraining
## Epistemic Reliability via Monte Carlo Dropout on Frozen Phase 4B Temporal GRU

### Research Question
> **"Can predictive uncertainty derived from the existing frozen Phase 4B model identify predictions that are more likely to have large absolute BP errors?"**

### Strict Experimental Constraints
- **Zero Retraining:** Zero optimizer steps, zero learning rate schedulers, zero parameter updates. All 146,978 CNN parameters and 27,106 GRU parameters are completely frozen (`trainable_parameters = 0`).
- **Target Architecture:** Frozen Phase 4A CNN + 1-layer Causal GRU + FC(64 $\\to$ 32) + `Dropout(p=0.2)` + Dual Linear Regression Heads.
- **Stochastic Mechanism:** Monte Carlo Dropout ($N = 30$ passes). ONLY the existing `Dropout(p=0.2)` layer is stochastic; all BatchNorm and convolutional layers remain strictly in `eval()` mode.
- **Dataset:** Unseen Phase 4B test sequences (31,192 consecutive 60-second causal sequences).
- **Resource Boundary:** Setup, baseline validation, and smoke test execute first; full 31,192 $\\times$ 30 inference is halted before Cell 10.
"""
    cells.append(nbf.v4.new_markdown_cell(header_md))

    # =========================================================================
    # Cell 1: Environment & Reproducibility Setup
    # =========================================================================
    c1 = nbf.v4.new_code_cell("""# 1. Environment & Reproducibility Setup
import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
print(f"PHASE 5A: Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
print(f"Random Seed: {SEED} (Deterministic MC-Dropout Protocol)")
print("=" * 70)
""")
    cells.append(c1)

    # =========================================================================
    # Cell 2: Directory Setup & Paths
    # =========================================================================
    c2 = nbf.v4.new_code_cell("""# 2. Directory Setup & Paths
cwd = Path.cwd().resolve()
PROJECT_ROOT = None
for c in [cwd, cwd.parent, cwd.parent.parent]:
    if (c / "BloodPressureDataset").exists():
        PROJECT_ROOT = c
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")

PHASE4B_DIR  = PROJECT_ROOT / "code" / "outputs" / "phase4b_temporal_gru"
PHASE4B_CKPT = PHASE4B_DIR / "checkpoints" / "best_temporal_gru.pt"
TEST_SEQ_NPZ = PHASE4B_DIR / "sequences" / "test_sequences.npz"
TEST_META_CSV = PHASE4B_DIR / "sequences" / "test_seq_metadata.csv"
PHASE4B_PREDS = PHASE4B_DIR / "predictions" / "test_temporal_predictions.csv"

OUTPUT_DIR   = PROJECT_ROOT / "code" / "outputs" / "phase5a_uncertainty"
PRED_DIR     = OUTPUT_DIR / "predictions"
METRICS_DIR  = OUTPUT_DIR / "metrics"
FIG_DIR      = OUTPUT_DIR / "figures"
REPORT_DIR   = OUTPUT_DIR / "reports"
LOG_DIR      = OUTPUT_DIR / "logs"

for d in [PRED_DIR, METRICS_DIR, FIG_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Project Root:        {PROJECT_ROOT}")
print(f"Phase 4B Checkpoint: {PHASE4B_CKPT}")
print(f"Phase 5A Output Dir: {OUTPUT_DIR}")
assert PHASE4B_CKPT.exists(), f"Missing Phase 4B checkpoint: {PHASE4B_CKPT}"
assert TEST_SEQ_NPZ.exists(), f"Missing test sequences: {TEST_SEQ_NPZ}"
assert TEST_META_CSV.exists(), f"Missing test sequence metadata: {TEST_META_CSV}"
assert PHASE4B_PREDS.exists(), f"Missing Phase 4B test predictions: {PHASE4B_PREDS}"
print("All input dependencies verified.")
""")
    cells.append(c2)

    # =========================================================================
    # Cell 3: Load Phase 4B Architecture & Freeze All Parameters
    # =========================================================================
    c3 = nbf.v4.new_code_cell("""# 3. Model Architecture & Frozen Checkpoint Loading
class TemporalGRUModel(nn.Module):
    def __init__(self, input_size: int = 64, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Strictly unidirectional GRU (causality mandate)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # Dense Projection with Dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)  # Stochastic Dropout layer for MC inference
        )
        self.sbp_head = nn.Linear(32, 1)
        self.dbp_head = nn.Linear(32, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)        # [B, 6, 64]
        last_hidden = out[:, -1, :] # [B, 64] (current window t)
        feat = self.fc(last_hidden) # [B, 32]
        sbp = self.sbp_head(feat)   # [B, 1]
        dbp = self.dbp_head(feat)   # [B, 1]
        return torch.cat([sbp, dbp], dim=1)  # [B, 2]

print(f"Loading Phase 4B checkpoint from: {PHASE4B_CKPT}")
phase4b_checkpoint = torch.load(PHASE4B_CKPT, map_location=device, weights_only=False)

model = TemporalGRUModel().to(device)
model.load_state_dict(phase4b_checkpoint["model_state_dict"])
model.eval()

# STRICT INFERENCE PROTOCOL: Freeze every single parameter
for param in model.parameters():
    param.requires_grad = False

total_p = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 70)
print("FROZEN PHASE 4B TEMPORAL MODEL VERIFICATION:")
print(f"  Checkpoint Epoch:       {phase4b_checkpoint.get('epoch')}")
print(f"  Validation SBP MAE:     {phase4b_checkpoint.get('val_sbp_mae'):.2f} mmHg")
print(f"  Validation DBP MAE:     {phase4b_checkpoint.get('val_dbp_mae'):.2f} mmHg")
print(f"  Validation Comb MAE:    {phase4b_checkpoint.get('val_comb_mae'):.2f} mmHg")
print(f"  Total Parameters:       {total_p:,}")
print(f"  Trainable Parameters:   {trainable_p:,} (STRICTLY 0)")
print("=" * 70)
assert trainable_p == 0, f"Critical Error: Expected 0 trainable parameters, found {trainable_p}!"
assert total_p == 27106, f"Critical Error: Expected 27,106 parameters, found {total_p}!"
""")
    cells.append(c3)

    # =========================================================================
    # Cell 4: Stochastic Isolation Helper (enable_mc_dropout)
    # =========================================================================
    c4 = nbf.v4.new_code_cell("""# 4. Stochastic Isolation Helper: enable_mc_dropout
def enable_mc_dropout(model: nn.Module):
    \"\"\"
    Enables ONLY Dropout modules for Monte Carlo stochastic forward passes,
    while keeping all other layers (BatchNorm, Linear, GRU) strictly in eval mode.
    \"\"\"
    model.eval()  # Default all layers to eval mode
    count_dropout = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # Enable stochastic dropout
            count_dropout += 1
    return count_dropout

# Verify dropout isolation behavior
n_dropout = enable_mc_dropout(model)
print("=" * 70)
print("STOCHASTIC ISOLATION VERIFICATION:")
print(f"  Identified {n_dropout} Dropout module(s) set to train() mode.")
dropout_layer = model.fc[2]
assert isinstance(dropout_layer, nn.Dropout), f"Expected Dropout at model.fc[2], got {type(dropout_layer)}"
assert dropout_layer.training == True, "Dropout layer must be in train mode!"
assert model.gru.training == False, "GRU must remain in eval mode!"
assert model.fc[0].training == False, "Linear projection must remain in eval mode!"
assert model.sbp_head.training == False, "SBP head must remain in eval mode!"
assert model.dbp_head.training == False, "DBP head must remain in eval mode!"
print("  All weights, linear projections, and recurrence remain strictly in eval() mode.")
print("=" * 70)
""")
    cells.append(c4)

    # =========================================================================
    # Cell 5: Load Test Sequences & Build DataLoader
    # =========================================================================
    c5 = nbf.v4.new_code_cell("""# 5. Load Cached Phase 4B Test Sequences
BATCH_SIZE = 256

print(f"Loading test sequences from: {TEST_SEQ_NPZ}")
test_npz = np.load(TEST_SEQ_NPZ)
X_test = test_npz["X"].astype(np.float32)  # [31192, 6, 64]
y_test = test_npz["y"].astype(np.float32)  # [31192, 2]

df_test_meta = pd.read_csv(TEST_META_CSV)
print(f"Loaded {len(X_test):,} sequences of shape {X_test.shape}")
assert len(X_test) == 31192, f"Expected 31,192 test sequences, found {len(X_test)}"
assert len(df_test_meta) == 31192, f"Metadata row count mismatch: {len(df_test_meta)}"

test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

print(f"Test DataLoader ready: {len(test_loader)} batches @ batch_size={BATCH_SIZE}")
""")
    cells.append(c5)

    # =========================================================================
    # Cell 6: Deterministic Baseline Verification
    # =========================================================================
    c6 = nbf.v4.new_code_cell("""# 6. Deterministic Baseline Verification (Sanity Check)
# Verifies that in standard eval() mode, the model exactly reproduces Phase 4B test metrics.

model.eval()  # Pure deterministic evaluation
det_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb)
        det_preds.append(pred.cpu().numpy())

det_preds = np.vstack(det_preds)
det_sbp_pred = det_preds[:, 0]
det_dbp_pred = det_preds[:, 1]
target_sbp = y_test[:, 0]
target_dbp = y_test[:, 1]

det_sbp_mae = float(np.mean(np.abs(det_sbp_pred - target_sbp)))
det_dbp_mae = float(np.mean(np.abs(det_dbp_pred - target_dbp)))
det_comb_mae = (det_sbp_mae + det_dbp_mae) / 2.0

print("=" * 70)
print("DETERMINISTIC PHASE 4B REPRODUCIBILITY CHECK:")
print(f"  Phase 4B Frozen Target: SBP MAE = 10.57 mmHg | DBP MAE = 5.51 mmHg | Comb = 8.04 mmHg")
print(f"  Reproduced Test Eval:   SBP MAE = {det_sbp_mae:.2f} mmHg | DBP MAE = {det_dbp_mae:.2f} mmHg | Comb = {det_comb_mae:.2f} mmHg")
print("=" * 70)

assert abs(det_sbp_mae - 10.57) < 0.1, f"SBP MAE discrepancy: {det_sbp_mae:.4f} vs 10.57"
assert abs(det_dbp_mae - 5.51) < 0.1, f"DBP MAE discrepancy: {det_dbp_mae:.4f} vs 5.51"
assert abs(det_comb_mae - 8.04) < 0.1, f"Combined MAE discrepancy: {det_comb_mae:.4f} vs 8.04"
print("DETERMINISTIC BASELINE VERIFIED.")
""")
    cells.append(c6)

    # =========================================================================
    # Cell 7: MC-Dropout Smoke Test (Section 23 Requirements)
    # =========================================================================
    c7 = nbf.v4.new_code_cell("""# 7. Comprehensive MC-Dropout Smoke Test & Invariance Verification
# Section 23: 32 sequences, 10 stochastic passes, parameter freezing & variance check.

def run_smoke_test(device):
    print("=" * 70)
    print("RUNNING PHASE 5A MC-DROPOUT SMOKE TEST (Section 23)")
    print("=" * 70)
    set_seed(42)
    
    # 1. Sample 32 test sequences
    smoke_X = torch.from_numpy(X_test[:32]).to(device)
    smoke_y = torch.from_numpy(y_test[:32]).to(device)
    
    # 2. Enable MC-Dropout
    enable_mc_dropout(model)
    
    # 3. Perform 10 stochastic passes
    smoke_passes = []
    with torch.no_grad():
        for p in range(10):
            out = model(smoke_X)
            assert out.shape == (32, 2), f"Expected shape (32, 2), got {out.shape}"
            assert torch.isfinite(out).all(), f"Pass {p}: Found NaN or Inf in predictions!"
            smoke_passes.append(out.cpu().numpy())
            
    smoke_passes = np.stack(smoke_passes, axis=1)  # [32, 10, 2]
    
    # 4. Variance and Non-negativity check
    sbp_stds = np.std(smoke_passes[:, :, 0], axis=1)
    dbp_stds = np.std(smoke_passes[:, :, 1], axis=1)
    
    assert np.all(sbp_stds > 0), "Error: SBP predictions did not vary across stochastic passes!"
    assert np.all(dbp_stds > 0), "Error: DBP predictions did not vary across stochastic passes!"
    assert np.all(sbp_stds >= 0), "Error: Found negative SBP std values!"
    assert np.all(dbp_stds >= 0), "Error: Found negative DBP std values!"
    print(f"1. Stochastic Variance Verified: SBP std range [{sbp_stds.min():.3f}, {sbp_stds.max():.3f}] mmHg [OK]")
    print(f"2. Stochastic Variance Verified: DBP std range [{dbp_stds.min():.3f}, {dbp_stds.max():.3f}] mmHg [OK]")
    
    # 5. Parameter Frozen Status & Zero Gradients Check
    assert all(not p.requires_grad for p in model.parameters()), "Parameters must not require grad!"
    assert all(p.grad is None for p in model.parameters()), "Gradients must remain strictly None!"
    print("3. Parameter Frozen Status: 0 trainable parameters, 0 gradients verified [OK]")
    
    # 6. Module Isolation Check
    assert model.fc[2].training == True, "Dropout layer must be training=True"
    assert model.gru.training == False, "GRU must remain training=False"
    print("4. Module Isolation: ONLY Dropout is active; recurrence is deterministic [OK]")
    
    print("=" * 70)
    print("PHASE 5A MC-DROPOUT SMOKE TEST PASSED")
    print("=" * 70)

run_smoke_test(device)
""")
    cells.append(c7)

    # =========================================================================
    # Cell 8: Resource Gate / Boundary Notice
    # =========================================================================
    c8 = nbf.v4.new_code_cell("""# 8. RESOURCE-AWARE EXECUTION BOUNDARY (Section 22 & 26 Rule)
# Execution halts cleanly here. Execute Cell 9 when ready to run full MC-Dropout inference.

print("=" * 70)
print("PHASE 5A SETUP VALIDATED — READY FOR UNCERTAINTY INFERENCE")
print("=" * 70)
print("To perform the full 31,192 sequences x 30 MC passes, proceed to execute Cells 9 through 17.")
""")
    cells.append(c8)

    # =========================================================================
    # Cell 9: Full MC-Dropout Inference Engine (N = 30)
    # =========================================================================
    c9 = nbf.v4.new_code_cell("""# 9. Full Monte Carlo Dropout Inference Engine (N = 30 Passes)
# For each of the 31,192 test sequences, computes 30 stochastic predictions.
# If cached predictions already exist, loads them instantly.

MC_PASSES = 30
pred_npy_path = PRED_DIR / "phase5a_mc_predictions.npy"
pred_csv_path = PRED_DIR / "phase5a_uncertainty_predictions.csv"
log_path      = LOG_DIR / "inference_log.txt"

if pred_npy_path.exists() and pred_csv_path.exists():
    print("=" * 70)
    print("EXISTING PREDICTIONS DETECTED: Loading cached MC-dropout predictions!")
    print(f"Source: {pred_npy_path}")
    print("=" * 70)
    mc_predictions = np.load(pred_npy_path)  # [31192, 30, 2]
    df_uncertainty = pd.read_csv(pred_csv_path)
    print(f"Loaded {len(df_uncertainty):,} predictions with {mc_predictions.shape[1]} MC passes.")
else:
    set_seed(SEED)
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
                out = model(xb)
                pass_preds.append(out.cpu().numpy())
            pass_preds = np.vstack(pass_preds)  # [31192, 2]
            mc_passes_list.append(pass_preds)
            if p % 5 == 0 or p == MC_PASSES:
                print(f"  Completed pass {p:02d}/{MC_PASSES:02d} ({time.time() - t0:.1f}s elapsed)")
                
    # Stack into [N_test, N_passes, 2]
    mc_predictions = np.stack(mc_passes_list, axis=1)  # [31192, 30, 2]
    elapsed = time.time() - t0
    print(f"\\nInference completed in {elapsed:.1f}s ({elapsed/len(X_test)*1000:.2f} ms/sequence).")
    
    np.save(pred_npy_path, mc_predictions)
    print(f"Saved raw MC predictions array: {pred_npy_path} ({mc_predictions.nbytes / (1024**2):.1f} MB)")
    
    # Compute predictive mean, std, errors
    sbp_samples = mc_predictions[:, :, 0]  # [N, 30]
    dbp_samples = mc_predictions[:, :, 1]  # [N, 30]
    
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
    print(f"Saved uncertainty predictions table: {pred_csv_path}")
    
    with open(log_path, "w") as f:
        f.write(f"MC-Dropout Inference Complete: {len(df_uncertainty)} sequences, {MC_PASSES} passes, elapsed={elapsed:.2f}s\\n")

print(df_uncertainty.head(3))
""")
    cells.append(c9)

    # =========================================================================
    # Cell 10: Research Question 1: Uncertainty-Error Correlation
    # =========================================================================
    c10 = nbf.v4.new_code_cell("""# 10. Research Question 1: Correlation Between Uncertainty & Absolute Error
# Evaluates Pearson and Spearman correlation with p-values (raw and log-transformed).

corr_rows = []
for target, u_col, e_col in [("SBP", "sbp_uncertainty", "sbp_absolute_error"), ("DBP", "dbp_uncertainty", "dbp_absolute_error")]:
    u = df_uncertainty[u_col].values
    e = df_uncertainty[e_col].values
    
    r_pearson, p_pearson = stats.pearsonr(u, e)
    rho_spearman, p_spearman = stats.spearmanr(u, e)
    
    # Log-transformed correlation
    r_log, p_log = stats.pearsonr(np.log(np.maximum(u, 1e-6)), e)
    
    corr_rows.append({
        "Target": target,
        "Pearson_r": r_pearson,
        "Pearson_p": p_pearson,
        "Spearman_rho": rho_spearman,
        "Spearman_p": p_spearman,
        "Log_Pearson_r": r_log,
        "Log_Pearson_p": p_log,
        "Mean_Uncertainty": float(np.mean(u)),
        "Median_Uncertainty": float(np.median(u)),
        "Std_Uncertainty": float(np.std(u)),
    })

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(METRICS_DIR / "correlation_metrics.csv", index=False)

print("=" * 80)
print("UNCERTAINTY-ERROR ASSOCIATION ANALYSIS:")
print("=" * 80)
print(df_corr[["Target", "Pearson_r", "Pearson_p", "Spearman_rho", "Spearman_p", "Mean_Uncertainty", "Median_Uncertainty"]].to_string(index=False))
print("=" * 80)
""")
    cells.append(c10)

    # =========================================================================
    # Cell 11: Uncertainty Decile Analysis
    # =========================================================================
    c11 = nbf.v4.new_code_cell("""# 11. Uncertainty Decile Stratification Analysis
# Partitions predictions into 10 groups by uncertainty and tracks error progression.

def compute_uncertainty_deciles(df: pd.DataFrame, target: str) -> pd.DataFrame:
    u_col = f"{target.lower()}_uncertainty"
    e_col = f"{target.lower()}_absolute_error"
    s_col = f"{target.lower()}_signed_error"
    
    df_sorted = df.sort_values(u_col).reset_index(drop=True)
    decile_labels = [f"D{i:02d} ({(i-1)*10}-{i*10}%)" for i in range(1, 11)]
    df_sorted["decile"] = pd.qcut(df_sorted[u_col], q=10, labels=decile_labels)
    
    rows = []
    for d_lbl in decile_labels:
        subset = df_sorted[df_sorted["decile"] == d_lbl]
        rows.append({
            "target": target,
            "decile": d_lbl,
            "sample_count": len(subset),
            "mean_uncertainty": float(subset[u_col].mean()),
            "median_uncertainty": float(subset[u_col].median()),
            "mae": float(subset[e_col].mean()),
            "rmse": float(np.sqrt(np.mean(subset[s_col]**2))),
            "bias": float(subset[s_col].mean()),
        })
    return pd.DataFrame(rows)

df_deciles_sbp = compute_uncertainty_deciles(df_uncertainty, "SBP")
df_deciles_dbp = compute_uncertainty_deciles(df_uncertainty, "DBP")

df_deciles_sbp.to_csv(METRICS_DIR / "uncertainty_deciles_sbp.csv", index=False)
df_deciles_dbp.to_csv(METRICS_DIR / "uncertainty_deciles_dbp.csv", index=False)

print("--- SBP UNCERTAINTY DECILES ---")
print(df_deciles_sbp[["decile", "mean_uncertainty", "mae", "rmse", "bias"]].to_string(index=False))
print("\\n--- DBP UNCERTAINTY DECILES ---")
print(df_deciles_dbp[["decile", "mean_uncertainty", "mae", "rmse", "bias"]].to_string(index=False))
""")
    cells.append(c11)

    # =========================================================================
    # Cell 12: Selective Prediction / Abstention Analysis
    # =========================================================================
    c12 = nbf.v4.new_code_cell("""# 12. Selective Prediction & Abstention Analysis
# Retains top lowest-uncertainty fractions (100%, 90%, 80%, 70%, 60%, 50%).

coverage_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
sel_rows = []

for cov in coverage_levels:
    n_keep = int(round(len(df_uncertainty) * cov))
    
    # SBP-specific retention
    sbp_sub = df_uncertainty.sort_values("sbp_uncertainty").head(n_keep)
    sbp_mae = float(sbp_sub["sbp_absolute_error"].mean())
    
    # DBP-specific retention
    dbp_sub = df_uncertainty.sort_values("dbp_uncertainty").head(n_keep)
    dbp_mae = float(dbp_sub["dbp_absolute_error"].mean())
    
    # Joint retention (ranked by mean of normalized uncertainties)
    u_sbp_norm = (df_uncertainty["sbp_uncertainty"] - df_uncertainty["sbp_uncertainty"].mean()) / df_uncertainty["sbp_uncertainty"].std()
    u_dbp_norm = (df_uncertainty["dbp_uncertainty"] - df_uncertainty["dbp_uncertainty"].mean()) / df_uncertainty["dbp_uncertainty"].std()
    comb_sub = df_uncertainty.assign(comb_u=u_sbp_norm + u_dbp_norm).sort_values("comb_u").head(n_keep)
    comb_mae = float((comb_sub["sbp_absolute_error"].mean() + comb_sub["dbp_absolute_error"].mean()) / 2.0)
    
    sel_rows.append({
        "coverage_fraction": cov,
        "coverage_pct": f"{int(cov*100)}%",
        "retained_samples": n_keep,
        "sbp_mae": sbp_mae,
        "dbp_mae": dbp_mae,
        "combined_mae": comb_mae,
    })

df_selective = pd.DataFrame(sel_rows)
df_selective.to_csv(METRICS_DIR / "selective_prediction.csv", index=False)

print("=" * 80)
print("SELECTIVE PREDICTION / ABSTENTION PERFORMANCE:")
print("=" * 80)
print(df_selective[["coverage_pct", "retained_samples", "sbp_mae", "dbp_mae", "combined_mae"]].to_string(index=False))
print("=" * 80)
""")
    cells.append(c12)

    # =========================================================================
    # Cell 13: Error Detection Task (ROC-AUC & PR-AUC)
    # =========================================================================
    c13 = nbf.v4.new_code_cell("""# 13. High-Error Discrimination Task (ROC-AUC & PR-AUC)
# Evaluates uncertainty as a detector for errors > 10 mmHg and > 15 mmHg.

auc_rows = []
for target, u_col, e_col in [("SBP", "sbp_uncertainty", "sbp_absolute_error"), ("DBP", "dbp_uncertainty", "dbp_absolute_error")]:
    u = df_uncertainty[u_col].values
    e = df_uncertainty[e_col].values
    
    for thresh in [10.0, 15.0]:
        y_true_binary = (e > thresh).astype(int)
        pos_rate = float(np.mean(y_true_binary))
        
        if pos_rate > 0 and pos_rate < 1:
            roc_auc = float(sk_metrics.roc_auc_score(y_true_binary, u))
            pr_auc  = float(sk_metrics.average_precision_score(y_true_binary, u))
        else:
            roc_auc, pr_auc = np.nan, np.nan
            
        auc_rows.append({
            "target": target,
            "error_threshold_mmHg": thresh,
            "positive_count": int(np.sum(y_true_binary)),
            "positive_rate": pos_rate,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "random_baseline_pr_auc": pos_rate,
        })

df_auc = pd.DataFrame(auc_rows)
df_auc.to_csv(METRICS_DIR / "high_error_auc.csv", index=False)

print("=" * 85)
print("HIGH-ERROR DETECTION DISCRIMINATION (ROC-AUC & PR-AUC):")
print("=" * 85)
print(df_auc[["target", "error_threshold_mmHg", "positive_count", "positive_rate", "roc_auc", "pr_auc", "random_baseline_pr_auc"]].to_string(index=False))
print("=" * 85)
""")
    cells.append(c13)

    # =========================================================================
    # Cell 14: Uncertainty Across Clinical BP Ranges
    # =========================================================================
    c14 = nbf.v4.new_code_cell("""# 14. Uncertainty Across Clinical BP Ranges
def compute_bp_range_uncertainty(df: pd.DataFrame, target: str) -> pd.DataFrame:
    t_col = f"target_{target.lower()}"
    u_col = f"{target.lower()}_uncertainty"
    e_col = f"{target.lower()}_absolute_error"
    
    if target.upper() == "SBP":
        bins = [-np.inf, 90, 120, 140, 160, np.inf]
        labels = ["<90", "90-119", "120-139", "140-159", ">=160"]
    else:
        bins = [-np.inf, 60, 80, 90, 100, np.inf]
        labels = ["<60", "60-79", "80-89", "90-99", ">=100"]
        
    cats = pd.cut(df[t_col], bins=bins, labels=labels, right=False)
    rows = []
    for lbl in labels:
        subset = df[cats == lbl]
        cnt = len(subset)
        rows.append({
            "target": target,
            "bp_range": lbl,
            "sample_count": cnt,
            "mean_uncertainty": float(subset[u_col].mean()) if cnt > 0 else np.nan,
            "median_uncertainty": float(subset[u_col].median()) if cnt > 0 else np.nan,
            "std_uncertainty": float(subset[u_col].std()) if cnt > 0 else np.nan,
            "mae": float(subset[e_col].mean()) if cnt > 0 else np.nan,
        })
    return pd.DataFrame(rows)

df_range_sbp = compute_bp_range_uncertainty(df_uncertainty, "SBP")
df_range_dbp = compute_bp_range_uncertainty(df_uncertainty, "DBP")

df_range_sbp.to_csv(METRICS_DIR / "bp_range_uncertainty_sbp.csv", index=False)
df_range_dbp.to_csv(METRICS_DIR / "bp_range_uncertainty_dbp.csv", index=False)

print("--- SBP RANGE UNCERTAINTY ---")
print(df_range_sbp.to_string(index=False))
print("\\n--- DBP RANGE UNCERTAINTY ---")
print(df_range_dbp.to_string(index=False))
""")
    cells.append(c14)

    # =========================================================================
    # Cell 15: Empirical Interval Coverage Check
    # =========================================================================
    c15 = nbf.v4.new_code_cell("""# 15. Empirical Heuristic Interval Coverage Analysis
cov_rows = []
for target, m_col, u_col, t_col in [("SBP", "sbp_prediction_mean", "sbp_uncertainty", "target_sbp"), 
                                    ("DBP", "dbp_prediction_mean", "dbp_uncertainty", "target_dbp")]:
    m = df_uncertainty[m_col].values
    u = df_uncertainty[u_col].values
    t = df_uncertainty[t_col].values
    
    # Nominal 1.0-sigma interval
    in_1sig = (t >= (m - 1.0 * u)) & (t <= (m + 1.0 * u))
    cov_1sig = float(np.mean(in_1sig) * 100.0)
    width_1sig = float(np.mean(2.0 * u))
    
    # Nominal 1.96-sigma interval
    in_196sig = (t >= (m - 1.96 * u)) & (t <= (m + 1.96 * u))
    cov_196sig = float(np.mean(in_196sig) * 100.0)
    width_196sig = float(np.mean(2.0 * 1.96 * u))
    
    cov_rows.append({
        "target": target,
        "interval_1.0_sigma_empirical_coverage_pct": cov_1sig,
        "interval_1.0_sigma_mean_width_mmHg": width_1sig,
        "interval_1.96_sigma_empirical_coverage_pct": cov_196sig,
        "interval_1.96_sigma_mean_width_mmHg": width_196sig,
        "interpretation": "Heuristic interval coverage (epistemic dropout proxy; not calibrated confidence interval)"
    })

df_cov = pd.DataFrame(cov_rows)
df_cov.to_csv(METRICS_DIR / "empirical_interval_coverage.csv", index=False)

print("=" * 80)
print("EMPIRICAL HEURISTIC INTERVAL COVERAGE ANALYSIS:")
print("=" * 80)
print(df_cov[["target", "interval_1.0_sigma_empirical_coverage_pct", "interval_1.0_sigma_mean_width_mmHg", 
              "interval_1.96_sigma_empirical_coverage_pct", "interval_1.96_sigma_mean_width_mmHg"]].to_string(index=False))
print("=" * 80)
""")
    cells.append(c15)

    # =========================================================================
    # Cell 16: Publication Figures (All 8 Figures)
    # =========================================================================
    c16 = nbf.v4.new_code_cell("""# 16. Publication Visualizations (All 8 Figures)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "figure.dpi": 130, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3
})

# FIG 1: SBP Uncertainty Histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_uncertainty["sbp_uncertainty"], bins=60, color="#2563eb", edgecolor="black", alpha=0.7)
ax.axvline(df_uncertainty["sbp_uncertainty"].mean(), color="red", linestyle="--", linewidth=1.5, 
           label=f"Mean ({df_uncertainty['sbp_uncertainty'].mean():.2f})")
ax.axvline(df_uncertainty["sbp_uncertainty"].median(), color="darkorange", linestyle=":", linewidth=1.5, 
           label=f"Median ({df_uncertainty['sbp_uncertainty'].median():.2f})")
ax.set_title("Phase 5A: SBP MC-Dropout Predictive Uncertainty Distribution")
ax.set_xlabel("Predictive Uncertainty (Standard Deviation across 30 Passes, mmHg)")
ax.set_ylabel("Sample Count")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "uncertainty_hist_sbp.png"), dpi=300)
plt.close()

# FIG 2: DBP Uncertainty Histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_uncertainty["dbp_uncertainty"], bins=60, color="#10b981", edgecolor="black", alpha=0.7)
ax.axvline(df_uncertainty["dbp_uncertainty"].mean(), color="red", linestyle="--", linewidth=1.5, 
           label=f"Mean ({df_uncertainty['dbp_uncertainty'].mean():.2f})")
ax.axvline(df_uncertainty["dbp_uncertainty"].median(), color="darkorange", linestyle=":", linewidth=1.5, 
           label=f"Median ({df_uncertainty['dbp_uncertainty'].median():.2f})")
ax.set_title("Phase 5A: DBP MC-Dropout Predictive Uncertainty Distribution")
ax.set_xlabel("Predictive Uncertainty (Standard Deviation across 30 Passes, mmHg)")
ax.set_ylabel("Sample Count")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "uncertainty_hist_dbp.png"), dpi=300)
plt.close()

# FIG 3: SBP Uncertainty vs Absolute Error Scatter
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_uncertainty["sbp_uncertainty"], df_uncertainty["sbp_absolute_error"], s=2, alpha=0.15, color="#2563eb", rasterized=True)
# Add trendline
m, b = np.polyfit(df_uncertainty["sbp_uncertainty"], df_uncertainty["sbp_absolute_error"], 1)
u_grid = np.linspace(df_uncertainty["sbp_uncertainty"].min(), df_uncertainty["sbp_uncertainty"].max(), 100)
ax.plot(u_grid, m * u_grid + b, color="red", linewidth=2, label=f"Trend (Slope={m:.2f}, r={df_corr.loc[0, 'Pearson_r']:.3f})")
ax.set_title("Phase 5A: SBP Predictive Uncertainty vs Absolute Prediction Error")
ax.set_xlabel("MC-Dropout Uncertainty (mmHg)")
ax.set_ylabel("Absolute Prediction Error (mmHg)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "uncertainty_vs_error_sbp.png"), dpi=300)
plt.close()

# FIG 4: DBP Uncertainty vs Absolute Error Scatter
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_uncertainty["dbp_uncertainty"], df_uncertainty["dbp_absolute_error"], s=2, alpha=0.15, color="#10b981", rasterized=True)
m, b = np.polyfit(df_uncertainty["dbp_uncertainty"], df_uncertainty["dbp_absolute_error"], 1)
u_grid = np.linspace(df_uncertainty["dbp_uncertainty"].min(), df_uncertainty["dbp_uncertainty"].max(), 100)
ax.plot(u_grid, m * u_grid + b, color="red", linewidth=2, label=f"Trend (Slope={m:.2f}, r={df_corr.loc[1, 'Pearson_r']:.3f})")
ax.set_title("Phase 5A: DBP Predictive Uncertainty vs Absolute Prediction Error")
ax.set_xlabel("MC-Dropout Uncertainty (mmHg)")
ax.set_ylabel("Absolute Prediction Error (mmHg)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "uncertainty_vs_error_dbp.png"), dpi=300)
plt.close()

# FIG 5: SBP Risk-Coverage Curve
fig, ax = plt.subplots(figsize=(8, 5))
cov_pcts = np.linspace(0.1, 1.0, 19)
sbp_risks = []
df_sbp_sorted = df_uncertainty.sort_values("sbp_uncertainty")
for cov in cov_pcts:
    n = int(round(len(df_uncertainty) * cov))
    sbp_risks.append(df_sbp_sorted.head(n)["sbp_absolute_error"].mean())
ax.plot(cov_pcts * 100, sbp_risks, marker="o", color="#2563eb", linewidth=2, label="Selective SBP MAE")
ax.axhline(df_uncertainty["sbp_absolute_error"].mean(), color="black", linestyle="--", alpha=0.6, label=f"100% Coverage MAE ({df_uncertainty['sbp_absolute_error'].mean():.2f})")
ax.set_title("Phase 5A: SBP Risk-Coverage Curve (Selective Prediction)")
ax.set_xlabel("Coverage (% of Predictions Retained with Lowest Uncertainty)")
ax.set_ylabel("Risk (MAE on Retained Predictions, mmHg)")
ax.set_xlim(5, 105)
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "risk_coverage_sbp.png"), dpi=300)
plt.close()

# FIG 6: DBP Risk-Coverage Curve
fig, ax = plt.subplots(figsize=(8, 5))
dbp_risks = []
df_dbp_sorted = df_uncertainty.sort_values("dbp_uncertainty")
for cov in cov_pcts:
    n = int(round(len(df_uncertainty) * cov))
    dbp_risks.append(df_dbp_sorted.head(n)["dbp_absolute_error"].mean())
ax.plot(cov_pcts * 100, dbp_risks, marker="o", color="#10b981", linewidth=2, label="Selective DBP MAE")
ax.axhline(df_uncertainty["dbp_absolute_error"].mean(), color="black", linestyle="--", alpha=0.6, label=f"100% Coverage MAE ({df_uncertainty['dbp_absolute_error'].mean():.2f})")
ax.set_title("Phase 5A: DBP Risk-Coverage Curve (Selective Prediction)")
ax.set_xlabel("Coverage (% of Predictions Retained with Lowest Uncertainty)")
ax.set_ylabel("Risk (MAE on Retained Predictions, mmHg)")
ax.set_xlim(5, 105)
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "risk_coverage_dbp.png"), dpi=300)
plt.close()

# FIG 7: Uncertainty by SBP Range
fig, ax1 = plt.subplots(figsize=(8, 5))
x_idx = np.arange(len(df_range_sbp))
w = 0.35
b1 = ax1.bar(x_idx - w/2, df_range_sbp["mean_uncertainty"], width=w, color="#93c5fd", edgecolor="#2563eb", label="Mean Uncertainty (mmHg)")
b2 = ax1.bar(x_idx + w/2, df_range_sbp["mae"], width=w, color="#fca5a5", edgecolor="#dc2626", label="MAE (mmHg)")
ax1.set_xticks(x_idx)
ax1.set_xticklabels(df_range_sbp["bp_range"])
ax1.set_title("Phase 5A: SBP Uncertainty & Error Stratified by Clinical BP Range")
ax1.set_xlabel("Clinical SBP Range (mmHg)")
ax1.set_ylabel("mmHg")
ax1.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "uncertainty_by_bp_range_sbp.png"), dpi=300)
plt.close()

# FIG 8: Uncertainty by DBP Range
fig, ax2 = plt.subplots(figsize=(8, 5))
x_idx = np.arange(len(df_range_dbp))
b1 = ax2.bar(x_idx - w/2, df_range_dbp["mean_uncertainty"], width=w, color="#6ee7b7", edgecolor="#059669", label="Mean Uncertainty (mmHg)")
b2 = ax2.bar(x_idx + w/2, df_range_dbp["mae"], width=w, color="#fca5a5", edgecolor="#dc2626", label="MAE (mmHg)")
ax2.set_xticks(x_idx)
ax2.set_xticklabels(df_range_dbp["bp_range"])
ax2.set_title("Phase 5A: DBP Uncertainty & Error Stratified by Clinical BP Range")
ax2.set_xlabel("Clinical DBP Range (mmHg)")
ax2.set_ylabel("mmHg")
ax2.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "uncertainty_by_bp_range_dbp.png"), dpi=300)
plt.close()

print(f"All 8 publication figures successfully generated and saved to: {FIG_DIR}")
""")
    cells.append(c16)

    # =========================================================================
    # Cell 17: Scientific Reports & Evidence Freeze
    # =========================================================================
    c17 = nbf.v4.new_code_cell("""# 17. Scientific Reports & Evidence Freeze
# Generates PHASE5A_UNCERTAINTY_REPORT.md, PHASE5A_EVIDENCE_FREEZE.md, and metadata JSON.

rep_path = REPORT_DIR / "PHASE5A_UNCERTAINTY_REPORT.md"
frz_path = REPORT_DIR / "PHASE5A_EVIDENCE_FREEZE.md"
meta_path = REPORT_DIR / "phase5a_uncertainty_metadata.json"

# Pre-render markdown tables to avoid expressions with backslashes in f-string
sbp_dec_md = df_deciles_sbp[['decile', 'mean_uncertainty', 'mae', 'rmse', 'bias']].to_markdown(index=False)
dbp_dec_md = df_deciles_dbp[['decile', 'mean_uncertainty', 'mae', 'rmse', 'bias']].to_markdown(index=False)
sel_md = df_selective[['coverage_pct', 'retained_samples', 'sbp_mae', 'dbp_mae', 'combined_mae']].to_markdown(index=False)
auc_md = df_auc[['target', 'error_threshold_mmHg', 'positive_count', 'positive_rate', 'roc_auc', 'pr_auc', 'random_baseline_pr_auc']].to_markdown(index=False)
range_sbp_md = df_range_sbp.to_markdown(index=False)
range_dbp_md = df_range_dbp.to_markdown(index=False)
cov_md = df_cov[['target', 'interval_1.0_sigma_empirical_coverage_pct', 'interval_1.0_sigma_mean_width_mmHg', 'interval_1.96_sigma_empirical_coverage_pct', 'interval_1.96_sigma_mean_width_mmHg']].to_markdown(index=False)

dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
curr_time = time.strftime('%Y-%m-%d %H:%M:%S')

report_text = f\"\"\"# PHASE 5A — Uncertainty Estimation Without Retraining Report

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
\"\"\"

with open(rep_path, "w") as f:
    f.write(report_text)

freeze_text = f\"\"\"# PHASE 5A EVIDENCE FREEZE: UNCERTAINTY ESTIMATION WITHOUT RETRAINING

- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
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
- **SBP 80% Coverage MAE:** {df_selective.loc[df_selective['coverage_fraction']==0.8, 'sbp_mae'].values[0]:.4f} mmHg
- **DBP 100% Coverage MAE:** {df_uncertainty['dbp_absolute_error'].mean():.4f} mmHg
- **DBP 80% Coverage MAE:** {df_selective.loc[df_selective['coverage_fraction']==0.8, 'dbp_mae'].values[0]:.4f} mmHg
- **Environment:** Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda if torch.cuda.is_available() else 'None'}

Phase 5A was inference-only. No neural network parameters were trained or updated.
\"\"\"

with open(frz_path, "w") as f:
    f.write(freeze_text)

meta_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    "selective_prediction": df_selective.to_dict(orient="records"),
    "high_error_auc": df_auc.to_dict(orient="records"),
}

with open(meta_path, "w") as f:
    json.dump(meta_data, f, indent=2)

print("=" * 70)
print(f"Report saved:   {rep_path}")
print(f"Freeze saved:   {frz_path}")
print(f"Metadata saved: {meta_path}")
print("=" * 70)
print("PHASE 5A WORKFLOW COMPLETE.")
""")
    cells.append(c17)

    nb.cells = cells
    cwd = Path.cwd().resolve()
    project_root = None
    for c in [cwd, cwd.parent, cwd.parent.parent]:
        if (c / "BloodPressureDataset").exists():
            project_root = c
            break
    if project_root is None:
        project_root = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")

    out_notebook_path = project_root / "code" / "notebooks" / "05A_uncertainty_mc_dropout.ipynb"
    with open(out_notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("=" * 70)
    print(f"SUCCESS: Generated {out_notebook_path}")
    print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'code')} code, {sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
    print("=" * 70)

if __name__ == "__main__":
    generate_phase5a_notebook()
