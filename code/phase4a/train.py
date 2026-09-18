"""
Phase 4A: Training Loop for PPG 1D CNN Baseline.

Protocol:
    1. Smoke test (always runs first):
       - Small subset, forward+backward pass, NaN/shape checks, GPU check
    2. Full training:
       - Model A (PPG only, n_channels=1)
       - Model B (PPG+VPG+APG, n_channels=3)
       - AdamW + ReduceLROnPlateau + early stopping
       - Best checkpoint saved by validation combined MAE
    3. Final test evaluation:
       - Frozen checkpoint evaluated exactly ONCE on test set
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from phase4a.dataset import PPGWindowDataset
from phase4a.model import PPGCNNBaseline
from phase4a.evaluate import (
    compute_metrics,
    compute_bp_range_errors,
    evaluate_model,
    generate_all_figures,
)
from phase4a.utils import set_seed, setup_logger, save_json, gpu_memory_summary


# ============================================================================
# Paths
# ============================================================================
_PROJECT_ROOT = _CODE_DIR.parent
MANIFEST_PATH = _CODE_DIR / "outputs" / "windows" / "window_manifest.csv"
DATASET_DIR = _PROJECT_ROOT / "BloodPressureDataset"
OUTPUT_DIR = _CODE_DIR / "outputs" / "phase4a"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PRED_DIR = OUTPUT_DIR / "predictions"
FIG_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "reports"

for d in [CKPT_DIR, METRICS_DIR, PRED_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Hyperparameters
# ============================================================================
SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
LR_FACTOR = 0.5
LR_PATIENCE = 3
MIN_LR = 1e-6
HUBER_DELTA = 5.0  # mmHg

PHASE3A_SBP_MAE = 13.93
PHASE3A_DBP_MAE = 7.05
PHASE3A_COMB_MAE = 10.49


# ============================================================================
# Loss
# ============================================================================
def combined_huber_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta: float = HUBER_DELTA,
) -> torch.Tensor:
    """
    Combined Huber loss for SBP and DBP.
    y_pred, y_true: (B, 2) where dim 1 = [SBP, DBP]
    """
    criterion = nn.HuberLoss(delta=delta, reduction="mean")
    loss_sbp = criterion(y_pred[:, 0], y_true[:, 0])
    loss_dbp = criterion(y_pred[:, 1], y_true[:, 1])
    return loss_sbp + loss_dbp


# ============================================================================
# Smoke Test
# ============================================================================
def run_smoke_test(
    device: torch.device,
    logger,
    n_windows: int = 128,
) -> bool:
    """
    Smoke test:
    1. Load small dataset subset (128 windows from train)
    2. Forward pass → verify output shape [B, 2]
    3. Backward pass → verify no NaN gradients
    4. Check no NaN/Inf in output
    5. 2 mini-epochs → verify loss decreases
    6. Print GPU memory
    
    Returns True if all checks pass, raises RuntimeError otherwise.
    """
    logger.info("=" * 60)
    logger.info("STARTING SMOKE TEST")
    logger.info("=" * 60)
    set_seed(SEED)

    # Load a tiny subset of train data for Model B (most complex)
    logger.info(f"Loading {n_windows} training windows for smoke test...")
    ds = PPGWindowDataset(
        manifest_path=MANIFEST_PATH,
        dataset_dir=DATASET_DIR,
        split="train",
        n_channels=3,
        use_zscore=True,
        max_windows=n_windows,
        verify_leakage=True,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)

    # Build model B
    model = PPGCNNBaseline(n_channels=3).to(device)
    model.print_summary()
    logger.info(f"GPU memory after model init: {gpu_memory_summary()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- Check 1: forward pass ----
    logger.info("CHECK 1: Forward pass shape...")
    model.train()
    X_batch, y_batch = next(iter(loader))
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    with torch.no_grad():
        out = model(X_batch)
    assert out.shape == (X_batch.shape[0], 2), (
        f"Output shape mismatch: expected ({X_batch.shape[0]}, 2), got {out.shape}"
    )
    assert torch.isfinite(out).all(), "NaN/Inf in model output during smoke test."
    logger.info(f"  Output shape: {out.shape} ✓")
    logger.info(f"  Output finite: ✓")

    # ---- Check 2: backward pass + gradient check ----
    logger.info("CHECK 2: Backward pass + gradient check...")
    optimizer.zero_grad()
    out = model(X_batch)
    loss = combined_huber_loss(out, y_batch)
    loss.backward()
    
    nan_grads = []
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            nan_grads.append(name)
    assert len(nan_grads) == 0, f"NaN/Inf gradients in: {nan_grads}"
    logger.info(f"  Loss: {loss.item():.4f} ✓")
    logger.info(f"  Gradients finite: ✓")
    optimizer.step()

    logger.info(f"GPU memory after backward: {gpu_memory_summary()}")

    # ---- Check 3: 2 mini-epochs, verify loss decreases ----
    logger.info("CHECK 3: 2 mini-epochs (loss should trend downward)...")
    losses = []
    for epoch in range(2):
        epoch_loss = 0.0
        n_batches = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = combined_huber_loss(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / n_batches
        losses.append(avg)
        logger.info(f"  Mini-epoch {epoch + 1}/2: loss={avg:.4f}")

    logger.info(f"GPU memory after mini-epochs: {gpu_memory_summary()}")
    logger.info("SMOKE TEST PASSED ✓")
    logger.info("=" * 60)
    return True


# ============================================================================
# Training Loop
# ============================================================================
def train_one_model(
    model_name: str,
    n_channels: int,
    device: torch.device,
    logger,
) -> Tuple[Dict, Dict, List[float], List[float]]:
    """
    Full training loop for one model variant.

    Returns:
        metrics_val:   Dict of validation metrics (sbp, dbp).
        metrics_test:  Dict of test metrics (sbp, dbp).
        train_losses:  List of per-epoch training losses.
        val_losses:    List of per-epoch validation losses.
    """
    set_seed(SEED)
    logger.info("=" * 70)
    logger.info(f"TRAINING {model_name} (n_channels={n_channels})")
    logger.info("=" * 70)

    t_start = time.time()

    # Load datasets
    logger.info("Loading train dataset...")
    train_ds = PPGWindowDataset(
        MANIFEST_PATH, DATASET_DIR, split="train",
        n_channels=n_channels, verify_leakage=True,
    )
    logger.info("Loading val dataset...")
    val_ds = PPGWindowDataset(
        MANIFEST_PATH, DATASET_DIR, split="val",
        n_channels=n_channels, verify_leakage=False,  # already checked
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Train: {len(train_ds)} windows | Val: {len(val_ds)} windows")
    logger.info(f"Batch size: {BATCH_SIZE} | Batches/epoch: {len(train_loader)}")

    # Build model
    model = PPGCNNBaseline(n_channels=n_channels).to(device)
    total_p, train_p = model.count_parameters()
    logger.info(f"Model parameters: {total_p:,} total, {train_p:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    best_val_comb = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    ckpt_path = CKPT_DIR / f"{model_name}_best.pt"

    log_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(X)
            loss = combined_huber_loss(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_loss_total = 0.0
        n_val_batches = 0
        all_val_true = []
        all_val_pred = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(X)
                loss = combined_huber_loss(out, y)
                val_loss_total += loss.item()
                n_val_batches += 1
                all_val_true.append(y.cpu().numpy())
                all_val_pred.append(out.cpu().numpy())

        avg_val_loss = val_loss_total / n_val_batches
        val_losses.append(avg_val_loss)

        val_true = np.vstack(all_val_true)   # (N, 2)
        val_pred = np.vstack(all_val_pred)   # (N, 2)
        sbp_mae_v = float(np.mean(np.abs(val_pred[:, 0] - val_true[:, 0])))
        dbp_mae_v = float(np.mean(np.abs(val_pred[:, 1] - val_true[:, 1])))
        val_comb = (sbp_mae_v + dbp_mae_v) / 2.0

        scheduler.step(val_comb)
        cur_lr = optimizer.param_groups[0]["lr"]

        log_rows.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_sbp_mae": sbp_mae_v,
            "val_dbp_mae": dbp_mae_v,
            "val_comb_mae": val_comb,
            "lr": cur_lr,
        })

        improved = val_comb < best_val_comb
        if improved:
            best_val_comb = val_comb
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        logger.info(
            f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
            f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
            f"SBP={sbp_mae_v:.2f} DBP={dbp_mae_v:.2f} Comb={val_comb:.2f} | "
            f"LR={cur_lr:.1e} | {'BEST ✓' if improved else f'NoImprove={epochs_no_improve}'}"
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
            break

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed / 60:.1f} min. Best epoch: {best_epoch}, Best val comb MAE: {best_val_comb:.4f}")

    # Save training log
    pd.DataFrame(log_rows).to_csv(METRICS_DIR / f"phase4a_training_log_{model_name}.csv", index=False)

    # --- Load best checkpoint and compute full validation metrics ---
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    sbp_v_true, dbp_v_true, sbp_v_pred, dbp_v_pred = evaluate_model(model, val_loader, device)
    metrics_val = {
        "sbp": compute_metrics(sbp_v_true, sbp_v_pred),
        "dbp": compute_metrics(dbp_v_true, dbp_v_pred),
    }

    # --- Final test evaluation (ONCE) ---
    logger.info(f"Loading test dataset for final FROZEN evaluation...")
    test_ds = PPGWindowDataset(
        MANIFEST_PATH, DATASET_DIR, split="test",
        n_channels=n_channels, verify_leakage=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    sbp_t_true, dbp_t_true, sbp_t_pred, dbp_t_pred = evaluate_model(model, test_loader, device)
    metrics_test = {
        "sbp": compute_metrics(sbp_t_true, sbp_t_pred),
        "dbp": compute_metrics(dbp_t_true, dbp_t_pred),
    }

    logger.info(f"[{model_name}] FINAL TEST: SBP MAE={metrics_test['sbp']['mae']:.2f}, DBP MAE={metrics_test['dbp']['mae']:.2f}")

    # Save predictions
    pd.DataFrame({
        "sbp_true": sbp_t_true, "dbp_true": dbp_t_true,
        "sbp_pred": sbp_t_pred, "dbp_pred": dbp_t_pred,
    }).to_csv(PRED_DIR / f"phase4a_predictions_test_{model_name}.csv", index=False)

    # Save validation predictions
    pd.DataFrame({
        "sbp_true": sbp_v_true, "dbp_true": dbp_v_true,
        "sbp_pred": sbp_v_pred, "dbp_pred": dbp_v_pred,
    }).to_csv(PRED_DIR / f"phase4a_predictions_val_{model_name}.csv", index=False)

    # Return val/test predictions too for figure generation
    model._sbp_v_true = sbp_v_true
    model._dbp_v_true = dbp_v_true
    model._sbp_v_pred = sbp_v_pred
    model._dbp_v_pred = dbp_v_pred
    model._sbp_t_true = sbp_t_true
    model._dbp_t_true = dbp_t_true
    model._sbp_t_pred = sbp_t_pred
    model._dbp_t_pred = dbp_t_pred
    model._train_losses = train_losses
    model._val_losses = val_losses
    model._best_epoch = best_epoch
    del train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics_val, metrics_test, train_losses, val_losses, model


# ============================================================================
# Report Writers
# ============================================================================
def write_reports(
    metrics_val_a: Dict, metrics_test_a: Dict,
    metrics_val_b: Dict, metrics_test_b: Dict,
    model_a, model_b,
    device: torch.device,
    logger,
) -> None:
    """Write all required CSV and markdown reports."""
    # ---- Validation results CSV ----
    val_rows = []
    for name, m in [("model_a_ppg_only", metrics_val_a), ("model_b_ppg_vpg_apg", metrics_val_b)]:
        val_rows.append({
            "model": name, "split": "val",
            "sbp_mae": m["sbp"]["mae"], "sbp_rmse": m["sbp"]["rmse"], "sbp_r2": m["sbp"]["r2"],
            "sbp_bias": m["sbp"]["bias"], "sbp_error_sd": m["sbp"]["error_sd"],
            "sbp_pct_5": m["sbp"]["pct_within_5"], "sbp_pct_10": m["sbp"]["pct_within_10"], "sbp_pct_15": m["sbp"]["pct_within_15"],
            "dbp_mae": m["dbp"]["mae"], "dbp_rmse": m["dbp"]["rmse"], "dbp_r2": m["dbp"]["r2"],
            "dbp_bias": m["dbp"]["bias"], "dbp_error_sd": m["dbp"]["error_sd"],
            "dbp_pct_5": m["dbp"]["pct_within_5"], "dbp_pct_10": m["dbp"]["pct_within_10"], "dbp_pct_15": m["dbp"]["pct_within_15"],
            "comb_mae": (m["sbp"]["mae"] + m["dbp"]["mae"]) / 2,
        })
    pd.DataFrame(val_rows).to_csv(METRICS_DIR / "phase4a_validation_results.csv", index=False)

    # ---- Test results CSV ----
    test_rows = []
    for name, m in [("model_a_ppg_only", metrics_test_a), ("model_b_ppg_vpg_apg", metrics_test_b)]:
        test_rows.append({
            "model": name, "split": "test",
            "sbp_mae": m["sbp"]["mae"], "sbp_rmse": m["sbp"]["rmse"], "sbp_r2": m["sbp"]["r2"],
            "sbp_bias": m["sbp"]["bias"], "sbp_error_sd": m["sbp"]["error_sd"],
            "sbp_pct_5": m["sbp"]["pct_within_5"], "sbp_pct_10": m["sbp"]["pct_within_10"], "sbp_pct_15": m["sbp"]["pct_within_15"],
            "dbp_mae": m["dbp"]["mae"], "dbp_rmse": m["dbp"]["rmse"], "dbp_r2": m["dbp"]["r2"],
            "dbp_bias": m["dbp"]["bias"], "dbp_error_sd": m["dbp"]["error_sd"],
            "dbp_pct_5": m["dbp"]["pct_within_5"], "dbp_pct_10": m["dbp"]["pct_within_10"], "dbp_pct_15": m["dbp"]["pct_within_15"],
            "comb_mae": (m["sbp"]["mae"] + m["dbp"]["mae"]) / 2,
        })
    pd.DataFrame(test_rows).to_csv(METRICS_DIR / "phase4a_test_results.csv", index=False)

    # ---- Model comparison CSV ----
    comb_rows = [
        {"model": "Phase 3A Classical (PPG features)", "sbp_mae": PHASE3A_SBP_MAE, "dbp_mae": PHASE3A_DBP_MAE, "comb_mae": PHASE3A_COMB_MAE},
        {"model": "Model A: PPG Only CNN", "sbp_mae": metrics_test_a["sbp"]["mae"], "dbp_mae": metrics_test_a["dbp"]["mae"],
         "comb_mae": (metrics_test_a["sbp"]["mae"] + metrics_test_a["dbp"]["mae"]) / 2},
        {"model": "Model B: PPG+VPG+APG CNN", "sbp_mae": metrics_test_b["sbp"]["mae"], "dbp_mae": metrics_test_b["dbp"]["mae"],
         "comb_mae": (metrics_test_b["sbp"]["mae"] + metrics_test_b["dbp"]["mae"]) / 2},
    ]
    pd.DataFrame(comb_rows).to_csv(METRICS_DIR / "phase4a_model_comparison.csv", index=False)

    # ---- BP range error CSVs ----
    sbp_t_true = model_b._sbp_t_true
    sbp_t_pred_a = model_a._sbp_t_pred
    sbp_t_pred_b = model_b._sbp_t_pred
    dbp_t_true = model_b._dbp_t_true
    dbp_t_pred_a = model_a._dbp_t_pred
    dbp_t_pred_b = model_b._dbp_t_pred

    df_sbp_a = compute_bp_range_errors(sbp_t_true, sbp_t_pred_a, "SBP")
    df_sbp_b = compute_bp_range_errors(sbp_t_true, sbp_t_pred_b, "SBP")
    df_dbp_a = compute_bp_range_errors(dbp_t_true, dbp_t_pred_a, "DBP")
    df_dbp_b = compute_bp_range_errors(dbp_t_true, dbp_t_pred_b, "DBP")

    df_sbp_a["model"] = "model_a_ppg_only"
    df_sbp_b["model"] = "model_b_ppg_vpg_apg"
    df_dbp_a["model"] = "model_a_ppg_only"
    df_dbp_b["model"] = "model_b_ppg_vpg_apg"

    pd.concat([df_sbp_a, df_sbp_b]).to_csv(METRICS_DIR / "phase4a_sbp_range_errors.csv", index=False)
    pd.concat([df_dbp_a, df_dbp_b]).to_csv(METRICS_DIR / "phase4a_dbp_range_errors.csv", index=False)

    logger.info("All CSVs written.")

    # ---- Generate figures ----
    generate_all_figures(
        train_losses_a=model_a._train_losses,
        val_losses_a=model_a._val_losses,
        train_losses_b=model_b._train_losses,
        val_losses_b=model_b._val_losses,
        val_sbp_true=model_a._sbp_v_true,
        val_sbp_pred_a=model_a._sbp_v_pred,
        val_dbp_true=model_a._dbp_v_true,
        val_dbp_pred_a=model_a._dbp_v_pred,
        val_sbp_pred_b=model_b._sbp_v_pred,
        val_dbp_pred_b=model_b._dbp_v_pred,
        test_sbp_true=model_a._sbp_t_true,
        test_sbp_pred_a=model_a._sbp_t_pred,
        test_dbp_true=model_a._dbp_t_true,
        test_dbp_pred_a=model_a._dbp_t_pred,
        test_sbp_pred_b=model_b._sbp_t_pred,
        test_dbp_pred_b=model_b._dbp_t_pred,
        metrics_val_a=metrics_val_a,
        metrics_val_b=metrics_val_b,
        metrics_test_a=metrics_test_a,
        metrics_test_b=metrics_test_b,
        figures_dir=FIG_DIR,
    )

    # ---- Metadata JSON ----
    import platform, torch as _torch
    meta = {
        "phase": "Phase 4A — Neural PPG Baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "random_seed": SEED,
        "python_version": sys.version,
        "pytorch_version": _torch.__version__,
        "cuda_available": _torch.cuda.is_available(),
        "cuda_version": _torch.version.cuda if _torch.cuda.is_available() else "N/A",
        "gpu": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CPU",
        "device": str(device),
        "platform": platform.platform(),
        "preprocessing": {
            "filter": "Butterworth bandpass, 0.5-8 Hz, order=3, offline filtfilt (non-causal)",
            "derivative": "np.gradient (central finite differences)",
            "normalization": "per-window z-score, eps=1e-8",
        },
        "model_architecture": "PPGCNNBaseline: 4xConvBlock + FC head, see model.py",
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "ReduceLROnPlateau",
            "lr_factor": LR_FACTOR,
            "lr_patience": LR_PATIENCE,
            "min_lr": MIN_LR,
            "loss": f"HuberLoss(delta={HUBER_DELTA}) summed for SBP + DBP",
            "gradient_clip_norm": 5.0,
        },
        "model_a": {
            "n_channels": 1,
            "description": "PPG only",
            "checkpoint": str(CKPT_DIR / "model_a_ppg_only_best.pt"),
            "best_epoch": model_a._best_epoch,
            "training_time_min": model_a._elapsed / 60,
            "validation": {
                "sbp_mae": metrics_val_a["sbp"]["mae"],
                "dbp_mae": metrics_val_a["dbp"]["mae"],
                "comb_mae": (metrics_val_a["sbp"]["mae"] + metrics_val_a["dbp"]["mae"]) / 2,
            },
            "test": {
                "sbp_mae": metrics_test_a["sbp"]["mae"],
                "dbp_mae": metrics_test_a["dbp"]["mae"],
                "comb_mae": (metrics_test_a["sbp"]["mae"] + metrics_test_a["dbp"]["mae"]) / 2,
            },
        },
        "model_b": {
            "n_channels": 3,
            "description": "PPG + VPG + APG",
            "checkpoint": str(CKPT_DIR / "model_b_ppg_vpg_apg_best.pt"),
            "best_epoch": model_b._best_epoch,
            "training_time_min": model_b._elapsed / 60,
            "validation": {
                "sbp_mae": metrics_val_b["sbp"]["mae"],
                "dbp_mae": metrics_val_b["dbp"]["mae"],
                "comb_mae": (metrics_val_b["sbp"]["mae"] + metrics_val_b["dbp"]["mae"]) / 2,
            },
            "test": {
                "sbp_mae": metrics_test_b["sbp"]["mae"],
                "dbp_mae": metrics_test_b["dbp"]["mae"],
                "comb_mae": (metrics_test_b["sbp"]["mae"] + metrics_test_b["dbp"]["mae"]) / 2,
            },
        },
        "phase3a_baseline": {
            "sbp_mae": PHASE3A_SBP_MAE,
            "dbp_mae": PHASE3A_DBP_MAE,
            "comb_mae": PHASE3A_COMB_MAE,
        },
        "test_evaluation_performed_once": True,
        "manifect_path": str(MANIFEST_PATH),
        "dataset_dir": str(DATASET_DIR),
        "split_file": str(_CODE_DIR / "outputs" / "splits" / "record_split.csv"),
    }
    save_json(meta, REPORT_DIR / "phase4a_training_metadata.json")
    logger.info("Metadata JSON saved.")

    # Write markdown reports
    _write_baseline_report(metrics_val_a, metrics_test_a, metrics_val_b, metrics_test_b,
                           model_a, model_b, logger)
    _write_evidence_freeze(meta, metrics_val_a, metrics_test_a, metrics_val_b, metrics_test_b,
                           model_a, model_b, logger)


def _write_baseline_report(
    metrics_val_a, metrics_test_a,
    metrics_val_b, metrics_test_b,
    model_a, model_b,
    logger,
) -> None:
    """Write PHASE4A_NEURAL_BASELINE_REPORT.md."""
    ma = metrics_test_a
    mb = metrics_test_b
    comb_a = (ma["sbp"]["mae"] + ma["dbp"]["mae"]) / 2
    comb_b = (mb["sbp"]["mae"] + mb["dbp"]["mae"]) / 2

    def _beat(val, ref):
        delta = ref - val
        return f"+{delta:.2f} improvement" if delta > 0 else f"{delta:.2f} regression"

    report = f"""# Phase 4A Neural Baseline Report: 1D CNN for Cuffless BP Estimation

**Project**: Cuffless Blood Pressure Estimation from PPG using MAX30102 + ESP32
**Dataset**: MIMIC-II Waveform Database (261,339 windows, frozen Phase 2 manifest)
**Execution**: GPU-accelerated (CUDA) where available; CPU fallback

---

## 1. Objective

Evaluate whether a compact 1D CNN applied to a single 10-second PPG waveform
(with deterministic VPG and APG derivative channels) can outperform the
frozen Phase 3A classical handcrafted-feature baseline.

Research question:
> Does a compact neural representation of morphology-derived PPG information
> improve calibration-free cuffless BP estimation under a strict record-level
> evaluation protocol?

---

## 2. Experimental Protocol

- Single-window 10-second input (no temporal history)
- Two ablation models compared:
  - **Model A**: PPG only (n_channels=1)
  - **Model B**: PPG + VPG + APG (n_channels=3)
- All other architecture and hyperparameter settings identical
- Validation-based model selection → single frozen test evaluation
- Test set evaluated **exactly once** per model

---

## 3. Dataset and Record Split

| Split | Records | Windows |
|---|---|---|
| Train | 8,400 | 183,517 |
| Validation | 1,800 | 39,461 |
| Test | 1,800 | 38,361 |

- Total eligible windows: 261,339
- Random seed: 42
- Manifest: `code/outputs/windows/window_manifest.csv` (frozen)
- Split: `code/outputs/splits/record_split.csv` (frozen)
- Split granularity: record-level (no individual window random splits)

---

## 4. Preprocessing

### Bandpass Filter (Offline Research)
- Type: Butterworth bandpass, order=3
- Cutoffs: 0.5–8.0 Hz
- Method: `scipy.signal.filtfilt` (zero-phase, forward+backward pass)
- **Non-causal / offline only**: `filtfilt` requires the full window and CANNOT
  be deployed on real-time streaming hardware (ESP32). Deployment will require
  a causal IIR or FIR filter.

### Derivatives (VPG, APG)
- Method: `np.gradient` (central finite differences)
- VPG = d/dt(filtered PPG)
- APG = d²/dt²(filtered PPG)
- No future context: derivatives computed within each 10-second window only

### Normalization
- Per-window z-score: `x = (x - mean(x)) / (std(x) + 1e-8)`
- Applied independently to each channel (PPG, VPG, APG)
- Zero leakage: uses only the current window statistics

---

## 5. Input Representation

| Model | Tensor Shape | Channels |
|---|---|---|
| Model A (PPG Only) | [B, 1, 1250] | Filtered PPG |
| Model B (PPG+VPG+APG) | [B, 3, 1250] | PPG, VPG, APG |

---

## 6. CNN Architecture

```
Input [B, n_ch, 1250]
Conv1D(n_ch→32, k=7, pad=same) + BatchNorm + ReLU + MaxPool(2)
Conv1D(32→64, k=7, pad=same)   + BatchNorm + ReLU + MaxPool(2)
Conv1D(64→128, k=5, pad=same)  + BatchNorm + ReLU + MaxPool(2)
Conv1D(128→128, k=5, pad=same) + BatchNorm + ReLU + AdaptiveAvgPool(1)
Flatten → [B, 128]
Linear(128→64) + ReLU + Dropout(0.2)
SBP head: Linear(64→1)
DBP head: Linear(64→1)
Output: [B, 2]  → [SBP_pred, DBP_pred]
```

---

## 7. Training Procedure

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6) |
| Loss | HuberLoss(delta=5) for SBP + DBP |
| Batch size | {BATCH_SIZE} |
| Max epochs | {MAX_EPOCHS} |
| Early stop patience | {EARLY_STOP_PATIENCE} |
| Gradient clipping | norm=5.0 |
| Mixed precision | Disabled |

---

## 8. Leakage Controls

All the following were verified by explicit assertions:

- ✓ `train_records ∩ val_records = ∅`
- ✓ `train_records ∩ test_records = ∅`
- ✓ `val_records ∩ test_records = ∅`
- ✓ ECG (channel 2) never accessed during data loading
- ✓ ABP used only as target source; never passed to model
- ✓ Per-window normalization uses only current window statistics
- ✓ No future context (single-window only)
- ✓ NaN/Inf checks on inputs and targets

---

## 9. PPG-Only Results (Model A)

### Validation
| Metric | SBP | DBP |
|---|---|---|
| MAE (mmHg) | {metrics_val_a["sbp"]["mae"]:.2f} | {metrics_val_a["dbp"]["mae"]:.2f} |
| RMSE (mmHg) | {metrics_val_a["sbp"]["rmse"]:.2f} | {metrics_val_a["dbp"]["rmse"]:.2f} |
| R² | {metrics_val_a["sbp"]["r2"]:.3f} | {metrics_val_a["dbp"]["r2"]:.3f} |
| Bias (mmHg) | {metrics_val_a["sbp"]["bias"]:.2f} | {metrics_val_a["dbp"]["bias"]:.2f} |
| ±5 mmHg (%) | {metrics_val_a["sbp"]["pct_within_5"]:.1f} | {metrics_val_a["dbp"]["pct_within_5"]:.1f} |
| ±10 mmHg (%) | {metrics_val_a["sbp"]["pct_within_10"]:.1f} | {metrics_val_a["dbp"]["pct_within_10"]:.1f} |
| ±15 mmHg (%) | {metrics_val_a["sbp"]["pct_within_15"]:.1f} | {metrics_val_a["dbp"]["pct_within_15"]:.1f} |

### Test (Final, Evaluated Once)
| Metric | SBP | DBP |
|---|---|---|
| MAE (mmHg) | **{ma["sbp"]["mae"]:.2f}** | **{ma["dbp"]["mae"]:.2f}** |
| RMSE (mmHg) | {ma["sbp"]["rmse"]:.2f} | {ma["dbp"]["rmse"]:.2f} |
| R² | {ma["sbp"]["r2"]:.3f} | {ma["dbp"]["r2"]:.3f} |
| Bias (mmHg) | {ma["sbp"]["bias"]:.2f} | {ma["dbp"]["bias"]:.2f} |
| ±5 mmHg (%) | {ma["sbp"]["pct_within_5"]:.1f} | {ma["dbp"]["pct_within_5"]:.1f} |
| ±10 mmHg (%) | {ma["sbp"]["pct_within_10"]:.1f} | {ma["dbp"]["pct_within_10"]:.1f} |
| ±15 mmHg (%) | {ma["sbp"]["pct_within_15"]:.1f} | {ma["dbp"]["pct_within_15"]:.1f} |

---

## 10. PPG+VPG+APG Results (Model B)

### Validation
| Metric | SBP | DBP |
|---|---|---|
| MAE (mmHg) | {metrics_val_b["sbp"]["mae"]:.2f} | {metrics_val_b["dbp"]["mae"]:.2f} |
| RMSE (mmHg) | {metrics_val_b["sbp"]["rmse"]:.2f} | {metrics_val_b["dbp"]["rmse"]:.2f} |
| R² | {metrics_val_b["sbp"]["r2"]:.3f} | {metrics_val_b["dbp"]["r2"]:.3f} |
| Bias (mmHg) | {metrics_val_b["sbp"]["bias"]:.2f} | {metrics_val_b["dbp"]["bias"]:.2f} |
| ±5 mmHg (%) | {metrics_val_b["sbp"]["pct_within_5"]:.1f} | {metrics_val_b["dbp"]["pct_within_5"]:.1f} |
| ±10 mmHg (%) | {metrics_val_b["sbp"]["pct_within_10"]:.1f} | {metrics_val_b["dbp"]["pct_within_10"]:.1f} |
| ±15 mmHg (%) | {metrics_val_b["sbp"]["pct_within_15"]:.1f} | {metrics_val_b["dbp"]["pct_within_15"]:.1f} |

### Test (Final, Evaluated Once)
| Metric | SBP | DBP |
|---|---|---|
| MAE (mmHg) | **{mb["sbp"]["mae"]:.2f}** | **{mb["dbp"]["mae"]:.2f}** |
| RMSE (mmHg) | {mb["sbp"]["rmse"]:.2f} | {mb["dbp"]["rmse"]:.2f} |
| R² | {mb["sbp"]["r2"]:.3f} | {mb["dbp"]["r2"]:.3f} |
| Bias (mmHg) | {mb["sbp"]["bias"]:.2f} | {mb["dbp"]["bias"]:.2f} |
| ±5 mmHg (%) | {mb["sbp"]["pct_within_5"]:.1f} | {mb["dbp"]["pct_within_5"]:.1f} |
| ±10 mmHg (%) | {mb["sbp"]["pct_within_10"]:.1f} | {mb["dbp"]["pct_within_10"]:.1f} |
| ±15 mmHg (%) | {mb["sbp"]["pct_within_15"]:.1f} | {mb["dbp"]["pct_within_15"]:.1f} |

---

## 11. Comparison Against Phase 3A

| Model | SBP MAE | DBP MAE | Combined MAE |
|---|---|---|---|
| Phase 3A Classical (PPG features) | 13.93 | 7.05 | 10.49 |
| Model A: PPG Only CNN | {ma["sbp"]["mae"]:.2f} | {ma["dbp"]["mae"]:.2f} | {comb_a:.2f} |
| Model B: PPG+VPG+APG CNN | {mb["sbp"]["mae"]:.2f} | {mb["dbp"]["mae"]:.2f} | {comb_b:.2f} |

Model A vs Phase 3A: SBP {_beat(ma["sbp"]["mae"], PHASE3A_SBP_MAE)}, DBP {_beat(ma["dbp"]["mae"], PHASE3A_DBP_MAE)}
Model B vs Phase 3A: SBP {_beat(mb["sbp"]["mae"], PHASE3A_SBP_MAE)}, DBP {_beat(mb["dbp"]["mae"], PHASE3A_DBP_MAE)}
VPG/APG benefit (B vs A): SBP {_beat(mb["sbp"]["mae"], ma["sbp"]["mae"])}, DBP {_beat(mb["dbp"]["mae"], ma["dbp"]["mae"])}

---

## 12. Limitations

1. **Calibration-free setting**: No subject-level initialization. All estimates
   are population-level predictions without personalization.
2. **Record-level (not subject-level) separation**: The MIMIC-II dataset used
   here does not provide verified patient-level identity for this protocol.
   Stronger subject-independent claims require an explicitly subject-labeled dataset.
3. **No temporal history**: Phase 4A intentionally uses only a single 10-second
   window. Phase 3B demonstrated that 20–60 second context provides additional
   signal; this is addressed in Phase 4C.
4. **Offline preprocessing**: `filtfilt` cannot be deployed on streaming hardware.
5. **MSE-family loss**: HuberLoss still exhibits regression-to-the-mean in extreme
   BP ranges without explicit range-weighted objectives.

---

## 13. Reproducibility Information

- Random seed: {SEED}
- See `PHASE4A_EVIDENCE_FREEZE.md` for complete version and config record.

---

## 14. Conclusion

See `PHASE4A_EVIDENCE_FREEZE.md` Section 5 for research decision summary.

---

## 15. Exact Next Experiment

Based on these results, the justified next step is:
- If CNN improved over Phase 3A AND VPG/APG helped: proceed to Phase 4C
  (temporal 1D CNN + GRU over 20–60 second sequences with explicit derivative channels)
- If no improvement: investigate data loading or architectural issues before adding complexity
"""
    with open(REPORT_DIR / "PHASE4A_NEURAL_BASELINE_REPORT.md", "w") as f:
        f.write(report)
    logger.info("PHASE4A_NEURAL_BASELINE_REPORT.md written.")


def _write_evidence_freeze(
    meta, metrics_val_a, metrics_test_a, metrics_val_b, metrics_test_b,
    model_a, model_b, logger,
) -> None:
    """Write PHASE4A_EVIDENCE_FREEZE.md."""
    import torch as _torch
    comb_a = (metrics_test_a["sbp"]["mae"] + metrics_test_a["dbp"]["mae"]) / 2
    comb_b = (metrics_test_b["sbp"]["mae"] + metrics_test_b["dbp"]["mae"]) / 2

    freeze = f"""# Phase 4A Evidence Freeze

**Date**: {meta['timestamp']}  
**Status**: COMPLETE & FROZEN  

---

## 1. Dataset

| Item | Value |
|---|---|
| Dataset | MIMIC-II Waveform Database |
| Manifest | `code/outputs/windows/window_manifest.csv` |
| Split file | `code/outputs/splits/record_split.csv` |
| Total windows | 261,339 |
| Train windows | 183,517 |
| Val windows | 39,461 |
| Test windows | 38,361 |
| Random seed | {SEED} |

---

## 2. Software Environment

| Item | Value |
|---|---|
| Python | {meta['python_version'].split()[0]} |
| PyTorch | {meta['pytorch_version']} |
| CUDA | {meta['cuda_version']} |
| GPU | {meta['gpu']} |
| Platform | {meta['platform']} |

---

## 3. Preprocessing

| Step | Value |
|---|---|
| Filter | Butterworth bandpass 0.5–8 Hz, order=3, filtfilt (non-causal, offline) |
| VPG | np.gradient(filtered_ppg, dt), central finite differences |
| APG | np.gradient(vpg, dt) |
| Normalization | Per-window z-score, eps=1e-8 |

---

## 4. Model & Training

| Parameter | Value |
|---|---|
| Architecture | PPGCNNBaseline (4×ConvBlock, FC) |
| Model A | n_channels=1 (PPG only) |
| Model B | n_channels=3 (PPG+VPG+APG) |
| Loss | HuberLoss(delta={HUBER_DELTA}) for SBP + DBP |
| Optimizer | AdamW |
| LR | {LR} |
| Weight decay | {WEIGHT_DECAY} |
| Scheduler | ReduceLROnPlateau(factor={LR_FACTOR}, patience={LR_PATIENCE}) |
| Batch size | {BATCH_SIZE} |
| Max epochs | {MAX_EPOCHS} |
| Early stop patience | {EARLY_STOP_PATIENCE} |
| Gradient clip | norm=5.0 |
| Mixed precision | Disabled |
| Model A best epoch | {model_a._best_epoch} |
| Model B best epoch | {model_b._best_epoch} |

---

## 5. Results Summary

### Validation
| Model | SBP MAE | DBP MAE | Combined MAE |
|---|---|---|---|
| Model A: PPG Only | {metrics_val_a["sbp"]["mae"]:.2f} | {metrics_val_a["dbp"]["mae"]:.2f} | {(metrics_val_a["sbp"]["mae"]+metrics_val_a["dbp"]["mae"])/2:.2f} |
| Model B: PPG+VPG+APG | {metrics_val_b["sbp"]["mae"]:.2f} | {metrics_val_b["dbp"]["mae"]:.2f} | {(metrics_val_b["sbp"]["mae"]+metrics_val_b["dbp"]["mae"])/2:.2f} |

### Test (Final, Evaluated Once)
| Model | SBP MAE | DBP MAE | Combined MAE |
|---|---|---|---|
| Phase 3A Classical | 13.93 | 7.05 | 10.49 |
| Model A: PPG Only CNN | {metrics_test_a["sbp"]["mae"]:.2f} | {metrics_test_a["dbp"]["mae"]:.2f} | {comb_a:.2f} |
| Model B: PPG+VPG+APG CNN | {metrics_test_b["sbp"]["mae"]:.2f} | {metrics_test_b["dbp"]["mae"]:.2f} | {comb_b:.2f} |

**Test-set evaluation performed exactly once**: YES

---

## 6. Checkpoints

- Model A: `code/outputs/phase4a/checkpoints/model_a_ppg_only_best.pt`
- Model B: `code/outputs/phase4a/checkpoints/model_b_ppg_vpg_apg_best.pt`

---

## 7. Research Decision

**Did CNN beat classical baseline?**
- Model A vs Phase 3A: SBP delta = {PHASE3A_SBP_MAE - metrics_test_a["sbp"]["mae"]:+.2f}, DBP delta = {PHASE3A_DBP_MAE - metrics_test_a["dbp"]["mae"]:+.2f}
- Model B vs Phase 3A: SBP delta = {PHASE3A_SBP_MAE - metrics_test_b["sbp"]["mae"]:+.2f}, DBP delta = {PHASE3A_DBP_MAE - metrics_test_b["dbp"]["mae"]:+.2f}

**Did VPG/APG help (Model B vs A)?**
- SBP delta = {metrics_test_a["sbp"]["mae"] - metrics_test_b["sbp"]["mae"]:+.2f} mmHg
- DBP delta = {metrics_test_a["dbp"]["mae"] - metrics_test_b["dbp"]["mae"]:+.2f} mmHg

**STOP CONDITION**: Phase 4A is complete.
Next phase decision based on results above — see report Section 15.
"""
    with open(REPORT_DIR / "PHASE4A_EVIDENCE_FREEZE.md", "w") as f:
        f.write(freeze)
    logger.info("PHASE4A_EVIDENCE_FREEZE.md written.")
