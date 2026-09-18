"""
Phase 4A: Evaluation, Metrics, and Figure Generation.

Provides:
    - compute_metrics: Standard regression metrics for BP estimation
    - compute_bp_range_errors: Stratified error analysis by clinical BP ranges
    - evaluate_model: Run model on a DataLoader, return predictions + metrics
    - generate_all_figures: 12 publication-quality figures
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics for BP estimation.

    Args:
        y_true: Ground truth values, shape (N,).
        y_pred: Predicted values, shape (N,).

    Returns:
        Dict with keys: mae, rmse, r2, bias, error_sd,
                        pct_within_5, pct_within_10, pct_within_15, n_samples.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    n = len(y_true)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    var_true = float(np.var(y_true))
    r2 = float(1.0 - np.sum(errors ** 2) / (n * var_true)) if var_true > 1e-9 else 0.0
    bias = float(np.mean(errors))
    error_sd = float(np.std(errors))

    pct_5 = float(np.mean(abs_errors <= 5.0) * 100)
    pct_10 = float(np.mean(abs_errors <= 10.0) * 100)
    pct_15 = float(np.mean(abs_errors <= 15.0) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "error_sd": error_sd,
        "pct_within_5": pct_5,
        "pct_within_10": pct_10,
        "pct_within_15": pct_15,
        "n_samples": n,
    }


def compute_bp_range_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str = "SBP",
) -> pd.DataFrame:
    """
    Compute stratified error metrics by clinical BP ranges.

    SBP bins: <90, 90-119, 120-139, 140-159, >=160
    DBP bins: <60, 60-79, 80-89, 90-99, >=100

    Args:
        y_true: Ground truth BP values.
        y_pred: Predicted BP values.
        target: 'SBP' or 'DBP'.

    Returns:
        DataFrame with columns: range, sample_count, mae, rmse, bias, error_sd.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    if target.upper() == "SBP":
        bins = [-np.inf, 90, 120, 140, 160, np.inf]
        labels = ["<90", "90-119", "120-139", "140-159", ">=160"]
    else:
        bins = [-np.inf, 60, 80, 90, 100, np.inf]
        labels = ["<60", "60-79", "80-89", "90-99", ">=100"]

    bp_cat = pd.cut(y_true, bins=bins, labels=labels, right=False)
    rows = []
    for lbl in labels:
        mask = bp_cat == lbl
        cnt = int(np.sum(mask))
        if cnt > 0:
            rows.append({
                "target": target,
                "range": lbl,
                "sample_count": cnt,
                "mae": float(np.mean(abs_errors[mask])),
                "rmse": float(np.sqrt(np.mean(errors[mask] ** 2))),
                "bias": float(np.mean(errors[mask])),
                "error_sd": float(np.std(errors[mask])),
            })
        else:
            rows.append({
                "target": target, "range": lbl, "sample_count": 0,
                "mae": np.nan, "rmse": np.nan, "bias": np.nan, "error_sd": np.nan,
            })
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference on a DataLoader.

    Returns:
        sbp_true, dbp_true, sbp_pred, dbp_pred — all as (N,) float64 arrays.
    """
    model.eval()
    all_true = []
    all_pred = []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y_hat = model(X).cpu().numpy()   # (B, 2)
        all_true.append(y.numpy())
        all_pred.append(y_hat)

    all_true = np.vstack(all_true)   # (N, 2)
    all_pred = np.vstack(all_pred)   # (N, 2)

    return (
        all_true[:, 0].astype(np.float64),
        all_true[:, 1].astype(np.float64),
        all_pred[:, 0].astype(np.float64),
        all_pred[:, 1].astype(np.float64),
    )


# ============================================================================
# Figure Generation
# ============================================================================

def _scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    split: str,
    mae: float,
    r2: float,
    ax: plt.Axes,
    color: str = "#2563EB",
) -> None:
    """Scatter plot with identity line and metric annotation."""
    ax.scatter(y_true, y_pred, s=2, alpha=0.15, color=color, rasterized=True)
    lims = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Reference {target} (mmHg)")
    ax.set_ylabel(f"Predicted {target} (mmHg)")
    ax.set_title(f"{split} {target}: MAE={mae:.2f} mmHg, R²={r2:.3f}")
    ax.legend(loc="upper left", fontsize=9)


def _bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    split: str,
    ax: plt.Axes,
    color: str = "#DC2626",
) -> None:
    """Bland–Altman plot."""
    mean_bp = (y_true + y_pred) / 2.0
    diff = y_pred - y_true
    bias = np.mean(diff)
    sd = np.std(diff)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    ax.scatter(mean_bp, diff, s=2, alpha=0.15, color=color, rasterized=True)
    ax.axhline(bias, color="navy", linewidth=1.5, linestyle="-", label=f"Bias: {bias:.2f}")
    ax.axhline(loa_upper, color="firebrick", linewidth=1.2, linestyle="--",
               label=f"+1.96 SD: {loa_upper:.2f}")
    ax.axhline(loa_lower, color="firebrick", linewidth=1.2, linestyle="--",
               label=f"-1.96 SD: {loa_lower:.2f}")
    ax.set_xlabel(f"Mean {target} (mmHg)")
    ax.set_ylabel("Predicted − Reference (mmHg)")
    ax.set_title(f"Bland–Altman: {split} {target}")
    ax.legend(loc="upper right", fontsize=9)


def generate_all_figures(
    # Loss curves
    train_losses_a: List[float],
    val_losses_a: List[float],
    train_losses_b: List[float],
    val_losses_b: List[float],
    # Val predictions
    val_sbp_true: np.ndarray,
    val_sbp_pred_a: np.ndarray,
    val_dbp_true: np.ndarray,
    val_dbp_pred_a: np.ndarray,
    val_sbp_pred_b: np.ndarray,
    val_dbp_pred_b: np.ndarray,
    # Test predictions
    test_sbp_true: np.ndarray,
    test_sbp_pred_a: np.ndarray,
    test_dbp_true: np.ndarray,
    test_dbp_pred_a: np.ndarray,
    test_sbp_pred_b: np.ndarray,
    test_dbp_pred_b: np.ndarray,
    # Metrics for labels
    metrics_val_a: Dict,
    metrics_val_b: Dict,
    metrics_test_a: Dict,
    metrics_test_b: Dict,
    # Output dir
    figures_dir: Path,
) -> None:
    """Generate all 12 required figures for Phase 4A."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Training vs Validation Loss (Model A) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_a = range(1, len(train_losses_a) + 1)
    ax.plot(epochs_a, train_losses_a, label="Train Loss", color="#1E40AF")
    ax.plot(epochs_a, val_losses_a, label="Val Loss", color="#DC2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Figure 1: Training vs Validation Loss — Model A (PPG Only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig01_loss_model_a.png", dpi=150)
    plt.close(fig)

    # ---- Figure 1b: Training vs Validation Loss (Model B) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_b = range(1, len(train_losses_b) + 1)
    ax.plot(epochs_b, train_losses_b, label="Train Loss", color="#1E40AF")
    ax.plot(epochs_b, val_losses_b, label="Val Loss", color="#DC2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Figure 1b: Training vs Validation Loss — Model B (PPG+VPG+APG)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig01b_loss_model_b.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: Val SBP Predicted vs True (Model B) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_plot(val_sbp_true, val_sbp_pred_b, "SBP", "Validation",
                  metrics_val_b["sbp"]["mae"], metrics_val_b["sbp"]["r2"], ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig02_val_sbp_scatter.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: Val DBP Predicted vs True (Model B) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_plot(val_dbp_true, val_dbp_pred_b, "DBP", "Validation",
                  metrics_val_b["dbp"]["mae"], metrics_val_b["dbp"]["r2"], ax, color="#7C3AED")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig03_val_dbp_scatter.png", dpi=150)
    plt.close(fig)

    # ---- Figure 4: Test SBP Predicted vs True (Model B) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_plot(test_sbp_true, test_sbp_pred_b, "SBP", "Test",
                  metrics_test_b["sbp"]["mae"], metrics_test_b["sbp"]["r2"], ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig04_test_sbp_scatter.png", dpi=150)
    plt.close(fig)

    # ---- Figure 5: Test DBP Predicted vs True (Model B) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_plot(test_dbp_true, test_dbp_pred_b, "DBP", "Test",
                  metrics_test_b["dbp"]["mae"], metrics_test_b["dbp"]["r2"], ax, color="#7C3AED")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig05_test_dbp_scatter.png", dpi=150)
    plt.close(fig)

    # ---- Figure 6: SBP Error Distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_, pred, label, color in [
        (axes[0], test_sbp_pred_a, "Model A (PPG Only)", "#2563EB"),
        (axes[1], test_sbp_pred_b, "Model B (PPG+VPG+APG)", "#059669"),
    ]:
        errs = pred - test_sbp_true
        ax_.hist(errs, bins=80, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax_.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax_.axvline(np.mean(errs), color="red", linewidth=1.2, linestyle="-",
                    label=f"Bias={np.mean(errs):.2f}")
        ax_.set_xlabel("Error (mmHg)")
        ax_.set_ylabel("Count")
        ax_.set_title(f"Fig 6: Test SBP Error — {label}")
        ax_.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig06_test_sbp_error_dist.png", dpi=150)
    plt.close(fig)

    # ---- Figure 7: DBP Error Distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_, pred, label, color in [
        (axes[0], test_dbp_pred_a, "Model A (PPG Only)", "#7C3AED"),
        (axes[1], test_dbp_pred_b, "Model B (PPG+VPG+APG)", "#D97706"),
    ]:
        errs = pred - test_dbp_true
        ax_.hist(errs, bins=80, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax_.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax_.axvline(np.mean(errs), color="red", linewidth=1.2, linestyle="-",
                    label=f"Bias={np.mean(errs):.2f}")
        ax_.set_xlabel("Error (mmHg)")
        ax_.set_ylabel("Count")
        ax_.set_title(f"Fig 7: Test DBP Error — {label}")
        ax_.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig07_test_dbp_error_dist.png", dpi=150)
    plt.close(fig)

    # ---- Figure 8: SBP Bland-Altman (Model B) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    _bland_altman(test_sbp_true, test_sbp_pred_b, "SBP", "Test", ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig08_bland_altman_sbp.png", dpi=150)
    plt.close(fig)

    # ---- Figure 9: DBP Bland-Altman (Model B) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    _bland_altman(test_dbp_true, test_dbp_pred_b, "DBP", "Test", ax, color="#7C3AED")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig09_bland_altman_dbp.png", dpi=150)
    plt.close(fig)

    # ---- Figure 10: Error by SBP Range ----
    df_sbp_a = compute_bp_range_errors(test_sbp_true, test_sbp_pred_a, "SBP")
    df_sbp_b = compute_bp_range_errors(test_sbp_true, test_sbp_pred_b, "SBP")
    _plot_range_errors(df_sbp_a, df_sbp_b, "SBP", figures_dir / "fig10_sbp_range_errors.png")

    # ---- Figure 11: Error by DBP Range ----
    df_dbp_a = compute_bp_range_errors(test_dbp_true, test_dbp_pred_a, "DBP")
    df_dbp_b = compute_bp_range_errors(test_dbp_true, test_dbp_pred_b, "DBP")
    _plot_range_errors(df_dbp_a, df_dbp_b, "DBP", figures_dir / "fig11_dbp_range_errors.png")

    # ---- Figure 12: Model A vs B Comparison ----
    _plot_model_comparison(metrics_test_a, metrics_test_b, figures_dir / "fig12_model_comparison.png")

    print(f"[Figures] All 12 figures saved to {figures_dir}")


def _plot_range_errors(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    target: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ranges = df_a["range"].values
    x = np.arange(len(ranges))
    width = 0.35

    for ax_, metric, ylabel in [
        (axes[0], "mae", "MAE (mmHg)"),
        (axes[1], "bias", "Bias (mmHg)"),
    ]:
        vals_a = df_a[metric].fillna(0).values
        vals_b = df_b[metric].fillna(0).values
        bars_a = ax_.bar(x - width / 2, vals_a, width, label="Model A (PPG Only)", color="#2563EB", alpha=0.85)
        bars_b = ax_.bar(x + width / 2, vals_b, width, label="Model B (PPG+VPG+APG)", color="#059669", alpha=0.85)
        ax_.set_xticks(x)
        ax_.set_xticklabels(ranges, rotation=20, ha="right")
        ax_.set_xlabel(f"{target} Range (mmHg)")
        ax_.set_ylabel(ylabel)
        ax_.set_title(f"{target} {ylabel} by Clinical Range")
        ax_.legend(fontsize=9)
        if metric == "bias":
            ax_.axhline(0, color="black", linewidth=1.0, linestyle="--")

    fig.suptitle(f"Figure: {target} Error by Clinical BP Range — Test Set", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_model_comparison(
    metrics_a: Dict,
    metrics_b: Dict,
    save_path: Path,
) -> None:
    """Bar chart comparing Model A vs Model B vs Phase 3A classical baseline."""
    labels = ["SBP MAE", "DBP MAE", "Combined MAE"]
    phase3a = [13.93, 7.05, 10.49]
    model_a = [
        metrics_a["sbp"]["mae"],
        metrics_a["dbp"]["mae"],
        (metrics_a["sbp"]["mae"] + metrics_a["dbp"]["mae"]) / 2,
    ]
    model_b = [
        metrics_b["sbp"]["mae"],
        metrics_b["dbp"]["mae"],
        (metrics_b["sbp"]["mae"] + metrics_b["dbp"]["mae"]) / 2,
    ]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, phase3a, width, label="Phase 3A Classical", color="#6B7280", alpha=0.85)
    ax.bar(x, model_a, width, label="Model A: PPG Only CNN", color="#2563EB", alpha=0.85)
    ax.bar(x + width, model_b, width, label="Model B: PPG+VPG+APG CNN", color="#059669", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("Figure 12: Phase 4A CNN vs Phase 3A Classical Baseline — Test Set")
    ax.legend()
    ax.set_ylim(0, max(max(phase3a), max(model_a), max(model_b)) * 1.3)

    # Annotate values
    for bars in [
        ax.bar(x - width, phase3a, width, color="#6B7280", alpha=0),
        ax.bar(x, model_a, width, color="#2563EB", alpha=0),
        ax.bar(x + width, model_b, width, color="#059669", alpha=0),
    ]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
