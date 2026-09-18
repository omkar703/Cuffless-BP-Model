"""
Publication-Grade Visualizations for Phase 3A Baseline Modeling.

Generates all 18 mandated figures:
1. model_01_sbp_true_vs_pred.png
2. model_02_dbp_true_vs_pred.png
3. model_03_sbp_residuals.png
4. model_04_dbp_residuals.png
5. model_05_sbp_bland_altman.png
6. model_06_dbp_bland_altman.png
7. model_07_sbp_error_vs_ref.png
8. model_08_dbp_error_vs_ref.png
9. model_09_sbp_mae_by_bp_range.png
10. model_10_dbp_mae_by_bp_range.png
11. model_11_record_level_mae_distribution.png
12. model_12_model_comparison_bar.png
13. model_13_error_vs_heart_rate.png
14. model_14_error_vs_ppg_quality.png
15. model_15_error_vs_pulse_amplitude_variation.png
16. model_16_record_weighted_vs_window_weighted.png
17. model_17_feature_group_importance.png
18. model_18_feature_group_ablation.png
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config.config import FIGURES_DIR
from utils.logging_utils import setup_logger

logger = setup_logger("model_plots")

# Publication style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    save_path: Path,
    r2_val: float,
    mae_val: float,
) -> None:
    """True vs Predicted scatter plot with identity line."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    # Subsample points if large to maintain crisp rendering
    n_pts = len(y_true)
    if n_pts > 5000:
        idx = np.random.choice(n_pts, 5000, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred

    ax.scatter(yt, yp, alpha=0.25, color="#1f77b4", edgecolors="none", s=18, label="Test Windows")

    # Limits and identity line
    min_val = min(np.min(yt), np.min(yp)) - 5
    max_val = max(np.max(yt), np.max(yp)) + 5
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.8, label="Identity (y = x)")

    # Linear fit line
    if np.std(yt) > 1e-6:
        m, c = np.polyfit(yt, yp, 1)
        x_grid = np.linspace(min_val, max_val, 100)
        ax.plot(x_grid, m * x_grid + c, color="#2ca02c", lw=1.5, label=f"Fit: y = {m:.2f}x + {c:.1f}")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(f"Reference True {target_name} (mmHg)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Predicted {target_name} (mmHg)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"True vs Predicted {target_name} (Test Set)\nMAE: {mae_val:.2f} mmHg | R²: {r2_val:.3f}",
        fontsize=12,
        pad=10,
    )
    ax.legend(frameon=True, loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    save_path: Path,
) -> None:
    """Residual distribution histogram and error metrics."""
    errors = y_pred - y_true
    bias = np.mean(errors)
    sd = np.std(errors)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    bins = np.linspace(bias - 4 * sd, bias + 4 * sd, 60)
    ax.hist(errors, bins=bins, color="#3b528b", edgecolor="white", alpha=0.8, density=True)

    # Reference zero and bias lines
    ax.axvline(0, color="black", linestyle="-", lw=1.2, label="Zero Error")
    ax.axvline(bias, color="red", linestyle="--", lw=1.5, label=f"Mean Bias: {bias:.2f} mmHg")
    ax.axvline(bias - 1.96 * sd, color="orange", linestyle=":", lw=1.2, label=f"-1.96 SD: {bias-1.96*sd:.1f}")
    ax.axvline(bias + 1.96 * sd, color="orange", linestyle=":", lw=1.2, label=f"+1.96 SD: {bias+1.96*sd:.1f}")

    ax.set_xlabel(f"Prediction Error (Pred - True, mmHg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{target_name} Residual Distribution (Test Set)\nBias: {bias:.2f} mmHg | SD: {sd:.2f} mmHg",
        fontsize=12,
        pad=10,
    )
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    save_path: Path,
) -> None:
    """Bland-Altman agreement plot."""
    means = 0.5 * (y_true + y_pred)
    diffs = y_pred - y_true
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs))
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    n_pts = len(means)
    if n_pts > 5000:
        idx = np.random.choice(n_pts, 5000, replace=False)
        m_plot, d_plot = means[idx], diffs[idx]
    else:
        m_plot, d_plot = means, diffs

    ax.scatter(m_plot, d_plot, alpha=0.25, color="#21918c", edgecolors="none", s=16)

    # Bias and LoA lines
    ax.axhline(bias, color="red", linestyle="-", lw=1.6, label=f"Mean Bias: {bias:.2f} mmHg")
    ax.axhline(
        upper_loa,
        color="darkblue",
        linestyle="--",
        lw=1.4,
        label=f"+1.96 SD (Upper LoA): {upper_loa:.2f} mmHg",
    )
    ax.axhline(
        lower_loa,
        color="darkblue",
        linestyle="--",
        lw=1.4,
        label=f"-1.96 SD (Lower LoA): {lower_loa:.2f} mmHg",
    )

    ax.set_xlabel(f"Mean of True and Predicted {target_name} (mmHg)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Difference (Pred - True, mmHg)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{target_name} Bland-Altman Agreement Plot (Test Set)\n95% Limits of Agreement [{lower_loa:.1f}, {upper_loa:.1f}] mmHg",
        fontsize=12,
        pad=10,
    )
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_error_vs_ref(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    save_path: Path,
) -> None:
    """Prediction error vs Reference BP value (demonstrates regression to the mean)."""
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    n_pts = len(y_true)
    if n_pts > 5000:
        idx = np.random.choice(n_pts, 5000, replace=False)
        yt, err = y_true[idx], errors[idx]
    else:
        yt, err = y_true, errors

    ax.scatter(yt, err, alpha=0.25, color="#440154", edgecolors="none", s=16)
    ax.axhline(0, color="black", linestyle="-", lw=1.2)

    # Trend line
    if np.std(yt) > 1e-6:
        m, c = np.polyfit(yt, err, 1)
        x_grid = np.linspace(np.min(yt), np.max(yt), 100)
        ax.plot(x_grid, m * x_grid + c, color="red", lw=2.0, label=f"Error Trend (slope={m:.2f})")

    ax.set_xlabel(f"Reference {target_name} (mmHg)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Prediction Error (Pred - True, mmHg)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{target_name} Prediction Error vs Reference Value\nDemonstrates Systematic Overestimation at Low BP & Underestimation at High BP",
        fontsize=11,
        pad=10,
    )
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_mae_by_bp_range(
    df_range_error: pd.DataFrame,
    target_name: str,
    save_path: Path,
) -> None:
    """Bar chart of MAE by BP range with sample counts."""
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    ranges = df_range_error["range"].tolist()
    maes = df_range_error["MAE"].tolist()
    counts = df_range_error["sample_count"].tolist()

    x = np.arange(len(ranges))
    width = 0.55

    bars = ax1.bar(x, maes, width, color="#2b5c8f", edgecolor="black", alpha=0.85, label="MAE (mmHg)")
    ax1.set_xlabel(f"Physiological {target_name} Range (mmHg)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error (mmHg)", fontsize=11, fontweight="bold", color="#2b5c8f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranges, fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    valid_maes = [m for m in maes if (m is not None and not np.isnan(m))]
    top_limit = max(valid_maes) * 1.25 if valid_maes else 30.0
    ax1.set_ylim(0, max(10.0, top_limit))

    # Display sample count and MAE value on top of each bar
    for bar, mae, cnt in zip(bars, maes, counts):
        if not np.isnan(mae):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{mae:.1f}\n(N={cnt:,})",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax1.set_title(f"{target_name} Error Stratification by Blood Pressure Range", fontsize=12, pad=12)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_record_level_distribution(
    rec_df_sbp: pd.DataFrame,
    rec_df_dbp: pd.DataFrame,
    save_path: Path,
) -> None:
    """Boxplot and KDE of per-record MAE for SBP and DBP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    ax1.hist(rec_df_sbp["record_mae"], bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
    ax1.axvline(
        rec_df_sbp["record_mae"].mean(),
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Mean: {rec_df_sbp['record_mae'].mean():.2f}",
    )
    ax1.axvline(
        rec_df_sbp["record_mae"].median(),
        color="black",
        linestyle=":",
        lw=1.5,
        label=f"Median: {rec_df_sbp['record_mae'].median():.2f}",
    )
    ax1.set_title("SBP Per-Record MAE Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Record MAE (mmHg)", fontsize=10)
    ax1.set_ylabel("Record Count", fontsize=10)
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.hist(rec_df_dbp["record_mae"], bins=40, color="#2ca02c", edgecolor="white", alpha=0.8)
    ax2.axvline(
        rec_df_dbp["record_mae"].mean(),
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Mean: {rec_df_dbp['record_mae'].mean():.2f}",
    )
    ax2.axvline(
        rec_df_dbp["record_mae"].median(),
        color="black",
        linestyle=":",
        lw=1.5,
        label=f"Median: {rec_df_dbp['record_mae'].median():.2f}",
    )
    ax2.set_title("DBP Per-Record MAE Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Record MAE (mmHg)", fontsize=10)
    ax2.set_ylabel("Record Count", fontsize=10)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Distribution of Cross-Record Generalization Error", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_model_comparison(
    df_results: pd.DataFrame,
    save_path: Path,
) -> None:
    """Grouped bar chart comparing validation MAE across all 5 models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    sbp_res = df_results[df_results["target"] == "SBP"].copy()
    dbp_res = df_results[df_results["target"] == "DBP"].copy()

    models = sbp_res["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    # SBP
    bars1 = ax1.bar(x - width / 2, sbp_res["mae"], width, label="MAE (mmHg)", color="#3182bd")
    bars2 = ax1.bar(x + width / 2, sbp_res["rmse"], width, label="RMSE (mmHg)", color="#9ecae1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax1.set_title("SBP Model Comparison (Validation)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Error (mmHg)", fontsize=10)
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    # DBP
    bars3 = ax2.bar(x - width / 2, dbp_res["mae"], width, label="MAE (mmHg)", color="#31a354")
    bars4 = ax2.bar(x + width / 2, dbp_res["rmse"], width, label="RMSE (mmHg)", color="#a1d99b")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax2.set_title("DBP Model Comparison (Validation)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Error (mmHg)", fontsize=10)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_error_vs_hr(
    df_eval: pd.DataFrame,
    save_path: Path,
) -> None:
    """Absolute error vs Estimated Heart Rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    sub = df_eval[df_eval["hr_bpm"].between(40, 180)].copy()
    if len(sub) > 5000:
        sub = sub.sample(5000, random_state=42)

    # SBP
    ax1.scatter(sub["hr_bpm"], sub["sbp_abs_error"], alpha=0.2, color="#756bb1", s=15)
    if len(sub) >= 2 and np.std(sub["hr_bpm"]) > 1e-6:
        m1, c1 = np.polyfit(sub["hr_bpm"], sub["sbp_abs_error"], 1)
        x1 = np.linspace(sub["hr_bpm"].min(), sub["hr_bpm"].max(), 100)
        ax1.plot(x1, m1 * x1 + c1, "r-", lw=2, label=f"Trend (slope={m1:.3f})")
    ax1.set_xlabel("Estimated Heart Rate (bpm)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("SBP Absolute Error (mmHg)", fontsize=10, fontweight="bold")
    ax1.set_title("SBP Error vs Heart Rate", fontsize=11)
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # DBP
    ax2.scatter(sub["hr_bpm"], sub["dbp_abs_error"], alpha=0.2, color="#2b8cbe", s=15)
    if len(sub) >= 2 and np.std(sub["hr_bpm"]) > 1e-6:
        m2, c2 = np.polyfit(sub["hr_bpm"], sub["dbp_abs_error"], 1)
        x2 = np.linspace(sub["hr_bpm"].min(), sub["hr_bpm"].max(), 100)
        ax2.plot(x2, m2 * x2 + c2, "r-", lw=2, label=f"Trend (slope={m2:.3f})")
    ax2.set_xlabel("Estimated Heart Rate (bpm)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("DBP Absolute Error (mmHg)", fontsize=10, fontweight="bold")
    ax2.set_title("DBP Error vs Heart Rate", fontsize=11)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_error_vs_quality(
    df_quality_sbp: pd.DataFrame,
    df_quality_dbp: pd.DataFrame,
    save_path: Path,
) -> None:
    """MAE across quality strata."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    strata = df_quality_sbp["stratum"].tolist()
    x = np.arange(len(strata))

    ax1.bar(x, df_quality_sbp["MAE"], color="#fa8c16", alpha=0.85, edgecolor="black", width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strata, rotation=25, ha="right", fontsize=9)
    ax1.set_title("SBP MAE by Signal Quality Stratum", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MAE (mmHg)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    ax2.bar(x, df_quality_dbp["MAE"], color="#faad14", alpha=0.85, edgecolor="black", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strata, rotation=25, ha="right", fontsize=9)
    ax2.set_title("DBP MAE by Signal Quality Stratum", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MAE (mmHg)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_error_vs_amplitude_variation(
    df_eval: pd.DataFrame,
    save_path: Path,
) -> None:
    """Absolute error vs pulse amplitude variability (CV)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    sub = df_eval[df_eval["pulse_amp_cv_a"].between(0, 1.0)].copy()
    if len(sub) > 5000:
        sub = sub.sample(5000, random_state=42)

    ax1.scatter(sub["pulse_amp_cv_a"], sub["sbp_abs_error"], alpha=0.2, color="#e6550d", s=15)
    ax1.set_xlabel("Pulse Amplitude Coefficient of Variation (CV)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("SBP Absolute Error (mmHg)", fontsize=10, fontweight="bold")
    ax1.set_title("SBP Error vs Pulse Amplitude Variation", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.scatter(sub["pulse_amp_cv_a"], sub["dbp_abs_error"], alpha=0.2, color="#31a354", s=15)
    ax2.set_xlabel("Pulse Amplitude Coefficient of Variation (CV)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("DBP Absolute Error (mmHg)", fontsize=10, fontweight="bold")
    ax2.set_title("DBP Error vs Pulse Amplitude Variation", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_record_vs_window_weighted(
    win_mae_sbp: float,
    rec_mae_mean_sbp: float,
    rec_mae_med_sbp: float,
    win_mae_dbp: float,
    rec_mae_mean_dbp: float,
    rec_mae_med_dbp: float,
    save_path: Path,
) -> None:
    """Model 16: Compares Window-Weighted vs Record-Weighted performance."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    targets = ["SBP", "DBP"]
    x = np.arange(len(targets))
    width = 0.25

    win_vals = [win_mae_sbp, win_mae_dbp]
    rec_mean_vals = [rec_mae_mean_sbp, rec_mae_mean_dbp]
    rec_med_vals = [rec_mae_med_sbp, rec_mae_med_dbp]

    r1 = ax.bar(x - width, win_vals, width, label="Window-Weighted MAE", color="#1f77b4", edgecolor="black")
    r2 = ax.bar(x, rec_mean_vals, width, label="Record-Weighted Mean MAE", color="#ff7f0e", edgecolor="black")
    r3 = ax.bar(x + width, rec_med_vals, width, label="Record-Weighted Median MAE", color="#2ca02c", edgecolor="black")

    ax.set_ylabel("MAE (mmHg)", fontsize=11, fontweight="bold")
    ax.set_title("Window-Weighted vs Record-Weighted Evaluation (Model 16)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    for rects in [r1, r2, r3]:
        for r in rects:
            h = r.get_height()
            ax.text(
                r.get_x() + r.get_width() / 2,
                h + 0.2,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_feature_group_importance(
    df_grp_imp: pd.DataFrame,
    save_path: Path,
) -> None:
    """Model 17: Total and normalized importance percentage by feature group."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    df_sorted = df_grp_imp.sort_values(by="importance_percentage", ascending=True)
    groups = df_sorted["group"].tolist()
    pcts = df_sorted["importance_percentage"].tolist()

    y = np.arange(len(groups))
    bars = ax.barh(y, pcts, color="#3b528b", edgecolor="black", alpha=0.85, height=0.55)

    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=10, fontweight="bold")
    ax.set_xlabel("Relative Importance Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Feature Group Importance Distribution (Model 17)", fontsize=12, pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    for bar, pct in zip(bars, pcts):
        ax.text(
            pct + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


def plot_feature_group_ablation(
    df_ablation: pd.DataFrame,
    save_path: Path,
) -> None:
    """Model 18: Cumulative feature group ablation curves for SBP and DBP."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    sbp_abl = df_ablation[df_ablation["target"] == "SBP"].copy()
    dbp_abl = df_ablation[df_ablation["target"] == "DBP"].copy()

    x_labels = [
        "Dummy (0)",
        "+ Basic (A)",
        "+ Timing (B)",
        "+ Morph (C)",
        "+ VPG (D)",
        "+ APG (E)",
        "+ Spectral (F)",
    ]
    x = np.arange(len(x_labels))

    ax.plot(x, sbp_abl["MAE"], "o-", color="#d62728", lw=2.2, ms=7, label="SBP MAE (mmHg)")
    ax.plot(x, dbp_abl["MAE"], "s-", color="#1f77b4", lw=2.2, ms=7, label="DBP MAE (mmHg)")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Validation MAE (mmHg)", fontsize=11, fontweight="bold")
    ax.set_title("Cumulative Feature-Group Ablation Performance (Model 18)", fontsize=12, pad=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Annotate points
    for xi, yi in zip(x, sbp_abl["MAE"]):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#d62728",
        )

    for xi, yi in zip(x, dbp_abl["MAE"]):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1f77b4",
        )

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")
