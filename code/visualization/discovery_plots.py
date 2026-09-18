"""
Visualization Engine for Phase 3B Research Discovery 1.

Generates the 10 mandated publication-grade figures in code/outputs/figures/phase3b_discovery/:
01_context_length_vs_sbp_mae.png
02_context_length_vs_dbp_mae.png
03_context_length_vs_sbp_extreme_bias.png  (MOST CRITICAL)
04_context_length_vs_dbp_extreme_bias.png
05_context_length_vs_r2.png
06_regression_to_mean_comparison.png
07_record_level_mae_comparison.png
08_ppg_vpg_apg_ablation.png
09_temporal_feature_importance.png
10_temporal_context_example.png
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure publication-grade styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#2c3e50",
        "axes.linewidth": 1.2,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Custom color palette
NAVY = "#1a365d"
TEAL = "#0d9488"
CORAL = "#e11d48"
AMBER = "#d97706"
PURPLE = "#7c3aed"
SLATE = "#475569"
LIGHT_BG = "#f8fafc"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_01_context_vs_sbp_mae(df_context: pd.DataFrame, out_dir: Path) -> None:
    """Figure 01: Context Length vs SBP MAE."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = df_context["context_sec"]
    y_all = df_context["sbp_mae_all"]
    y_matched = df_context.get("sbp_mae_matched", y_all)

    ax.plot(x, y_all, marker="o", markersize=8, linewidth=2.2, color=NAVY, label="All Eligible Windows")
    if "sbp_mae_matched" in df_context.columns:
        ax.plot(x, y_matched, marker="s", markersize=7, linewidth=2.0, linestyle="--", color=TEAL, label="Matched Windows (≥5 History)")

    ax.set_title("Context Length vs. SBP Estimation Error (MAE)")
    ax.set_xlabel("Sequential Context Duration (seconds)")
    ax.set_ylabel("Validation SBP MAE (mmHg)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}s\n(Ctx-{k})" for s, k in zip(x, df_context["context_k"])])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor=LIGHT_BG, edgecolor="#cbd5e1")

    # Annotate delta vs Context-0
    c0_mae = y_all.iloc[0]
    for i, (xi, yi) in enumerate(zip(x, y_all)):
        diff = yi - c0_mae
        sign = "+" if diff > 0 else ""
        ax.annotate(
            f"{yi:.2f}\n({sign}{diff:.2f})",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10 if i % 2 == 0 else -25),
            ha="center",
            fontsize=9,
            weight="bold",
            color=NAVY,
        )

    y_min, y_max = min(y_all.min(), y_matched.min()), max(y_all.max(), y_matched.max())
    ax.set_ylim(y_min - 1.0, y_max + 1.5)
    plt.tight_layout()
    fig.savefig(out_dir / "01_context_length_vs_sbp_mae.png")
    plt.close(fig)


def plot_02_context_vs_dbp_mae(df_context: pd.DataFrame, out_dir: Path) -> None:
    """Figure 02: Context Length vs DBP MAE."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = df_context["context_sec"]
    y_all = df_context["dbp_mae_all"]
    y_matched = df_context.get("dbp_mae_matched", y_all)

    ax.plot(x, y_all, marker="o", markersize=8, linewidth=2.2, color=TEAL, label="All Eligible Windows")
    if "dbp_mae_matched" in df_context.columns:
        ax.plot(x, y_matched, marker="s", markersize=7, linewidth=2.0, linestyle="--", color=PURPLE, label="Matched Windows (≥5 History)")

    ax.set_title("Context Length vs. DBP Estimation Error (MAE)")
    ax.set_xlabel("Sequential Context Duration (seconds)")
    ax.set_ylabel("Validation DBP MAE (mmHg)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}s\n(Ctx-{k})" for s, k in zip(x, df_context["context_k"])])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor=LIGHT_BG, edgecolor="#cbd5e1")

    c0_mae = y_all.iloc[0]
    for i, (xi, yi) in enumerate(zip(x, y_all)):
        diff = yi - c0_mae
        sign = "+" if diff > 0 else ""
        ax.annotate(
            f"{yi:.2f}\n({sign}{diff:.2f})",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10 if i % 2 == 0 else -25),
            ha="center",
            fontsize=9,
            weight="bold",
            color=TEAL,
        )

    y_min, y_max = min(y_all.min(), y_matched.min()), max(y_all.max(), y_matched.max())
    ax.set_ylim(y_min - 0.6, y_max + 1.0)
    plt.tight_layout()
    fig.savefig(out_dir / "02_context_length_vs_dbp_mae.png")
    plt.close(fig)


def plot_03_context_vs_sbp_extreme_bias(df_extreme_sbp: pd.DataFrame, out_dir: Path) -> None:
    """
    Figure 03: MOST CRITICAL PLOT
    Context length vs SBP Extreme-Range Bias (<90 mmHg Hypotension and >=160 mmHg Hypertension).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    contexts = df_extreme_sbp["context_label"].unique()
    x = np.arange(len(contexts))
    width = 0.45

    # Subplot 1: Hypotension (<90 mmHg) - where bias is positive
    sub_hypo = df_extreme_sbp[df_extreme_sbp["range"] == "<90"]
    bars1 = ax1.bar(
        x,
        sub_hypo["bias"],
        width=width,
        color=CORAL,
        edgecolor="#9f1239",
        alpha=0.85,
    )
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_title("Hypotension (SBP < 90 mmHg)\nOverestimation Bias", pad=10)
    ax1.set_xlabel("Temporal Context")
    ax1.set_ylabel("Mean Bias (mmHg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(contexts)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

    for b in bars1:
        h = b.get_height()
        ax1.annotate(
            f"+{h:.1f}",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            weight="bold",
        )

    # Subplot 2: Hypertension (>=160 mmHg) - where bias is negative
    sub_hyper = df_extreme_sbp[df_extreme_sbp["range"] == ">=160"]
    bars2 = ax2.bar(
        x,
        sub_hyper["bias"],
        width=width,
        color=NAVY,
        edgecolor="#0f172a",
        alpha=0.85,
    )
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_title("Severe Hypertension (SBP ≥ 160 mmHg)\nUnderestimation Bias", pad=10)
    ax2.set_xlabel("Temporal Context")
    ax2.set_ylabel("Mean Bias (mmHg)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(contexts)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    for b in bars2:
        h = b.get_height()
        ax2.annotate(
            f"{h:.1f}",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            weight="bold",
        )

    fig.suptitle("Impact of Temporal Context on SBP Extreme-Range Bias", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "03_context_length_vs_sbp_extreme_bias.png")
    plt.close(fig)


def plot_04_context_vs_dbp_extreme_bias(df_extreme_dbp: pd.DataFrame, out_dir: Path) -> None:
    """Figure 04: Context length vs DBP Extreme-Range Bias (<60 mmHg Low and >=100 mmHg High)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    contexts = df_extreme_dbp["context_label"].unique()
    x = np.arange(len(contexts))
    width = 0.45

    sub_low = df_extreme_dbp[df_extreme_dbp["range"] == "<60"]
    bars1 = ax1.bar(x, sub_low["bias"], width=width, color=AMBER, edgecolor="#b45309", alpha=0.85)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_title("Low Diastolic (DBP < 60 mmHg)\nOverestimation Bias", pad=10)
    ax1.set_xlabel("Temporal Context")
    ax1.set_ylabel("Mean Bias (mmHg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(contexts)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"+{h:.1f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, weight="bold")

    sub_high = df_extreme_dbp[df_extreme_dbp["range"] == ">=100"]
    bars2 = ax2.bar(x, sub_high["bias"], width=width, color=PURPLE, edgecolor="#5b21b6", alpha=0.85)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_title("Stage 2 Diastolic (DBP ≥ 100 mmHg)\nUnderestimation Bias", pad=10)
    ax2.set_xlabel("Temporal Context")
    ax2.set_ylabel("Mean Bias (mmHg)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(contexts)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9, weight="bold")

    fig.suptitle("Impact of Temporal Context on DBP Extreme-Range Bias", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "04_context_length_vs_dbp_extreme_bias.png")
    plt.close(fig)


def plot_05_context_vs_r2(df_context: pd.DataFrame, out_dir: Path) -> None:
    """Figure 05: Context Length vs R² for SBP and DBP."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = df_context["context_sec"]

    ax.plot(x, df_context["sbp_r2"], marker="o", markersize=8, linewidth=2.2, color=NAVY, label="SBP $R^2$")
    ax.plot(x, df_context["dbp_r2"], marker="s", markersize=8, linewidth=2.2, color=TEAL, label="DBP $R^2$")

    ax.set_title("Variance Explained ($R^2$) across Temporal Context Lengths")
    ax.set_xlabel("Sequential Context Duration (seconds)")
    ax.set_ylabel("Coefficient of Determination ($R^2$)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}s\n(Ctx-{k})" for s, k in zip(x, df_context["context_k"])])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor=LIGHT_BG, edgecolor="#cbd5e1")

    for xi, y_sbp, y_dbp in zip(x, df_context["sbp_r2"], df_context["dbp_r2"]):
        ax.annotate(f"{y_sbp:.3f}", (xi, y_sbp), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, weight="bold", color=NAVY)
        ax.annotate(f"{y_dbp:.3f}", (xi, y_dbp), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8.5, weight="bold", color=TEAL)

    ax.set_ylim(min(df_context["sbp_r2"].min(), df_context["dbp_r2"].min()) - 0.05, max(df_context["sbp_r2"].max(), df_context["dbp_r2"].max()) + 0.05)
    plt.tight_layout()
    fig.savefig(out_dir / "05_context_length_vs_r2.png")
    plt.close(fig)


def plot_06_regression_to_mean(
    rtm_data: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Figure 06: Regression-to-the-Mean Comparison across contexts (Slope & Trend)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = [NAVY, TEAL, AMBER, PURPLE]
    contexts = list(rtm_data.keys())

    # SBP Error Slope
    for i, c in enumerate(contexts):
        d = rtm_data[c]["sbp"]
        bins = d["bins"]
        mean_errors = d["mean_errors"]
        slope = d["slope"]
        r = d["r"]
        ax1.plot(
            bins,
            mean_errors,
            marker="o",
            linewidth=2.0,
            color=colors[i % len(colors)],
            label=f"{c} (slope={slope:.3f}, r={r:.2f})",
        )

    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.set_title("SBP Error vs Reference Pressure\n(Regression-to-the-Mean)")
    ax1.set_xlabel("True SBP Binned (mmHg)")
    ax1.set_ylabel("Mean Error (Pred - True) (mmHg)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=8.5, facecolor=LIGHT_BG)

    # DBP Error Slope
    for i, c in enumerate(contexts):
        d = rtm_data[c]["dbp"]
        bins = d["bins"]
        mean_errors = d["mean_errors"]
        slope = d["slope"]
        r = d["r"]
        ax2.plot(
            bins,
            mean_errors,
            marker="s",
            linewidth=2.0,
            color=colors[i % len(colors)],
            label=f"{c} (slope={slope:.3f}, r={r:.2f})",
        )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.set_title("DBP Error vs Reference Pressure\n(Regression-to-the-Mean)")
    ax2.set_xlabel("True DBP Binned (mmHg)")
    ax2.set_ylabel("Mean Error (Pred - True) (mmHg)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=8.5, facecolor=LIGHT_BG)

    fig.suptitle("Regression-to-the-Mean Diagnostics Across Context Lengths", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "06_regression_to_mean_comparison.png")
    plt.close(fig)


def plot_07_record_level_mae(
    df_record_stats: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Figure 07: Per-Record MAE Comparison (Mean, Median, SD)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    x = np.arange(len(df_record_stats))
    width = 0.35

    # SBP
    bars1 = ax1.bar(x - width / 2, df_record_stats["sbp_rec_mean"], width=width, color=NAVY, label="Record Mean MAE")
    bars2 = ax1.bar(x + width / 2, df_record_stats["sbp_rec_median"], width=width, color=TEAL, label="Record Median MAE")
    ax1.set_title("SBP Record-Level Error Distribution")
    ax1.set_ylabel("MAE (mmHg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_record_stats["context_label"])
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(frameon=True, facecolor=LIGHT_BG)

    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
    for b in bars2:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)

    # DBP
    bars3 = ax2.bar(x - width / 2, df_record_stats["dbp_rec_mean"], width=width, color=NAVY, label="Record Mean MAE")
    bars4 = ax2.bar(x + width / 2, df_record_stats["dbp_rec_median"], width=width, color=TEAL, label="Record Median MAE")
    ax2.set_title("DBP Record-Level Error Distribution")
    ax2.set_ylabel("MAE (mmHg)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_record_stats["context_label"])
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(frameon=True, facecolor=LIGHT_BG)

    for b in bars3:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
    for b in bars4:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)

    fig.suptitle("Record-Level MAE Across Context Lengths", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "07_record_level_mae_comparison.png")
    plt.close(fig)


def plot_08_ppg_vpg_apg_ablation(df_deriv: pd.DataFrame, out_dir: Path) -> None:
    """Figure 08: PPG vs PPG+VPG vs PPG+VPG+APG (Experiment C & D comparison)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    x = np.arange(len(df_deriv))
    labels = df_deriv["label"]

    colors = [NAVY if "PPG" in l and "VPG" not in l else TEAL if "VPG" in l and "APG" not in l else PURPLE for l in labels]

    # SBP MAE
    bars1 = ax1.bar(x, df_deriv["sbp_mae"], color=colors, width=0.55, edgecolor="#1e293b", alpha=0.85)
    ax1.set_title("Derivative Ablation: SBP MAE")
    ax1.set_ylabel("Validation MAE (mmHg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, weight="bold")

    y_min1, y_max1 = df_deriv["sbp_mae"].min() - 1.0, df_deriv["sbp_mae"].max() + 1.0
    ax1.set_ylim(y_min1, y_max1)

    # DBP MAE
    bars2 = ax2.bar(x, df_deriv["dbp_mae"], color=colors, width=0.55, edgecolor="#1e293b", alpha=0.85)
    ax2.set_title("Derivative Ablation: DBP MAE")
    ax2.set_ylabel("Validation MAE (mmHg)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, weight="bold")

    y_min2, y_max2 = df_deriv["dbp_mae"].min() - 0.5, df_deriv["dbp_mae"].max() + 0.5
    ax2.set_ylim(y_min2, y_max2)

    fig.suptitle("Explicit Derivative Representation Comparison (C1 vs C2 vs C3 vs D-series)", fontsize=13.5, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "08_ppg_vpg_apg_ablation.png")
    plt.close(fig)


def plot_09_temporal_feature_importance(df_imp: pd.DataFrame, out_dir: Path) -> None:
    """Figure 09: Top 20 Feature Importance (Permutation or Tree split), highlighting temporal features."""
    fig, ax = plt.subplots(figsize=(9, 7))

    top20 = df_imp.head(20).iloc[::-1]  # Bottom to top
    y = np.arange(len(top20))

    # Color code static vs temporal
    is_temporal = top20["feature"].apply(
        lambda f: any(suffix in f for suffix in ["_hist_mean", "_hist_std", "_hist_min", "_hist_max", "_delta_prev", "_delta_hist_mean", "_slope"])
    )
    bar_colors = [CORAL if t else NAVY for t in is_temporal]

    bars = ax.barh(y, top20["importance"], color=bar_colors, edgecolor="#1e293b", height=0.65, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top20["feature"], fontsize=9.5)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Top 20 Features in Temporal Model (Red = Temporal Dynamic, Blue = Static)")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NAVY, label="Static Window Feature"),
        Patch(facecolor=CORAL, label="Temporal Dynamic Feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", facecolor=LIGHT_BG, edgecolor="#cbd5e1")

    plt.tight_layout()
    fig.savefig(out_dir / "09_temporal_feature_importance.png")
    plt.close(fig)


def plot_10_temporal_context_example(
    time_series: Dict[str, np.ndarray],
    features_series: Dict[str, List[float]],
    out_dir: Path,
) -> None:
    """
    Figure 10: Multi-Window Temporal Context Example.
    Shows consecutive 10s windows over a 60-second context, illustrating PPG waveforms,
    derivative profiles, and temporal evolution of HR and pulse amplitude.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)

    t_sec = time_series["time_sec"]
    ppg_raw = time_series["ppg"]
    vpg_raw = time_series["vpg"]

    # Subplot 1: Sequential PPG Windows
    ax1.plot(t_sec, ppg_raw, color=NAVY, linewidth=1.2, label="Filtered PPG")
    # Mark window boundaries
    for w_t in [10, 20, 30, 40, 50, 60]:
        ax1.axvline(w_t, color="#94a3b8", linestyle="--", linewidth=1.2)
    # Highlight current target window (50-60s)
    ax1.axvspan(50, 60, color=CORAL, alpha=0.15, label="Target Window $W_t$ (Current)")
    ax1.axvspan(0, 50, color=SLATE, alpha=0.08, label="Historical Context $W_{t-5} \dots W_{t-1}$")
    ax1.set_ylabel("PPG Amplitude (V)")
    ax1.set_title("Multi-Window Temporal Context Sequence (60 Seconds Total Duration)")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, facecolor=LIGHT_BG)

    # Subplot 2: First Derivative (VPG)
    ax2.plot(t_sec, vpg_raw, color=TEAL, linewidth=1.0, label="First Derivative (VPG)")
    for w_t in [10, 20, 30, 40, 50, 60]:
        ax2.axvline(w_t, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax2.axvspan(50, 60, color=CORAL, alpha=0.15)
    ax2.axvspan(0, 50, color=SLATE, alpha=0.08)
    ax2.set_ylabel("VPG (V/s)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5, facecolor=LIGHT_BG)

    # Subplot 3: Discrete Extracted Feature Trends across the 6 windows
    win_centers = np.array([5, 15, 25, 35, 45, 55])
    hr_vals = features_series["hr_bpm"]
    amp_vals = features_series["pulse_amp_median_a"]

    ax3_twin = ax3.twinx()
    l1 = ax3.plot(win_centers, hr_vals, marker="o", color=AMBER, linewidth=2.0, label="Window HR (bpm)")
    l2 = ax3_twin.plot(win_centers, amp_vals, marker="s", color=PURPLE, linewidth=2.0, label="Pulse Amplitude")

    for w_t in [10, 20, 30, 40, 50, 60]:
        ax3.axvline(w_t, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax3.axvspan(50, 60, color=CORAL, alpha=0.15)
    ax3.axvspan(0, 50, color=SLATE, alpha=0.08)

    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Heart Rate (bpm)", color=AMBER)
    ax3_twin.set_ylabel("Pulse Amplitude", color=PURPLE)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right", fontsize=8.5, facecolor=LIGHT_BG)

    plt.tight_layout()
    fig.savefig(out_dir / "10_temporal_context_example.png")
    plt.close(fig)
