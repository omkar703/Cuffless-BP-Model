"""
Dataset-Level Distribution Visualizations.

Generates 7 standard scientific distribution plots:
1. Plot 1: Record duration distribution (seconds)
2. Plot 2: PPG amplitude distribution (peak-to-peak V)
3. Plot 3: ABP continuous distribution (mmHg)
4. Plot 4: SBP distribution histogram & KDE (mmHg)
5. Plot 5: DBP distribution histogram & KDE (mmHg)
6. Plot 6: Decoupled record validity breakdown (Paired, PPG-only, ABP-only, Rejected)
7. Plot 7: Rejection reasons breakdown (PPG vs ABP)
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import FIGURES_DIR

# Set scientific plotting style
sns.set_theme(style="whitegrid", font="sans-serif")


def plot_record_duration_distribution(df_manifest: pd.DataFrame, save_dir: Optional[Path] = None) -> Path:
    """Plot 1: Distribution of record durations in seconds."""
    out_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    durations = df_manifest["duration_seconds"].values
    sns.histplot(durations, bins=40, kde=True, color="#2c3e50", ax=ax, edgecolor="black", alpha=0.7)

    median_dur = np.median(durations)
    ax.axvline(median_dur, color="#e74c3c", linestyle="--", lw=1.8, label=f"Median Duration: {median_dur:.1f}s (~{median_dur/60:.1f} min)")

    ax.set_title("Dataset Diagnostic: Record Duration Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Signal Duration (seconds)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Records", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    p = out_dir / "01_record_duration_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_ppg_amplitude_distribution(df_manifest: pd.DataFrame, save_dir: Optional[Path] = None) -> Path:
    """Plot 2: Distribution of PPG peak-to-peak amplitudes (ptp)."""
    out_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    df_valid_ppg = df_manifest[df_manifest["ppg_valid"]].dropna(subset=["ppg_ptp"])
    ptps = df_valid_ppg["ppg_ptp"].values

    sns.histplot(ptps, bins=50, kde=True, color="#2980b9", ax=ax, edgecolor="black", alpha=0.7)

    median_ptp = np.median(ptps) if len(ptps) > 0 else 0
    p95_ptp = np.percentile(ptps, 95) if len(ptps) > 0 else 0

    ax.axvline(median_ptp, color="#e67e22", linestyle="--", lw=1.8, label=f"Median PTP: {median_ptp:.3f} V")
    ax.axvline(p95_ptp, color="#8e44ad", linestyle=":", lw=1.5, label=f"95th Percentile: {p95_ptp:.3f} V")

    ax.set_title("Dataset Diagnostic: PPG Peak-to-Peak Amplitude Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("PPG Peak-to-Peak Amplitude (V / uncalibrated units)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Record Count", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    p = out_dir / "02_ppg_amplitude_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_abp_continuous_distribution(df_manifest: pd.DataFrame, save_dir: Optional[Path] = None) -> Path:
    """Plot 3: Distribution of mean and extrema ABP levels across records."""
    out_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    df_valid_abp = df_manifest[df_manifest["abp_valid"]].dropna(subset=["abp_mean", "abp_min", "abp_max"])

    sns.kdeplot(df_valid_abp["abp_mean"], color="#d35400", fill=True, alpha=0.4, label="Mean ABP (MAP Integral)", ax=ax, lw=2)
    sns.kdeplot(df_valid_abp["abp_p05"], color="#27ae60", fill=True, alpha=0.2, label="DBP Proxy (5th Percentile)", ax=ax, lw=1.5)
    sns.kdeplot(df_valid_abp["abp_p95"], color="#c0392b", fill=True, alpha=0.2, label="SBP Proxy (95th Percentile)", ax=ax, lw=1.5)

    ax.set_title("Dataset Diagnostic: Arterial Blood Pressure Continuous Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Pressure (mmHg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    p = out_dir / "03_abp_continuous_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_sbp_distribution(df_manifest: pd.DataFrame, save_dir: Optional[Path] = None) -> Path:
    """Plot 4: Distribution of beat-by-beat SBP means."""
    out_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    sbps = df_manifest[df_manifest["abp_valid"]]["sbp_mean"].dropna().values

    if len(sbps) > 0:
        sns.histplot(sbps, bins=40, kde=True, color="#e74c3c", ax=ax, edgecolor="black", alpha=0.7)
        mean_sbp = np.mean(sbps)
        std_sbp = np.std(sbps)
        ax.axvline(mean_sbp, color="#2c3e50", linestyle="--", lw=2, label=f"Mean SBP: {mean_sbp:.1f} ± {std_sbp:.1f} mmHg")
        ax.axvline(120, color="#27ae60", linestyle=":", lw=1.5, label="Normal SBP Threshold (120 mmHg)")
        ax.axvline(140, color="#d35400", linestyle=":", lw=1.5, label="Stage 1 Hypertension (140 mmHg)")

    ax.set_title("Dataset Diagnostic: Systolic Blood Pressure (SBP) Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Systolic Blood Pressure (mmHg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Record Count", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    p = out_dir / "04_sbp_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_dbp_distribution(df_manifest: pd.DataFrame, save_dir: Optional[Path] = None) -> Path:
    """Plot 5: Distribution of beat-by-beat DBP means."""
    out_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    dbps = df_manifest[df_manifest["abp_valid"]]["dbp_mean"].dropna().values

    if len(dbps) > 0:
        sns.histplot(dbps, bins=40, kde=True, color="#16a085", ax=ax, edgecolor="black", alpha=0.7)
        mean_dbp = np.mean(dbps)
        std_dbp = np.std(dbps)
        ax.axvline(mean_dbp, color="#2c3e50", linestyle="--", lw=2, label=f"Mean DBP: {mean_dbp:.1f} ± {std_dbp:.1f} mmHg")
        ax.axvline(80, color="#2980b9", linestyle=":", lw=1.5, label="Normal DBP Threshold (80 mmHg)")
        ax.axvline(90, color="#c0392b", linestyle=":", lw=1.5, label="Stage 1 Hypertension (90 mmHg)")

    ax.set_title("Dataset Diagnostic: Diastolic Blood Pressure (DBP) Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Diastolic Blood Pressure (mmHg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Record Count", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    p = out_dir / "05_dbp_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_quality_status_breakdown(df_manifest: pd.DataFrame, save_dir: Optional[Path] = None) -> Path:
    """Plot 6: Decoupled record validity categories."""
    out_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    counts = df_manifest["quality_status"].value_counts()
    colors = {
        "valid_paired": "#27ae60",
        "valid_ppg_only": "#2980b9",
        "valid_abp_only": "#f39c12",
        "rejected_both": "#c0392b",
    }
    bar_colors = [colors.get(k, "#95a5a6") for k in counts.index]

    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="black", alpha=0.85)

    # Add count labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}\n({h/len(df_manifest)*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Decoupled Quality Control Breakdown (Preserving Usable PPG)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Record Quality Status Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Records", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.18)

    fig.tight_layout()
    p = out_dir / "06_quality_status_breakdown.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_rejection_reasons_breakdown(df_manifest: pd.DataFrame, save_dir: Optional[Path] = None) -> Path:
    """Plot 7: Diagnostic breakdown of rejection reasons for PPG and ABP."""
    out_dir = save_dir or FIGURES_DIR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # PPG Rejections
    df_ppg_bad = df_manifest[~df_manifest["ppg_valid"]]
    ppg_reasons = df_ppg_bad["ppg_rejection_reason"].value_counts().head(7)
    if len(ppg_reasons) > 0:
        ax1.barh(ppg_reasons.index, ppg_reasons.values, color="#e74c3c", edgecolor="black", alpha=0.8)
        ax1.set_title("PPG Signal Rejection Reasons", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Count", fontsize=10, fontweight="bold")
        for i, v in enumerate(ppg_reasons.values):
            ax1.text(v + 1, i, f" {v}", va="center", fontweight="bold", fontsize=9)
    else:
        ax1.text(0.5, 0.5, "Zero PPG Rejections", ha="center", va="center")

    # ABP Rejections
    df_abp_bad = df_manifest[~df_manifest["abp_valid"]]
    abp_reasons = df_abp_bad["abp_rejection_reason"].value_counts().head(7)
    if len(abp_reasons) > 0:
        ax2.barh(abp_reasons.index, abp_reasons.values, color="#e67e22", edgecolor="black", alpha=0.8)
        ax2.set_title("ABP Reference Rejection Reasons", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Count", fontsize=10, fontweight="bold")
        for i, v in enumerate(abp_reasons.values):
            ax2.text(v + 1, i, f" {v}", va="center", fontweight="bold", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Zero ABP Rejections", ha="center", va="center")

    fig.suptitle("Diagnostic Analysis of Signal Anomaly Causes", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "07_rejection_reasons_breakdown.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def generate_all_dataset_distribution_plots(df_manifest: pd.DataFrame, save_dir: Optional[Path] = None) -> List[Path]:
    """Generates all 7 dataset distribution plots."""
    out_dir = save_dir or FIGURES_DIR
    plots = [
        plot_record_duration_distribution(df_manifest, out_dir),
        plot_ppg_amplitude_distribution(df_manifest, out_dir),
        plot_abp_continuous_distribution(df_manifest, out_dir),
        plot_sbp_distribution(df_manifest, out_dir),
        plot_dbp_distribution(df_manifest, out_dir),
        plot_quality_status_breakdown(df_manifest, out_dir),
        plot_rejection_reasons_breakdown(df_manifest, out_dir),
    ]
    return plots
