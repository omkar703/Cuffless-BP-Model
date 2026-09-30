"""
Phase 6C Research Visualization Module.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Generates research-grade diagnostic plots conforming strictly to publication standards:
1. Raw IR optical waveform timeline with reference BP event overlay
2. Causal 3-channel (PPG, VPG, APG) waveform traces
3. Prediction timeline with 95% conformal uncertainty bounds vs reference measurements
4. SBP & DBP Scatter plots with identity line and correlation metrics
5. SBP & DBP Error distribution histograms
6. SBP & DBP Bland-Altman agreement plots with Limits of Agreement (LoA)
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from phase6c_app.config import DEMO_WATERMARK_TEXT


def _add_watermark_if_demo(ax: plt.Axes, is_demo: bool):
    """Overlay a clear watermark if plotting synthetic demo data."""
    if is_demo:
        ax.text(
            0.5, 0.5, DEMO_WATERMARK_TEXT,
            transform=ax.transAxes,
            fontsize=14, color="red", alpha=0.35,
            ha="center", va="center", rotation=25,
            weight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.15, ec="red")
        )


def plot_raw_ir_timeline(
    df_ppg: pd.DataFrame,
    df_bp_events: Optional[pd.DataFrame] = None,
    is_demo: bool = False
) -> plt.Figure:
    """Plot full raw IR waveform with reference BP event timestamps overlaid."""
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)

    t_sec = df_ppg["elapsed_s"].to_numpy(dtype=float)
    ir_vals = df_ppg["ir"].to_numpy(dtype=float)

    # Downsample for smooth plotting if very dense
    stride = max(1, len(t_sec) // 4000)
    ax.plot(t_sec[::stride], ir_vals[::stride], color="#1f77b4", lw=1.0, label="Raw MAX30102 IR")

    # Overlay BP events
    if df_bp_events is not None and len(df_bp_events) > 0:
        for _, evt in df_bp_events.iterrows():
            evt_t = float(evt["elapsed_s"]) if pd.notna(evt["elapsed_s"]) else np.nan
            if np.isnan(evt_t):
                continue
            etype = str(evt["event_type"]).upper()
            if "START" in etype:
                ax.axvline(x=evt_t, color="#ff7f0e", linestyle="--", lw=1.5, alpha=0.85, label="Cuff Inflation Start")
            elif "RESULT" in etype or pd.notna(evt["sbp"]):
                sbp_txt = f"{evt['sbp']:.0f}" if pd.notna(evt["sbp"]) else "?"
                dbp_txt = f"{evt['dbp']:.0f}" if pd.notna(evt["dbp"]) else "?"
                ax.axvline(x=evt_t, color="#d62728", linestyle="-", lw=2.0, alpha=0.9, label=f"Ref BP Result ({sbp_txt}/{dbp_txt})")
                ax.text(evt_t, ax.get_ylim()[1] * 0.92, f" Ref BP:\n {sbp_txt}/{dbp_txt}", color="#d62728", fontsize=8, weight="bold")

    # Clean up duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9, fontsize=8)

    ax.set_title("MAX30102 Raw Optical PPG & Reference Cuff Timeline", fontsize=11, fontweight="bold")
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=9)
    ax.set_ylabel("Raw ADC Amplitude (counts)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    _add_watermark_if_demo(ax, is_demo)
    plt.tight_layout()
    return fig


def plot_derivatives_timeline(traces: Dict[str, np.ndarray], is_demo: bool = False) -> plt.Figure:
    """Plot causal 3-channel waveforms: Filtered PPG, VPG (1st deriv), APG (2nd deriv)."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True, dpi=150)

    t_sec = traces.get("time_s", np.array([]))
    ppg = traces.get("ppg", np.array([]))
    vpg = traces.get("vpg", np.array([]))
    apg = traces.get("apg", np.array([]))

    stride = max(1, len(t_sec) // 3000)

    if len(t_sec) > 0:
        ax1.plot(t_sec[::stride], ppg[::stride], color="#2ca02c", lw=0.9)
        ax2.plot(t_sec[::stride], vpg[::stride], color="#ff7f0e", lw=0.8)
        ax3.plot(t_sec[::stride], apg[::stride], color="#9467bd", lw=0.8)

    ax1.set_title("Causal 3-Channel Signal Pipeline (125 Hz Resampled & Normalized)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("PPG (Z-Score)", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)
    _add_watermark_if_demo(ax1, is_demo)

    ax2.set_ylabel("VPG (Z-Score)", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3.set_ylabel("APG (Z-Score)", fontsize=8)
    ax3.set_xlabel("Elapsed Time (seconds)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig


def plot_prediction_vs_reference_timeline(
    df_predictions: pd.DataFrame,
    df_pairs: Optional[pd.DataFrame] = None,
    is_demo: bool = False
) -> plt.Figure:
    """Plot continuous model predictions with 95% conformal bounds alongside reference cuff markers."""
    fig, (ax_sbp, ax_dbp) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, dpi=150)

    if df_predictions is not None and len(df_predictions) > 0:
        t_pred = df_predictions["prediction_elapsed_s"].to_numpy(dtype=float)
        sbp_cal = df_predictions["calibrated_sbp"].to_numpy(dtype=float)
        sbp_lo = df_predictions["conformal_lower_sbp"].to_numpy(dtype=float)
        sbp_hi = df_predictions["conformal_upper_sbp"].to_numpy(dtype=float)

        dbp_cal = df_predictions["calibrated_dbp"].to_numpy(dtype=float)
        dbp_lo = df_predictions["conformal_lower_dbp"].to_numpy(dtype=float)
        dbp_hi = df_predictions["conformal_upper_dbp"].to_numpy(dtype=float)

        # SBP Prediction curve & conformal ribbon
        ax_sbp.plot(t_pred, sbp_cal, "o-", color="#1f77b4", lw=1.8, markersize=4, label="Predicted SBP (Calibrated)")
        ax_sbp.fill_between(t_pred, sbp_lo, sbp_hi, color="#1f77b4", alpha=0.2, label="95% Conformal Interval")

        # DBP Prediction curve & conformal ribbon
        ax_dbp.plot(t_pred, dbp_cal, "s-", color="#2ca02c", lw=1.8, markersize=4, label="Predicted DBP (Calibrated)")
        ax_dbp.fill_between(t_pred, dbp_lo, dbp_hi, color="#2ca02c", alpha=0.2, label="95% Conformal Interval")

    # Reference BP markers
    if df_pairs is not None and len(df_pairs) > 0:
        valid_pairs = df_pairs[df_pairs["included_in_metrics"] == True]
        if len(valid_pairs) > 0:
            t_ref = valid_pairs["prediction_timestamp"].to_numpy(dtype=float)
            # Map back to elapsed seconds if prediction_timestamp is ms
            t0 = df_predictions["prediction_timestamp_ms"].iloc[0] if len(df_predictions) > 0 else 0.0
            t_ref_s = (t_ref - t0) / 1000.0 + df_predictions["prediction_elapsed_s"].iloc[0] if len(df_predictions) > 0 else np.zeros_like(t_ref)

            ref_sbp = valid_pairs["reference_sbp"].to_numpy(dtype=float)
            ref_dbp = valid_pairs["reference_dbp"].to_numpy(dtype=float)

            ax_sbp.scatter(t_ref_s, ref_sbp, color="#d62728", s=65, zorder=5, marker="^", label="Reference Cuff SBP")
            ax_dbp.scatter(t_ref_s, ref_dbp, color="#d62728", s=65, zorder=5, marker="^", label="Reference Cuff DBP")

    ax_sbp.set_title("Temporal BP Predictions with Conformal Bounds vs Reference Cuff", fontsize=11, fontweight="bold")
    ax_sbp.set_ylabel("Systolic BP (mmHg)", fontsize=9)
    ax_sbp.grid(True, linestyle=":", alpha=0.6)
    ax_sbp.legend(loc="upper right", fontsize=8)
    _add_watermark_if_demo(ax_sbp, is_demo)

    ax_dbp.set_ylabel("Diastolic BP (mmHg)", fontsize=9)
    ax_dbp.set_xlabel("Elapsed Time (seconds)", fontsize=9)
    ax_dbp.grid(True, linestyle=":", alpha=0.6)
    ax_dbp.legend(loc="upper right", fontsize=8)
    _add_watermark_if_demo(ax_dbp, is_demo)

    plt.tight_layout()
    return fig


def plot_scatter_sbp_dbp(df_pairs: pd.DataFrame, is_demo: bool = False) -> plt.Figure:
    """Plot SBP and DBP predicted vs reference scatter plots with line of identity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=150)

    valid_pairs = df_pairs[df_pairs["included_in_metrics"] == True] if len(df_pairs) > 0 else pd.DataFrame()
    n = len(valid_pairs)

    # --- SBP Scatter ---
    if n > 0:
        y_ref_sbp = valid_pairs["reference_sbp"].to_numpy(dtype=float)
        y_pred_sbp = valid_pairs["predicted_sbp_calibrated"].to_numpy(dtype=float)
        ax1.scatter(y_ref_sbp, y_pred_sbp, color="#1f77b4", edgecolor="black", s=50, alpha=0.85, label="Matched Pairs")

        all_vals = np.concatenate([y_ref_sbp, y_pred_sbp])
        lo = np.floor(np.min(all_vals) / 10.0) * 10.0 - 5.0
        hi = np.ceil(np.max(all_vals) / 10.0) * 10.0 + 5.0
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Identity (y = x)")
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)

        mae = np.mean(np.abs(y_pred_sbp - y_ref_sbp))
        ax1.text(0.05, 0.90, f"N = {n}\nMAE = {mae:.2f} mmHg", transform=ax1.transAxes,
                 fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax1.text(0.5, 0.5, "No Valid Matched SBP Pairs", ha="center", va="center", transform=ax1.transAxes)

    ax1.set_title("Systolic Blood Pressure (SBP)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Reference Cuff SBP (mmHg)", fontsize=9)
    ax1.set_ylabel("Predicted Calibrated SBP (mmHg)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8)
    _add_watermark_if_demo(ax1, is_demo)

    # --- DBP Scatter ---
    if n > 0:
        y_ref_dbp = valid_pairs["reference_dbp"].to_numpy(dtype=float)
        y_pred_dbp = valid_pairs["predicted_dbp_calibrated"].to_numpy(dtype=float)
        ax2.scatter(y_ref_dbp, y_pred_dbp, color="#2ca02c", edgecolor="black", s=50, alpha=0.85, label="Matched Pairs")

        all_vals_dbp = np.concatenate([y_ref_dbp, y_pred_dbp])
        lo_d = np.floor(np.min(all_vals_dbp) / 10.0) * 10.0 - 5.0
        hi_d = np.ceil(np.max(all_vals_dbp) / 10.0) * 10.0 + 5.0
        ax2.plot([lo_d, hi_d], [lo_d, hi_d], "k--", lw=1.5, label="Identity (y = x)")
        ax2.set_xlim(lo_d, hi_d)
        ax2.set_ylim(lo_d, hi_d)

        mae_d = np.mean(np.abs(y_pred_dbp - y_ref_dbp))
        ax2.text(0.05, 0.90, f"N = {n}\nMAE = {mae_d:.2f} mmHg", transform=ax2.transAxes,
                 fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "No Valid Matched DBP Pairs", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("Diastolic Blood Pressure (DBP)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Reference Cuff DBP (mmHg)", fontsize=9)
    ax2.set_ylabel("Predicted Calibrated DBP (mmHg)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8)
    _add_watermark_if_demo(ax2, is_demo)

    plt.tight_layout()
    return fig


def plot_error_distributions(df_pairs: pd.DataFrame, is_demo: bool = False) -> plt.Figure:
    """Plot error histograms (Predicted - Reference) for SBP and DBP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    valid_pairs = df_pairs[df_pairs["included_in_metrics"] == True] if len(df_pairs) > 0 else pd.DataFrame()
    n = len(valid_pairs)

    if n > 0:
        err_sbp = valid_pairs["sbp_error_calibrated"].to_numpy(dtype=float)
        bias_s = float(np.mean(err_sbp))
        std_s = float(np.std(err_sbp)) if n > 1 else 0.0

        ax1.hist(err_sbp, bins=max(5, n // 2), color="#1f77b4", edgecolor="black", alpha=0.7)
        ax1.axvline(bias_s, color="red", linestyle="--", lw=1.8, label=f"Bias: {bias_s:+.2f} mmHg")
        ax1.axvline(0.0, color="black", linestyle=":", lw=1.2)
        ax1.set_title(f"SBP Error Distribution (N={n})", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Prediction Error: Pred - Ref (mmHg)", fontsize=9)
        ax1.set_ylabel("Frequency", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No Valid SBP Pairs", ha="center", va="center", transform=ax1.transAxes)

    if n > 0:
        err_dbp = valid_pairs["dbp_error_calibrated"].to_numpy(dtype=float)
        bias_d = float(np.mean(err_dbp))
        std_d = float(np.std(err_dbp)) if n > 1 else 0.0

        ax2.hist(err_dbp, bins=max(5, n // 2), color="#2ca02c", edgecolor="black", alpha=0.7)
        ax2.axvline(bias_d, color="red", linestyle="--", lw=1.8, label=f"Bias: {bias_d:+.2f} mmHg")
        ax2.axvline(0.0, color="black", linestyle=":", lw=1.2)
        ax2.set_title(f"DBP Error Distribution (N={n})", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Prediction Error: Pred - Ref (mmHg)", fontsize=9)
        ax2.set_ylabel("Frequency", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No Valid DBP Pairs", ha="center", va="center", transform=ax2.transAxes)

    _add_watermark_if_demo(ax1, is_demo)
    _add_watermark_if_demo(ax2, is_demo)
    plt.tight_layout()
    return fig


def plot_bland_altman(df_pairs: pd.DataFrame, is_demo: bool = False) -> plt.Figure:
    """Generate Bland-Altman agreement plots with Mean Bias and 95% Limits of Agreement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    valid_pairs = df_pairs[df_pairs["included_in_metrics"] == True] if len(df_pairs) > 0 else pd.DataFrame()
    n = len(valid_pairs)

    def _render_ba(ax: plt.Axes, ref: np.ndarray, pred: np.ndarray, title: str, color: str):
        mean_bp = (ref + pred) / 2.0
        diff = pred - ref
        bias = float(np.mean(diff))
        std = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        upper_loa = bias + 1.96 * std
        lower_loa = bias - 1.96 * std

        ax.scatter(mean_bp, diff, color=color, edgecolor="black", s=50, alpha=0.85)
        ax.axhline(bias, color="red", linestyle="-", lw=1.8, label=f"Mean Bias: {bias:+.2f}")
        ax.axhline(upper_loa, color="navy", linestyle="--", lw=1.5, label=f"+1.96 SD: {upper_loa:+.2f}")
        ax.axhline(lower_loa, color="navy", linestyle="--", lw=1.5, label=f"-1.96 SD: {lower_loa:+.2f}")
        ax.axhline(0.0, color="gray", linestyle=":", lw=1.0)

        ax.set_title(f"{title} (N={len(diff)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean of Reference and Predicted (mmHg)", fontsize=9)
        ax.set_ylabel("Difference: Pred - Ref (mmHg)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    if n > 0:
        y_ref_sbp = valid_pairs["reference_sbp"].to_numpy(dtype=float)
        y_pred_sbp = valid_pairs["predicted_sbp_calibrated"].to_numpy(dtype=float)
        _render_ba(ax1, y_ref_sbp, y_pred_sbp, "Bland-Altman SBP", "#1f77b4")

        y_ref_dbp = valid_pairs["reference_dbp"].to_numpy(dtype=float)
        y_pred_dbp = valid_pairs["predicted_dbp_calibrated"].to_numpy(dtype=float)
        _render_ba(ax2, y_ref_dbp, y_pred_dbp, "Bland-Altman DBP", "#2ca02c")
    else:
        ax1.text(0.5, 0.5, "No Valid SBP Pairs", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No Valid DBP Pairs", ha="center", va="center", transform=ax2.transAxes)

    _add_watermark_if_demo(ax1, is_demo)
    _add_watermark_if_demo(ax2, is_demo)
    plt.tight_layout()
    return fig


def save_all_figures(
    fig_dir: Path,
    df_ppg: pd.DataFrame,
    df_bp_events: pd.DataFrame,
    df_predictions: pd.DataFrame,
    df_pairs: pd.DataFrame,
    traces: Dict[str, np.ndarray],
    is_demo: bool = False
) -> Dict[str, Path]:
    """Generate and save all diagnostic figures to disk."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # 1. Raw IR Timeline
    f1 = plot_raw_ir_timeline(df_ppg, df_bp_events, is_demo=is_demo)
    p1 = fig_dir / "01_raw_ir_bp_timeline.png"
    f1.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(f1)
    saved_paths["raw_ir_timeline"] = p1

    # 2. Derivatives Timeline
    f2 = plot_derivatives_timeline(traces, is_demo=is_demo)
    p2 = fig_dir / "02_causal_derivatives.png"
    f2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(f2)
    saved_paths["derivatives_timeline"] = p2

    # 3. Prediction vs Reference Timeline
    f3 = plot_prediction_vs_reference_timeline(df_predictions, df_pairs, is_demo=is_demo)
    p3 = fig_dir / "03_prediction_conformal_timeline.png"
    f3.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(f3)
    saved_paths["prediction_timeline"] = p3

    # 4. Scatter SBP & DBP
    f4 = plot_scatter_sbp_dbp(df_pairs, is_demo=is_demo)
    p4 = fig_dir / "04_scatter_predicted_vs_reference.png"
    f4.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(f4)
    saved_paths["scatter_plots"] = p4

    # 5. Error Distributions
    f5 = plot_error_distributions(df_pairs, is_demo=is_demo)
    p5 = fig_dir / "05_error_distributions.png"
    f5.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close(f5)
    saved_paths["error_distributions"] = p5

    # 6. Bland-Altman Plots
    f6 = plot_bland_altman(df_pairs, is_demo=is_demo)
    p6 = fig_dir / "06_bland_altman_agreement.png"
    f6.savefig(p6, dpi=200, bbox_inches="tight")
    plt.close(f6)
    saved_paths["bland_altman"] = p6

    return saved_paths
