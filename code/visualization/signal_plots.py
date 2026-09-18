"""
Signal Waveform Visualization Module for PPG and ABP signals.

Generates high-resolution publication-quality plots:
- Plot A: Raw PPG waveform
- Plot B: OFFLINE RESEARCH FILTER PPG waveform (0.5-8 Hz)
- Plot C: Raw vs Filtered PPG overlay
- Plot D: ABP reference waveform with detected systolic peaks and diastolic troughs
- Plot E: Aligned PPG and ABP waveforms on synchronized time axis
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from config.config import FIGURES_DIR, SAMPLING_RATE
from data.preprocessing import (
    apply_offline_research_ppg_filter,
    extract_physiological_bp_targets,
)


def plot_single_record_inspection(
    record: Dict[str, Any],
    window_sec: float = 10.0,
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Creates a comprehensive 4-panel diagnostic plot for a single record:
    1. Raw PPG vs Filtered PPG (0.5-8 Hz zero-phase)
    2. Reference ABP with detected systolic peaks and diastolic troughs
    3. Synchronized PPG & ABP overlay demonstrating pulse morphology
    4. PPG first derivative (Velocity Plethysmogram - VPG) demonstrating ejection slope
    """
    out_dir = save_dir or FIGURES_DIR
    rec_id = record["record_id"]
    fs = record.get("sampling_frequency", SAMPLING_RATE)

    ppg_raw = record["ppg"]
    abp_raw = record["abp"]

    # Limit to specified window length for visual clarity
    num_samples = int(min(len(ppg_raw), window_sec * fs))
    t = np.arange(num_samples) / fs

    p_raw_slice = ppg_raw[:num_samples]
    a_raw_slice = abp_raw[:num_samples]

    # Apply offline research filter
    p_filt_slice = apply_offline_research_ppg_filter(p_raw_slice, fs=fs)

    # Extract beat diagnostics on ABP slice
    bp_diag = extract_physiological_bp_targets(a_raw_slice, fs=fs)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, dpi=150)
    fig.patch.set_facecolor("white")

    # Panel 1: Raw vs Filtered PPG
    axes[0].plot(t, p_raw_slice, color="#95a5a6", alpha=0.7, label="Raw PPG (Unfiltered)", lw=1.2)
    axes[0].plot(t, p_filt_slice, color="#2980b9", label="PPG (OFFLINE RESEARCH FILTER: 0.5–8 Hz)", lw=1.6)
    axes[0].set_ylabel("PPG Amplitude", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Record: {rec_id} — PPG Filtering & Waveform Inspection ({window_sec:.0f}s Window)", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", framealpha=0.9)

    # Panel 2: Reference ABP with detected cardiac peaks and troughs
    axes[1].plot(t, a_raw_slice, color="#c0392b", label="Reference Arterial Blood Pressure (ABP)", lw=1.5)
    if bp_diag["is_valid"]:
        p_idx = bp_diag["peak_indices"]
        t_idx = bp_diag["trough_indices"]
        valid_p = p_idx[p_idx < num_samples]
        valid_t = t_idx[t_idx < num_samples]
        axes[1].scatter(valid_p / fs, a_raw_slice[valid_p], color="#e74c3c", marker="o", s=45, zorder=5, label="Systolic Peak (SBP)")
        axes[1].scatter(valid_t / fs, a_raw_slice[valid_t], color="#27ae60", marker="v", s=45, zorder=5, label="Diastolic Trough (DBP)")
    axes[1].axhline(120, color="gray", linestyle=":", alpha=0.5, label="120/80 mmHg Normotensive Guide")
    axes[1].axhline(80, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("Pressure (mmHg)", fontsize=11, fontweight="bold")
    axes[1].set_title(f"ABP Ground Truth: Mean SBP={bp_diag.get('sbp_mean', np.nan):.1f} mmHg, Mean DBP={bp_diag.get('dbp_mean', np.nan):.1f} mmHg", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", framealpha=0.9)

    # Panel 3: Synchronized Normalized PPG and ABP (Dual-Axis)
    ax3_ppg = axes[2]
    ax3_abp = ax3_ppg.twinx()

    line1 = ax3_ppg.plot(t, p_filt_slice, color="#2980b9", lw=1.5, label="Filtered PPG")
    line2 = ax3_abp.plot(t, a_raw_slice, color="#c0392b", lw=1.4, alpha=0.85, label="Reference ABP")

    ax3_ppg.set_ylabel("PPG (V)", color="#2980b9", fontsize=11, fontweight="bold")
    ax3_abp.set_ylabel("ABP (mmHg)", color="#c0392b", fontsize=11, fontweight="bold")
    ax3_ppg.tick_params(axis="y", labelcolor="#2980b9")
    ax3_abp.tick_params(axis="y", labelcolor="#c0392b")
    ax3_ppg.set_title("Synchronized PPG vs ABP Waveforms (Cardiovascular Pulse Dynamics)", fontsize=11)
    ax3_ppg.grid(True, linestyle="--", alpha=0.5)

    # Combined legend for twinx
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3_ppg.legend(lines, labels, loc="upper right", framealpha=0.9)

    # Panel 4: Velocity Plethysmogram (VPG - 1st derivative)
    vpg = np.gradient(p_filt_slice, t)
    axes[3].plot(t, vpg, color="#8e44ad", lw=1.3, label="VPG: d(PPG)/dt (Systolic Ejection Velocity)")
    axes[3].axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.6)
    axes[3].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    axes[3].set_ylabel("d(PPG)/dt", fontsize=11, fontweight="bold")
    axes[3].set_title("First Derivative (VPG) — Characteristic Slope & Inflexion Features", fontsize=11)
    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out_path = out_dir / f"signal_inspection_{rec_id}_{int(window_sec)}s.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def generate_representative_signal_plots(
    records: list,
    save_dir: Optional[Path] = None,
    window_sec: float = 10.0,
) -> list:
    """Generates inspection plots for multiple representative records."""
    out_dir = save_dir or FIGURES_DIR
    generated_files = []
    for rec in records:
        try:
            path = plot_single_record_inspection(rec, window_sec=window_sec, save_dir=out_dir)
            generated_files.append(path)
        except Exception as e:
            print(f"Failed to plot record {rec.get('record_id')}: {e}")
    return generated_files
