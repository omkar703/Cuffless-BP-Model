#!/usr/bin/env python3
"""
MAX30102 hardware-capture inspection
------------------------------------
Use on an EXISTING CSV. This script does not modify the input CSV.

Usage:
    python inspect_hardware_capture.py final_dataset_ready.csv

Expected columns:
    sample_index, timestamp_ms, expected_timestamp_ms, host_timestamp_ms, ir, red

The script is intentionally tolerant of older CSVs:
- If host_timestamp_ms is missing but timestamp_ms exists, timestamp_ms is used.
- expected_timestamp_ms is NOT trusted for validation.
- sample_index is checked using its observed step (e.g. 10), not assuming step=1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def robust_mode(values: np.ndarray) -> float:
    """Return the most common rounded value; fallback to median."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    rounded = np.round(values, 6)
    unique, counts = np.unique(rounded, return_counts=True)
    return float(unique[np.argmax(counts)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Input hardware CSV")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <csv folder>/hardware_inspection)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else csv_path.parent / "hardware_inspection"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    n = len(df)

    required = ["sample_index", "ir", "red"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Prefer host_timestamp_ms for the actual observed acquisition timing.
    if "host_timestamp_ms" in df.columns:
        host_col = "host_timestamp_ms"
    elif "timestamp_ms" in df.columns:
        host_col = "timestamp_ms"
    else:
        host_col = None

    sample_index = pd.to_numeric(df["sample_index"], errors="coerce").to_numpy(float)
    ir = pd.to_numeric(df["ir"], errors="coerce").to_numpy(float)
    red = pd.to_numeric(df["red"], errors="coerce").to_numpy(float)

    if host_col:
        host_ts = pd.to_numeric(df[host_col], errors="coerce").to_numpy(float)
    else:
        host_ts = None

    # -------------------------
    # Sample-index integrity
    # -------------------------
    sdiff = np.diff(sample_index)
    finite_sdiff = sdiff[np.isfinite(sdiff)]
    median_sstep = float(np.median(finite_sdiff)) if finite_sdiff.size else float("nan")
    mode_sstep = robust_mode(finite_sdiff)

    # Treat the observed modal step as the expected step.
    expected_step = mode_sstep if np.isfinite(mode_sstep) and mode_sstep > 0 else median_sstep
    step_bad = (
        np.isfinite(sdiff)
        & np.isfinite(expected_step)
        & (sdiff != expected_step)
    )

    sample_gap_count = int(np.sum(step_bad))
    sample_duplicate_or_reverse = int(np.sum(np.isfinite(sdiff) & (sdiff <= 0)))

    # If sample_index is in milliseconds, 10 units = 10 ms.
    # We do not silently call it 100 Hz unless the observed step is ~10 and
    # host timing agrees.
    inferred_sample_interval_ms = float(expected_step) if np.isfinite(expected_step) else float("nan")
    inferred_sample_rate_hz = (
        1000.0 / inferred_sample_interval_ms
        if np.isfinite(inferred_sample_interval_ms) and inferred_sample_interval_ms > 0
        else float("nan")
    )

    # -------------------------
    # Host timing
    # -------------------------
    host_diff = None
    host_mean_ms = host_median_ms = host_min_ms = host_max_ms = host_rate_hz = float("nan")
    host_bad_count = 0
    host_duration_s = float("nan")

    if host_ts is not None:
        host_diff = np.diff(host_ts)
        fhd = host_diff[np.isfinite(host_diff)]
        if fhd.size:
            host_mean_ms = float(np.mean(fhd))
            host_median_ms = float(np.median(fhd))
            host_min_ms = float(np.min(fhd))
            host_max_ms = float(np.max(fhd))
            host_duration_s = float((host_ts[-1] - host_ts[0]) / 1000.0)
            if host_mean_ms > 0:
                host_rate_hz = 1000.0 / host_mean_ms

            # 9–11 ms is reported for diagnostics only.
            # It is not used to delete samples.
            host_bad_count = int(np.sum((fhd < 9.0) | (fhd > 11.0)))

    # -------------------------
    # Signal statistics
    # -------------------------
    finite_ir = ir[np.isfinite(ir)]
    finite_red = red[np.isfinite(red)]

    ir_above_40k = (
        float(np.mean(finite_ir > 40000.0) * 100.0) if finite_ir.size else float("nan")
    )

    # A rough descriptive range check only; no samples are removed.
    ir_zero = int(np.sum(np.isfinite(ir) & (ir == 0)))
    ir_nan_inf = int(np.sum(~np.isfinite(ir)))
    red_zero = int(np.sum(np.isfinite(red) & (red == 0)))
    red_nan_inf = int(np.sum(~np.isfinite(red)))

    # -------------------------
    # Overall timing verdict
    # -------------------------
    timing_pass = (
        np.isfinite(host_rate_hz)
        and abs(host_rate_hz - 100.0) <= 1.0
        and host_bad_count <= max(5, int(0.01 * max(1, n - 1)))
    )

    # -------------------------
    # Write report
    # -------------------------
    report = {
        "input_file": str(csv_path),
        "rows": int(n),
        "columns": list(df.columns),

        "sample_index": {
            "first": float(sample_index[0]) if np.isfinite(sample_index[0]) else None,
            "last": float(sample_index[-1]) if np.isfinite(sample_index[-1]) else None,
            "median_step": median_sstep,
            "modal_step": mode_sstep,
            "expected_step_used_for_check": expected_step,
            "discontinuities_vs_modal_step": sample_gap_count,
            "duplicate_or_reverse_steps": sample_duplicate_or_reverse,
            "inferred_interval_ms_if_index_is_ms": inferred_sample_interval_ms,
            "inferred_rate_hz_if_index_is_ms": inferred_sample_rate_hz,
        },

        "host_timing": {
            "source_column": host_col,
            "mean_interval_ms": host_mean_ms,
            "median_interval_ms": host_median_ms,
            "min_interval_ms": host_min_ms,
            "max_interval_ms": host_max_ms,
            "duration_s": host_duration_s,
            "effective_rate_hz": host_rate_hz,
            "intervals_outside_9_to_11_ms": host_bad_count,
        },

        "ir": {
            "mean": float(np.mean(finite_ir)) if finite_ir.size else None,
            "median": float(np.median(finite_ir)) if finite_ir.size else None,
            "min": float(np.min(finite_ir)) if finite_ir.size else None,
            "max": float(np.max(finite_ir)) if finite_ir.size else None,
            "std": float(np.std(finite_ir, ddof=1)) if finite_ir.size > 1 else None,
            "percent_above_40000": ir_above_40k,
            "zeros": ir_zero,
            "nan_or_inf": ir_nan_inf,
        },

        "red": {
            "mean": float(np.mean(finite_red)) if finite_red.size else None,
            "median": float(np.median(finite_red)) if finite_red.size else None,
            "min": float(np.min(finite_red)) if finite_red.size else None,
            "max": float(np.max(finite_red)) if finite_red.size else None,
            "std": float(np.std(finite_red, ddof=1)) if finite_red.size > 1 else None,
            "zeros": red_zero,
            "nan_or_inf": red_nan_inf,
        },

        "verdict": {
            "timing_pass": bool(timing_pass),
            "input_csv_modified": False,
            "notes": [
                "No rows are deleted by this script.",
                "expected_timestamp_ms is not used as the source of truth.",
                "sample_index is checked against its observed modal step, not assumed to increment by 1.",
                "Host timestamp timing is used for the primary 100-Hz acquisition check when available.",
                "A timing pass does not establish BP accuracy or clinical validity.",
            ],
        },
    }

    with open(outdir / "hardware_inspection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Human-readable text report
    lines = [
        "=" * 72,
        "MAX30102 HARDWARE CAPTURE INSPECTION",
        "=" * 72,
        f"Input: {csv_path.name}",
        f"Rows: {n}",
        "",
        "--- SAMPLE INDEX ---",
        f"First: {sample_index[0]:.0f}",
        f"Last:  {sample_index[-1]:.0f}",
        f"Median step: {median_sstep:.3f}",
        f"Modal step:  {mode_sstep:.3f}",
        f"Discontinuities vs modal step: {sample_gap_count}",
        f"Duplicate/reverse steps: {sample_duplicate_or_reverse}",
        f"Inferred interval (if index units are ms): {inferred_sample_interval_ms:.3f} ms",
        f"Inferred rate (if index units are ms): {inferred_sample_rate_hz:.3f} Hz",
        "",
        "--- HOST TIMING ---",
        f"Source column: {host_col}",
        f"Mean interval:   {host_mean_ms:.3f} ms",
        f"Median interval: {host_median_ms:.3f} ms",
        f"Min interval:    {host_min_ms:.3f} ms",
        f"Max interval:    {host_max_ms:.3f} ms",
        f"Duration:        {host_duration_s:.3f} s",
        f"Effective rate:  {host_rate_hz:.3f} Hz",
        f"Intervals outside 9–11 ms: {host_bad_count}",
        "",
        "--- IR SIGNAL ---",
        f"Mean: {np.mean(finite_ir):.2f}",
        f"Median: {np.median(finite_ir):.2f}",
        f"Min: {np.min(finite_ir):.2f}",
        f"Max: {np.max(finite_ir):.2f}",
        f"Std: {np.std(finite_ir, ddof=1):.2f}" if finite_ir.size > 1 else "Std: n/a",
        f"Above 40,000 counts: {ir_above_40k:.2f}%",
        f"Zeros: {ir_zero}",
        f"NaN/Inf: {ir_nan_inf}",
        "",
        "--- RED CHANNEL ---",
        f"Mean: {np.mean(finite_red):.2f}",
        f"Median: {np.median(finite_red):.2f}",
        f"Min: {np.min(finite_red):.2f}",
        f"Max: {np.max(finite_red):.2f}",
        f"Std: {np.std(finite_red, ddof=1):.2f}" if finite_red.size > 1 else "Std: n/a",
        f"Zeros: {red_zero}",
        f"NaN/Inf: {red_nan_inf}",
        "",
        "--- VERDICT ---",
        f"Primary host-timing check: {'PASS' if timing_pass else 'REVIEW'}",
        "Raw input CSV modified: NO",
        "",
        "Do not use this report as evidence of BP accuracy or clinical validity.",
    ]

    with open(outdir / "hardware_inspection_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # -------------------------
    # Corrected raw IR plot
    # -------------------------
    if host_ts is not None and np.isfinite(host_ts).all():
        x = (host_ts - host_ts[0]) / 1000.0
        xlabel = "Relative time (s) — host timestamp"
    else:
        # Fall back to sample index relative to first sample.
        x = (sample_index - sample_index[0]) / 1000.0
        xlabel = "Relative time (s) — sample_index assuming ms"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, ir, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("IR ADC Counts")
    ax.set_title("MAX30102 Raw IR PPG — Corrected Time Axis")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "raw_ir_corrected.png", dpi=200)
    plt.close(fig)

    # Last 60 s (or all data if shorter) — easier to inspect steady-state signal.
    start_t = max(0.0, float(x[-1]) - 60.0)
    mask = x >= start_t

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x[mask], ir[mask], linewidth=0.8)
    ax.set_xlabel("Relative time (s)")
    ax.set_ylabel("IR ADC Counts")
    ax.set_title("MAX30102 Raw IR PPG — Last 60 Seconds")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "raw_ir_last60s.png", dpi=200)
    plt.close(fig)

    # Optional short signal summary file for easy copy/paste into chat.
    with open(outdir / "UPLOAD_ME.txt", "w", encoding="utf-8") as f:
        f.write(
            "Send these files/results for review:\n"
            "1. hardware_inspection_report.txt\n"
            "2. hardware_inspection_report.json\n"
            "3. raw_ir_corrected.png\n"
            "4. raw_ir_last60s.png\n"
            "5. The original CSV filename and row count\n"
        )

    print("\n" + "=" * 72)
    print("INSPECTION COMPLETE")
    print("=" * 72)
    print(f"Input: {csv_path}")
    print(f"Rows: {n}")
    print(f"Sample-index modal step: {mode_sstep:.3f}")
    print(f"Host median interval: {host_median_ms:.3f} ms")
    print(f"Host effective rate: {host_rate_hz:.3f} Hz")
    print(f"Sample-index discontinuities vs modal step: {sample_gap_count}")
    print(f"Host intervals outside 9–11 ms: {host_bad_count}")
    print(f"Primary timing check: {'PASS' if timing_pass else 'REVIEW'}")
    print(f"Output directory: {outdir}")
    print("\nSend the contents/files listed in UPLOAD_ME.txt.")


if __name__ == "__main__":
    main()
