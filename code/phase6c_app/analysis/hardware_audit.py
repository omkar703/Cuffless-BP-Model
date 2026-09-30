"""
Hardware Data Integrity Audit Module.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Reuses and formalizes the Phase 6B validation checks on sample continuity,
timing jitter, sensor saturation/disconnect, and channel statistics.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import scipy.stats as stats


def audit_hardware_data(df: pd.DataFrame, dataset_name: str = "raw_hardware_data") -> Dict[str, Any]:
    """
    Perform a comprehensive integrity audit on incoming hardware acquisition data.
    
    Args:
        df: DataFrame containing sample_index, host_timestamp_ms, ir, red.
        dataset_name: Identifier for the source dataset.
        
    Returns:
        Structured audit dictionary with metrics and PASS/WARN/FAIL verdict.
    """
    n_rows = len(df)
    if n_rows == 0:
        return {
            "dataset_filename": dataset_name,
            "row_count": 0,
            "audit_verdict": "FAIL",
            "failure_reasons": ["Empty dataset"],
        }

    # 1. Sample Index Analysis
    s_idx = df["sample_index"].to_numpy(dtype=float)
    first_idx = float(s_idx[0])
    last_idx = float(s_idx[-1])
    diff_s = np.diff(s_idx)

    if len(diff_s) > 0:
        modal_res = stats.mode(diff_s, keepdims=False)
        modal_s_step = float(modal_res.mode)
        s_discont = int(np.sum(diff_s != modal_s_step))
        s_dup_rev = int(np.sum(diff_s <= 0))
    else:
        modal_s_step = 10.0
        s_discont = 0
        s_dup_rev = 0

    if modal_s_step == 1.0:
        inferred_interval_ms = 10.0  # Sequential sample counter (1 count per 10 ms sample @ 100 Hz)
        inferred_nominal_rate = 100.0
    elif modal_s_step > 0:
        inferred_interval_ms = modal_s_step
        inferred_nominal_rate = 1000.0 / inferred_interval_ms
    else:
        inferred_interval_ms = 10.0
        inferred_nominal_rate = 100.0

    # 2. Host Timestamp Timing & Jitter
    host_ts = df["host_timestamp_ms"].to_numpy(dtype=float)
    host_intervals = np.diff(host_ts)

    if len(host_intervals) > 0:
        host_mean_int = float(np.mean(host_intervals))
        host_median_int = float(np.median(host_intervals))
        host_min_int = float(np.min(host_intervals))
        host_max_int = float(np.max(host_intervals))
        host_std_int = float(np.std(host_intervals))
        host_duration_s = float((host_ts[-1] - host_ts[0]) / 1000.0)
        effective_rate_hz = float(len(host_intervals) / host_duration_s) if host_duration_s > 0 else 0.0
        outside_9_11_ms = int(np.sum((host_intervals < 9.0) | (host_intervals > 11.0)))
        percent_outside_9_11 = float(outside_9_11_ms / len(host_intervals) * 100.0)
    else:
        host_mean_int = 10.0
        host_median_int = 10.0
        host_min_int = 10.0
        host_max_int = 10.0
        host_std_int = 0.0
        host_duration_s = 0.0
        effective_rate_hz = 100.0
        outside_9_11_ms = 0
        percent_outside_9_11 = 0.0

    # 3. IR Channel Statistics
    ir_vals = df["ir"].to_numpy(dtype=float)
    ir_nan_inf = int(np.sum(np.isnan(ir_vals) | np.isinf(ir_vals)))
    clean_ir = ir_vals[~(np.isnan(ir_vals) | np.isinf(ir_vals))]

    if len(clean_ir) > 0:
        ir_mean = float(np.mean(clean_ir))
        ir_median = float(np.median(clean_ir))
        ir_min = float(np.min(clean_ir))
        ir_max = float(np.max(clean_ir))
        ir_std = float(np.std(clean_ir))
        ir_zeros = int(np.sum(clean_ir == 0.0))
        ir_above_40k = float(np.mean(clean_ir >= 40000.0) * 100.0)
    else:
        ir_mean, ir_median, ir_min, ir_max, ir_std = 0.0, 0.0, 0.0, 0.0, 0.0
        ir_zeros = 0
        ir_above_40k = 0.0

    # 4. Red Channel Statistics
    red_vals = df["red"].to_numpy(dtype=float)
    clean_red = red_vals[~(np.isnan(red_vals) | np.isinf(red_vals))]
    if len(clean_red) > 0:
        red_mean = float(np.mean(clean_red))
        red_median = float(np.median(clean_red))
        red_min = float(np.min(clean_red))
        red_max = float(np.max(clean_red))
        red_std = float(np.std(clean_red))
        red_zeros = int(np.sum(clean_red == 0.0))
    else:
        red_mean, red_median, red_min, red_max, red_std = 0.0, 0.0, 0.0, 0.0, 0.0
        red_zeros = 0

    # 5. Verdict Determination
    failure_reasons = []
    warning_reasons = []

    if ir_nan_inf > 0:
        failure_reasons.append(f"Detected {ir_nan_inf} NaN/Inf values in IR signal")
    if s_dup_rev > 0:
        failure_reasons.append(f"Detected {s_dup_rev} duplicate or reverse sample indices")
    if host_duration_s < 10.0:
        failure_reasons.append("Recording duration is shorter than a single 10-second model window")

    if s_discont > 0:
        warning_reasons.append(f"Detected {s_discont} sample-index discontinuities")
    if outside_9_11_ms > 0:
        warning_reasons.append(f"{outside_9_11_ms} host intervals ({percent_outside_9_11:.1f}%) outside 9–11 ms")
    if ir_above_40k < 50.0:
        warning_reasons.append(f"Only {ir_above_40k:.1f}% of IR samples are >= 40,000 counts (low signal/contact)")
    if red_mean < 100.0:
        warning_reasons.append(f"Red channel mean is low ({red_mean:.1f} counts); indicates ambient/single-wavelength mode")

    if failure_reasons:
        verdict = "FAIL"
    elif warning_reasons:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "dataset_filename": dataset_name,
        "row_count": n_rows,
        "sample_index": {
            "first": first_idx,
            "last": last_idx,
            "modal_step": modal_s_step,
            "discontinuities_vs_modal_step": s_discont,
            "duplicate_or_reverse_steps": s_dup_rev,
            "inferred_interval_ms": inferred_interval_ms,
            "inferred_nominal_rate_hz": inferred_nominal_rate,
        },
        "host_timing": {
            "mean_interval_ms": host_mean_int,
            "median_interval_ms": host_median_int,
            "min_interval_ms": host_min_int,
            "max_interval_ms": host_max_int,
            "std_interval_ms": host_std_int,
            "duration_s": host_duration_s,
            "effective_rate_hz": effective_rate_hz,
            "intervals_outside_9_11_ms": outside_9_11_ms,
            "percent_outside_9_11_ms": percent_outside_9_11,
        },
        "ir_signal": {
            "mean": ir_mean,
            "median": ir_median,
            "min": ir_min,
            "max": ir_max,
            "std": ir_std,
            "zeros": ir_zeros,
            "nan_or_inf": ir_nan_inf,
            "percent_above_40k": ir_above_40k,
        },
        "red_signal": {
            "mean": red_mean,
            "median": red_median,
            "min": red_min,
            "max": red_max,
            "std": red_std,
            "zeros": red_zeros,
        },
        "audit_verdict": verdict,
        "failure_reasons": failure_reasons,
        "warning_reasons": warning_reasons,
    }
