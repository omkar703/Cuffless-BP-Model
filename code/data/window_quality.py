"""
Window-Level PPG Quality Control Engine.

Applies decoupled, multi-stage physiological and signal-integrity quality gates
independently to each 10-second PPG window (1250 samples at 125 Hz).

Quality Checks:
1. Finite data check (NaN, Inf) -> REJECT
2. Sufficient sample length (< 1250 samples) -> REJECT
3. Flatline / zero-variance check (std < 1e-5) -> REJECT
4. Clipping / ADC saturation (samples at extremum) -> WARN / REJECT
5. Excessive sample-to-sample discontinuities / step jumps -> REJECT
6. Pulse structure & physiological heart rate limits (40-220 bpm) -> WARN / REJECT
7. Pulse amplitude morphological consistency -> PASS / WARN

Quality Statuses:
- PASS: Clean waveform, well-formed pulses, normal physiological HR.
- WARN: Borderline HR, mild baseline fluctuation, or mild amplitude variation (retained as valid).
- REJECT: Unusable, corrupted, or non-physiological window.

PPG Validity:
ppg_valid = True if quality_status in ["PASS", "WARN"] else False
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.signal import find_peaks

from config.config import (
    SAMPLING_RATE,
    WINDOW_SAMPLES,
    PPG_WINDOW_MIN_STD,
    PPG_WINDOW_MAX_CLIPPING_RATIO,
    PPG_WINDOW_WARN_CLIPPING_RATIO,
    PPG_WINDOW_MAX_JUMP_FACTOR,
    PPG_HR_PLAUSIBLE_RANGE,
    PPG_PULSE_MIN_DISTANCE,
    PPG_PULSE_PROMINENCE_RATIO,
)
from data.preprocessing import apply_offline_research_ppg_filter


def validate_ppg_window(
    ppg_window: np.ndarray,
    fs: int = SAMPLING_RATE,
    window_samples: int = WINDOW_SAMPLES,
) -> Tuple[bool, str, str, Dict[str, Any]]:
    """
    Validates a single 10-second PPG window.

    Parameters:
        ppg_window: 1D numpy array of raw PPG samples (expected length: 1250).
        fs: Sampling frequency in Hz (default: 125 Hz).
        window_samples: Required sample count (default: 1250).

    Returns:
        (ppg_valid: bool, quality_status: str, rejection_reason: str, diagnostics: dict)
    """
    diagnostics = {
        "ppg_mean": np.nan,
        "ppg_std": np.nan,
        "ppg_ptp": np.nan,
        "ppg_min": np.nan,
        "ppg_max": np.nan,
        "clipped_fraction_low": 0.0,
        "clipped_fraction_high": 0.0,
        "ppg_clipped_fraction": 0.0,
        "ppg_max_jump": 0.0,
        "ppg_pulse_count": 0,
        "estimated_hr_bpm": np.nan,
        "median_ibi_sec": np.nan,
        "pulse_amp_cv": np.nan,
    }

    # 1. Check for empty or None
    if ppg_window is None or len(ppg_window) == 0:
        return False, "REJECT", "EMPTY_WINDOW", diagnostics

    # 2. Check sufficient length
    if len(ppg_window) < window_samples:
        return False, "REJECT", f"INSUFFICIENT_SAMPLES_{len(ppg_window)}", diagnostics

    # 3. Check for non-finite values (NaN / Inf)
    if np.isnan(ppg_window).any():
        nan_count = int(np.isnan(ppg_window).sum())
        return False, "REJECT", f"NAN_PRESENT_{nan_count}", diagnostics

    if np.isinf(ppg_window).any():
        return False, "REJECT", "INF_PRESENT", diagnostics

    # Basic signal statistics
    mean_val = float(np.mean(ppg_window))
    std_val = float(np.std(ppg_window))
    min_val = float(np.min(ppg_window))
    max_val = float(np.max(ppg_window))
    ptp_val = float(max_val - min_val)

    diagnostics["ppg_mean"] = mean_val
    diagnostics["ppg_std"] = std_val
    diagnostics["ppg_ptp"] = ptp_val
    diagnostics["ppg_min"] = min_val
    diagnostics["ppg_max"] = max_val

    # 4. Flatline check (near-zero variance)
    if std_val < PPG_WINDOW_MIN_STD:
        return False, "REJECT", f"FLATLINE_STD_{std_val:.2e}", diagnostics

    # 5. Clipping / ADC saturation detection
    clip_atol = 1e-4
    clipped_top = float(np.isclose(ppg_window, max_val, atol=clip_atol).mean())
    clipped_bot = float(np.isclose(ppg_window, min_val, atol=clip_atol).mean())
    max_clip = max(clipped_top, clipped_bot)

    diagnostics["clipped_fraction_low"] = clipped_bot
    diagnostics["clipped_fraction_high"] = clipped_top
    diagnostics["ppg_clipped_fraction"] = max_clip

    if max_clip > PPG_WINDOW_MAX_CLIPPING_RATIO:
        return False, "REJECT", f"CLIPPING_SATURATION_{max_clip:.2%}", diagnostics

    # 6. Discontinuity / Step jump check
    abs_diffs = np.abs(np.diff(ppg_window))
    max_jump = float(np.max(abs_diffs))
    diagnostics["ppg_max_jump"] = max_jump

    if std_val > 0 and (max_jump / std_val) > PPG_WINDOW_MAX_JUMP_FACTOR:
        return False, "REJECT", f"STEP_DISCONTINUITY_{max_jump / std_val:.1f}X_STD", diagnostics

    # 7. Pulse Structure & Morphological Inspection using OFFLINE research filter
    # NOTE: This is strictly for QC validation diagnostics; pulse features are NOT ML inputs!
    ppg_filtered = apply_offline_research_ppg_filter(ppg_window, fs=fs)
    std_filt = float(np.std(ppg_filtered))

    prominence = max(0.01, PPG_PULSE_PROMINENCE_RATIO * std_filt)
    peaks, properties = find_peaks(
        ppg_filtered,
        distance=PPG_PULSE_MIN_DISTANCE,
        prominence=prominence,
    )
    pulse_count = len(peaks)
    diagnostics["ppg_pulse_count"] = pulse_count

    # Calculate pulse rate and morphology diagnostics
    estimated_hr = np.nan
    median_ibi_sec = np.nan
    pulse_amp_cv = np.nan

    if pulse_count >= 2:
        ibi_samples = np.diff(peaks)
        median_ibi_sec = float(np.median(ibi_samples)) / fs
        diagnostics["median_ibi_sec"] = median_ibi_sec
        if median_ibi_sec > 0:
            estimated_hr = float(60.0 / median_ibi_sec)
            diagnostics["estimated_hr_bpm"] = estimated_hr

    if "prominences" in properties and len(properties["prominences"]) >= 2:
        proms = properties["prominences"]
        mean_prom = float(np.mean(proms))
        if mean_prom > 0:
            pulse_amp_cv = float(np.std(proms) / mean_prom)
            diagnostics["pulse_amp_cv"] = pulse_amp_cv

    # Heart Rate & Pulse Count Evaluation for 10-second window:
    # 40 bpm corresponds to ~6.7 beats in 10s; 220 bpm corresponds to ~36.7 beats in 10s
    # Hard rejection thresholds: pulse_count < 4 or pulse_count > 42
    if pulse_count < 4:
        return False, "REJECT", f"IMPLAUSIBLE_PULSE_COUNT_TOO_LOW_{pulse_count}", diagnostics

    if pulse_count > 42:
        return False, "REJECT", f"IMPLAUSIBLE_PULSE_COUNT_TOO_HIGH_{pulse_count}", diagnostics

    # Warning thresholds for borderline cases
    warn_reasons = []
    if max_clip > PPG_WINDOW_WARN_CLIPPING_RATIO:
        warn_reasons.append(f"MILD_CLIPPING_{max_clip:.1%}")

    if pulse_count in [4, 5]:
        warn_reasons.append(f"BORDERLINE_BRADYCARDIA_{pulse_count}_PULSES")
    elif pulse_count >= 37:
        warn_reasons.append(f"BORDERLINE_TACHYCARDIA_{pulse_count}_PULSES")

    if not np.isnan(pulse_amp_cv) and pulse_amp_cv > 0.65:
        warn_reasons.append(f"HIGH_AMPLITUDE_VARIABILITY_CV_{pulse_amp_cv:.2f}")

    if warn_reasons:
        return True, "WARN", "; ".join(warn_reasons), diagnostics

    return True, "PASS", "VALID", diagnostics
