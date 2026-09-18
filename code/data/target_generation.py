"""
Physiological ABP Target Generation & Normalization Engine.

Contains:
1. Normalization functions for PPG windows (Diagnostic & Experimental):
   - Raw filtered PPG
   - Per-window z-score normalization
   - Robust normalization (median / IQR)
2. Beat-by-beat ABP target generator:
   - Detects systolic peaks and diastolic valleys in 10-second ABP windows
   - Validates each individual cardiac cycle
   - Computes window-level SBP, DBP, MAP, and pulse pressure targets
   - Implements target sanity checks (SBP > DBP, valid beat ratio, plausible ranges)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.signal import find_peaks

from config.config import (
    SAMPLING_RATE,
    ABP_PEAK_MIN_DISTANCE,
    ABP_PEAK_PROMINENCE,
    ABP_MIN_VALID_BEATS,
    ABP_MIN_VALID_BEAT_RATIO,
    ABP_WARN_VALID_BEAT_RATIO,
    SBP_WINDOW_PLAUSIBLE_RANGE,
    DBP_WINDOW_PLAUSIBLE_RANGE,
    ABP_WINDOW_MIN_PULSE_PRESSURE,
    ABP_ABSOLUTE_MIN,
    ABP_ABSOLUTE_MAX,
    ABP_MIN_STD,
)
from data.preprocessing import apply_offline_research_ppg_filter


# ==============================================================================
# Normalization Functions (Diagnostic & Modeling Exploration)
# ==============================================================================

def normalize_raw_filtered(
    ppg_window: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """
    Applies offline research zero-phase bandpass filter (0.5 to 8.0 Hz)
    without rescaling.
    """
    return apply_offline_research_ppg_filter(ppg_window, fs=fs)


def normalize_zscore(
    ppg_window: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Per-window zero-mean unit-variance (z-score) normalization.
    """
    mean_val = np.mean(ppg_window)
    std_val = np.std(ppg_window)
    if std_val < eps:
        return np.zeros_like(ppg_window)
    return (ppg_window - mean_val) / (std_val + eps)


def normalize_robust(
    ppg_window: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Robust scaling using median and Interquartile Range (IQR).
    Resistant to isolated motion spikes.
    """
    median_val = np.median(ppg_window)
    p25 = np.percentile(ppg_window, 25)
    p75 = np.percentile(ppg_window, 75)
    iqr_val = p75 - p25
    if iqr_val < eps:
        # Fall back to std or return zeros if completely flat
        std_val = np.std(ppg_window)
        if std_val < eps:
            return np.zeros_like(ppg_window)
        return (ppg_window - median_val) / (std_val + eps)
    return (ppg_window - median_val) / (iqr_val + eps)


# ==============================================================================
# Beat-Wise ABP Target Extraction & Validation
# ==============================================================================

def extract_window_abp_targets(
    abp_window: Optional[np.ndarray],
    fs: int = SAMPLING_RATE,
    min_distance: int = ABP_PEAK_MIN_DISTANCE,
    prominence: float = ABP_PEAK_PROMINENCE,
) -> Tuple[bool, str, str, Dict[str, float], Dict[str, Any]]:
    """
    Extracts beat-by-beat physiological SBP, DBP, MAP targets from an ABP window.

    Parameters:
        abp_window: 1D numpy array of raw ABP pressure samples (mmHg).
        fs: Sampling rate (125 Hz).
        min_distance: Minimum samples between candidate systolic peaks (~44 samples).
        prominence: Minimum peak prominence in mmHg (10.0 mmHg).

    Returns:
        (abp_valid: bool, quality_status: str, rejection_reason: str, targets: dict, diagnostics: dict)
    """
    empty_targets = {
        "sbp": np.nan,
        "dbp": np.nan,
        "map": np.nan,
        "pulse_pressure": np.nan,
    }
    empty_diag = {
        "abp_mean": np.nan,
        "abp_std": np.nan,
        "abp_min": np.nan,
        "abp_max": np.nan,
        "candidate_beats": 0,
        "valid_beat_count": 0,
        "valid_beat_ratio": 0.0,
        "peak_indices": np.array([], dtype=int),
        "trough_indices": np.array([], dtype=int),
        "valid_sbp_beats": np.array([], dtype=float),
        "valid_dbp_beats": np.array([], dtype=float),
    }

    if abp_window is None or len(abp_window) == 0:
        return False, "REJECT", "EMPTY_ABP_WINDOW", empty_targets, empty_diag

    if np.isnan(abp_window).any():
        return False, "REJECT", "NAN_IN_ABP", empty_targets, empty_diag

    if np.isinf(abp_window).any():
        return False, "REJECT", "INF_IN_ABP", empty_targets, empty_diag

    mean_abp = float(np.mean(abp_window))
    std_abp = float(np.std(abp_window))
    min_abp = float(np.min(abp_window))
    max_abp = float(np.max(abp_window))

    empty_diag["abp_mean"] = mean_abp
    empty_diag["abp_std"] = std_abp
    empty_diag["abp_min"] = min_abp
    empty_diag["abp_max"] = max_abp

    # Catheter decoupling or flatline check
    if std_abp < ABP_MIN_STD:
        return False, "REJECT", f"FLATLINE_ABP_STD_{std_abp:.2f}", empty_targets, empty_diag

    if min_abp < ABP_ABSOLUTE_MIN:
        return False, "REJECT", f"ABP_BELOW_MIN_LIMIT_{min_abp:.1f}", empty_targets, empty_diag

    if max_abp > ABP_ABSOLUTE_MAX:
        return False, "REJECT", f"ABP_ABOVE_MAX_LIMIT_{max_abp:.1f}", empty_targets, empty_diag

    # 1. Detect candidate systolic peaks
    peaks, _ = find_peaks(abp_window, distance=min_distance, prominence=prominence)

    if len(peaks) < 2:
        return False, "REJECT", "INSUFFICIENT_PEAKS_DETECTED", empty_targets, empty_diag

    # 2. Detect corresponding diastolic troughs between adjacent peaks
    valid_sbp_list = []
    valid_dbp_list = []
    valid_map_list = []
    trough_indices = []

    for i in range(len(peaks) - 1):
        idx_curr = peaks[i]
        idx_next = peaks[i + 1]

        # Diastolic foot is the local minimum in the interval between systolic peaks
        foot_rel = int(np.argmin(abp_window[idx_curr:idx_next]))
        foot_abs = idx_curr + foot_rel
        foot_val = float(abp_window[foot_abs])
        peak_val = float(abp_window[idx_curr])
        pp_val = peak_val - foot_val

        # Beat-wise physiological sanity check
        is_beat_valid = (
            SBP_WINDOW_PLAUSIBLE_RANGE[0] <= peak_val <= SBP_WINDOW_PLAUSIBLE_RANGE[1]
            and DBP_WINDOW_PLAUSIBLE_RANGE[0] <= foot_val <= DBP_WINDOW_PLAUSIBLE_RANGE[1]
            and pp_val >= ABP_WINDOW_MIN_PULSE_PRESSURE
        )

        if is_beat_valid:
            valid_sbp_list.append(peak_val)
            valid_dbp_list.append(foot_val)
            # Beat-wise MAP formula: DBP + 1/3 * (SBP - DBP)
            beat_map = foot_val + (peak_val - foot_val) / 3.0
            valid_map_list.append(beat_map)
            trough_indices.append(foot_abs)

    total_candidate_beats = len(peaks) - 1
    valid_beat_count = len(valid_sbp_list)
    valid_beat_ratio = (
        float(valid_beat_count / total_candidate_beats)
        if total_candidate_beats > 0
        else 0.0
    )

    empty_diag["candidate_beats"] = total_candidate_beats
    empty_diag["valid_beat_count"] = valid_beat_count
    empty_diag["valid_beat_ratio"] = valid_beat_ratio
    empty_diag["peak_indices"] = peaks
    empty_diag["trough_indices"] = np.array(trough_indices, dtype=int)
    empty_diag["valid_sbp_beats"] = np.array(valid_sbp_list, dtype=float)
    empty_diag["valid_dbp_beats"] = np.array(valid_dbp_list, dtype=float)

    # 3. Validation thresholds for window-level ABP target integrity
    if valid_beat_count < ABP_MIN_VALID_BEATS:
        return (
            False,
            "REJECT",
            f"INSUFFICIENT_VALID_BEATS_{valid_beat_count}_OF_{total_candidate_beats}",
            empty_targets,
            empty_diag,
        )

    if valid_beat_ratio < ABP_MIN_VALID_BEAT_RATIO:
        return (
            False,
            "REJECT",
            f"LOW_VALID_BEAT_RATIO_{valid_beat_ratio:.1%}",
            empty_targets,
            empty_diag,
        )

    # 4. Compute primary window targets as mean of valid beats
    sbp_target = float(np.mean(valid_sbp_list))
    dbp_target = float(np.mean(valid_dbp_list))
    map_target = float(np.mean(valid_map_list))
    pulse_pressure_target = float(sbp_target - dbp_target)

    # Final sanity check on window targets
    if pulse_pressure_target < ABP_WINDOW_MIN_PULSE_PRESSURE:
        return (
            False,
            "REJECT",
            f"LOW_PULSE_PRESSURE_{pulse_pressure_target:.1f}",
            empty_targets,
            empty_diag,
        )

    if not (SBP_WINDOW_PLAUSIBLE_RANGE[0] <= sbp_target <= SBP_WINDOW_PLAUSIBLE_RANGE[1]):
        return (
            False,
            "REJECT",
            f"SBP_TARGET_OUT_OF_RANGE_{sbp_target:.1f}",
            empty_targets,
            empty_diag,
        )

    if not (DBP_WINDOW_PLAUSIBLE_RANGE[0] <= dbp_target <= DBP_WINDOW_PLAUSIBLE_RANGE[1]):
        return (
            False,
            "REJECT",
            f"DBP_TARGET_OUT_OF_RANGE_{dbp_target:.1f}",
            empty_targets,
            empty_diag,
        )

    targets = {
        "sbp": sbp_target,
        "dbp": dbp_target,
        "map": map_target,
        "pulse_pressure": pulse_pressure_target,
    }

    # Assign PASS vs WARN
    if valid_beat_ratio < ABP_WARN_VALID_BEAT_RATIO:
        status = "WARN"
        reason = f"MODERATE_BEAT_QUALITY_{valid_beat_ratio:.1%}"
    else:
        status = "PASS"
        reason = "VALID"

    return True, status, reason, targets, empty_diag


def evaluate_window_quality_status(
    ppg_valid: bool,
    abp_valid: bool,
) -> Tuple[str, bool]:
    """
    Decoupled four-way classification of window quality status and modeling eligibility.

    Returns:
        (window_quality_status: str, modeling_eligible: bool)
    """
    if ppg_valid and abp_valid:
        return "PPG_VALID_ABP_VALID", True
    elif ppg_valid and not abp_valid:
        return "PPG_VALID_ABP_INVALID", False
    elif not ppg_valid and abp_valid:
        return "PPG_INVALID_ABP_VALID", False
    else:
        return "PPG_INVALID_ABP_INVALID", False
