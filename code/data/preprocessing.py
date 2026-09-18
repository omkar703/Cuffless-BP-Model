"""
Signal Preprocessing & Physiological Ground Truth Extraction.

Contains:
1. OFFLINE RESEARCH FILTER: Zero-phase Butterworth bandpass filter (0.5–8.0 Hz).
2. Diagnostic beat-by-beat ABP peak & foot detector for accurate SBP/DBP ground truth.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from config.config import (
    SAMPLING_RATE,
    PPG_BANDPASS_LOWCUT,
    PPG_BANDPASS_HIGHCUT,
    PPG_FILTER_ORDER,
    ABP_PEAK_MIN_DISTANCE,
    ABP_PEAK_PROMINENCE,
    SBP_PLAUSIBLE_RANGE,
    DBP_PLAUSIBLE_RANGE,
    MIN_PULSE_PRESSURE,
)


def apply_offline_research_ppg_filter(
    ppg_raw: np.ndarray,
    fs: int = SAMPLING_RATE,
    lowcut: float = PPG_BANDPASS_LOWCUT,
    highcut: float = PPG_BANDPASS_HIGHCUT,
    order: int = PPG_FILTER_ORDER,
) -> np.ndarray:
    """
    Applies zero-phase 3rd-order Butterworth bandpass filter (0.5 to 8.0 Hz).
    
    WARNING / RESEARCH NOTE:
        This function uses `scipy.signal.filtfilt` (forward-backward zero-phase filtering).
        It is designed strictly as an OFFLINE RESEARCH FILTER for dataset cleaning, validation,
        and visualization. Real-time embedded deployment on an ESP32 will require a causal
        IIR or FIR filter (e.g. streaming biquad IIR) to avoid non-causal lookahead latency.
    """
    if len(ppg_raw) < 3 * order:
        return ppg_raw.copy()

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    ppg_filtered = filtfilt(b, a, ppg_raw)
    return ppg_filtered


def extract_physiological_bp_targets(
    abp: np.ndarray,
    fs: int = SAMPLING_RATE,
    min_distance: int = ABP_PEAK_MIN_DISTANCE,
    prominence: float = ABP_PEAK_PROMINENCE,
) -> Dict[str, Any]:
    """
    Extracts medically grounded beat-by-beat SBP, DBP, and MAP diagnostics from ABP.
    
    Rather than blindly taking record global extrema, this detector identifies each
    systolic peak, finds the diastolic valley preceding or following each peak, and reports:
        - num_detected_beats
        - estimated_hr_bpm
        - valid_beat_percentage
        - ambiguous_beats_count
        - sbp_mean, sbp_std, dbp_mean, dbp_std, map_mean
    """
    empty_result = {
        "is_valid": False,
        "reason": "EMPTY_OR_SHORT",
        "num_detected_beats": 0,
        "estimated_hr_bpm": np.nan,
        "valid_beat_percentage": 0.0,
        "ambiguous_beats_count": 0,
        "sbp_mean": np.nan,
        "sbp_std": np.nan,
        "dbp_mean": np.nan,
        "dbp_std": np.nan,
        "map_mean": np.nan,
        "peak_indices": np.array([], dtype=int),
        "trough_indices": np.array([], dtype=int),
    }

    if len(abp) < fs:
        return empty_result

    # 1. Detect systolic pressure peaks with prominence threshold
    peaks, _ = find_peaks(abp, distance=min_distance, prominence=prominence)

    if len(peaks) < 2:
        empty_result["reason"] = "INSUFFICIENT_PEAKS_DETECTED"
        return empty_result

    # Compute inter-beat intervals and estimated heart rate
    rr_samples = np.diff(peaks)
    mean_rr_sec = float(np.mean(rr_samples)) / fs
    estimated_hr_bpm = float(60.0 / mean_rr_sec) if mean_rr_sec > 0 else np.nan

    # 2. Find diastolic valleys between peaks
    trough_indices = []
    valid_sbp_list = []
    valid_dbp_list = []
    ambiguous_count = 0

    for i in range(len(peaks) - 1):
        idx_peak = peaks[i]
        idx_next = peaks[i + 1]

        # Diastolic foot is the local minimum between systolic peaks
        trough_rel = int(np.argmin(abp[idx_peak:idx_next]))
        trough_abs = idx_peak + trough_rel
        trough_val = float(abp[trough_abs])
        peak_val = float(abp[idx_peak])

        pulse_pressure = peak_val - trough_val

        # Physiological sanity check on this individual beat
        is_beat_plausible = (
            SBP_PLAUSIBLE_RANGE[0] <= peak_val <= SBP_PLAUSIBLE_RANGE[1]
            and DBP_PLAUSIBLE_RANGE[0] <= trough_val <= DBP_PLAUSIBLE_RANGE[1]
            and pulse_pressure >= MIN_PULSE_PRESSURE
        )

        if is_beat_plausible:
            trough_indices.append(trough_abs)
            valid_sbp_list.append(peak_val)
            valid_dbp_list.append(trough_val)
        else:
            ambiguous_count += 1

    total_candidate_beats = len(peaks) - 1
    valid_beat_pct = (
        float(len(valid_sbp_list) / total_candidate_beats * 100.0)
        if total_candidate_beats > 0
        else 0.0
    )

    is_valid = len(valid_sbp_list) >= 3 and valid_beat_pct >= 50.0

    sbp_mean = float(np.mean(valid_sbp_list)) if valid_sbp_list else np.nan
    sbp_std = float(np.std(valid_sbp_list)) if valid_sbp_list else np.nan
    dbp_mean = float(np.mean(valid_dbp_list)) if valid_dbp_list else np.nan
    dbp_std = float(np.std(valid_dbp_list)) if valid_dbp_list else np.nan
    map_mean = float(np.mean(abp))

    return {
        "is_valid": is_valid,
        "reason": "VALID" if is_valid else "LOW_VALID_BEAT_PERCENTAGE",
        "num_detected_beats": len(peaks),
        "estimated_hr_bpm": estimated_hr_bpm,
        "valid_beat_percentage": valid_beat_pct,
        "ambiguous_beats_count": ambiguous_count,
        "sbp_mean": sbp_mean,
        "sbp_std": sbp_std,
        "dbp_mean": dbp_mean,
        "dbp_std": dbp_std,
        "map_mean": map_mean,
        "peak_indices": peaks,
        "trough_indices": np.array(trough_indices, dtype=int),
        "sbp_beat_values": np.array(valid_sbp_list),
        "dbp_beat_values": np.array(valid_dbp_list),
    }
