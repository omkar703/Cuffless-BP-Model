"""
Quality Control and Validation Engine for PPG and ABP signals.

Implements decoupled quality gates:
1. structural_valid: Record is structurally intact and meets minimum duration.
2. ppg_valid: Photoplethysmography waveform is clean (no NaN, not flat, no clipping).
3. abp_valid: Arterial blood pressure reference catheter integrity is confirmed.
4. paired_valid: Both PPG and ABP are valid (ready for paired supervised training).

IMPORTANT SCIENTIFIC PRINCIPLE:
Corrupted ABP does NOT invalidate PPG data. Useful PPG signals are preserved
independently for unsupervised, morphological, or self-supervised modeling.
"""

from typing import Dict, Any, Tuple
import numpy as np

from config.config import (
    MIN_SIGNAL_LENGTH_SAMPLES,
    PPG_MIN_STD,
    PPG_MAX_CLIPPING_RATIO,
    PPG_MAX_JUMP_STD_FACTOR,
    ABP_MIN_STD,
    ABP_ABSOLUTE_MIN,
    ABP_ABSOLUTE_MAX,
    SBP_PLAUSIBLE_RANGE,
    DBP_PLAUSIBLE_RANGE,
    MIN_PULSE_PRESSURE,
)
from data.preprocessing import extract_physiological_bp_targets


def validate_ppg_signal(ppg: np.ndarray) -> Tuple[bool, str, Dict[str, float]]:
    """
    Validates a raw 1D PPG signal independently of ABP.
    
    Returns:
        (is_valid: bool, reason: str, diagnostics: dict)
    """
    diagnostics = {
        "ppg_mean": np.nan,
        "ppg_std": np.nan,
        "ppg_ptp": np.nan,
        "ppg_clipping_ratio": 0.0,
        "ppg_max_jump": 0.0,
    }

    if ppg is None or len(ppg) == 0:
        return False, "EMPTY_SIGNAL", diagnostics

    if len(ppg) < MIN_SIGNAL_LENGTH_SAMPLES:
        return False, f"SHORT_DURATION_{len(ppg)}_SAMPLES", diagnostics

    if np.isnan(ppg).any():
        nan_count = int(np.isnan(ppg).sum())
        return False, f"NAN_PRESENT_{nan_count}", diagnostics

    if np.isinf(ppg).any():
        return False, "INF_PRESENT", diagnostics

    std_val = float(np.std(ppg))
    diagnostics["ppg_std"] = std_val
    diagnostics["ppg_mean"] = float(np.mean(ppg))
    ptp_val = float(np.ptp(ppg))
    diagnostics["ppg_ptp"] = ptp_val

    # Reject zero-variance / sensor disconnection flatlines
    if std_val < PPG_MIN_STD:
        return False, f"FLATLINE_STD_{std_val:.2e}", diagnostics

    # Rail clipping / ADC saturation diagnostic
    min_val, max_val = np.min(ppg), np.max(ppg)
    clipped_top = float(np.isclose(ppg, max_val, atol=1e-4).mean())
    clipped_bot = float(np.isclose(ppg, min_val, atol=1e-4).mean())
    max_clip = max(clipped_top, clipped_bot)
    diagnostics["ppg_clipping_ratio"] = max_clip

    if max_clip > PPG_MAX_CLIPPING_RATIO:
        return False, f"CLIPPING_SATURATION_MAX_{max_clip:.2%}", diagnostics

    # Excessive non-physiological step jump check
    max_jump = float(np.max(np.abs(np.diff(ppg))))
    diagnostics["ppg_max_jump"] = max_jump
    if std_val > 0 and (max_jump / std_val) > PPG_MAX_JUMP_STD_FACTOR:
        return False, f"STEP_DISCONTINUITY_{max_jump / std_val:.1f}X_STD", diagnostics

    return True, "VALID", diagnostics


def validate_abp_signal(abp: np.ndarray) -> Tuple[bool, str, Dict[str, float]]:
    """
    Validates a raw 1D arterial blood pressure (ABP) reference waveform in mmHg.
    
    Returns:
        (is_valid: bool, reason: str, diagnostics: dict)
    """
    diagnostics = {
        "abp_mean": np.nan,
        "abp_std": np.nan,
        "abp_min": np.nan,
        "abp_max": np.nan,
        "abp_p05": np.nan,
        "abp_p95": np.nan,
    }

    if abp is None or len(abp) == 0:
        return False, "EMPTY_SIGNAL", diagnostics

    if len(abp) < MIN_SIGNAL_LENGTH_SAMPLES:
        return False, f"SHORT_DURATION_{len(abp)}_SAMPLES", diagnostics

    if np.isnan(abp).any():
        nan_count = int(np.isnan(abp).sum())
        return False, f"NAN_PRESENT_{nan_count}", diagnostics

    if np.isinf(abp).any():
        return False, "INF_PRESENT", diagnostics

    std_val = float(np.std(abp))
    diagnostics["abp_std"] = std_val
    diagnostics["abp_mean"] = float(np.mean(abp))
    min_val, max_val = float(np.min(abp)), float(np.max(abp))
    diagnostics["abp_min"] = min_val
    diagnostics["abp_max"] = max_val

    # Reject zero-pressure or flatline arterial line
    if std_val < ABP_MIN_STD:
        return False, f"FLATLINE_ABP_STD_{std_val:.2f}", diagnostics

    # Reject catastrophic sensor decoupling or flushing spikes
    if min_val < ABP_ABSOLUTE_MIN:
        return False, f"ABP_BELOW_MIN_LIMIT_{min_val:.1f}_MMHG", diagnostics

    if max_val > ABP_ABSOLUTE_MAX:
        return False, f"ABP_ABOVE_MAX_LIMIT_{max_val:.1f}_MMHG", diagnostics

    # Percentile based range inspection (robust against small motion ripples)
    p05 = float(np.percentile(abp, 5))
    p95 = float(np.percentile(abp, 95))
    diagnostics["abp_p05"] = p05
    diagnostics["abp_p95"] = p95

    if p95 < SBP_PLAUSIBLE_RANGE[0] or p95 > SBP_PLAUSIBLE_RANGE[1]:
        return False, f"SBP_OUT_OF_RANGE_P95_{p95:.1f}_MMHG", diagnostics

    if p05 < DBP_PLAUSIBLE_RANGE[0] or p05 > DBP_PLAUSIBLE_RANGE[1]:
        return False, f"DBP_OUT_OF_RANGE_P05_{p05:.1f}_MMHG", diagnostics

    if (p95 - p05) < MIN_PULSE_PRESSURE:
        return False, f"LOW_PULSE_PRESSURE_{p95 - p05:.1f}_MMHG", diagnostics

    return True, "VALID", diagnostics


def perform_record_quality_control(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates record structural integrity, independent PPG quality, independent
    ABP reference quality, and computes beat-by-beat ground truth diagnostics.
    """
    ppg = record["ppg"]
    abp = record["abp"]

    structural_valid = (
        ppg is not None
        and abp is not None
        and len(ppg) >= MIN_SIGNAL_LENGTH_SAMPLES
        and len(abp) >= MIN_SIGNAL_LENGTH_SAMPLES
    )

    ppg_valid, ppg_reason, ppg_diag = validate_ppg_signal(ppg)
    abp_valid, abp_reason, abp_diag = validate_abp_signal(abp)

    paired_valid = ppg_valid and abp_valid

    # Determine decoupled quality status category
    if paired_valid:
        quality_status = "valid_paired"
    elif ppg_valid and not abp_valid:
        quality_status = "valid_ppg_only"
    elif not ppg_valid and abp_valid:
        quality_status = "valid_abp_only"
    else:
        quality_status = "rejected_both"

    # Compute ABP beat detection diagnostics if ABP is generally usable
    abp_beat_diag = {
        "num_detected_beats": 0,
        "estimated_hr_bpm": np.nan,
        "valid_beat_percentage": 0.0,
        "ambiguous_beats_count": 0,
        "sbp_mean": np.nan,
        "sbp_std": np.nan,
        "dbp_mean": np.nan,
        "dbp_std": np.nan,
        "map_mean": np.nan,
        "is_abp_beat_valid": False,
    }

    if abp_valid:
        try:
            bp_res = extract_physiological_bp_targets(abp, fs=record["sampling_frequency"])
            if bp_res["is_valid"]:
                abp_beat_diag.update({
                    "num_detected_beats": bp_res["num_detected_beats"],
                    "estimated_hr_bpm": bp_res["estimated_hr_bpm"],
                    "valid_beat_percentage": bp_res["valid_beat_percentage"],
                    "ambiguous_beats_count": bp_res["ambiguous_beats_count"],
                    "sbp_mean": bp_res["sbp_mean"],
                    "sbp_std": bp_res["sbp_std"],
                    "dbp_mean": bp_res["dbp_mean"],
                    "dbp_std": bp_res["dbp_std"],
                    "map_mean": bp_res["map_mean"],
                    "is_abp_beat_valid": True,
                })
            else:
                abp_beat_diag["ambiguous_beats_count"] = bp_res.get("ambiguous_beats_count", 0)
        except Exception:
            pass

    return {
        "record_id": record["record_id"],
        "part_id": record["part_id"],
        "part_file": record.get("part_file", ""),
        "record_index": record["record_index"],
        "global_record_index": record["global_record_index"],
        "sampling_frequency": record["sampling_frequency"],
        "num_samples": record["num_samples"],
        "duration_seconds": record["duration_seconds"],
        "structural_valid": structural_valid,
        "ppg_valid": ppg_valid,
        "abp_valid": abp_valid,
        "paired_valid": paired_valid,
        "quality_status": quality_status,
        "ppg_rejection_reason": ppg_reason,
        "abp_rejection_reason": abp_reason,
        # PPG Diagnostics
        "ppg_mean": ppg_diag["ppg_mean"],
        "ppg_std": ppg_diag["ppg_std"],
        "ppg_ptp": ppg_diag["ppg_ptp"],
        "ppg_clipping_ratio": ppg_diag["ppg_clipping_ratio"],
        "ppg_max_jump": ppg_diag["ppg_max_jump"],
        # ABP Continuous Waveform Diagnostics
        "abp_mean": abp_diag["abp_mean"],
        "abp_std": abp_diag["abp_std"],
        "abp_min": abp_diag["abp_min"],
        "abp_max": abp_diag["abp_max"],
        "abp_p05": abp_diag["abp_p05"],
        "abp_p95": abp_diag["abp_p95"],
        # ABP Beat-by-Beat Physiological Targets & Diagnostics
        "abp_num_detected_beats": abp_beat_diag["num_detected_beats"],
        "abp_estimated_hr_bpm": abp_beat_diag["estimated_hr_bpm"],
        "abp_valid_beat_pct": abp_beat_diag["valid_beat_percentage"],
        "abp_ambiguous_beats_count": abp_beat_diag["ambiguous_beats_count"],
        "sbp_mean": abp_beat_diag["sbp_mean"],
        "sbp_std": abp_beat_diag["sbp_std"],
        "dbp_mean": abp_beat_diag["dbp_mean"],
        "dbp_std": abp_beat_diag["dbp_std"],
        "map_mean": abp_beat_diag["map_mean"],
        "is_abp_beat_valid": abp_beat_diag["is_abp_beat_valid"],
    }
