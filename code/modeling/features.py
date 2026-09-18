"""
Feature Engineering Engine for Phase 3A PPG-Only Baseline Modeling.

Implements three physiological representation branches:
- Branch A: Amplitude-Preserving PPG (0.5–8.0 Hz bandpass, no amplitude normalization)
- Branch B: Normalized Morphology PPG (Per-window Z-score normalized)
- Branch C: Combined Representation (Branch A + Branch B)

Also organizes features into cumulative ablation groups:
- Group A: Basic waveform statistics (11 features)
- Group B: Pulse rate & timing (6 features)
- Group C: Pulse morphology (7 features)
- Group D: First derivative / VPG (5 features)
- Group E: Second derivative / APG (4 features)
- Group F: Spectral properties (3 features)

STRICT INTEGRITY ASSERTION:
Rejects any feature name containing forbidden substrings:
['abp', 'ecg', 'sbp', 'dbp', 'map', 'ptt', 'pat', 'calibration'].
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis

from config.config import (
    SAMPLING_RATE,
    PPG_BANDPASS_LOWCUT,
    PPG_BANDPASS_HIGHCUT,
    PPG_FILTER_ORDER,
)
from utils.logging_utils import setup_logger

logger = setup_logger("feature_engineering")

# Forbidden substrings for PPG-only calibration-free baseline
FORBIDDEN_SUBSTRINGS = ["abp", "ecg", "sbp", "dbp", "map", "ptt", "pat", "calibration"]

# Precompute filter coefficients
_nyq = 0.5 * SAMPLING_RATE
_b_band, _a_band = butter(
    PPG_FILTER_ORDER,
    [PPG_BANDPASS_LOWCUT / _nyq, PPG_BANDPASS_HIGHCUT / _nyq],
    btype="band",
)


def assert_feature_integrity(feature_names: List[str]) -> None:
    """Asserts that no feature name contains forbidden substrings."""
    for feat in feature_names:
        lower_feat = feat.lower()
        for forbidden in FORBIDDEN_SUBSTRINGS:
            assert (
                forbidden not in lower_feat
            ), f"INTEGRITY VIOLATION: Feature '{feat}' contains forbidden term '{forbidden}'!"


# Explicit Group Definitions
FEATURE_GROUPS = {
    "A_basic": [
        "mean_a",
        "std_a",
        "var_a",
        "rms_a",
        "ptp_a",
        "min_a",
        "max_a",
        "median_a",
        "iqr_a",
        "skew_b",
        "kurt_b",
    ],
    "B_timing": [
        "pulse_count",
        "hr_bpm",
        "ibi_mean",
        "ibi_median",
        "ibi_std",
        "ibi_cv",
    ],
    "C_morphology": [
        "pulse_amp_median_a",
        "pulse_amp_iqr_a",
        "pulse_amp_cv_a",
        "pulse_width_median",
        "rise_time_median",
        "decay_time_median",
        "max_upstroke_slope_median",
    ],
    "D_vpg": [
        "vpg_max",
        "vpg_min",
        "vpg_std",
        "vpg_rms",
        "vpg_max_upstroke",
    ],
    "E_apg": [
        "apg_max",
        "apg_min",
        "apg_std",
        "apg_b_to_a_ratio",
    ],
    "F_spectral": [
        "dominant_freq",
        "pulse_band_power",
        "spectral_entropy",
    ],
}

# Branch Definitions
BRANCH_A_FEATURES = [
    "mean_a",
    "std_a",
    "var_a",
    "rms_a",
    "ptp_a",
    "min_a",
    "max_a",
    "median_a",
    "iqr_a",
    "p10_a",
    "p90_a",
    "pulse_amp_median_a",
    "pulse_amp_iqr_a",
    "pulse_amp_cv_a",
]

BRANCH_B_FEATURES = [
    "skew_b",
    "kurt_b",
    "pulse_count",
    "hr_bpm",
    "ibi_mean",
    "ibi_median",
    "ibi_std",
    "ibi_cv",
    "pulse_width_median",
    "rise_time_median",
    "decay_time_median",
    "max_upstroke_slope_median",
    "max_downstroke_slope_median",
    "pulse_area_median",
    "vpg_max",
    "vpg_min",
    "vpg_std",
    "vpg_rms",
    "vpg_max_upstroke",
    "apg_max",
    "apg_min",
    "apg_std",
    "apg_b_to_a_ratio",
    "dominant_freq",
    "pulse_band_power",
    "spectral_entropy",
]

# Combined Representation: All distinct features across Branch A and Branch B
BRANCH_C_FEATURES = sorted(list(set(BRANCH_A_FEATURES + BRANCH_B_FEATURES)))

# Run static assertion on all defined feature sets
assert_feature_integrity(BRANCH_A_FEATURES)
assert_feature_integrity(BRANCH_B_FEATURES)
assert_feature_integrity(BRANCH_C_FEATURES)
for grp_name, grp_feats in FEATURE_GROUPS.items():
    assert_feature_integrity(grp_feats)


def extract_features_from_ppg_window(ppg_raw: np.ndarray) -> Dict[str, float]:
    """
    Extracts complete set of handcrafted PPG features from a 1250-sample window.
    
    Returns a dictionary of feature_name -> float value.
    Handles NaN/Inf gracefully by substituting np.nan.
    """
    if len(ppg_raw) < 100:
        return {feat: np.nan for feat in BRANCH_C_FEATURES}

    # -------------------------------------------------------------------------
    # 1. Branch A: Amplitude-Preserving Filtered PPG
    # -------------------------------------------------------------------------
    try:
        ppg_filt = filtfilt(_b_band, _a_band, ppg_raw)
    except Exception:
        ppg_filt = ppg_raw.copy()

    mean_a = float(np.mean(ppg_filt))
    std_a = float(np.std(ppg_filt))
    var_a = float(np.var(ppg_filt))
    rms_a = float(np.sqrt(np.mean(ppg_filt**2)))
    ptp_a = float(np.ptp(ppg_filt))
    min_a = float(np.min(ppg_filt))
    max_a = float(np.max(ppg_filt))

    p10_a, q25, q50, q75, p90_a = np.percentile(ppg_filt, [10, 25, 50, 75, 90])
    median_a = float(q50)
    iqr_a = float(q75 - q25)

    # -------------------------------------------------------------------------
    # 2. Branch B: Normalized Morphology PPG (Z-score)
    # -------------------------------------------------------------------------
    if std_a > 1e-6:
        ppg_norm = (ppg_filt - mean_a) / std_a
    else:
        ppg_norm = ppg_filt - mean_a

    skew_b = float(skew(ppg_norm)) if not np.isnan(std_a) else 0.0
    kurt_b = float(kurtosis(ppg_norm)) if not np.isnan(std_a) else 0.0

    # -------------------------------------------------------------------------
    # 3. Pulse Peak & Foot Detection (Timing & Heart Rate)
    # -------------------------------------------------------------------------
    # Minimum distance 40 samples (0.32 s = 187 bpm max plausible HR)
    peaks, _ = find_peaks(ppg_norm, distance=40, prominence=0.3)
    pulse_count = len(peaks)

    if pulse_count >= 2:
        ibis = np.diff(peaks) / float(SAMPLING_RATE)
        hr_bpm = float(60.0 / np.mean(ibis))
        ibi_mean = float(np.mean(ibis))
        ibi_median = float(np.median(ibis))
        ibi_std = float(np.std(ibis))
        ibi_cv = float(ibi_std / (ibi_mean + 1e-6))
    else:
        hr_bpm = np.nan
        ibi_mean = np.nan
        ibi_median = np.nan
        ibi_std = np.nan
        ibi_cv = np.nan

    # -------------------------------------------------------------------------
    # 4. Pulse Morphology (Amplitudes, Rise/Decay times, Slopes, Area)
    # -------------------------------------------------------------------------
    troughs, _ = find_peaks(-ppg_norm, distance=35, prominence=0.2)

    amps_a = []
    rise_times = []
    decay_times = []
    up_slopes = []
    down_slopes = []
    areas = []

    if pulse_count >= 2 and len(troughs) >= 2:
        for p_idx in peaks:
            # find trough preceding peak
            tr_before = [tr for tr in troughs if tr < p_idx]
            # find trough following peak
            tr_after = [tr for tr in troughs if tr > p_idx]

            if tr_before and tr_after:
                tb = tr_before[-1]
                ta = tr_after[0]
                # Amplitude in Branch A (unnormalized filtered PPG)
                amp_val = ppg_filt[p_idx] - ppg_filt[tb]
                if amp_val > 0:
                    amps_a.append(amp_val)

                # Timing in seconds
                dt_rise = (p_idx - tb) / float(SAMPLING_RATE)
                dt_decay = (ta - p_idx) / float(SAMPLING_RATE)

                if dt_rise > 0 and dt_decay > 0:
                    rise_times.append(dt_rise)
                    decay_times.append(dt_decay)
                    up_slopes.append((ppg_norm[p_idx] - ppg_norm[tb]) / dt_rise)
                    down_slopes.append((ppg_norm[p_idx] - ppg_norm[ta]) / dt_decay)

                    # Pulse area under normalized curve from tb to ta
                    pulse_segment = ppg_norm[tb:ta] - min(ppg_norm[tb], ppg_norm[ta])
                    areas.append(float(np.trapz(pulse_segment, dx=1.0 / SAMPLING_RATE)))

    if amps_a:
        pulse_amp_median_a = float(np.median(amps_a))
        pulse_amp_iqr_a = float(np.percentile(amps_a, 75) - np.percentile(amps_a, 25))
        pulse_amp_cv_a = float(np.std(amps_a) / (np.mean(amps_a) + 1e-6))
    else:
        pulse_amp_median_a = ptp_a
        pulse_amp_iqr_a = 0.0
        pulse_amp_cv_a = 0.0

    if rise_times and decay_times:
        pulse_width_median = float(np.median(rise_times) + np.median(decay_times))
        rise_time_median = float(np.median(rise_times))
        decay_time_median = float(np.median(decay_times))
        max_upstroke_slope_median = float(np.median(up_slopes))
        max_downstroke_slope_median = float(np.median(down_slopes))
        pulse_area_median = float(np.median(areas)) if areas else 0.0
    else:
        pulse_width_median = np.nan
        rise_time_median = np.nan
        decay_time_median = np.nan
        max_upstroke_slope_median = np.nan
        max_downstroke_slope_median = np.nan
        pulse_area_median = np.nan

    # -------------------------------------------------------------------------
    # 5. First Derivative (VPG) & Second Derivative (APG)
    # -------------------------------------------------------------------------
    dt = 1.0 / float(SAMPLING_RATE)
    vpg = np.gradient(ppg_norm, dt)
    apg = np.gradient(vpg, dt)

    vpg_max = float(np.max(vpg))
    vpg_min = float(np.min(vpg))
    vpg_std = float(np.std(vpg))
    vpg_rms = float(np.sqrt(np.mean(vpg**2)))
    vpg_max_upstroke = float(np.percentile(vpg, 95))

    apg_max = float(np.max(apg))
    apg_min = float(np.min(apg))
    apg_std = float(np.std(apg))

    # APG b/a ratio estimation
    apg_peaks, _ = find_peaks(apg, distance=40, prominence=5.0)
    apg_troughs, _ = find_peaks(-apg, distance=40, prominence=5.0)
    if len(apg_peaks) > 0 and len(apg_troughs) > 0:
        a_val = float(np.median(apg[apg_peaks]))
        b_val = float(np.median(np.abs(apg[apg_troughs])))
        apg_b_to_a_ratio = float(b_val / (a_val + 1e-6))
    else:
        apg_b_to_a_ratio = np.nan

    # -------------------------------------------------------------------------
    # 6. Spectral Features (Welch PSD)
    # -------------------------------------------------------------------------
    f, psd = welch(ppg_norm, fs=SAMPLING_RATE, nperseg=min(256, len(ppg_norm)))
    dom_freq = float(f[np.argmax(psd)])
    band_mask = (f >= 0.5) & (f <= 3.5)
    total_power = float(np.sum(psd)) + 1e-12
    pulse_band_power = float(np.sum(psd[band_mask]) / total_power)

    norm_psd = psd / total_power
    spectral_entropy = float(-np.sum(norm_psd * np.log(norm_psd + 1e-12)))

    # Assemble dictionary
    features_dict = {
        # Branch A
        "mean_a": mean_a,
        "std_a": std_a,
        "var_a": var_a,
        "rms_a": rms_a,
        "ptp_a": ptp_a,
        "min_a": min_a,
        "max_a": max_a,
        "median_a": median_a,
        "iqr_a": iqr_a,
        "p10_a": float(p10_a),
        "p90_a": float(p90_a),
        "pulse_amp_median_a": pulse_amp_median_a,
        "pulse_amp_iqr_a": pulse_amp_iqr_a,
        "pulse_amp_cv_a": pulse_amp_cv_a,
        # Branch B
        "skew_b": skew_b,
        "kurt_b": kurt_b,
        "pulse_count": float(pulse_count),
        "hr_bpm": hr_bpm,
        "ibi_mean": ibi_mean,
        "ibi_median": ibi_median,
        "ibi_std": ibi_std,
        "ibi_cv": ibi_cv,
        "pulse_width_median": pulse_width_median,
        "rise_time_median": rise_time_median,
        "decay_time_median": decay_time_median,
        "max_upstroke_slope_median": max_upstroke_slope_median,
        "max_downstroke_slope_median": max_downstroke_slope_median,
        "pulse_area_median": pulse_area_median,
        "vpg_max": vpg_max,
        "vpg_min": vpg_min,
        "vpg_std": vpg_std,
        "vpg_rms": vpg_rms,
        "vpg_max_upstroke": vpg_max_upstroke,
        "apg_max": apg_max,
        "apg_min": apg_min,
        "apg_std": apg_std,
        "apg_b_to_a_ratio": apg_b_to_a_ratio,
        "dominant_freq": dom_freq,
        "pulse_band_power": pulse_band_power,
        "spectral_entropy": spectral_entropy,
    }

    return features_dict
