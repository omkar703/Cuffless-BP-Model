"""
Phase 6B: Live Engineering Quality Gating Module
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Computes transparent engineering quality metrics for 10-second PPG windows:
- Peak-to-peak amplitude
- Standard deviation
- Clipping fraction
- Zero fraction
- NaN / Inf checks
- Estimated heart rate via systolic peak detection
- Baseline wander / drift indicator

Classifies window into PASS / WARN / REJECT.
Explicitly avoids any arbitrary IR >= 50,000 threshold.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import find_peaks


class WindowQualityAssessor:
    """
    Evaluates engineering signal quality on a 10-second PPG window.
    
    Classification Rules:
    - REJECT:
        * Any NaN or Inf in raw or filtered window
        * Fraction of zero/negative samples > 5%
        * Fraction of clipped/saturated samples > 5% (ADC extremum)
        * Extremely low variance (std < 20 counts, disconnected sensor/flatline)
        * Fewer than 3 valid systolic pulses detected (<20 bpm or gross motion disruption)
    - WARN:
        * Borderline pulse amplitude (ptp < 300 counts)
        * Excessive baseline drift (>15,000 counts change across window)
        * Borderline heart rate (<45 bpm or >175 bpm)
    - PASS:
        * Clean pulsatile morphology with stable heart rate (45–175 bpm) and adequate amplitude.
    """

    def __init__(self, fs: float = 125.0, adc_max: float = 262143.0, adc_min: float = 0.0):
        self.fs = fs
        self.adc_max = adc_max
        self.adc_min = adc_min

    def assess_window(
        self,
        raw_slice: np.ndarray,
        normalized_tensor: np.ndarray,
        window_idx: int,
        start_time_sec: float,
        end_time_sec: float
    ) -> Dict[str, Any]:
        """
        Assess a 10-second window.
        
        Args:
            raw_slice: 1D array of 1250 raw resampled PPG samples.
            normalized_tensor: Array of shape [3, 1250] containing [PPG, VPG, APG] (z-scored).
            window_idx: Window index counter.
            start_time_sec: Window start time in seconds.
            end_time_sec: Window end time in seconds.
            
        Returns:
            Dictionary containing metrics and status (PASS / WARN / REJECT).
        """
        raw = np.asarray(raw_slice, dtype=np.float64)
        ppg_z = normalized_tensor[0]
        
        has_nan = bool(np.isnan(raw).any() or np.isnan(normalized_tensor).any())
        has_inf = bool(np.isinf(raw).any() or np.isinf(normalized_tensor).any())
        
        raw_min = float(np.min(raw))
        raw_max = float(np.max(raw))
        raw_ptp = float(raw_max - raw_min)
        raw_std = float(np.std(raw))
        
        # Clipping: within 1% of top or bottom ADC boundary
        clip_frac = float(np.mean((raw >= self.adc_max * 0.99) | (raw <= self.adc_min + 100.0)))
        zero_frac = float(np.mean(raw <= 0.0))
        
        # Baseline drift: difference between initial 1s and final 1s mean
        n_1s = int(self.fs)
        drift_delta = float(np.abs(np.mean(raw[-n_1s:]) - np.mean(raw[:n_1s])))
        
        # Systolic peak detection on normalized PPG
        peaks, _ = find_peaks(ppg_z, distance=int(0.35 * self.fs), prominence=0.5)
        num_peaks = len(peaks)
        
        if num_peaks >= 3:
            ibi_sec = np.diff(peaks) / self.fs
            hr_bpm = float(60.0 / np.median(ibi_sec))
        else:
            hr_bpm = np.nan
            
        # Classification logic
        if has_nan or has_inf:
            status = "REJECT"
            reason = "NaN or Inf detected in window"
        elif zero_frac > 0.05:
            status = "REJECT"
            reason = f"Zero/negative sample fraction excessive ({zero_frac*100:.1f}%)"
        elif clip_frac > 0.05:
            status = "REJECT"
            reason = f"ADC saturation/clipping excessive ({clip_frac*100:.1f}%)"
        elif raw_std < 20.0:
            status = "REJECT"
            reason = f"Near-zero variance (std={raw_std:.1f} counts, sensor flatline)"
        elif num_peaks < 3:
            status = "REJECT"
            reason = f"Insufficient pulsatile cycles ({num_peaks} peaks detected)"
        elif raw_ptp < 300.0:
            status = "WARN"
            reason = f"Low pulsatile amplitude (ptp={raw_ptp:.0f} counts)"
        elif drift_delta > 15000.0:
            status = "WARN"
            reason = f"High baseline wander/drift ({drift_delta:.0f} counts)"
        elif not np.isnan(hr_bpm) and (hr_bpm < 45.0 or hr_bpm > 175.0):
            status = "WARN"
            reason = f"Borderline heart rate ({hr_bpm:.1f} bpm)"
        else:
            status = "PASS"
            reason = "Valid pulsatile morphology within physiological bounds"
            
        return {
            "window_index": window_idx,
            "start_time_sec": start_time_sec,
            "end_time_sec": end_time_sec,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_ptp": raw_ptp,
            "raw_std": raw_std,
            "clip_fraction": clip_frac,
            "zero_fraction": zero_frac,
            "baseline_drift_delta": drift_delta,
            "detected_peaks": num_peaks,
            "estimated_hr_bpm": hr_bpm,
            "qc_status": status,
            "qc_reason": reason,
        }
