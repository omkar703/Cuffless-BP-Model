"""
Window Generation Engine for Cuffless Blood Pressure Estimation.

Extracts deterministic, non-overlapping (or configurable overlap) fixed-duration
windows (default: 10 seconds = 1250 samples at 125 Hz) from continuous PPG and ABP
records, preserving parent record provenance for leakage-safe partitioning.

CRITICAL ARCHITECTURAL RULES:
1. Records shorter than the window duration (< 1250 samples) yield 0 windows,
   but remain preserved in the master record-level QC manifest.
2. Every window retains its parent `record_id` and has a unique deterministic
   `window_id` formatted as: `{record_id}_win_{window_index:03d}`.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from config.config import (
    SAMPLING_RATE,
    WINDOW_SECONDS,
    WINDOW_SAMPLES,
    PRIMARY_OVERLAP,
)
from utils.logging_utils import setup_logger

logger = setup_logger("windowing")


def calculate_expected_window_count(
    num_samples: int,
    window_samples: int = WINDOW_SAMPLES,
    overlap: float = PRIMARY_OVERLAP,
) -> int:
    """
    Calculates the exact number of windows that will be generated for a given
    signal length.
    
    If num_samples < window_samples, returns 0.
    """
    if num_samples < window_samples:
        return 0
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError(f"Overlap must be in [0.0, 1.0), got {overlap}")
    
    step_samples = int(round(window_samples * (1.0 - overlap)))
    step_samples = max(1, step_samples)
    return (num_samples - window_samples) // step_samples + 1


def slice_record_into_windows(
    record: Dict[str, Any],
    window_samples: int = WINDOW_SAMPLES,
    overlap: float = PRIMARY_OVERLAP,
    sampling_rate: int = SAMPLING_RATE,
) -> List[Dict[str, Any]]:
    """
    Slices a continuous record into discrete, fixed-duration windows.
    
    Parameters:
        record: Dictionary yielded by data_loader containing at least:
            - record_id: str
            - part_id: str
            - record_index: int
            - ppg: 1D np.ndarray
            - abp: 1D np.ndarray
            - num_samples: int
            - sampling_frequency: int
        window_samples: Window length in samples (default: 1250 for 10 s @ 125 Hz)
        overlap: Overlap fraction in [0.0, 1.0) (default: 0.0 for independent windows)
        sampling_rate: Acquisition frequency (default: 125 Hz)
        
    Returns:
        List of window dictionaries. Returns [] if record length < window_samples.
    """
    num_samples = record.get("num_samples", 0)
    ppg = record.get("ppg")
    abp = record.get("abp")

    # Gracefully handle short records (< 1250 samples): return 0 windows
    if num_samples < window_samples or ppg is None or len(ppg) < window_samples:
        return []

    fs = record.get("sampling_frequency", sampling_rate)
    step_samples = int(round(window_samples * (1.0 - overlap)))
    step_samples = max(1, step_samples)
    n_windows = (num_samples - window_samples) // step_samples + 1

    record_id = record["record_id"]
    part_id = record["part_id"]
    record_index = record.get("record_index", 0)

    windows = []
    for win_idx in range(n_windows):
        start_samp = win_idx * step_samples
        end_samp = start_samp + window_samples

        ppg_slice = ppg[start_samp:end_samp]
        abp_slice = abp[start_samp:end_samp] if abp is not None and len(abp) >= end_samp else None

        win_id = f"{record_id}_win_{win_idx:03d}"
        start_sec = float(start_samp / fs)
        end_sec = float(end_samp / fs)

        window_dict = {
            "record_id": record_id,
            "part_id": part_id,
            "record_index": record_index,
            "window_id": win_id,
            "window_index": win_idx,
            "start_sample": start_samp,
            "end_sample": end_samp,
            "start_time_seconds": start_sec,
            "end_time_seconds": end_sec,
            "sampling_frequency": fs,
            "window_samples": window_samples,
            "ppg": ppg_slice,
            "abp": abp_slice,
        }
        windows.append(window_dict)

    return windows
