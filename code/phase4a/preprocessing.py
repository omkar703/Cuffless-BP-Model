"""
Phase 4A: PPG Preprocessing Pipeline.

Implements:
  - Offline zero-phase Butterworth bandpass filter (filtfilt)
  - VPG = first derivative of filtered PPG (np.gradient)
  - APG = second derivative of filtered PPG
  - Per-window z-score normalization (zero information leakage)

IMPORTANT — Offline Research Preprocessing:
    This module uses scipy.signal.filtfilt for zero-phase (non-causal) filtering.
    filtfilt processes the signal forward AND backward, requiring access to the
    ENTIRE signal window before filtering. This is suitable ONLY for offline
    research and CANNOT be deployed on real-time streaming hardware (e.g., ESP32).
    The final embedded deployment must use a causal IIR or FIR filter instead.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple

# ============================================================================
# Filter Constants (matching Phase 1/2 research convention)
# ============================================================================
FS: float = 125.0
LOWCUT: float = 0.5
HIGHCUT: float = 8.0
FILTER_ORDER: int = 3
WINDOW_SAMPLES: int = 1250
ZSCORE_EPS: float = 1e-8


def bandpass_filter(
    ppg: np.ndarray,
    fs: float = FS,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Apply offline zero-phase Butterworth bandpass filter to a PPG window.

    OFFLINE RESEARCH FILTER (non-causal):
        Uses scipy.signal.filtfilt (forward+backward pass). This requires the
        entire signal window and cannot be used in real-time streaming.
        Effective filter order is 2*order (6th order) due to the double pass.

    Args:
        ppg:     Raw PPG signal array of shape (N,), float64.
        fs:      Sampling frequency in Hz. Default: 125.0 Hz.
        lowcut:  Low cutoff frequency in Hz. Default: 0.5 Hz.
        highcut: High cutoff frequency in Hz. Default: 8.0 Hz.
        order:   Butterworth filter order. Default: 3.

    Returns:
        Filtered PPG signal of same shape as input.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, ppg)


def compute_vpg(ppg_filtered: np.ndarray, dt: float = 1.0 / FS) -> np.ndarray:
    """
    Compute the Velocity PhotoPlethysmoGraph (VPG) as the first time derivative
    of the filtered PPG signal using central finite differences (np.gradient).

    Args:
        ppg_filtered: Bandpass-filtered PPG window of shape (N,), float64.
        dt:           Sampling interval in seconds. Default: 1/125 = 0.008 s.

    Returns:
        VPG signal of same shape as input.
    """
    return np.gradient(ppg_filtered, dt)


def compute_apg(vpg: np.ndarray, dt: float = 1.0 / FS) -> np.ndarray:
    """
    Compute the Acceleration PhotoPlethysmoGraph (APG) as the second time
    derivative (derivative of VPG).

    Args:
        vpg: First derivative (VPG) signal of shape (N,), float64.
        dt:  Sampling interval in seconds. Default: 1/125 = 0.008 s.

    Returns:
        APG signal of same shape as input.
    """
    return np.gradient(vpg, dt)


def per_window_zscore(
    channel: np.ndarray,
    eps: float = ZSCORE_EPS,
) -> np.ndarray:
    """
    Apply per-window z-score normalization to a single channel.

    Formula:
        x_norm = (x - mean(x)) / (std(x) + eps)

    Zero-leakage guarantee:
        Only the mean and std of the CURRENT window are used.

    Args:
        channel: 1D numpy array of shape (N,), float64.
        eps:     Small constant for numerical stability. Default: 1e-8.

    Returns:
        Normalized channel of same shape, float64.
    """
    mu = np.mean(channel)
    sigma = np.std(channel)
    return (channel - mu) / (sigma + eps)


def preprocess_window(
    raw_ppg: np.ndarray,
    fs: float = FS,
    use_zscore: bool = True,
    n_channels: int = 3,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single 10-second PPG window.

    Steps:
        1. Bandpass filter: raw_ppg -> filtered_ppg  (offline filtfilt)
        2. Derivative: filtered_ppg -> VPG (np.gradient)
        3. Derivative: VPG -> APG (np.gradient)
        4. Optional per-window z-score normalization of each channel
        5. Stack channels -> [n_channels, 1250]

    Args:
        raw_ppg:    Raw PPG signal of shape (1250,), float64.
        fs:         Sampling frequency. Default: 125 Hz.
        use_zscore: If True, apply per-window z-score to each channel.
        n_channels: 1 for PPG-only (Model A), 3 for PPG+VPG+APG (Model B).

    Returns:
        np.ndarray of shape (n_channels, 1250), float32.

    Raises:
        ValueError: If raw_ppg has wrong shape or contains NaN/Inf.
    """
    raw_ppg = np.asarray(raw_ppg, dtype=np.float64)
    if raw_ppg.ndim != 1 or len(raw_ppg) != WINDOW_SAMPLES:
        raise ValueError(
            f"Expected 1D PPG of length {WINDOW_SAMPLES}, got shape {raw_ppg.shape}"
        )
    if not np.isfinite(raw_ppg).all():
        raise ValueError("Raw PPG contains NaN or Inf values.")

    dt = 1.0 / fs

    # Step 1: Bandpass filter
    ppg_filt = bandpass_filter(raw_ppg, fs=fs)

    if n_channels == 1:
        ch_ppg = per_window_zscore(ppg_filt) if use_zscore else ppg_filt
        return np.stack([ch_ppg], axis=0).astype(np.float32)

    # Steps 2-3: Derivatives
    vpg = compute_vpg(ppg_filt, dt=dt)
    apg = compute_apg(vpg, dt=dt)

    # Step 4: Per-window z-score normalization
    if use_zscore:
        ppg_filt = per_window_zscore(ppg_filt)
        vpg = per_window_zscore(vpg)
        apg = per_window_zscore(apg)

    # Step 5: Stack channels [3, 1250]
    tensor = np.stack([ppg_filt, vpg, apg], axis=0)  # (3, 1250)
    return tensor.astype(np.float32)
