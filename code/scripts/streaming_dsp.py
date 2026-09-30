"""
Phase 6B: Streaming Causal DSP Pipeline
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Implements stateful causal bandpass filtering (0.5–8.0 Hz) and causal backward
derivatives (VPG, APG) operating incrementally across streaming chunks with zero future access.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.signal as signal

from streaming_resampler import StatefulRationalResampler


class StreamingCausalFilter:
    """
    Stateful 3rd-order Butterworth bandpass filter (0.5–8.0 Hz at Fs = 125 Hz).
    
    Uses Second-Order Sections (SOS) with persistent state zi.
    No filtfilt (strictly causal, zero future access).
    """

    def __init__(self, fs: float = 125.0, lowcut: float = 0.5, highcut: float = 8.0, order: int = 3):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        
        nyq = 0.5 * fs
        self.sos = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
        self.zi_base = signal.sosfilt_zi(self.sos)
        self.zi: Optional[np.ndarray] = None
        self.initialized = False
        self.total_samples_filtered = 0

    def reset(self) -> None:
        """Reset internal filter state."""
        self.zi = None
        self.initialized = False
        self.total_samples_filtered = 0

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of 125-Hz samples.
        
        Args:
            chunk: 1D array of incoming samples.
            
        Returns:
            1D array of filtered samples of identical length.
        """
        if len(chunk) == 0:
            return np.array([], dtype=np.float64)
            
        chunk = np.asarray(chunk, dtype=np.float64)
        
        if not self.initialized:
            # Initialize state with steady-state response for initial sample
            self.zi = self.zi_base * chunk[0]
            self.initialized = True
            
        y, self.zi = signal.sosfilt(self.sos, chunk, zi=self.zi)
        self.total_samples_filtered += len(y)
        return y


class StreamingCausalDerivatives:
    """
    Causal backward finite-difference derivatives:
        VPG[n] = (PPG[n] - PPG[n-1]) / dt
        APG[n] = (VPG[n] - VPG[n-1]) / dt
    
    where dt = 1.0 / fs (8 ms at 125 Hz).
    Maintains previous PPG and previous VPG samples across chunks.
    No np.gradient (strictly causal, zero future access).
    """

    def __init__(self, fs: float = 125.0):
        self.fs = fs
        self.dt = 1.0 / fs
        self.prev_ppg: Optional[float] = None
        self.prev_vpg: Optional[float] = None
        self.total_samples_processed = 0

    def reset(self) -> None:
        """Reset derivative boundary states."""
        self.prev_ppg = None
        self.prev_vpg = None
        self.total_samples_processed = 0

    def process_chunk(self, ppg_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute VPG and APG for a chunk of filtered PPG samples.
        
        Args:
            ppg_chunk: 1D array of filtered PPG samples.
            
        Returns:
            Tuple of (vpg_chunk, apg_chunk) arrays.
        """
        if len(ppg_chunk) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            
        vpg_out = []
        apg_out = []
        
        for p in ppg_chunk:
            p_val = float(p)
            if self.prev_ppg is None:
                v = 0.0  # Initial sample boundary condition
            else:
                v = (p_val - self.prev_ppg) / self.dt
            self.prev_ppg = p_val
            
            if self.prev_vpg is None:
                a = 0.0  # Initial sample boundary condition
            else:
                a = (v - self.prev_vpg) / self.dt
            self.prev_vpg = v
            
            vpg_out.append(v)
            apg_out.append(a)
            
        self.total_samples_processed += len(ppg_chunk)
        return np.array(vpg_out, dtype=np.float64), np.array(apg_out, dtype=np.float64)


class StreamingDSPPipeline:
    """
    Complete end-to-end streaming DSP pipeline:
        Raw 100 Hz IR samples
            ↓ (Stateful Rational Resampler)
        125 Hz resampled stream
            ↓ (Stateful Causal 0.5–8 Hz Filter)
        Filtered PPG stream
            ↓ (Stateful Backward Differences)
        VPG & APG streams
            ↓ (Window Accumulation)
        10-Second Windows (1250 samples @ 125 Hz)
            ↓ (Per-Window Z-Score Normalization)
        Ready Modeling Tensors: [3, 1250] float32
    """

    def __init__(self, window_samples: int = 1250, fs_out: float = 125.0):
        self.window_samples = window_samples
        self.fs_out = fs_out
        
        self.resampler = StatefulRationalResampler(up=5, down=4)
        self.filter = StreamingCausalFilter(fs=fs_out, lowcut=0.5, highcut=8.0, order=3)
        self.derivatives = StreamingCausalDerivatives(fs=fs_out)
        
        # Buffer accumulating processed samples for the current window
        self.buf_ppg: List[float] = []
        self.buf_vpg: List[float] = []
        self.buf_apg: List[float] = []
        self.buf_raw: List[float] = []  # Raw resampled for quality checking
        
        self.completed_windows: List[np.ndarray] = []
        self.raw_window_slices: List[np.ndarray] = []
        self.latest_filtered_chunk: np.ndarray = np.array([], dtype=np.float64)
        self.window_count = 0
        self.total_100hz_in = 0
        self.total_125hz_out = 0

    def reset(self) -> None:
        """Reset all stages and buffers."""
        self.resampler.reset()
        self.filter.reset()
        self.derivatives.reset()
        self.buf_ppg.clear()
        self.buf_vpg.clear()
        self.buf_apg.clear()
        self.buf_raw.clear()
        self.completed_windows.clear()
        self.raw_window_slices.clear()
        self.latest_filtered_chunk = np.array([], dtype=np.float64)
        self.window_count = 0
        self.total_100hz_in = 0
        self.total_125hz_out = 0

    def process_raw_samples(self, raw_100hz_chunk: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Feed raw 100-Hz IR samples and emit any 10-second windows that become complete.
        
        Args:
            raw_100hz_chunk: Array of incoming 100-Hz optical IR samples.
            
        Returns:
            List of tuples: (window_index, normalized_tensor [3, 1250], raw_window_slice [1250])
        """
        if len(raw_100hz_chunk) == 0:
            return []
            
        self.total_100hz_in += len(raw_100hz_chunk)
        
        # Stage 1: Resample 100 Hz -> 125 Hz
        resampled_125 = self.resampler.process_chunk(raw_100hz_chunk)
        if len(resampled_125) == 0:
            return []
            
        self.total_125hz_out += len(resampled_125)
        
        # Stage 2: Causal Filter
        filtered_ppg = self.filter.process_chunk(resampled_125)
        self.latest_filtered_chunk = filtered_ppg
        
        # Stage 3: Causal Derivatives
        vpg, apg = self.derivatives.process_chunk(filtered_ppg)
        
        # Stage 4: Accumulate into current window buffer
        self.buf_ppg.extend(filtered_ppg)
        self.buf_vpg.extend(vpg)
        self.buf_apg.extend(apg)
        self.buf_raw.extend(resampled_125)
        
        new_windows = []
        while len(self.buf_ppg) >= self.window_samples:
            # Pop exactly 1250 samples for the window
            w_ppg = np.array(self.buf_ppg[:self.window_samples], dtype=np.float64)
            w_vpg = np.array(self.buf_vpg[:self.window_samples], dtype=np.float64)
            w_apg = np.array(self.buf_apg[:self.window_samples], dtype=np.float64)
            w_raw = np.array(self.buf_raw[:self.window_samples], dtype=np.float64)
            
            # Slice buffer
            self.buf_ppg = self.buf_ppg[self.window_samples:]
            self.buf_vpg = self.buf_vpg[self.window_samples:]
            self.buf_apg = self.buf_apg[self.window_samples:]
            self.buf_raw = self.buf_raw[self.window_samples:]
            
            # Per-window z-score normalization
            mu_p, sig_p = np.mean(w_ppg), np.std(w_ppg) + 1e-8
            z_ppg = (w_ppg - mu_p) / sig_p
            
            # Causal backward differences on normalized PPG (Phase 6A Bit-Exact Causal Path)
            dt = 1.0 / self.fs_out
            v_causal = np.zeros_like(z_ppg)
            v_causal[1:] = (z_ppg[1:] - z_ppg[:-1]) / dt
            v_causal[0] = 0.0
            mu_v, sig_v = np.mean(v_causal), np.std(v_causal) + 1e-8
            z_vpg = (v_causal - mu_v) / sig_v
            
            a_causal = np.zeros_like(z_vpg)
            a_causal[1:] = (z_vpg[1:] - z_vpg[:-1]) / dt
            a_causal[0] = 0.0
            mu_a, sig_a = np.mean(a_causal), np.std(a_causal) + 1e-8
            z_apg = (a_causal - mu_a) / sig_a
            
            tensor_3x1250 = np.stack([z_ppg, z_vpg, z_apg], axis=0).astype(np.float32)
            
            w_idx = self.window_count
            self.window_count += 1
            self.completed_windows.append(tensor_3x1250)
            self.raw_window_slices.append(w_raw)
            
            new_windows.append((w_idx, tensor_3x1250, w_raw))
            
        return new_windows


def validate_filter_state(test_signal: np.ndarray = None, chunk_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Validate that streaming causal SOS filter matches batch causal SOS filter bit-for-bit.
    """
    if test_signal is None:
        np.random.seed(42)
        t = np.arange(1000) / 125.0
        test_signal = np.sin(2 * np.pi * 1.5 * t) + np.cos(2 * np.pi * 4.0 * t) + 100.0
        
    if chunk_sizes is None:
        chunk_sizes = [1, 7, 16, 32, 100, 137]
        
    # Batch reference
    ref_filter = StreamingCausalFilter()
    batch_out = ref_filter.process_chunk(test_signal)
    
    results = {}
    for cs in chunk_sizes:
        st_filter = StreamingCausalFilter()
        chunks = []
        for i in range(0, len(test_signal), cs):
            c_out = st_filter.process_chunk(test_signal[i:i+cs])
            chunks.append(c_out)
        stream_out = np.concatenate(chunks)
        
        diff = np.max(np.abs(batch_out - stream_out))
        results[f"chunk_{cs}"] = {
            "chunk_size": cs,
            "max_absolute_error": float(diff),
            "status": "PASS" if diff < 1e-12 else "FAIL"
        }
    return results


def validate_derivative_state(test_signal: np.ndarray = None, chunk_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Validate that streaming backward finite differences match batch backward differences.
    """
    if test_signal is None:
        np.random.seed(42)
        t = np.arange(1000) / 125.0
        test_signal = np.sin(2 * np.pi * 1.5 * t)
        
    if chunk_sizes is None:
        chunk_sizes = [1, 7, 16, 32, 100, 137]
        
    # Batch reference
    ref_deriv = StreamingCausalDerivatives()
    batch_v, batch_a = ref_deriv.process_chunk(test_signal)
    
    results = {}
    for cs in chunk_sizes:
        st_deriv = StreamingCausalDerivatives()
        v_chunks, a_chunks = [], []
        for i in range(0, len(test_signal), cs):
            vc, ac = st_deriv.process_chunk(test_signal[i:i+cs])
            v_chunks.append(vc)
            a_chunks.append(ac)
        stream_v = np.concatenate(v_chunks)
        stream_a = np.concatenate(a_chunks)
        
        diff_v = float(np.max(np.abs(batch_v - stream_v)))
        diff_a = float(np.max(np.abs(batch_a - stream_a)))
        results[f"chunk_{cs}"] = {
            "chunk_size": cs,
            "max_abs_err_vpg": diff_v,
            "max_abs_err_apg": diff_a,
            "status": "PASS" if (diff_v < 1e-12 and diff_a < 1e-12) else "FAIL"
        }
    return results


if __name__ == "__main__":
    print("Testing Causal Filter Chunk Invariance...")
    f_res = validate_filter_state()
    f_pass = all(v["status"] == "PASS" for v in f_res.values())
    for k, v in f_res.items():
        print(f"  {k:<10}: MaxAE = {v['max_absolute_error']:.2e} | [{v['status']}]")
    print(f"Filter Chunk Invariance: {'ALL PASS' if f_pass else 'FAIL'}")
    
    print("\nTesting Causal Derivative Chunk Invariance...")
    d_res = validate_derivative_state()
    d_pass = all(v["status"] == "PASS" for v in d_res.values())
    for k, v in d_res.items():
        print(f"  {k:<10}: VPG MaxAE = {v['max_abs_err_vpg']:.2e} | APG MaxAE = {v['max_abs_err_apg']:.2e} | [{v['status']}]")
    print(f"Derivative Chunk Invariance: {'ALL PASS' if d_pass else 'FAIL'}")
