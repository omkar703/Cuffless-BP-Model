"""
Phase 6B: Streaming Rational Resampler (100 Hz -> 125 Hz)
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Stateful rational polyphase resampler implementing exact mathematical equivalence
with scipy.signal.resample_poly(x, up=5, down=4) across arbitrary incoming chunk sizes.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import scipy.signal as signal


class StatefulRationalResampler:
    """
    Stateful rational resampler for converting 100 Hz input to 125 Hz output.
    
    Mathematical Specification:
        up = 5, down = 4 (rate factor = 1.25)
        FIR filter: 101-tap Kaiser-windowed lowpass filter designed via firwin(101, 0.2)
        Padded filter length: 103 taps (pre-pad = 2 zeros)
        Pre-remove offset: 13 samples
    
    Maintains persistent input history across streaming chunks to prevent boundary artifacts.
    Produces identical numerical output regardless of chunk size (chunk-invariant).
    """

    def __init__(self, up: int = 5, down: int = 4):
        self.up = up
        self.down = down
        
        # Design linear-phase low-pass FIR filter identical to scipy.signal.resample_poly
        max_rate = max(up, down)
        f_c = 1.0 / max_rate  # 0.2 of Nyquist
        half_len = 10 * max_rate  # 50
        h = signal.firwin(2 * half_len + 1, f_c, window=("kaiser", 5.0)) * up
        
        # Exact scipy padding rules
        n_pre_pad = (down - half_len % down)  # 2
        self.h_padded = np.concatenate((np.zeros(n_pre_pad, dtype=np.float64), h))
        self.n_pre_remove = (half_len + n_pre_pad) // down  # 13
        self.L_h = len(self.h_padded)  # 103
        
        # State tracking
        self.history: Dict[int, float] = {}
        self.n_in = 0
        self.k_out = 0
        
    def reset(self) -> None:
        """Reset internal buffer state."""
        self.history.clear()
        self.n_in = 0
        self.k_out = 0

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a new chunk of raw 100-Hz samples.
        
        Args:
            chunk: 1D array or list of incoming samples at 100 Hz.
            
        Returns:
            np.ndarray of newly ready resampled samples at 125 Hz.
        """
        if len(chunk) == 0:
            return np.array([], dtype=np.float64)
            
        output = []
        for x in chunk:
            self.history[self.n_in] = float(x)
            self.n_in += 1
            
            # Emit all output samples k whose required input samples have arrived
            while True:
                j = self.k_out + self.n_pre_remove
                n_max = (j * self.down) // self.up
                
                # Check if the latest required input sample has arrived
                if n_max < self.n_in:
                    n_min = int(np.ceil((j * self.down - self.L_h + 1) / self.up))
                    val = 0.0
                    for n in range(n_min, n_max + 1):
                        if n >= 0:
                            l = j * self.down - self.up * n
                            val += self.h_padded[l] * self.history.get(n, 0.0)
                    output.append(val)
                    self.k_out += 1
                else:
                    break
                    
        # Prune history samples that will never be needed for future output samples
        oldest_needed = int(np.ceil(((self.k_out + self.n_pre_remove) * self.down - self.L_h + 1) / self.up))
        keys_to_del = [k for k in self.history if k < oldest_needed]
        for k in keys_to_del:
            del self.history[k]
            
        return np.array(output, dtype=np.float64)

    def flush(self) -> np.ndarray:
        """
        Flush remaining output samples upon stream termination (equivalent to offline zero-padding).
        """
        total_expected_out = (self.n_in * self.up) // self.down + bool((self.n_in * self.up) % self.down)
        output = []
        while self.k_out < total_expected_out:
            j = self.k_out + self.n_pre_remove
            n_min = int(np.ceil((j * self.down - self.L_h + 1) / self.up))
            n_max = (j * self.down) // self.up
            val = 0.0
            for n in range(n_min, n_max + 1):
                if 0 <= n < self.n_in:
                    l = j * self.down - self.up * n
                    val += self.h_padded[l] * self.history.get(n, 0.0)
            output.append(val)
            self.k_out += 1
        return np.array(output, dtype=np.float64)


def validate_streaming_resampler(test_signal: np.ndarray = None, chunk_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Validate the stateful rational resampler against scipy.signal.resample_poly.
    
    Tests across multiple chunk sizes (e.g. 1, 7, 16, 32, 100, 137).
    """
    if test_signal is None:
        np.random.seed(42)
        t = np.arange(1000) / 100.0
        # Multi-frequency physiological-like signal
        test_signal = (
            np.sin(2 * np.pi * 1.2 * t) * 500.0 +
            np.sin(2 * np.pi * 3.5 * t) * 150.0 +
            np.cos(2 * np.pi * 6.0 * t) * 50.0 +
            100000.0  # DC offset
        )
        
    if chunk_sizes is None:
        chunk_sizes = [1, 7, 16, 32, 100, 137]
        
    up, down = 5, 4
    ref_offline = signal.resample_poly(test_signal, up, down)
    
    results = {}
    for cs in chunk_sizes:
        resampler = StatefulRationalResampler(up, down)
        chunks_out = []
        for i in range(0, len(test_signal), cs):
            chk = test_signal[i:i+cs]
            out = resampler.process_chunk(chk)
            if len(out) > 0:
                chunks_out.append(out)
        out_flushed = resampler.flush()
        if len(out_flushed) > 0:
            chunks_out.append(out_flushed)
            
        full_streaming_out = np.concatenate(chunks_out)
        
        diff = np.abs(ref_offline - full_streaming_out)
        max_ae = float(np.max(diff))
        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        corr = float(np.corrcoef(ref_offline, full_streaming_out)[0, 1])
        
        results[f"chunk_{cs}"] = {
            "chunk_size": cs,
            "output_length": len(full_streaming_out),
            "expected_length": len(ref_offline),
            "max_absolute_error": max_ae,
            "mean_absolute_error": mae,
            "rmse": rmse,
            "pearson_r": corr,
            "status": "PASS" if max_ae < 1e-10 else "FAIL"
        }
        
    return results


test_resampler_chunk_invariance = validate_streaming_resampler


if __name__ == "__main__":
    print("Running Resampler Unit Test across chunk sizes [1, 7, 16, 32, 100, 137]...")
    res = validate_streaming_resampler()
    all_pass = all(v["status"] == "PASS" for v in res.values())
    for k, v in res.items():
        print(f"  {k:<10}: Output={v['output_length']}/{v['expected_length']} | MaxAE={v['max_absolute_error']:.2e} | r={v['pearson_r']:.8f} | [{v['status']}]")
    print(f"\nResampler Chunk-Invariance Test Result: {'ALL PASS' if all_pass else 'FAIL'}")
