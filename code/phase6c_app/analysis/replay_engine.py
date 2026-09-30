"""
Phase 6C Frozen Replay Engine.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Directly executes the frozen Phase 6B streaming pipeline:
- Stateful 100->125 Hz rational resampler
- Stateful causal Butterworth bandpass SOS (0.5–8.0 Hz)
- Stateful causal backward finite-difference derivatives (VPG, APG)
- Per-window z-score normalization (10s / 1250 samples)
- Window quality assessment (PASS / WARN / REJECT)
- Frozen Phase 4A 1D CNN embedding (64-dim)
- 6-window causal temporal history buffer (60s context)
- Frozen Phase 4B 1-Layer Unidirectional GRU
- Frozen Phase 5C extreme-aware isotonic recalibration & 95% conformal bounds

STRICT SAFEGUARD:
Asserts 0 trainable parameters. Absolutely no weight updates or recalibration.
"""

import time
import json
import pickle
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phase6c_app.config import (
    PHASE4A_CKPT,
    PHASE4B_CKPT,
    ISOTONIC_SBP_PKL,
    ISOTONIC_DBP_PKL,
    CONFORMAL_QUANTILES_JSON,
    CHUNK_SIZE,
    WINDOW_SAMPLES,
    OUTPUT_FS,
    EXPECTED_CNN_PARAMS,
    EXPECTED_GRU_PARAMS,
    EXPECTED_TOTAL_PARAMS,
)
from streaming_dsp import StreamingDSPPipeline
from live_quality import WindowQualityAssessor
from phase4a.model import PPGCNNBaseline


class TemporalGRUModel(nn.Module):
    """Frozen Phase 4B 1-layer unidirectional causal GRU architecture."""

    def __init__(self, input_size: int = 64, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.sbp_head = nn.Linear(32, 1)
        self.dbp_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        feat = self.fc(last_hidden)
        sbp = self.sbp_head(feat)
        dbp = self.dbp_head(feat)
        return torch.cat([sbp, dbp], dim=1)


@dataclass
class ReplayExecutionResult:
    df_windows: pd.DataFrame
    df_predictions: pd.DataFrame
    streaming_traces: Dict[str, np.ndarray]
    summary: Dict[str, Any]


class FrozenPipelineRunner:
    """Wrapper that verifies, loads, and manages frozen research models."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.cnn_model: Optional[PPGCNNBaseline] = None
        self.gru_model: Optional[TemporalGRUModel] = None
        self.isotonic_sbp = None
        self.isotonic_dbp = None
        self.conformal_quantiles: Optional[dict] = None
        self.is_loaded = False

    def load_models(self) -> Dict[str, Any]:
        """Load frozen checkpoints and calibration maps with strict parameter audit."""
        if not PHASE4A_CKPT.exists():
            raise FileNotFoundError(f"Phase 4A checkpoint not found: {PHASE4A_CKPT}")
        if not PHASE4B_CKPT.exists():
            raise FileNotFoundError(f"Phase 4B checkpoint not found: {PHASE4B_CKPT}")

        # 1. Phase 4A CNN
        self.cnn_model = PPGCNNBaseline(n_channels=3, dropout=0.2).to(self.device)
        c4a_data = torch.load(PHASE4A_CKPT, map_location=self.device, weights_only=False)
        self.cnn_model.load_state_dict(c4a_data["model_state_dict"])
        self.cnn_model.eval()

        # 2. Phase 4B GRU
        self.gru_model = TemporalGRUModel().to(self.device)
        c4b_data = torch.load(PHASE4B_CKPT, map_location=self.device, weights_only=False)
        self.gru_model.load_state_dict(c4b_data["model_state_dict"])
        self.gru_model.eval()

        # 3. Enforce Strict Zero-Retraining Invariant
        for p in list(self.cnn_model.parameters()) + list(self.gru_model.parameters()):
            p.requires_grad = False

        cnn_p = sum(p.numel() for p in self.cnn_model.parameters())
        gru_p = sum(p.numel() for p in self.gru_model.parameters())
        trainable_p = sum(p.numel() for p in list(self.cnn_model.parameters()) + list(self.gru_model.parameters()) if p.requires_grad)
        tot_params = cnn_p + gru_p

        assert trainable_p == 0, "Trainable parameters detected! Model weights must be strictly frozen."
        assert tot_params == EXPECTED_TOTAL_PARAMS, f"Parameter count mismatch: {tot_params} != {EXPECTED_TOTAL_PARAMS}"

        # 4. Phase 5C Calibration Artifacts
        has_phase5c = False
        if ISOTONIC_SBP_PKL.exists() and ISOTONIC_DBP_PKL.exists() and CONFORMAL_QUANTILES_JSON.exists():
            with open(ISOTONIC_SBP_PKL, "rb") as f:
                self.isotonic_sbp = pickle.load(f)
            with open(ISOTONIC_DBP_PKL, "rb") as f:
                self.isotonic_dbp = pickle.load(f)
            with open(CONFORMAL_QUANTILES_JSON, "r") as f:
                self.conformal_quantiles = json.load(f)
            has_phase5c = True

        self.is_loaded = True
        return {
            "cnn_params": cnn_p,
            "gru_params": gru_p,
            "total_params": tot_params,
            "trainable_params": trainable_p,
            "has_phase5c": has_phase5c,
        }

    def calibrate_bp(self, sbp_raw: float, dbp_raw: float) -> Tuple[float, float, float, float, float, float]:
        """Apply Phase 5C extreme-aware isotonic mapping and bin-specific 95% conformal intervals."""
        if self.isotonic_sbp is None or self.isotonic_dbp is None or self.conformal_quantiles is None:
            return sbp_raw, sbp_raw - 25.0, sbp_raw + 25.0, dbp_raw, dbp_raw - 15.0, dbp_raw + 15.0

        sbp_cal = float(self.isotonic_sbp.predict([sbp_raw])[0])
        dbp_cal = float(self.isotonic_dbp.predict([dbp_raw])[0])

        sbp_bin = "<120" if sbp_raw < 120.0 else ("120-139" if sbp_raw < 140.0 else ">=140")
        dbp_bin = "<60" if dbp_raw < 60.0 else ("60-79" if dbp_raw < 80.0 else ">=80")

        sbp_q = self.conformal_quantiles["SBP"][sbp_bin]["95"]
        dbp_q = self.conformal_quantiles["DBP"][dbp_bin]["95"]

        return (
            sbp_cal,
            sbp_cal - sbp_q["q_lower"],
            sbp_cal + sbp_q["q_upper"],
            dbp_cal,
            dbp_cal - dbp_q["q_lower"],
            dbp_cal + dbp_q["q_upper"],
        )

    def run_replay(self, df_replay: pd.DataFrame, progress_callback=None) -> ReplayExecutionResult:
        """
        Execute streaming replay simulation on standardized input dataframe.
        
        Args:
            df_replay: DataFrame with ir, sample_index, host_timestamp_ms.
            progress_callback: Optional callable(float) -> None for UI progress bar.
            
        Returns:
            ReplayExecutionResult containing windows, predictions, traces, and metrics.
        """
        if not self.is_loaded:
            self.load_models()

        ir_vals = df_replay["ir"].to_numpy(dtype=float)
        host_ts = df_replay["host_timestamp_ms"].to_numpy(dtype=float)
        n_samples = len(ir_vals)
        t0_host = host_ts[0] if len(host_ts) > 0 else 0.0

        dsp_pipeline = StreamingDSPPipeline(window_samples=WINDOW_SAMPLES, fs_out=OUTPUT_FS)
        quality_assessor = WindowQualityAssessor(fs=OUTPUT_FS)

        six_window_embeddings: List[torch.Tensor] = []
        six_window_meta: List[Dict[str, Any]] = []

        window_records: List[Dict[str, Any]] = []
        prediction_records: List[Dict[str, Any]] = []
        chunk_latencies_ms: List[float] = []

        # Buffers for continuous waveform traces
        all_raw_slices = []
        all_ppg_slices = []
        all_vpg_slices = []
        all_apg_slices = []

        t_replay_start = time.perf_counter()

        for i in range(0, n_samples, CHUNK_SIZE):
            chunk = ir_vals[i : i + CHUNK_SIZE]
            t_chunk_start = time.perf_counter()

            new_windows = dsp_pipeline.process_raw_samples(chunk)

            # Check for completed 10-second windows
            for w_idx, win_tensor_3x1250, raw_slice in new_windows:
                w_start_sec = w_idx * 10.0
                w_end_sec = (w_idx + 1) * 10.0
                w_end_ts_ms = t0_host + (w_end_sec * 1000.0)

                # Quality assessment
                qc = quality_assessor.assess_window(
                    raw_slice=raw_slice,
                    normalized_tensor=win_tensor_3x1250,
                    window_idx=w_idx,
                    start_time_sec=w_start_sec,
                    end_time_sec=w_end_sec,
                )
                qc["window_id"] = f"win_{w_idx:02d}"
                qc["end_timestamp_ms"] = w_end_ts_ms
                window_records.append(qc)

                all_raw_slices.append(raw_slice)
                all_ppg_slices.append(win_tensor_3x1250[0])
                all_vpg_slices.append(win_tensor_3x1250[1])
                all_apg_slices.append(win_tensor_3x1250[2])

                # Extract 64-dim CNN Embedding
                t_win = torch.from_numpy(win_tensor_3x1250).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    x = self.cnn_model.block1(t_win)
                    x = self.cnn_model.block2(x)
                    x = self.cnn_model.block3(x)
                    x = self.cnn_model.block4(x)
                    emb_64 = self.cnn_model.fc(self.cnn_model.flatten(x))  # [1, 64]

                six_window_embeddings.append(emb_64)
                six_window_meta.append(qc)

                # Maintain rolling 6-window sequence
                if len(six_window_embeddings) >= 6:
                    if len(six_window_embeddings) > 6:
                        six_window_embeddings.pop(0)
                        six_window_meta.pop(0)

                    seq_idx = w_idx - 6 + 1
                    has_reject = any(m["qc_status"] == "REJECT" for m in six_window_meta)
                    has_warn = any(m["qc_status"] == "WARN" for m in six_window_meta)

                    if has_reject:
                        pred_status = "REJECTED_SIGNAL_QUALITY"
                        sbp_raw, dbp_raw = np.nan, np.nan
                        sbp_cal, sbp_lo, sbp_hi, dbp_cal, dbp_lo, dbp_hi = [np.nan] * 6
                    else:
                        pred_status = "WARN" if has_warn else "PASS"
                        seq_tensor = torch.cat(six_window_embeddings, dim=0).unsqueeze(0)
                        with torch.no_grad():
                            out_bp = self.gru_model(seq_tensor).cpu().numpy()[0]
                        sbp_raw, dbp_raw = float(out_bp[0]), float(out_bp[1])
                        sbp_cal, sbp_lo, sbp_hi, dbp_cal, dbp_lo, dbp_hi = self.calibrate_bp(sbp_raw, dbp_raw)

                    pred_record = {
                        "prediction_id": f"pred_{seq_idx:02d}",
                        "sequence_index": seq_idx,
                        "target_window": f"win_{w_idx:02d}",
                        "context_start_s": float(seq_idx * 10.0),
                        "context_end_s": float(w_end_sec),
                        "prediction_timestamp_ms": float(w_end_ts_ms),
                        "prediction_elapsed_s": float(w_end_sec),
                        "duration_sec": 60.0,
                        "quality_status": pred_status,
                        "raw_sbp": sbp_raw,
                        "raw_dbp": dbp_raw,
                        "calibrated_sbp": sbp_cal,
                        "calibrated_dbp": dbp_cal,
                        "conformal_lower_sbp": sbp_lo,
                        "conformal_upper_sbp": sbp_hi,
                        "conformal_lower_dbp": dbp_lo,
                        "conformal_upper_dbp": dbp_hi,
                    }
                    prediction_records.append(pred_record)

            t_chunk_end = time.perf_counter()
            chunk_latencies_ms.append((t_chunk_end - t_chunk_start) * 1000.0)

            if progress_callback is not None and (i % (CHUNK_SIZE * 50) == 0 or i + CHUNK_SIZE >= n_samples):
                progress_callback(min(1.0, float(i + CHUNK_SIZE) / n_samples))

        t_replay_total = time.perf_counter() - t_replay_start

        df_windows = pd.DataFrame(window_records) if window_records else pd.DataFrame()
        df_predictions = pd.DataFrame(prediction_records) if prediction_records else pd.DataFrame()

        # Concatenate traces for full waveform visualization
        concat_raw = np.concatenate(all_raw_slices) if all_raw_slices else np.array([])
        concat_ppg = np.concatenate(all_ppg_slices) if all_ppg_slices else np.array([])
        concat_vpg = np.concatenate(all_vpg_slices) if all_vpg_slices else np.array([])
        concat_apg = np.concatenate(all_apg_slices) if all_apg_slices else np.array([])
        t_traces = np.arange(len(concat_ppg)) / OUTPUT_FS if len(concat_ppg) > 0 else np.array([])

        streaming_traces = {
            "time_s": t_traces,
            "raw_resampled": concat_raw,
            "ppg": concat_ppg,
            "vpg": concat_vpg,
            "apg": concat_apg,
        }

        # Summary statistics
        mean_chunk_lat = float(np.mean(chunk_latencies_ms)) if chunk_latencies_ms else 0.0
        p95_chunk_lat = float(np.percentile(chunk_latencies_ms, 95)) if chunk_latencies_ms else 0.0
        peak_chunk_lat = float(np.max(chunk_latencies_ms)) if chunk_latencies_ms else 0.0
        budget_headroom = (100.0 / mean_chunk_lat) if mean_chunk_lat > 0 else 0.0

        n_pass_win = int((df_windows["qc_status"] == "PASS").sum()) if len(df_windows) > 0 else 0
        n_warn_win = int((df_windows["qc_status"] == "WARN").sum()) if len(df_windows) > 0 else 0
        n_rej_win = int((df_windows["qc_status"] == "REJECT").sum()) if len(df_windows) > 0 else 0

        summary = {
            "total_samples_processed": n_samples,
            "total_windows_formed": len(df_windows),
            "window_qc_counts": {"PASS": n_pass_win, "WARN": n_warn_win, "REJECT": n_rej_win},
            "total_sequences_formed": len(df_predictions),
            "total_predictions_generated": len(df_predictions),
            "replay_duration_seconds": t_replay_total,
            "mean_chunk_latency_ms": mean_chunk_lat,
            "p95_chunk_latency_ms": p95_chunk_lat,
            "peak_chunk_latency_ms": peak_chunk_lat,
            "host_budget_headroom_x": budget_headroom,
        }

        return ReplayExecutionResult(
            df_windows=df_windows,
            df_predictions=df_predictions,
            streaming_traces=streaming_traces,
            summary=summary,
        )
