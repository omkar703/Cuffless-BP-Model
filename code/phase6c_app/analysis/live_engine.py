"""
Phase 6C / Phase 7 Live Streaming & Experiment Engine.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Unifies physical ESP32 MAX30102 live serial streaming, recorded session replay,
and demo session replay into a single, identical, strictly causal pipeline:
    Raw PPG (100 Hz)
      ↓ (Stateful Rational Resampler 100→125 Hz)
    Filtered PPG (0.5–8.0 Hz Causal SOS Filter)
      ↓ (Causal Backward Differences)
    VPG & APG
      ↓ (10-second Windowing & Z-Score Normalization)
    Quality Gate (PASS / WARN / REJECT)
      ↓ (Frozen Phase 4A 1D CNN Encoder)
    60-Second Sequence Buffer (6 x 64-dim embeddings)
      ↓ (Frozen Phase 4B Unidirectional Causal GRU)
    Phase 5C Extreme-Aware Recalibration & 95% Conformal Bounds
      ↓
    Phase 7 Multi-Domain Reliability Engine (TRUST / REVIEW / ABSTAIN)

STRICT SAFEGUARDS:
- Zero Retraining: Models, calibration maps, and reliability engine remain 100% frozen.
- Convergence: Live serial data and recorded files execute the exact same DSP & inference code.
"""

import sys
import time
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Ensure path imports
APP_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = APP_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

for p in [CODE_DIR, CODE_DIR / "scripts", CODE_DIR / "phase4a"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase6c_app.config import (
    PHASE4A_CKPT,
    PHASE4B_CKPT,
    ISOTONIC_SBP_PKL,
    ISOTONIC_DBP_PKL,
    CONFORMAL_QUANTILES_JSON,
    OUTPUT_FS,
    WINDOW_SAMPLES,
    SEQUENCE_WINDOWS,
    EXPECTED_TOTAL_PARAMS,
)
from streaming_dsp import StreamingDSPPipeline
from live_quality import WindowQualityAssessor
from phase4a.model import PPGCNNBaseline
from phase6c_app.analysis.replay_engine import TemporalGRUModel


def discover_serial_ports() -> List[Dict[str, str]]:
    """Scan and return available serial ports with human-readable descriptions."""
    if not SERIAL_AVAILABLE:
        return []
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append({
            "device": p.device,
            "description": p.description or p.device,
            "hwid": p.hwid or ""
        })
    return ports


class LiveExperimentEngine:
    """
    Stateful streaming experiment engine supporting live serial hardware,
    recorded session files, and demo replay through the exact same causal pipeline.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.is_loaded = False
        
        # Load Frozen Research Models & Calibration Artifacts
        self._load_frozen_artifacts()
        
        # DSP Pipeline & Quality Assessor
        self.dsp = StreamingDSPPipeline(window_samples=WINDOW_SAMPLES, fs_out=OUTPUT_FS)
        self.quality = WindowQualityAssessor(adc_max=262143.0, adc_min=0.0)
        
        # Serial Connection
        self.serial_conn: Optional[Any] = None
        self.is_serial_connected = False
        self.connected_port: Optional[str] = None
        
        # Stream & Context Buffers
        self.reset()

    def _load_frozen_artifacts(self):
        """Loads Phase 4A CNN, Phase 4B GRU, Phase 5C calibration, and Phase 7 reliability engine."""
        # 1. Phase 4A CNN
        self.cnn_model = PPGCNNBaseline(n_channels=3, dropout=0.2).to(self.device)
        c4a = torch.load(PHASE4A_CKPT, map_location=self.device, weights_only=False)
        self.cnn_model.load_state_dict(c4a["model_state_dict"])
        self.cnn_model.eval()

        # 2. Phase 4B GRU
        self.gru_model = TemporalGRUModel().to(self.device)
        c4b = torch.load(PHASE4B_CKPT, map_location=self.device, weights_only=False)
        self.gru_model.load_state_dict(c4b["model_state_dict"])
        self.gru_model.eval()

        # Strictly freeze parameters
        for p in list(self.cnn_model.parameters()) + list(self.gru_model.parameters()):
            p.requires_grad = False

        tot_params = sum(p.numel() for p in self.cnn_model.parameters()) + sum(p.numel() for p in self.gru_model.parameters())
        assert tot_params == EXPECTED_TOTAL_PARAMS, f"Parameter count mismatch: {tot_params} != {EXPECTED_TOTAL_PARAMS}"

        # 3. Phase 5C Isotonic Calibration & Conformal Quantiles
        with open(ISOTONIC_SBP_PKL, "rb") as f:
            self.isotonic_sbp = pickle.load(f)
        with open(ISOTONIC_DBP_PKL, "rb") as f:
            self.isotonic_dbp = pickle.load(f)
        with open(CONFORMAL_QUANTILES_JSON, "r") as f:
            self.conformal_quantiles = json.load(f)

        # 4. Phase 7 Reliability Engine
        rel_model_path = PROJECT_ROOT / "code" / "outputs" / "phase7_reliability" / "reliability_engine_model.pkl"
        with open(rel_model_path, "rb") as f:
            self.rel_engine = pickle.load(f)

        self.is_loaded = True

    def reset(self):
        """Reset internal streaming state, buffers, and sequence history."""
        self.dsp.reset()
        
        # Raw rolling buffer for visualization (last 600 samples @ 100 Hz = 6s)
        self.raw_display_ir: List[float] = []
        self.raw_display_times: List[float] = []
        
        # Filtered pulsatile PPG rolling buffer (last 750 samples @ 125 Hz = 6s)
        self.display_ppg: List[float] = []
        self.display_times: List[float] = []
        self.total_125hz_samples: int = 0
        
        # Capture buffer for recording export
        self.is_recording = False
        self.recorded_samples: List[Dict[str, Any]] = []
        
        # Stream counters
        self.samples_received = 0
        self.start_monotonic_s: Optional[float] = None
        self.last_sample_index: Optional[int] = None
        self.effective_rate_hz: float = 100.0
        
        # Windows & Temporal Context
        self.completed_windows: List[Dict[str, Any]] = []
        self.sequence_embeddings: List[torch.Tensor] = []
        self.sequence_meta: List[Dict[str, Any]] = []
        
        # State machine: IDLE, ACQUIRING, BUILDING_CONTEXT, ESTIMATING, COMPLETE, ERROR
        self.pipeline_state = "IDLE"
        self.status_message = "Ready to begin experiment."
        self.current_signal_qc = "PENDING"
        
        # Results
        self.latest_prediction: Optional[Dict[str, Any]] = None
        self.latest_reliability: Optional[Dict[str, Any]] = None
        self.history_predictions: List[Dict[str, Any]] = []
        self.latest_window_tensor: Optional[np.ndarray] = None
        
        # Exploratory Reference Cuff Comparison
        self.reference_comparison: Optional[Dict[str, Any]] = None

    def connect_serial(self, port: str, baud_rate: int = 921600, timeout: float = 0.1) -> bool:
        """Connect to physical ESP32 MAX30102 serial port."""
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is not installed in the environment.")
        try:
            self.disconnect_serial()
            self.serial_conn = serial.Serial(port, baud_rate, timeout=timeout)
            self.is_serial_connected = True
            self.connected_port = port
            self.pipeline_state = "CONNECTING"
            self.status_message = f"Connected to {port} @ {baud_rate:,} baud. Waiting for pulse stream."
            return True
        except Exception as e:
            self.is_serial_connected = False
            self.connected_port = None
            self.pipeline_state = "ERROR"
            self.status_message = f"Serial connection error: {e}"
            return False

    def disconnect_serial(self):
        """Disconnect active serial connection."""
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        self.is_serial_connected = False
        self.connected_port = None

    def parse_serial_line(self, line: str) -> Optional[Tuple[int, float, float, float, float]]:
        """
        Parses serial line supporting standard project formats:
        - 5-part: sample_index, expected_timestamp_ms, host_timestamp_ms, ir, red
        - 4-part: sample_index, timestamp_ms, ir, red
        - 3-part: sample_index, ir, red
        """
        parts = line.strip().split(",")
        if len(parts) >= 5:
            try:
                return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                return None
        elif len(parts) == 4:
            try:
                s_idx = int(parts[0])
                ts = float(parts[1])
                return s_idx, ts, ts, float(parts[2]), float(parts[3])
            except ValueError:
                return None
        elif len(parts) == 3:
            try:
                s_idx = int(parts[0])
                ts = s_idx * 10.0
                return s_idx, ts, ts, float(parts[1]), float(parts[2])
            except ValueError:
                return None
        return None

    def poll_serial(self, max_lines: int = 50) -> int:
        """Poll incoming lines from active serial connection and ingest them."""
        if not self.is_serial_connected or self.serial_conn is None:
            return 0
        
        lines_read = 0
        try:
            while self.serial_conn.in_waiting > 0 and lines_read < max_lines:
                raw_line = self.serial_conn.readline().decode("utf-8", errors="ignore")
                parsed = self.parse_serial_line(raw_line)
                if parsed:
                    s_idx, exp_ts, host_ts, ir, red = parsed
                    self.ingest_sample(s_idx, exp_ts, host_ts, ir, red)
                    lines_read += 1
        except Exception as e:
            self.pipeline_state = "ERROR"
            self.status_message = f"Serial reading error: {e}"
        return lines_read

    def ingest_sample(self, s_idx: int, exp_ts: float, host_ts: float, ir: float, red: float = 0.0):
        """Ingest a single raw sample through the streaming causal pipeline."""
        now_mono = time.monotonic()
        if self.start_monotonic_s is None:
            self.start_monotonic_s = now_mono
            
        self.samples_received += 1
        self.last_sample_index = s_idx
        elapsed_s = now_mono - self.start_monotonic_s
        
        if elapsed_s > 0.5:
            self.effective_rate_hz = float(self.samples_received / elapsed_s)
            
        # Maintain raw rolling buffer for live scrolling waveform (last 600 samples @ 100 Hz = 6s)
        raw_time_s = self.samples_received / 100.0
        self.raw_display_ir.append(ir)
        self.raw_display_times.append(raw_time_s)
        if len(self.raw_display_ir) > 600:
            self.raw_display_ir.pop(0)
            self.raw_display_times.pop(0)
            
        # Recording buffer
        if self.is_recording:
            self.recorded_samples.append({
                "sample_index": s_idx,
                "expected_timestamp_ms": exp_ts,
                "host_timestamp_ms": host_ts,
                "ir": ir,
                "red": red
            })

        # Update state message
        if self.samples_received < 100:
            self.pipeline_state = "ACQUIRING"
            self.status_message = "Waiting for the optical pulse signal."
        elif len(self.completed_windows) < SEQUENCE_WINDOWS:
            self.pipeline_state = "BUILDING_CONTEXT"
            n_w = len(self.completed_windows)
            if n_w == 0:
                self.status_message = "The sensor is receiving a usable pulse waveform. Accumulating window 1."
            elif n_w == 1:
                self.status_message = "First analysis window ready. The temporal model is building context (1/6)."
            else:
                self.status_message = f"{n_w}/6 windows ready. The temporal model requires a 60-second sequence."

        # Pass to Causal Streaming DSP (resampling 100->125 Hz, SOS filter, VPG, APG)
        new_windows = self.dsp.process_raw_samples(np.array([ir], dtype=np.float64))

        # Maintain filtered pulsatile PPG rolling buffer (last 750 samples @ 125 Hz = 6s)
        if hasattr(self.dsp, "latest_filtered_chunk") and len(self.dsp.latest_filtered_chunk) > 0:
            for val in self.dsp.latest_filtered_chunk:
                self.total_125hz_samples += 1
                self.display_ppg.append(float(val))
                self.display_times.append(self.total_125hz_samples / 125.0)
            while len(self.display_ppg) > 750:
                self.display_ppg.pop(0)
                self.display_times.pop(0)

        if not new_windows:
            return

        # A 10-second window completed!
        for w_idx, win_tensor_3x1250, raw_slice in new_windows:
            self.latest_window_tensor = win_tensor_3x1250
            w_start_sec = w_idx * 10.0
            w_end_sec = (w_idx + 1) * 10.0

            # Quality Assessment
            qc = self.quality.assess_window(
                raw_slice=raw_slice,
                normalized_tensor=win_tensor_3x1250,
                window_idx=w_idx,
                start_time_sec=w_start_sec,
                end_time_sec=w_end_sec
            )
            self.current_signal_qc = qc["qc_status"]

            win_record = {
                "window_idx": w_idx,
                "start_time_sec": w_start_sec,
                "end_time_sec": w_end_sec,
                "status": qc["qc_status"],
                "reason": qc["qc_reason"],
                "ptp": float(qc.get("raw_ptp", 0.0)),
                "std": float(qc.get("raw_std", 0.0)),
                "hr": float(qc.get("estimated_hr_bpm", 75.0)),
                "peaks": int(qc.get("detected_peaks", 0)),
            }
            self.completed_windows.append(win_record)

            # CNN Latent Feature Extraction
            t_win = torch.from_numpy(win_tensor_3x1250).unsqueeze(0).to(self.device)
            with torch.no_grad():
                x = self.cnn_model.block1(t_win)
                x = self.cnn_model.block2(x)
                x = self.cnn_model.block3(x)
                x = self.cnn_model.block4(x)
                emb_64 = self.cnn_model.fc(self.cnn_model.flatten(x))  # [1, 64]

            self.sequence_embeddings.append(emb_64)
            self.sequence_meta.append(qc)

            # Maintain sliding 6-window sequence buffer
            if len(self.sequence_embeddings) < SEQUENCE_WINDOWS:
                self.pipeline_state = "BUILDING_CONTEXT"
            else:
                if len(self.sequence_embeddings) > SEQUENCE_WINDOWS:
                    self.sequence_embeddings.pop(0)
                    self.sequence_meta.pop(0)

                # Check if 6 consecutive windows are valid
                has_reject = any(m["qc_status"] == "REJECT" for m in self.sequence_meta)
                has_warn = any(m["qc_status"] == "WARN" for m in self.sequence_meta)

                if has_reject:
                    self.pipeline_state = "REJECTED_QUALITY"
                    rej_idx = next(i for i, m in enumerate(self.sequence_meta) if m["qc_status"] == "REJECT")
                    rej_reason = self.sequence_meta[rej_idx].get("qc_reason", "Signal quality criteria failed")
                    self.status_message = f"Window {rej_idx + 1} rejected: {rej_reason}. Collecting next valid window."
                else:
                    # Execute Frozen Phase 4B GRU Inference!
                    self.pipeline_state = "ESTIMATING"
                    self.status_message = "Temporal context complete. Running the frozen BP model."
                    
                    seq_tensor = torch.cat(self.sequence_embeddings, dim=0).unsqueeze(0)  # [1, 6, 64]
                    with torch.no_grad():
                        out = self.gru_model(seq_tensor).cpu().numpy()[0]
                    sbp_raw, dbp_raw = float(out[0]), float(out[1])

                    # Phase 5C Extreme-Aware Recalibration & Conformal Bounds
                    sbp_cal, sbp_lo, sbp_hi, dbp_cal, dbp_lo, dbp_hi = self._calibrate(sbp_raw, dbp_raw)

                    # Phase 7 Reliability Feature Extraction
                    feat_row = self._build_reliability_features(
                        qc=qc,
                        sbp_cal=sbp_cal,
                        dbp_cal=dbp_cal,
                        sbp_lo=sbp_lo,
                        sbp_hi=sbp_hi,
                        dbp_lo=dbp_lo,
                        dbp_hi=dbp_hi
                    )

                    # Phase 7 Reliability Engine Execution
                    rel_payload = self.rel_engine.generate_explanation_payload(
                        sbp_pred=sbp_cal,
                        dbp_pred=dbp_cal,
                        features=feat_row
                    )

                    pred_record = {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "window_index": w_idx,
                        "timeline_sec": w_end_sec,
                        "raw_sbp": sbp_raw,
                        "raw_dbp": dbp_raw,
                        "calibrated_sbp": sbp_cal,
                        "calibrated_dbp": dbp_cal,
                        "sbp_lower": sbp_lo,
                        "sbp_upper": sbp_hi,
                        "dbp_lower": dbp_lo,
                        "dbp_upper": dbp_hi,
                        "qc_status": "WARN" if has_warn else "PASS",
                    }
                    self.latest_prediction = pred_record
                    self.latest_reliability = rel_payload
                    self.history_predictions.append(pred_record)
                    
                    self.pipeline_state = "COMPLETE"
                    self.status_message = "Blood pressure estimate generated from the 60-second causal sequence."

    def _calibrate(self, sbp_raw: float, dbp_raw: float) -> Tuple[float, float, float, float, float, float]:
        """Apply Phase 5C extreme-aware isotonic mapping and bin-specific 95% conformal intervals."""
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
            dbp_cal + dbp_q["q_upper"]
        )

    def _build_reliability_features(self, qc: Dict[str, Any], sbp_cal: float, dbp_cal: float,
                                    sbp_lo: float, sbp_hi: float, dbp_lo: float, dbp_hi: float) -> Dict[str, Any]:
        """Construct multi-domain feature vector for Phase 7 ReliabilityEngine."""
        # Temporal stability across recent predictions
        if len(self.history_predictions) >= 2:
            recent_sbp = [p["calibrated_sbp"] for p in self.history_predictions[-2:]] + [sbp_cal]
            recent_dbp = [p["calibrated_dbp"] for p in self.history_predictions[-2:]] + [dbp_cal]
            r_std_s = float(np.std(recent_sbp, ddof=1))
            r_std_d = float(np.std(recent_dbp, ddof=1))
            max_j_s = float(np.max(np.abs(np.diff(recent_sbp))))
            max_j_d = float(np.max(np.abs(np.diff(recent_dbp))))
        else:
            r_std_s, r_std_d, max_j_s, max_j_d = 0.0, 0.0, 0.0, 0.0

        return {
            "qc_pass": 1.0 if qc.get("qc_status") == "PASS" else 0.0,
            "ppg_ptp": float(qc.get("raw_ptp", 40000.0)),
            "ppg_std": float(qc.get("raw_std", 10000.0)),
            "ppg_clipped_fraction": float(qc.get("clip_fraction", 0.0)),
            "ppg_pulse_count": float(qc.get("detected_peaks", 14.0)),
            "estimated_hr_bpm": float(qc.get("estimated_hr_bpm", 75.0)),
            "sbp_uncertainty": 14.2,  # Empirical Phase 5A uncertainty proxy
            "dbp_uncertainty": 7.6,
            "sbp_conformal_width_90": float(sbp_hi - sbp_lo),
            "dbp_conformal_width_90": float(dbp_hi - dbp_lo),
            "rolling_sbp_std_3": r_std_s,
            "rolling_dbp_std_3": r_std_d,
            "max_abs_sbp_jump_3": max_j_s,
            "max_abs_dbp_jump_3": max_j_d,
            "median_abs_sbp_change_3": max_j_s,
            "median_abs_dbp_change_3": max_j_d,
        }

    def compare_reference(self, ref_sbp: float, ref_dbp: float) -> Optional[Dict[str, Any]]:
        """Compute exploratory comparison against external reference measurement without altering model."""
        if self.latest_prediction is None:
            return None
        
        pred_sbp = self.latest_prediction["calibrated_sbp"]
        pred_dbp = self.latest_prediction["calibrated_dbp"]
        
        diff_sbp = pred_sbp - ref_sbp
        diff_dbp = pred_dbp - ref_dbp
        
        self.reference_comparison = {
            "ref_sbp": ref_sbp,
            "ref_dbp": ref_dbp,
            "pred_sbp": pred_sbp,
            "pred_dbp": pred_dbp,
            "diff_sbp": diff_sbp,
            "diff_dbp": diff_dbp,
            "disclaimer": "Exploratory reference comparison. This reference value is not used by the prediction model."
        }
        return self.reference_comparison

    def start_recording(self):
        self.is_recording = True
        self.recorded_samples = []

    def stop_recording(self):
        self.is_recording = False

    def export_recorded_csv(self) -> str:
        """Export recorded session samples to standard CSV format."""
        if not self.recorded_samples:
            return ""
        df = pd.DataFrame(self.recorded_samples)
        return df.to_csv(index=False)
