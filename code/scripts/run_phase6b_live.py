#!/usr/bin/env python3
"""
================================================================================
Phase 6B — Real-Time Live MAX30102 Streaming Serial Pipeline
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only
================================================================================

This script connects to a live MAX30102 sensor stream via an ESP32 micro-controller
over UART / Serial (default 921,600 baud), executing host-side real-time streaming DSP
and causal inference with the frozen Phase 4A CNN + Phase 4B Temporal GRU models.

Protocol Format:
    sample_index,expected_timestamp_ms,host_timestamp_ms,ir,red

Safety & Integrity Rules:
- IR is used strictly as the single-channel input (red is recorded for diagnostic logging).
- Zero Retraining Rule: Phase 4A CNN (146,978 params) and Phase 4B GRU (27,106 params)
  are strictly frozen (0 trainable parameters).
- Scientific Scope: Hardware accuracy cannot be claimed without reference arm-cuff labels.
- Safe port handling: Discovers available ports or accepts explicit CLI argument.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

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

# Ensure local script imports work
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from streaming_resampler import StatefulRationalResampler
from streaming_dsp import StreamingDSPPipeline
from live_quality import WindowQualityAssessor

# -----------------------------------------------------------------------------
# Global Paths & Checkpoints
# -----------------------------------------------------------------------------
OUTPUT_BASE = PROJECT_ROOT / "code" / "outputs" / "phase6b_live_stream"
CAPTURES_DIR = OUTPUT_BASE / "captures"
PRED_DIR = OUTPUT_BASE / "predictions"
LOGS_DIR = OUTPUT_BASE / "logs"

for p in [CAPTURES_DIR, PRED_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

PHASE4A_CKPT = PROJECT_ROOT / "code" / "outputs" / "phase4a_single_model" / "checkpoints" / "best_model_ppg_vpg_apg.pt"
PHASE4B_CKPT = PROJECT_ROOT / "code" / "outputs" / "phase4b_temporal_gru" / "checkpoints" / "best_temporal_gru.pt"
PHASE5C_METRICS = PROJECT_ROOT / "code" / "outputs" / "phase5c_extreme_calibration" / "reports" / "phase5c_metrics.json"

# -----------------------------------------------------------------------------
# Neural Architecture Definitions (Frozen Checkpoints)
# -----------------------------------------------------------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, k=7, p=3, pool=2, drop=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.dropout(self.pool(self.relu(self.bn(self.conv(x)))))


class PPGCNNBaseline(nn.Module):
    def __init__(self, n_channels=3, dropout=0.2):
        super().__init__()
        self.block1 = ConvBlock1D(n_channels, 32, k=7, p=3, pool=2, drop=dropout)
        self.block2 = ConvBlock1D(32, 64, k=7, p=3, pool=2, drop=dropout)
        self.block3 = ConvBlock1D(64, 128, k=7, p=3, pool=2, drop=dropout)
        self.block4 = ConvBlock1D(128, 256, k=7, p=3, pool=2, drop=dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 78, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        feat = self.fc(self.flatten(self.block4(self.block3(self.block2(self.block1(x))))))
        return self.head(feat)


class TemporalGRUModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# -----------------------------------------------------------------------------
# Serial Discovery Helper
# -----------------------------------------------------------------------------
def discover_ports() -> List[str]:
    """Scan and return available serial ports."""
    if not SERIAL_AVAILABLE:
        return []
    ports = [p.device for p in serial.tools.list_ports.comports()]
    return sorted(ports)


# -----------------------------------------------------------------------------
# Live Stream Engine
# -----------------------------------------------------------------------------
class LiveStreamEngine:
    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: int = 921600,
        timeout: float = 1.0,
        device_type: str = "cpu"
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.device = torch.device(device_type if torch.cuda.is_available() and device_type == "cuda" else "cpu")
        
        # Load Frozen Models
        self._load_frozen_models()
        self._load_conformal_calibration()
        
        # DSP Pipeline & Quality Gating
        self.dsp = StreamingDSPPipeline(window_samples=1250, fs_out=125.0)
        self.quality = WindowQualityAssessor(adc_max=262143.0, adc_min=0.0)
        
        # 60-Second Sequence Buffer (6 x 64-dim embeddings)
        self.history_embeddings: List[torch.Tensor] = []
        self.history_meta: List[Dict[str, Any]] = []
        
        # Integrity & State Tracking
        self.samples_received = 0
        self.last_sample_index: Optional[int] = None
        self.dropped_samples = 0
        self.gap_count = 0
        self.duplicate_count = 0
        self.malformed_count = 0
        
        self.state = "RESAMPLER_WARMUP"
        self.capture_buffer: List[Dict[str, Any]] = []
        self.prediction_records: List[Dict[str, Any]] = []

    def _load_frozen_models(self):
        logging.info("Loading Frozen Phase 4A CNN and Phase 4B GRU Models...")
        self.cnn = PPGCNNBaseline(n_channels=3, dropout=0.2).to(self.device)
        c4a = torch.load(PHASE4A_CKPT, map_location=self.device, weights_only=False)
        self.cnn.load_state_dict(c4a["model_state_dict"])
        self.cnn.eval()

        self.gru = TemporalGRUModel().to(self.device)
        c4b = torch.load(PHASE4B_CKPT, map_location=self.device, weights_only=False)
        self.gru.load_state_dict(c4b["model_state_dict"])
        self.gru.eval()

        for p in list(self.cnn.parameters()) + list(self.gru.parameters()):
            p.requires_grad = False

        total_p = sum(p.numel() for p in self.cnn.parameters()) + sum(p.numel() for p in self.gru.parameters())
        logging.info(f"Loaded Models: Total parameters = {total_p:,} (Trainable: 0)")

    def _load_conformal_calibration(self):
        self.has_phase5c = False
        self.isotonic_sbp = None
        self.isotonic_dbp = None
        self.conformal_quantiles = None

        phase5c_dir = PROJECT_ROOT / "code" / "outputs" / "phase5c_extreme_aware"
        iso_sbp_pkl = phase5c_dir / "mappings" / "isotonic_sbp.pkl"
        iso_dbp_pkl = phase5c_dir / "mappings" / "isotonic_dbp.pkl"
        cq_json = phase5c_dir / "calibration" / "conformal_quantiles_by_bin.json"

        if iso_sbp_pkl.exists() and iso_dbp_pkl.exists() and cq_json.exists():
            import pickle
            import json
            try:
                with open(iso_sbp_pkl, "rb") as f:
                    self.isotonic_sbp = pickle.load(f)
                with open(iso_dbp_pkl, "rb") as f:
                    self.isotonic_dbp = pickle.load(f)
                with open(cq_json, "r") as f:
                    self.conformal_quantiles = json.load(f)
                self.has_phase5c = True
                logging.info("Phase 5C post-hoc calibration artifacts loaded successfully.")
            except Exception as e:
                logging.warning(f"Failed loading Phase 5C artifacts: {e}")

    def calibrate(self, raw_sbp: float, raw_dbp: float) -> Tuple[float, float, float, float, float, float]:
        if not self.has_phase5c:
            return raw_sbp, raw_sbp - 25.0, raw_sbp + 25.0, raw_dbp, raw_dbp - 15.0, raw_dbp + 15.0

        sbp_cal = float(self.isotonic_sbp.predict([raw_sbp])[0])
        dbp_cal = float(self.isotonic_dbp.predict([raw_dbp])[0])

        sbp_bin = "<120" if raw_sbp < 120.0 else ("120-139" if raw_sbp < 140.0 else ">=140")
        dbp_bin = "<60" if raw_dbp < 60.0 else ("60-79" if raw_dbp < 80.0 else ">=80")

        sbp_q = self.conformal_quantiles["SBP"][sbp_bin]["95"]
        dbp_q = self.conformal_quantiles["DBP"][dbp_bin]["95"]

        return (
            sbp_cal, sbp_cal - sbp_q["q_lower"], sbp_cal + sbp_q["q_upper"],
            dbp_cal, dbp_cal - dbp_q["q_lower"], dbp_cal + dbp_q["q_upper"]
        )

    def parse_line(self, line: str) -> Optional[Tuple[int, float, float, float, float]]:
        """Parse serial CSV line: sample_index,expected_timestamp_ms,host_timestamp_ms,ir,red"""
        parts = line.strip().split(",")
        if len(parts) >= 5:
            try:
                s_idx = int(parts[0])
                exp_ts = float(parts[1])
                host_ts = float(parts[2])
                ir_val = float(parts[3])
                red_val = float(parts[4])
                return s_idx, exp_ts, host_ts, ir_val, red_val
            except ValueError:
                self.malformed_count += 1
                return None
        elif len(parts) == 4:
            # Fallback 4-column format: sample_index, timestamp_ms, ir, red
            try:
                s_idx = int(parts[0])
                exp_ts = float(parts[1])
                host_ts = exp_ts
                ir_val = float(parts[2])
                red_val = float(parts[3])
                return s_idx, exp_ts, host_ts, ir_val, red_val
            except ValueError:
                self.malformed_count += 1
                return None
        else:
            self.malformed_count += 1
            return None

    def feed_sample(self, s_idx: int, exp_ts: float, host_ts: float, ir_val: float, red_val: float) -> Optional[Dict[str, Any]]:
        """Feed a single incoming sample to the pipeline and return a prediction dict if a 10s window was completed."""
        # 1. Sample Integrity Check
        if self.last_sample_index is not None:
            if s_idx == self.last_sample_index:
                self.duplicate_count += 1
                return None
            elif s_idx < self.last_sample_index:
                self.malformed_count += 1
                return None
            elif s_idx > self.last_sample_index + 1:
                gaps = s_idx - (self.last_sample_index + 1)
                self.gap_count += gaps
                self.dropped_samples += gaps

        self.last_sample_index = s_idx
        self.samples_received += 1

        self.capture_buffer.append({
            "sample_index": s_idx,
            "expected_timestamp_ms": exp_ts,
            "host_timestamp_ms": host_ts,
            "ir": ir_val,
            "red": red_val
        })

        # Update state machine
        if self.state == "RESAMPLER_WARMUP" and self.samples_received >= 25:
            self.state = "FILTER_WARMUP"
        if self.state == "FILTER_WARMUP" and self.samples_received >= 250:
            self.state = "BUFFER_FILLING"

        # 2. Feed to Streaming DSP Pipeline
        new_windows = self.dsp.process_raw_samples(np.array([ir_val], dtype=np.float64))
        if not new_windows:
            return None

        # A 10-second window completed!
        prediction_result = None
        for w_idx, win_tensor_3x1250, raw_slice in new_windows:
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

            # CNN Feature Extraction
            t_in = torch.from_numpy(win_tensor_3x1250).unsqueeze(0).to(self.device)
            with torch.no_grad():
                x = self.cnn.block1(t_in)
                x = self.cnn.block2(x)
                x = self.cnn.block3(x)
                x = self.cnn.block4(x)
                emb_64 = self.cnn.fc(self.cnn.flatten(x))  # [1, 64]

            self.history_embeddings.append(emb_64)
            self.history_meta.append(qc)

            if len(self.history_embeddings) < 6:
                self.state = "HISTORY_FILLING"
                status_str = f"BUFFERING ({len(self.history_embeddings)}/6 Windows)"
                prediction_result = {
                    "timeline_sec": w_end_sec,
                    "window_index": w_idx,
                    "sequence_index": None,
                    "status": status_str,
                    "qc": qc["qc_status"],
                    "sbp": None,
                    "dbp": None,
                }
            else:
                self.state = "READY"
                if len(self.history_embeddings) > 6:
                    self.history_embeddings.pop(0)
                    self.history_meta.pop(0)

                seq_idx = w_idx - 6 + 1
                has_reject = any(m["qc_status"] == "REJECT" for m in self.history_meta)
                has_warn = any(m["qc_status"] == "WARN" for m in self.history_meta)

                if has_reject:
                    pred_status = "REJECTED_SIGNAL_QUALITY"
                    sbp_raw, dbp_raw = np.nan, np.nan
                    sbp_cal, sbp_lo, sbp_hi, dbp_cal, dbp_lo, dbp_hi = [np.nan]*6
                else:
                    pred_status = "WARN" if has_warn else "PASS"
                    seq_tensor = torch.cat(self.history_embeddings, dim=0).unsqueeze(0)
                    with torch.no_grad():
                        out = self.gru(seq_tensor).cpu().numpy()[0]
                    sbp_raw, dbp_raw = float(out[0]), float(out[1])
                    sbp_cal, sbp_lo, sbp_hi, dbp_cal, dbp_lo, dbp_hi = self.calibrate(sbp_raw, dbp_raw)

                pred_record = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timeline_sec": w_end_sec,
                    "window_index": w_idx,
                    "sequence_index": seq_idx,
                    "state": self.state,
                    "status": pred_status,
                    "qc_latest": qc["qc_status"],
                    "raw_sbp": sbp_raw,
                    "raw_dbp": dbp_raw,
                    "calibrated_sbp": sbp_cal,
                    "calibrated_dbp": dbp_cal,
                    "sbp_lower": sbp_lo,
                    "sbp_upper": sbp_hi,
                    "dbp_lower": dbp_lo,
                    "dbp_upper": dbp_hi,
                    "samples_received": self.samples_received,
                    "dropped_samples": self.dropped_samples,
                }
                self.prediction_records.append(pred_record)
                prediction_result = pred_record

        return prediction_result

    def run_live(self):
        """Execute live serial reading loop."""
        if not SERIAL_AVAILABLE:
            print("ERROR: pyserial is not installed in the active environment.")
            return

        print("\n" + "=" * 70)
        print("  PHASE 6B: REAL-TIME MAX30102 LIVE STREAMING ENGINE")
        print("  Project: Calibration-Free Cuffless BP Estimation (PPG Only)")
        print("=" * 70)
        print(f"Connecting to Serial Port: {self.port} @ {self.baud_rate} baud (Timeout: {self.timeout}s)...")

        try:
            ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            time.sleep(1.0)
            ser.reset_input_buffer()
            print("Serial port opened successfully. Beginning acquisition stream...\n")
        except Exception as e:
            print(f"Failed to open serial port {self.port}: {e}")
            return

        run_id = time.strftime("%Y%m%d_%H%M%S")
        cap_file = CAPTURES_DIR / f"live_capture_{run_id}.csv"
        pred_file = PRED_DIR / f"live_predictions_{run_id}.csv"

        print(f"Logging raw data to:    {cap_file}")
        print(f"Logging predictions to: {pred_file}")
        print("Press Ctrl+C to terminate session safely.\n")

        print("-" * 75)
        print(f"{'TIME':<10} | {'STATE':<16} | {'QC':<6} | {'SBP (mmHg)':<18} | {'DBP (mmHg)':<18}")
        print("-" * 75)

        try:
            while True:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue
                try:
                    line_str = line_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                parsed = self.parse_line(line_str)
                if parsed is None:
                    continue

                s_idx, exp_ts, host_ts, ir_val, red_val = parsed
                res = self.feed_sample(s_idx, exp_ts, host_ts, ir_val, red_val)

                if res is not None:
                    t_str = f"{res['timeline_sec']:.0f}s"
                    state_str = self.state
                    qc_str = res.get("qc_latest", res.get("qc", "N/A"))
                    sbp_val = res.get("calibrated_sbp")
                    dbp_val = res.get("calibrated_dbp")

                    if sbp_val is not None and not np.isnan(sbp_val):
                        sbp_str = f"{sbp_val:.1f} [{res['sbp_lower']:.1f}-{res['sbp_upper']:.1f}]"
                        dbp_str = f"{dbp_val:.1f} [{res['dbp_lower']:.1f}-{res['dbp_upper']:.1f}]"
                    else:
                        sbp_str = "-- (WARMING UP)"
                        dbp_str = "-- (WARMING UP)"

                    print(f"{t_str:<10} | {state_str:<16} | {qc_str:<6} | {sbp_str:<18} | {dbp_str:<18}")

        except KeyboardInterrupt:
            print("\nUser interrupted acquisition. Shutting down gracefully...")
        finally:
            ser.close()
            # Save captures
            if self.capture_buffer:
                df_cap = pd.DataFrame(self.capture_buffer)
                df_cap.to_csv(cap_file, index=False)
                print(f"Saved {len(df_cap)} raw samples to {cap_file}")

            # Save predictions
            if self.prediction_records:
                df_preds = pd.DataFrame(self.prediction_records)
                df_preds.to_csv(pred_file, index=False)
                print(f"Saved {len(df_preds)} predictions to {pred_file}")

            print("\nSession Summary:")
            print(f"  Total Samples Received: {self.samples_received}")
            print(f"  Dropped / Missing:      {self.dropped_samples}")
            print(f"  Malformed Lines:        {self.malformed_count}")
            print(f"  Duplicate Samples:      {self.duplicate_count}")
            print("Session closed safely.")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 6B Real-Time MAX30102 Streaming Pipeline")
    parser.add_argument("--port", type=str, default=None, help="Serial device port (e.g. /dev/ttyUSB0, /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate (default: 921600)")
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout in seconds (default: 1.0)")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    available_ports = discover_ports()

    if args.list_ports:
        print("Available Serial Ports:")
        if not available_ports:
            print("  No serial devices detected.")
        else:
            for p in available_ports:
                print(f"  - {p}")
        return

    selected_port = args.port
    if selected_port is None:
        if len(available_ports) == 1:
            selected_port = available_ports[0]
            print(f"Auto-detected single available serial device: {selected_port}")
        elif len(available_ports) > 1:
            print("Multiple serial devices detected:")
            for i, p in enumerate(available_ports):
                print(f"  [{i+1}] {p}")
            print("\nPlease specify a port explicitly using `--port <device_path>` (e.g. --port /dev/ttyUSB0).")
            return
        else:
            print("No serial device connected or detected.")
            print("To run the live pipeline, connect your ESP32 device or specify `--port <device_path>`.")
            print("To test the streaming architecture offline, run: ./.venv/bin/python code/scripts/run_phase6b_replay.py")
            return

    engine = LiveStreamEngine(port=selected_port, baud_rate=args.baud, timeout=args.timeout, device_type=args.device)
    engine.run_live()


if __name__ == "__main__":
    main()
