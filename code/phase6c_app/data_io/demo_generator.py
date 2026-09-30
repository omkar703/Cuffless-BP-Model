"""
Synthetic Demo Data Generator for Phase 6C Software Testing.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Generates a fully synthetic PPG session with synchronized reference BP events
specifically for UI/pipeline verification prior to receipt of physical data.

MANDATORY WATERMARK:
"DEMO / NOT REAL HARDWARE DATA / NOT RESEARCH RESULT"
"""

import io
import json
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from phase6c_app.config import DEMO_WATERMARK_TEXT
from phase6c_app.data_io.session_loader import SessionData, _standardize_ppg_dataframe, _standardize_bp_events


def generate_synthetic_ppg_and_events(
    duration_s: float = 95.0,
    fs_nominal: float = 100.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generate synthetic 100 Hz optical PPG waveform and synchronized BP events.
    
    Waveform includes:
    - Realistic finger placement settling in first 3.5 seconds (Window 0 WARN)
    - Realistic arterial pulse morphology with dicrotic notch (~72 bpm)
    - Realistic host timestamp intervals (mean ~10.02 ms, bounded [9, 11] ms)
    - Synchronized reference BP events at ~70s and ~90s
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * fs_nominal)

    # 1. Generate Timestamps
    # Nominal 10 ms steps with small host jitter
    jitter_steps = rng.choice([9, 10, 10, 10, 11], size=n_samples)
    jitter_steps[0] = 0
    host_ts = 1727650000000.0 + np.cumsum(jitter_steps)
    sample_index = 100000 + np.arange(n_samples) * 10

    t_sec = (host_ts - host_ts[0]) / 1000.0

    # 2. Synthesize Pulsatile IR Waveform
    hr_hz = 1.2  # ~72 bpm
    # Pulse template with harmonics and dicrotic notch
    phase = 2 * np.pi * hr_hz * t_sec
    pulse = (
        np.sin(phase)
        + 0.5 * np.sin(2 * phase - 0.4)
        + 0.25 * np.sin(3 * phase - 0.8)
        + 0.12 * np.sin(4 * phase - 1.2)
    )
    # Scale to ~4,000 counts peak-to-peak AC
    ac_signal = pulse * 2000.0

    # Respiratory baseline wander (~0.25 Hz, 500 counts)
    resp_wander = np.sin(2 * np.pi * 0.25 * t_sec) * 500.0

    # High frequency noise
    noise = rng.normal(0, 30.0, size=n_samples)

    # Finger placement transition in first 3.5 seconds
    dc_level = 160000.0
    settling_mask = 1.0 / (1.0 + np.exp(-3.0 * (t_sec - 2.0)))
    raw_ir = (dc_level * settling_mask) + (ac_signal + resp_wander + noise) * settling_mask
    raw_ir = np.clip(raw_ir, 500.0, 260000.0)

    # Ambient red channel
    raw_red = rng.normal(42.0, 5.0, size=n_samples)
    raw_red = np.clip(raw_red, 10.0, 100.0)

    df_ppg = pd.DataFrame({
        "sample_index": sample_index,
        "host_timestamp_ms": host_ts,
        "ir": np.round(raw_ir, 1),
        "red": np.round(raw_red, 1),
    })

    # 3. Synchronized Reference BP Events
    # Placed after 60s history has filled so predictions exist!
    t0_ms = host_ts[0]
    bp_events = [
        {
            "event_id": "DEMO_EVT_01",
            "host_timestamp_ms": t0_ms + 62000.0,
            "elapsed_s": 62.0,
            "event_type": "REFERENCE_BP_START",
            "label": "Cuff Inflation Start",
            "sbp": np.nan,
            "dbp": np.nan,
            "notes": "Automated oscillometric cuff inflation start [DEMO]",
        },
        {
            "event_id": "DEMO_EVT_02",
            "host_timestamp_ms": t0_ms + 70000.0,
            "elapsed_s": 70.0,
            "event_type": "REFERENCE_BP_RESULT",
            "label": "Reference Cuff Measurement #1",
            "sbp": 122.0,
            "dbp": 78.0,
            "notes": "Resting seated measurement [DEMO]",
        },
        {
            "event_id": "DEMO_EVT_03",
            "host_timestamp_ms": t0_ms + 82000.0,
            "elapsed_s": 82.0,
            "event_type": "REFERENCE_BP_START",
            "label": "Cuff Inflation Start",
            "sbp": np.nan,
            "dbp": np.nan,
            "notes": "Automated cuff inflation start [DEMO]",
        },
        {
            "event_id": "DEMO_EVT_04",
            "host_timestamp_ms": t0_ms + 90000.0,
            "elapsed_s": 90.0,
            "event_type": "REFERENCE_BP_RESULT",
            "label": "Reference Cuff Measurement #2",
            "sbp": 125.0,
            "dbp": 81.0,
            "notes": "Second confirmation measurement [DEMO]",
        },
    ]
    df_bp = pd.DataFrame(bp_events)

    metadata = {
        "IS_DEMO": True,
        "DEMO_WATERMARK": DEMO_WATERMARK_TEXT,
        "session_id": "DEMO_SIMULATION_SESSION",
        "subject_id": "DEMO_USER_01",
        "sensor_model": "MAX30102 (Synthetic Simulation)",
        "reference_device": "OMRON M3 (Simulated Reference)",
        "protocol": "Resting Seated 90s Protocol",
        "nominal_fs_hz": 100.0,
        "generation_timestamp": "2026-09-29T23:45:00Z",
        "description": "Synthetic demonstration dataset to test the Phase 6C desktop validation pipeline.",
    }

    return df_ppg, df_bp, metadata


def generate_demo_session() -> SessionData:
    """Instantiate a ready-to-run SessionData object with synthetic test data."""
    df_ppg_raw, df_bp_raw, metadata = generate_synthetic_ppg_and_events()
    df_ppg_replay, _ = _standardize_ppg_dataframe(df_ppg_raw)
    t0_host = float(df_ppg_replay["host_timestamp_ms"].iloc[0])
    df_bp_events, _ = _standardize_bp_events(df_bp_raw, t0_host_ms=t0_host)

    readme_text = (
        "======================================================================\n"
        "DEMO / SIMULATION SESSION CAPTURE (PHASE 6C)\n"
        f"{DEMO_WATERMARK_TEXT}\n"
        "======================================================================\n"
        "This dataset was generated algorithmically for software verification.\n"
        "It DOES NOT represent real human hardware measurements or clinical accuracy.\n"
    )

    return SessionData(
        session_id="DEMO_SIMULATION_SESSION",
        source_type="DEMO",
        source_name="synthetic_demo_session.zip",
        df_ppg_raw=df_ppg_raw,
        df_ppg_replay=df_ppg_replay,
        df_bp_events=df_bp_events,
        metadata=metadata,
        readme_text=readme_text,
        is_demo=True,
        load_warnings=[DEMO_WATERMARK_TEXT],
    )


def create_demo_zip_bytes() -> bytes:
    """Create in-memory zip bytes of the demo session package."""
    df_ppg_raw, df_bp_raw, metadata = generate_synthetic_ppg_and_events()
    readme_text = (
        "======================================================================\n"
        "PHASE 6C SESSION CAPTURE PACKAGE (DEMO)\n"
        f"{DEMO_WATERMARK_TEXT}\n"
        "======================================================================\n"
        "Contains synthetic test files matching the expected physical format:\n"
        "- ppg_samples.csv\n"
        "- bp_events.csv\n"
        "- session_metadata.json\n"
        "- README.txt\n"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ppg_samples.csv", df_ppg_raw.to_csv(index=False))
        z.writestr("bp_events.csv", df_bp_raw.to_csv(index=False))
        z.writestr("session_metadata.json", json.dumps(metadata, indent=2))
        z.writestr("README.txt", readme_text)

    return buf.getvalue()
