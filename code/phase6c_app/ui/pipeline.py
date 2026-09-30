import sys
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = APP_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

for p in [APP_DIR, CODE_DIR, CODE_DIR / "scripts", CODE_DIR / "phase4a"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase6c_app.data_io.session_loader import load_session_from_zip
from phase6c_app.analysis.hardware_audit import audit_hardware_data
from phase6c_app.analysis.replay_engine import FrozenPipelineRunner

def load_and_run_demo_session():
    """Loads the demo session and runs the frozen pipeline silently."""
    if st.session_state.get("demo_data") is not None:
        return st.session_state.demo_data

    # Use existing hardware demo zip
    zip_path = PROJECT_ROOT / "hardware" / "samples" / "04_20260930_161820.zip"
    if not zip_path.exists():
        # Fallback to another zip if this specific one isn't there
        zip_files = list((PROJECT_ROOT / "hardware" / "samples").glob("*.zip"))
        if zip_files:
            zip_path = zip_files[0]
        else:
            raise FileNotFoundError("No demo zip found in hardware/samples/")

    # 1. Load Session
    sdata = load_session_from_zip(str(zip_path))
    
    # 2. Audit
    audit_results = audit_hardware_data(sdata.df_ppg_replay, sdata.source_name)
    
    # 3. Inference Replay
    runner = FrozenPipelineRunner()
    runner.load_models()
    replay_result = runner.run_replay(sdata.df_ppg_replay, progress_callback=lambda x: None)

    # 4. Generate Reliability Payload for a representative prediction
    import pickle
    rel_model_path = PROJECT_ROOT / "code/outputs/phase7_reliability/reliability_engine_model.pkl"
    with open(rel_model_path, "rb") as f:
        rel_engine = pickle.load(f)
    
    df_pred = replay_result.df_predictions
    
    payloads = []
    # Simplified feature extraction identical to app.py
    sbp_cal_vals = df_pred["calibrated_sbp"].values
    dbp_cal_vals = df_pred["calibrated_dbp"].values
    
    for i, r in df_pred.iterrows():
        target_win = r.get("target_window_index", 0)
        df_win_stats = replay_result.df_windows if hasattr(replay_result, 'df_windows') else None
        
        # We just need some dummy values for standard deviation / signal quality if not available, 
        # but let's try to grab them from df_win if possible, else 0
        qc_p = 1
        ptp_val = 0.5
        std_val = 0.1
        hr_val = 70.0
        
        if df_win_stats is not None and "window_index" in df_win_stats.columns:
            w_row = df_win_stats[df_win_stats["window_index"] == target_win]
            if len(w_row) > 0:
                qc_p = 1 if w_row.iloc[0].get("qc_status", "PASS") == "PASS" else 0
                ptp_val = w_row.iloc[0].get("ir_ptp", 0.5)
                std_val = w_row.iloc[0].get("ir_std", 0.1)
                hr_val = w_row.iloc[0].get("estimated_hr_bpm", 70.0)
        
        w3_s = max(0, i - 2)
        sub_s = sbp_cal_vals[w3_s:i+1]
        sub_d = dbp_cal_vals[w3_s:i+1]
        r_std_s = float(np.std(sub_s, ddof=1)) if len(sub_s) > 1 else 0.0
        r_std_d = float(np.std(sub_d, ddof=1)) if len(sub_d) > 1 else 0.0
        max_j_s = float(np.max(np.abs(np.diff(sub_s)))) if len(sub_s) > 1 else 0.0
        max_j_d = float(np.max(np.abs(np.diff(sub_d)))) if len(sub_d) > 1 else 0.0

        conf_w_s = float(r["conformal_upper_sbp"] - r["conformal_lower_sbp"]) if "conformal_upper_sbp" in r else 50.0
        conf_w_d = float(r["conformal_upper_dbp"] - r["conformal_lower_dbp"]) if "conformal_upper_dbp" in r else 25.0

        feat_row = {
            "qc_pass": qc_p,
            "ppg_ptp": ptp_val,
            "ppg_std": std_val,
            "ppg_clipped_fraction": 0.0,
            "ppg_pulse_count": 14.0,
            "estimated_hr_bpm": hr_val,
            "sbp_uncertainty": 14.2,
            "dbp_uncertainty": 7.6,
            "sbp_conformal_width_90": conf_w_s,
            "dbp_conformal_width_90": conf_w_d,
            "rolling_sbp_std_3": r_std_s,
            "rolling_dbp_std_3": r_std_d,
            "max_abs_sbp_jump_3": max_j_s,
            "max_abs_dbp_jump_3": max_j_d,
            "median_abs_sbp_change_3": max_j_s,
            "median_abs_dbp_change_3": max_j_d
        }

        p_load = rel_engine.generate_explanation_payload(
            sbp_pred=r["calibrated_sbp"],
            dbp_pred=r["calibrated_dbp"],
            features=feat_row
        )
        payloads.append(p_load)

    # Grab the middle prediction for the demo walkthrough
    mid_idx = len(payloads) // 2
    if mid_idx >= len(payloads): mid_idx = 0
    
    demo_data = {
        "sdata": sdata,
        "audit": audit_results,
        "replay": replay_result,
        "payloads": payloads,
        "demo_idx": mid_idx,
    }
    st.session_state.demo_data = demo_data
    return demo_data
