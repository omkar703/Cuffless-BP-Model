"""
Phase 6C: Desktop Reference-Cuff Validation & Analysis Dashboard
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

A Streamlit desktop application for researcher evaluation of physical MAX30102
recordings synchronized with reference oscillometric blood pressure events.

STRICT INVARIANTS:
- Zero Retraining: Models and calibration maps remain 100% frozen.
- Deterministic Pairing: Predefined pairing rule without manual error-dependent selection.
- Non-Destructive: Raw acquisition inputs are never modified or overwritten.
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure app package and project paths are importable
APP_DIR = Path(__file__).resolve().parent
CODE_DIR = APP_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

for p in [APP_DIR, CODE_DIR, CODE_DIR / "scripts", CODE_DIR / "phase4a"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase6c_app.config import (
    PHASE4A_CKPT,
    PHASE4B_CKPT,
    ISOTONIC_SBP_PKL,
    ISOTONIC_DBP_PKL,
    CONFORMAL_QUANTILES_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PAIRING_TOLERANCE_S,
    EXPECTED_TOTAL_PARAMS,
    DISCLAIMER_TEXT,
    DEMO_WATERMARK_TEXT,
)
from phase6c_app.data_io.session_loader import (
    SessionData,
    load_session_from_zip,
    load_session_from_files,
)
from phase6c_app.data_io.demo_generator import generate_demo_session, create_demo_zip_bytes
from phase6c_app.data_io.exporter import export_full_results
from phase6c_app.analysis.hardware_audit import audit_hardware_data
from phase6c_app.analysis.replay_engine import FrozenPipelineRunner, ReplayExecutionResult
from phase6c_app.analysis.metrics import compute_validation_summary
from phase6c_app.pairing.bp_pairing import pair_predictions_with_reference
from phase6c_app.visualization.plots import (
    plot_raw_ir_timeline,
    plot_derivatives_timeline,
    plot_prediction_vs_reference_timeline,
    plot_scatter_sbp_dbp,
    plot_error_distributions,
    plot_bland_altman,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Phase 6C — Reference-Cuff BP Validation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "session_data" not in st.session_state:
    st.session_state.session_data: Optional[SessionData] = None
if "audit_results" not in st.session_state:
    st.session_state.audit_results = None
if "replay_result" not in st.session_state:
    st.session_state.replay_result: Optional[ReplayExecutionResult] = None
if "pairing_tolerance" not in st.session_state:
    st.session_state.pairing_tolerance = DEFAULT_PAIRING_TOLERANCE_S
if "exclude_warn_windows" not in st.session_state:
    st.session_state.exclude_warn_windows = False
if "df_pairs" not in st.session_state:
    st.session_state.df_pairs = None
if "pairing_audit" not in st.session_state:
    st.session_state.pairing_audit = None
if "validation_metrics" not in st.session_state:
    st.session_state.validation_metrics = None
if "export_done" not in st.session_state:
    st.session_state.export_done = False
if "export_files" not in st.session_state:
    st.session_state.export_files = None
if "export_zip_bytes" not in st.session_state:
    st.session_state.export_zip_bytes = None


# -----------------------------------------------------------------------------
# Sidebar Navigation & Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 Phase 6C Validation")
    st.caption("Calibration-Free Cuffless BP Research Dashboard")
    st.markdown("---")

    steps = [
        "STEP 1: Load Session",
        "STEP 2: Inspect Hardware Data",
        "STEP 3: Review BP Events",
        "STEP 4: Run Frozen Pipeline",
        "STEP 5: Review Predictions",
        "STEP 6: Reference-BP Comparison",
        "STEP 7: Export Results",
        "STEP 8: Prediction Reliability",
    ]
    selected_step = st.radio("Navigation Steps", steps, index=0)

    st.markdown("---")
    st.subheader("📋 Session Status")
    if st.session_state.session_data is not None:
        sdata = st.session_state.session_data
        st.write(f"**Session ID:** `{sdata.session_id}`")
        st.write(f"**Source:** `{sdata.source_name}`")
        st.write(f"**Samples:** `{len(sdata.df_ppg_raw):,}`")
        st.write(f"**BP Events:** `{len(sdata.df_bp_events)}`")
        if sdata.is_demo:
            st.warning("⚠️ MODE: DEMO / SYNTHETIC DATA")
        else:
            st.success("🟢 MODE: PHYSICAL HARDWARE")
    else:
        st.info("No session loaded yet.")

    st.markdown("---")
    st.caption("**Scientific Safeguards:**")
    st.caption("• 100% Frozen Models (0 Trainable Parameters)")
    st.caption("• Zero Post-Hoc Tuning on Test Data")
    st.caption("• Predefined Deterministic Pairing (±15s)")


# -----------------------------------------------------------------------------
# STEP 1: Load Session
# -----------------------------------------------------------------------------
if selected_step == "STEP 1: Load Session":
    st.header("Step 1: Ingest Phase 6C Session Package")
    st.markdown(
        """
        Load the synchronized capture package received from the acquisition setup.
        The package contains 100 Hz optical PPG samples (`ppg_samples.csv`),
        reference oscillometric BP events (`bp_events.csv`), and session metadata (`session_metadata.json`).
        """
    )

    tab_zip, tab_files, tab_local, tab_demo = st.tabs([
        "📦 Upload Session ZIP",
        "📄 Upload Individual Files",
        "📁 Local Directory / File",
        "🧪 Software Demo Mode (Simulation)"
    ])

    with tab_zip:
        st.subheader("Ingest Session Archive (.zip)")
        uploaded_zip = st.file_uploader(
            "Select phase6c_session_capture.zip",
            type=["zip"],
            help="ZIP package containing ppg_samples.csv, bp_events.csv, and session_metadata.json"
        )
        if uploaded_zip is not None:
            if st.button("Load and Parse ZIP Package", key="btn_load_zip"):
                with st.spinner("Extracting and validating session archive..."):
                    try:
                        sdata = load_session_from_zip(uploaded_zip)
                        st.session_state.session_data = sdata
                        st.session_state.audit_results = audit_hardware_data(sdata.df_ppg_replay, sdata.source_name)
                        st.session_state.replay_result = None
                        st.session_state.df_pairs = None
                        st.session_state.export_done = False
                        st.success(f"Successfully loaded session `{sdata.session_id}` ({len(sdata.df_ppg_raw):,} samples, {len(sdata.df_bp_events)} BP events).")
                    except Exception as e:
                        st.error(f"Failed to load ZIP archive: {e}")

    with tab_files:
        st.subheader("Upload Individual Session Files")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            ppg_file = st.file_uploader("PPG Samples CSV (Required)", type=["csv"], key="upload_ppg_csv")
            meta_file = st.file_uploader("Session Metadata JSON (Optional)", type=["json"], key="upload_meta_json")
        with col_f2:
            bp_file = st.file_uploader("Reference BP Events CSV (Required for BP validation)", type=["csv"], key="upload_bp_csv")
            sess_id_in = st.text_input("Custom Session ID (Optional)", placeholder="e.g. session_subject01_rest")

        if ppg_file is not None:
            if st.button("Load Uploaded Files", key="btn_load_files"):
                with st.spinner("Validating and parsing uploaded files..."):
                    try:
                        sdata = load_session_from_files(
                            ppg_file=ppg_file,
                            bp_file=bp_file,
                            meta_file=meta_file,
                            session_id_override=sess_id_in if sess_id_in else None
                        )
                        st.session_state.session_data = sdata
                        st.session_state.audit_results = audit_hardware_data(sdata.df_ppg_replay, sdata.source_name)
                        st.session_state.replay_result = None
                        st.session_state.df_pairs = None
                        st.session_state.export_done = False
                        st.success(f"Successfully loaded session `{sdata.session_id}` ({len(sdata.df_ppg_raw):,} samples).")
                    except Exception as e:
                        st.error(f"Failed to load session files: {e}")

    with tab_local:
        st.subheader("Load Physical Data from Local Workspace Path")
        default_local_path = str(PROJECT_ROOT / "hardware" / "new report hardware" / "final_dataset_ready(1).csv")
        local_path_in = st.text_input("Enter local CSV or directory path:", value=default_local_path)
        if st.button("Load Local Dataset", key="btn_load_local"):
            p = Path(local_path_in)
            if not p.exists():
                st.error(f"Path does not exist: {p}")
            else:
                try:
                    sdata = load_session_from_files(ppg_file=p, bp_file=None, session_id_override=p.stem)
                    st.session_state.session_data = sdata
                    st.session_state.audit_results = audit_hardware_data(sdata.df_ppg_replay, sdata.source_name)
                    st.session_state.replay_result = None
                    st.session_state.df_pairs = None
                    st.session_state.export_done = False
                    st.success(f"Loaded local hardware file `{p.name}` ({len(sdata.df_ppg_raw):,} samples).")
                    st.info("Note: Loaded without a synchronized BP event file. Pipeline replay and waveform audits are available.")
                except Exception as e:
                    st.error(f"Error loading local dataset: {e}")

    with tab_demo:
        st.subheader("🧪 Synthetic Demo & Testing Mode")
        st.info(
            "Use this mode to test the complete validation pipeline, plots, metrics, reports, and downloads "
            "before your physical reference-cuff session capture arrives."
        )
        st.markdown(f"> **Notice**: Output generated in demo mode is explicitly watermarked `{DEMO_WATERMARK_TEXT}`.")

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            if st.button("▶️ Generate & Load Demo Session", key="btn_load_demo", type="primary"):
                with st.spinner("Synthesizing realistic 100 Hz PPG waveform and synchronized reference BP events..."):
                    demo_sdata = generate_demo_session()
                    st.session_state.session_data = demo_sdata
                    st.session_state.audit_results = audit_hardware_data(demo_sdata.df_ppg_replay, demo_sdata.source_name)
                    st.session_state.replay_result = None
                    st.session_state.df_pairs = None
                    st.session_state.export_done = False
                    st.success("Loaded synthetic demonstration session with 2 synchronized reference cuff measurements.")
        with col_d2:
            demo_bytes = create_demo_zip_bytes()
            st.download_button(
                label="📥 Download Sample phase6c_session_capture.zip",
                data=demo_bytes,
                file_name="phase6c_demo_session_capture.zip",
                mime="application/zip",
                help="Download this sample ZIP to inspect the exact file format expected from physical captures."
            )

    # Display Loaded Session Information
    if st.session_state.session_data is not None:
        sdata = st.session_state.session_data
        st.markdown("---")
        st.subheader("Loaded Session Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Session ID", sdata.session_id)
        c2.metric("Raw Samples", f"{len(sdata.df_ppg_raw):,}")
        duration_s = sdata.df_ppg_replay["elapsed_s"].iloc[-1] if len(sdata.df_ppg_replay) > 0 else 0.0
        c3.metric("Duration", f"{duration_s:.1f} s")
        c4.metric("Reference BP Events", len(sdata.df_bp_events))

        if sdata.load_warnings:
            with st.expander("Loader Warnings and Schema Adaptations", expanded=False):
                for w in sdata.load_warnings:
                    st.write(f"- {w}")


# -----------------------------------------------------------------------------
# STEP 2: Inspect Hardware Data
# -----------------------------------------------------------------------------
elif selected_step == "STEP 2: Inspect Hardware Data":
    st.header("Step 2: Hardware Data Integrity Audit")
    if st.session_state.session_data is None:
        st.warning("Please load a session in Step 1 first.")
    else:
        sdata = st.session_state.session_data
        if st.session_state.audit_results is None:
            st.session_state.audit_results = audit_hardware_data(sdata.df_ppg_replay, sdata.source_name)
        audit = st.session_state.audit_results

        verdict = audit.get("audit_verdict", "UNKNOWN")
        if verdict == "PASS":
            st.success(f"**Hardware Audit Status**: ✅ {verdict} — Signal integrity and timing are within nominal limits.")
        elif verdict == "WARN":
            st.warning(f"**Hardware Audit Status**: ⚠️ {verdict} — Anomalies detected, but compatible with streaming replay.")
        else:
            st.error(f"**Hardware Audit Status**: ❌ {verdict} — Severe discontinuities or missing channels detected.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Effective Acquisition Rate", f"{audit['host_timing']['effective_rate_hz']:.3f} Hz")
        col2.metric("Mean Host Interval", f"{audit['host_timing']['mean_interval_ms']:.3f} ms")
        col3.metric("Index Discontinuities", audit['sample_index']['discontinuities_vs_modal_step'])
        col4.metric("Jitter Outside 9–11ms", f"{audit['host_timing']['intervals_outside_9_11_ms']}")

        # Full Audit Details
        with st.expander("Detailed Acquisition Statistics Table", expanded=True):
            audit_table_data = [
                ("Total Rows", f"{audit['row_count']:,}"),
                ("Sample Index Range", f"{audit['sample_index']['first']:.0f} → {audit['sample_index']['last']:.0f}"),
                ("Modal Index Step", f"{audit['sample_index']['modal_step']:.0f} (Inferred: {audit['sample_index']['inferred_nominal_rate_hz']:.1f} Hz)"),
                ("Duplicate / Reverse Indices", f"{audit['sample_index']['duplicate_or_reverse_steps']}"),
                ("Host Interval Bounds", f"[{audit['host_timing']['min_interval_ms']:.1f}, {audit['host_timing']['max_interval_ms']:.1f}] ms"),
                ("Recording Duration", f"{audit['host_timing']['duration_s']:.3f} s"),
                ("IR Amplitude Range", f"[{audit['ir_signal']['min']:.0f}, {audit['ir_signal']['max']:.0f}] counts"),
                ("IR Mean ± Std", f"{audit['ir_signal']['mean']:.1f} ± {audit['ir_signal']['std']:.1f}"),
                ("IR Samples >= 40k Counts", f"{audit['ir_signal']['percent_above_40k']:.1f}%"),
                ("IR NaN or Inf Samples", f"{audit['ir_signal']['nan_or_inf']}"),
                ("Red Channel Mean Counts", f"{audit['red_signal']['mean']:.2f}"),
            ]
            st.table(pd.DataFrame(audit_table_data, columns=["Parameter", "Observed Value"]))

        # Waveform Visualization
        st.subheader("Raw IR Waveform & Event Timeline")
        fig_ir = plot_raw_ir_timeline(sdata.df_ppg_replay, sdata.df_bp_events, is_demo=sdata.is_demo)
        st.pyplot(fig_ir)
        plt.close(fig_ir)


# -----------------------------------------------------------------------------
# STEP 3: Review BP Events
# -----------------------------------------------------------------------------
elif selected_step == "STEP 3: Review BP Events":
    st.header("Step 3: Review Reference Blood Pressure Events")
    if st.session_state.session_data is None:
        st.warning("Please load a session in Step 1 first.")
    else:
        sdata = st.session_state.session_data
        df_events = sdata.df_bp_events

        if len(df_events) == 0:
            st.info("No BP events table was provided with this session. Add a `bp_events.csv` to evaluate validation metrics.")
        else:
            st.markdown(f"Found **{len(df_events)}** timestamped reference events in the session capture:")

            # Metrics on events
            ref_results = df_events[(df_events["event_type"] == "REFERENCE_BP_RESULT") | (df_events["sbp"].notna())]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Events Logged", len(df_events))
            c2.metric("Reference BP Measurements", len(ref_results))
            if len(ref_results) > 0 and ref_results["sbp"].notna().any():
                sbp_min = ref_results["sbp"].min()
                sbp_max = ref_results["sbp"].max()
                dbp_min = ref_results["dbp"].min()
                dbp_max = ref_results["dbp"].max()
                c3.metric("Reference BP Range", f"{sbp_min:.0f}–{sbp_max:.0f} / {dbp_min:.0f}–{dbp_max:.0f} mmHg")
            else:
                c3.metric("Reference BP Range", "N/A")

            st.dataframe(df_events, use_container_width=True)


# -----------------------------------------------------------------------------
# STEP 4: Run Frozen Pipeline
# -----------------------------------------------------------------------------
elif selected_step == "STEP 4: Run Frozen Pipeline":
    st.header("Step 4: Execute Frozen Phase 6B Streaming Pipeline")
    st.markdown(
        """
        Runs the exact causal research pipeline validated in Phase 6B:
        - **100→125 Hz** Rational Resampling ($P=5, Q=4$)
        - Stateful Causal **0.5–8.0 Hz** Butterworth SOS Bandpass
        - Causal Backward Finite Differences (**VPG**, **APG**)
        - 10-Second Windows ($1,250$ samples @ $125\\text{ Hz}$) with Quality Gating
        - **Frozen Phase 4A 1D CNN Encoder** ($146,978$ params, 0 trainable)
        - Rolling 6-Window Context ($60\\text{ s}$)
        - **Frozen Phase 4B 1-Layer Unidirectional Causal GRU** ($27,106$ params, 0 trainable)
        - **Phase 5C Extreme-Aware Recalibration** & 95% Conformal Intervals
        """
    )

    if st.session_state.session_data is None:
        st.warning("Please load a session in Step 1 first.")
    else:
        sdata = st.session_state.session_data

        with st.expander("Model Verification & Checkpoint Integrity", expanded=True):
            st.write(f"- Phase 4A CNN Checkpoint: `{PHASE4A_CKPT.name}`")
            st.write(f"- Phase 4B GRU Checkpoint: `{PHASE4B_CKPT.name}`")
            st.write(f"- Phase 5C Isotonic SBP/DBP: `{ISOTONIC_SBP_PKL.name}`, `{ISOTONIC_DBP_PKL.name}`")
            st.write(f"- Phase 5C Conformal Quantiles: `{CONFORMAL_QUANTILES_JSON.name}`")
            st.write(f"- **Total Model Parameters**: `{EXPECTED_TOTAL_PARAMS:,}` (**0 Trainable**)")

        if st.button("🚀 RUN PHASE 6C VALIDATION PIPELINE", type="primary", key="btn_run_pipeline"):
            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()
            status_placeholder.info("Initializing frozen neural models and DSP pipeline...")

            runner = FrozenPipelineRunner()
            try:
                load_meta = runner.load_models()
                status_placeholder.info(f"Models verified ({load_meta['total_params']:,} parameters, 0 trainable). Streaming replay in progress...")

                def update_progress(frac: float):
                    progress_bar.progress(frac)

                replay_result = runner.run_replay(sdata.df_ppg_replay, progress_callback=update_progress)
                st.session_state.replay_result = replay_result

                # Auto-run pairing if BP events are present
                if len(sdata.df_bp_events) > 0:
                    df_pairs, pairing_audit = pair_predictions_with_reference(
                        df_predictions=replay_result.df_predictions,
                        df_bp_events=sdata.df_bp_events,
                        session_id=sdata.session_id,
                        tolerance_s=st.session_state.pairing_tolerance,
                        exclude_warn_windows=st.session_state.exclude_warn_windows,
                    )
                    st.session_state.df_pairs = df_pairs
                    st.session_state.pairing_audit = pairing_audit
                    st.session_state.validation_metrics = compute_validation_summary(df_pairs)

                status_placeholder.success("Replay simulation completed successfully!")
            except Exception as e:
                status_placeholder.error(f"Execution error during replay: {e}")
                raise e

        # Show replay results if executed
        if st.session_state.replay_result is not None:
            rep = st.session_state.replay_result
            st.markdown("---")
            st.subheader("Replay Execution Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Execution Time", f"{rep.summary['replay_duration_seconds']:.3f} s")
            m2.metric("Mean Chunk Latency", f"{rep.summary['mean_chunk_latency_ms']:.3f} ms")
            m3.metric("Peak Chunk Latency", f"{rep.summary['peak_chunk_latency_ms']:.3f} ms")
            m4.metric("Host Headroom", f"{rep.summary['host_budget_headroom_x']:.1f}x")

            w1, w2, w3, w4 = st.columns(4)
            w1.metric("10s Windows Formed", rep.summary['total_windows_formed'])
            w1_qc = rep.summary['window_qc_counts']
            w2.metric("Windows PASS", w1_qc['PASS'])
            w3.metric("Windows WARN", w1_qc['WARN'])
            w4.metric("Predictions Generated", rep.summary['total_predictions_generated'])


# -----------------------------------------------------------------------------
# STEP 5: Review Predictions
# -----------------------------------------------------------------------------
elif selected_step == "STEP 5: Review Predictions":
    st.header("Step 5: Review Causal Model Predictions")
    if st.session_state.replay_result is None:
        st.warning("Please run the frozen pipeline in Step 4 first.")
    else:
        rep = st.session_state.replay_result
        df_pred = rep.df_predictions
        df_win = rep.df_windows

        tab_pred, tab_win, tab_traces = st.tabs(["📊 Temporal BP Predictions", "🪟 10s Window QC Table", "📈 Causal Waveforms"])

        with tab_pred:
            st.subheader("Generated SBP/DBP Predictions (60s Temporal Context)")
            if len(df_pred) == 0:
                st.info("No 60-second temporal sequences formed (recording duration was < 60 seconds).")
            else:
                pred_display_cols = [
                    "prediction_id", "target_window", "context_start_s", "context_end_s",
                    "quality_status", "raw_sbp", "raw_dbp", "calibrated_sbp", "calibrated_dbp",
                    "conformal_lower_sbp", "conformal_upper_sbp", "conformal_lower_dbp", "conformal_upper_dbp"
                ]
                avail_pred_cols = [c for c in pred_display_cols if c in df_pred.columns]
                st.dataframe(df_pred[avail_pred_cols], use_container_width=True)

        with tab_win:
            st.subheader("Window-by-Window Engineering Quality Gating")
            if len(df_win) > 0:
                win_display_cols = [
                    "window_id", "start_time_sec", "end_time_sec", "qc_status",
                    "estimated_hr_bpm", "detected_peaks", "baseline_drift_delta", "raw_ptp", "qc_reason"
                ]
                avail_cols = [c for c in win_display_cols if c in df_win.columns]
                st.dataframe(df_win[avail_cols], use_container_width=True)

        with tab_traces:
            st.subheader("Normalized Causal Derivatives (PPG, VPG, APG)")
            fig_traces = plot_derivatives_timeline(rep.streaming_traces, is_demo=st.session_state.session_data.is_demo)
            st.pyplot(fig_traces)
            plt.close(fig_traces)


# -----------------------------------------------------------------------------
# STEP 6: Reference-BP Comparison
# -----------------------------------------------------------------------------
elif selected_step == "STEP 6: Reference-BP Comparison":
    st.header("Step 6: Deterministic Reference-BP Comparison & Metrics")

    if st.session_state.replay_result is None:
        st.warning("Please run the frozen pipeline in Step 4 first.")
    elif st.session_state.session_data is None or len(st.session_state.session_data.df_bp_events) == 0:
        st.warning("No reference BP events found in loaded session. Validation metrics require a synchronized BP event table.")
    else:
        sdata = st.session_state.session_data
        rep = st.session_state.replay_result

        # Pairing Configuration Controls
        st.subheader("⚙️ Predefined Pairing Protocol Settings")
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            tol_val = st.slider(
                "Maximum Temporal Tolerance (seconds)",
                min_value=5.0,
                max_value=30.0,
                value=float(st.session_state.pairing_tolerance),
                step=1.0,
                help="Default research protocol tolerance is ±15.0 seconds. Matches nearest prediction to reference event."
            )
            if tol_val != st.session_state.pairing_tolerance:
                st.session_state.pairing_tolerance = tol_val
        with col_c2:
            warn_filter = st.checkbox(
                "Strict Quality Filter: Exclude WARN Windows",
                value=st.session_state.exclude_warn_windows,
                help="If checked, predictions containing WARN windows are excluded from metric calculations."
            )
            if warn_filter != st.session_state.exclude_warn_windows:
                st.session_state.exclude_warn_windows = warn_filter

        # Re-run pairing if settings changed or not run yet
        df_pairs, pairing_audit = pair_predictions_with_reference(
            df_predictions=rep.df_predictions,
            df_bp_events=sdata.df_bp_events,
            session_id=sdata.session_id,
            tolerance_s=st.session_state.pairing_tolerance,
            exclude_warn_windows=st.session_state.exclude_warn_windows,
        )
        st.session_state.df_pairs = df_pairs
        st.session_state.pairing_audit = pairing_audit
        val_metrics = compute_validation_summary(df_pairs)
        st.session_state.validation_metrics = val_metrics

        # Display Matched Pairs Table
        st.subheader("📋 Matched Prediction-Reference Pairs Audit")
        pairs_display_cols = [
            "reference_event_id", "reference_sbp", "reference_dbp",
            "prediction_id", "predicted_sbp_calibrated", "predicted_dbp_calibrated",
            "pairing_delta_s", "sbp_error_calibrated", "dbp_error_calibrated",
            "quality_status", "included_in_metrics", "exclusion_reason"
        ]
        avail_pairs_cols = [c for c in pairs_display_cols if c in df_pairs.columns]
        st.dataframe(df_pairs[avail_pairs_cols], use_container_width=True)

        # Validation Metrics Display
        n_matched = val_metrics["n_valid_matched_pairs"]
        st.markdown("---")
        st.subheader(f"📊 Statistical Validation Metrics (N = {n_matched} Valid Pairs)")

        if n_matched == 0:
            st.warning("No valid matched pairs found within the current temporal tolerance and quality filters.")
        else:
            cal_s = val_metrics["calibrated"]["sbp"]
            cal_d = val_metrics["calibrated"]["dbp"]
            raw_s = val_metrics["raw"]["sbp"]
            raw_d = val_metrics["raw"]["dbp"]

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("SBP MAE (Calibrated)", f"{cal_s['mae']:.2f} mmHg", delta=f"{cal_s['mae'] - raw_s['mae']:+.2f} vs raw", delta_color="inverse")
            m_col2.metric("SBP Bias", f"{cal_s['bias_mean_error']:+.2f} mmHg", f"SD: {cal_s['std_error']:.2f}")
            m_col3.metric("DBP MAE (Calibrated)", f"{cal_d['mae']:.2f} mmHg", delta=f"{cal_d['mae'] - raw_d['mae']:+.2f} vs raw", delta_color="inverse")
            m_col4.metric("DBP Bias", f"{cal_d['bias_mean_error']:+.2f} mmHg", f"SD: {cal_d['std_error']:.2f}")

            # Detailed Metrics Comparison Table
            with st.expander("Full Statistical Comparison (Raw vs Calibrated)", expanded=True):
                metrics_df = pd.DataFrame([
                    ("Mean Absolute Error (MAE)", f"{raw_s['mae']:.2f} mmHg", f"{cal_s['mae']:.2f} mmHg", f"{raw_d['mae']:.2f} mmHg", f"{cal_d['mae']:.2f} mmHg"),
                    ("Root Mean Squared Error (RMSE)", f"{raw_s['rmse']:.2f} mmHg", f"{cal_s['rmse']:.2f} mmHg", f"{raw_d['rmse']:.2f} mmHg", f"{cal_d['rmse']:.2f} mmHg"),
                    ("Mean Error / Bias", f"{raw_s['bias_mean_error']:+.2f} mmHg", f"{cal_s['bias_mean_error']:+.2f} mmHg", f"{raw_d['bias_mean_error']:+.2f} mmHg", f"{cal_d['bias_mean_error']:+.2f} mmHg"),
                    ("Standard Deviation of Error", f"{raw_s['std_error']:.2f} mmHg", f"{cal_s['std_error']:.2f} mmHg", f"{raw_d['std_error']:.2f} mmHg", f"{cal_d['std_error']:.2f} mmHg"),
                    ("Pearson Correlation (r)", f"{raw_s['pearson_r']:.3f}", f"{cal_s['pearson_r']:.3f}", f"{raw_d['pearson_r']:.3f}", f"{cal_d['pearson_r']:.3f}"),
                    ("Error <= 5 mmHg (BHS %)", f"{raw_s['bhs_grades']['percent_le_5mmHg']:.1f}%", f"{cal_s['bhs_grades']['percent_le_5mmHg']:.1f}%", f"{raw_d['bhs_grades']['percent_le_5mmHg']:.1f}%", f"{cal_d['bhs_grades']['percent_le_5mmHg']:.1f}%"),
                    ("Error <= 10 mmHg (BHS %)", f"{raw_s['bhs_grades']['percent_le_10mmHg']:.1f}%", f"{cal_s['bhs_grades']['percent_le_10mmHg']:.1f}%", f"{raw_d['bhs_grades']['percent_le_10mmHg']:.1f}%", f"{cal_d['bhs_grades']['percent_le_10mmHg']:.1f}%"),
                    ("AAMI Criteria Pass", "PASS" if raw_s['aami_compliant'] else "FAIL", "PASS" if cal_s['aami_compliant'] else "FAIL", "PASS" if raw_d['aami_compliant'] else "FAIL", "PASS" if cal_d['aami_compliant'] else "FAIL"),
                ], columns=["Metric", "SBP (Raw)", "SBP (Calibrated)", "DBP (Raw)", "DBP (Calibrated)"])
                st.table(metrics_df)

            # Diagnostic Plots
            st.markdown("---")
            st.subheader("Diagnostic Validation Visualizations")

            st.write("#### 1. Prediction Timeline with 95% Conformal Bounds vs Reference Cuff")
            fig_pred_timeline = plot_prediction_vs_reference_timeline(rep.df_predictions, df_pairs, is_demo=sdata.is_demo)
            st.pyplot(fig_pred_timeline)
            plt.close(fig_pred_timeline)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.write("#### 2. Predicted vs Reference Scatter")
                fig_scatter = plot_scatter_sbp_dbp(df_pairs, is_demo=sdata.is_demo)
                st.pyplot(fig_scatter)
                plt.close(fig_scatter)

            with col_p2:
                st.write("#### 3. Prediction Error Distributions")
                fig_err = plot_error_distributions(df_pairs, is_demo=sdata.is_demo)
                st.pyplot(fig_err)
                plt.close(fig_err)

            st.write("#### 4. Bland-Altman Agreement Plots")
            fig_ba = plot_bland_altman(df_pairs, is_demo=sdata.is_demo)
            st.pyplot(fig_ba)
            plt.close(fig_ba)


# -----------------------------------------------------------------------------
# STEP 7: Export Results
# -----------------------------------------------------------------------------
elif selected_step == "STEP 7: Export Results":
    st.header("Step 7: Generate Research Deliverables & Export Package")

    if st.session_state.replay_result is None or st.session_state.session_data is None:
        st.warning("Please complete Step 1 through Step 4 before exporting results.")
    else:
        sdata = st.session_state.session_data
        rep = st.session_state.replay_result
        df_pairs = st.session_state.df_pairs if st.session_state.df_pairs is not None else pd.DataFrame()
        pairing_audit = st.session_state.pairing_audit if st.session_state.pairing_audit is not None else {}
        val_metrics = st.session_state.validation_metrics if st.session_state.validation_metrics is not None else compute_validation_summary(df_pairs)
        audit_res = st.session_state.audit_results

        st.markdown(
            """
            Export a full, reproducible validation package to disk:
            - `session_audit.json`
            - `window_quality.csv`
            - `predictions.csv`
            - `reference_bp_events.csv`
            - `phase6c_prediction_reference_pairs.csv`
            - `figures/` (PNG plots 01 through 06)
            - `phase6c_validation_report.md`
            - `phase6c_validation_report.json`
            - `run_metadata.json`
            """
        )

        out_dir_in = st.text_input("Export Directory Path:", value=str(DEFAULT_OUTPUT_DIR))

        col_ex1, col_ex2 = st.columns([1, 1])
        with col_ex1:
            if st.button("💾 Generate & Save All Deliverables to Disk", type="primary", key="btn_save_deliverables"):
                with st.spinner("Generating reports, plots, and saving deliverables..."):
                    out_path = Path(out_dir_in)
                    saved_files, zip_bytes = export_full_results(
                        output_dir=out_path,
                        session_data=sdata,
                        audit_results=audit_res,
                        df_windows=rep.df_windows,
                        df_predictions=rep.df_predictions,
                        streaming_traces=rep.streaming_traces,
                        replay_summary=rep.summary,
                        df_pairs=df_pairs,
                        pairing_audit=pairing_audit,
                        validation_metrics=val_metrics,
                    )
                    st.session_state.export_done = True
                    st.session_state.export_files = saved_files
                    st.session_state.export_zip_bytes = zip_bytes
                    st.success(f"Successfully generated all validation deliverables in: `{out_path}`")

        with col_ex2:
            if st.session_state.export_zip_bytes is not None:
                st.download_button(
                    label="📦 Download Complete Validation ZIP Bundle",
                    data=st.session_state.export_zip_bytes,
                    file_name=f"phase6c_{sdata.session_id}_validation_export.zip",
                    mime="application/zip",
                    key="btn_download_zip"
                )

        # Markdown Report Preview
        if st.session_state.export_done and st.session_state.export_files:
            md_path = st.session_state.export_files.get("validation_report_md")
            if md_path and md_path.exists():
                st.markdown("---")
                st.subheader("📄 Generated Validation Report Preview")
                with open(md_path, "r") as f:
                    report_text = f.read()
                st.markdown(report_text)


# -----------------------------------------------------------------------------
# STEP 8: Prediction Reliability Monitor (Phase 7 Research Extension)
# -----------------------------------------------------------------------------
elif selected_step == "STEP 8: Prediction Reliability":
    st.header("Step 8: Prediction Reliability & Selective Abstention Monitor")
    st.caption("Phase 7 Research Extension: Trust-Aware Decision Support & Selective Prediction")

    st.markdown(
        """
        > [!IMPORTANT]
        > **Research Disclaimer**: This prediction reliability engine provides algorithmic confidence assessment 
        > and selective-abstention recommendations. It is **NOT** a disease diagnosis system and does not claim 
        > that a prediction is medically correct or incorrect in the absence of a synchronized reference cuff.
        """
    )

    if st.session_state.replay_result is None:
        st.warning("Please run the frozen pipeline in Step 4 first.")
    else:
        rep = st.session_state.replay_result
        df_pred = rep.df_predictions
        df_win = rep.df_windows

        if len(df_pred) == 0:
            st.info("No predictions available to evaluate reliability.")
        else:
            # Load fitted Phase 7 Reliability Engine
            rel_model_path = PROJECT_ROOT / "code/outputs/phase7_reliability/reliability_engine_model.pkl"
            
            try:
                import pickle
                with open(rel_model_path, "rb") as f:
                    rel_engine = pickle.load(f)
                engine_loaded = True
            except Exception as e:
                st.error(f"Could not load pre-fitted reliability model: {e}")
                engine_loaded = False

            if engine_loaded:
                # Compute multi-domain reliability features for session predictions
                payloads = []
                pred_rows = []

                # Rolling temporal features across session predictions
                sbp_cal_vals = df_pred["calibrated_sbp"].values
                dbp_cal_vals = df_pred["calibrated_dbp"].values
                n_preds = len(df_pred)

                for i, r in df_pred.iterrows():
                    # Window QC info
                    target_win = r.get("target_window", f"win_{i:02d}")
                    win_match = df_win[df_win["window_id"] == target_win]
                    
                    qc_p = 1.0 if r.get("quality_status") == "PASS" else 0.0
                    ptp_val = float(win_match["raw_ptp"].values[0]) if len(win_match) > 0 and "raw_ptp" in win_match else 40000.0
                    hr_val = float(win_match["estimated_hr_bpm"].values[0]) if len(win_match) > 0 and "estimated_hr_bpm" in win_match else 75.0
                    std_val = float(win_match["raw_std"].values[0]) if len(win_match) > 0 and "raw_std" in win_match else 10000.0

                    # Causal temporal stats up to i
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
                    pred_rows.append({
                        "Prediction ID": r["prediction_id"],
                        "Target Window": target_win,
                        "Context Time (s)": f"{r.get('context_start_s', 0):.0f}–{r.get('context_end_s', 60):.0f}s",
                        "Predicted SBP": f"{r['calibrated_sbp']:.1f} mmHg",
                        "Predicted DBP": f"{r['calibrated_dbp']:.1f} mmHg",
                        "Reliability State": p_load["reliability_state"],
                        "Risk Score": f"{p_load['reliability_score']:.3f}",
                        "Signal Quality": p_load["signal_quality"],
                        "Uncertainty": p_load["model_uncertainty"],
                        "Conformal Width": p_load["conformal_width"],
                        "Temporal Stability": p_load["temporal_stability"]
                    })

                df_rel_summary = pd.DataFrame(pred_rows)

                # Reliability Status Metrics
                n_trust = sum(1 for p in payloads if p["reliability_state"] == "TRUST")
                n_review = sum(1 for p in payloads if p["reliability_state"] == "REVIEW")
                n_abstain = sum(1 for p in payloads if p["reliability_state"] == "ABSTAIN")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Predictions", n_preds)
                c2.metric("🟢 TRUST (High Confidence)", f"{n_trust} ({n_trust/n_preds*100:.0f}%)")
                c3.metric("🟡 REVIEW (Moderate Uncertainty)", f"{n_review} ({n_review/n_preds*100:.0f}%)")
                c4.metric("🔴 ABSTAIN (Elevated Risk)", f"{n_abstain} ({n_abstain/n_preds*100:.0f}%)")

                st.markdown("---")

                # Interactive Prediction Inspector (Card Layout)
                st.subheader("🔍 Prediction Reliability Inspector")
                pred_options = [f"{p['prediction_id']} ({p['calibrated_sbp']:.1f}/{p['calibrated_dbp']:.1f} mmHg)" for _, p in df_pred.iterrows()]
                selected_pred_idx = st.selectbox("Select Prediction to Inspect:", range(len(pred_options)), format_func=lambda x: pred_options[x])

                sel_payload = payloads[selected_pred_idx]
                sel_row = df_pred.iloc[selected_pred_idx]

                # Render Card Layout
                st.markdown("### 📋 Prediction Reliability Summary Card")
                state_badge = {
                    "TRUST": "🟢 **STATUS: TRUST (High Confidence)**",
                    "REVIEW": "🟡 **STATUS: PREDICTION REVIEW (Moderate Uncertainty)**",
                    "ABSTAIN": "🔴 **STATUS: ABSTAIN (Elevated Error Risk)**"
                }.get(sel_payload["reliability_state"], "⚪ STATUS: UNKNOWN")

                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #ccc; border-radius: 8px; padding: 16px; background-color: #f9f9f9; margin-bottom: 16px;">
                            <h2 style="margin: 0; color: #333; text-align: center;">BP ESTIMATION: {sel_payload['prediction']['sbp']} / {sel_payload['prediction']['dbp']} mmHg</h2>
                            <p style="text-align: center; font-size: 1.2em; margin-top: 8px;">{state_badge}</p>
                            <hr style="margin: 12px 0;">
                            <table style="width: 100%; border-collapse: collapse; font-size: 1.05em;">
                                <tr>
                                    <td style="padding: 6px; font-weight: bold;">Signal Quality:</td>
                                    <td style="padding: 6px;">{'✓ PASS' if sel_payload['signal_quality']=='PASS' else '⚠ MARGINAL'}</td>
                                    <td style="padding: 6px; font-weight: bold;">Model Uncertainty:</td>
                                    <td style="padding: 6px;">{'✓ LOW' if sel_payload['model_uncertainty']=='LOW' else ('⚠ MODERATE' if sel_payload['model_uncertainty']=='MODERATE' else '❌ ELEVATED')}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 6px; font-weight: bold;">Conformal Width:</td>
                                    <td style="padding: 6px;">{'✓ STANDARD' if sel_payload['conformal_width']=='STANDARD' else '⚠ WIDE'}</td>
                                    <td style="padding: 6px; font-weight: bold;">Temporal Stability:</td>
                                    <td style="padding: 6px;">{'✓ STABLE' if sel_payload['temporal_stability']=='STABLE' else '⚠ UNSTABLE'}</td>
                                </tr>
                            </table>
                            <hr style="margin: 12px 0;">
                            <p style="margin: 4px 0;"><b>Estimated Risk Score:</b> <code>{sel_payload['reliability_score']:.4f}</code> (Operating range: &le; {rel_engine.tau_trust:.3f} = TRUST, &gt; {rel_engine.tau_abstain:.3f} = ABSTAIN)</p>
                            <p style="margin: 4px 0;"><b>Recommendation:</b> {sel_payload['recommendation']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with st.expander("🔬 View Diagnostic Evidence & Reasons", expanded=True):
                    st.write("**Specific Evidence Signals:**")
                    for rz in sel_payload["reliability_reasons"]:
                        st.write(f"- {rz}")

                # ------------------------------------------------------------------
                # AI Explanation Section (Optional Groq — explicit trigger only)
                # ------------------------------------------------------------------
                st.markdown("---")
                st.subheader("🤖 AI-Powered Explanation (Optional)")

                st.markdown(
                    """
                    > **Research Boundary Notice**: The deterministic reliability engine (above) has already
                    > made the reliability classification. The AI explanation below only translates the
                    > computed result into plain language. **The AI cannot change the TRUST / REVIEW / ABSTAIN
                    > state, the BP prediction, the reliability score, or any research metric.**
                    """
                )

                st.caption(
                    "⚠️ Groq is called ONLY when you click the button below. "
                    "The deterministic BP/reliability system works fully offline without Groq."
                )

                if st.button("✨ Generate AI Explanation", key="btn_groq_explain"):
                    with st.spinner("Generating explanation via Groq (or local fallback)…"):
                        try:
                            # Import here so app works even if module has import issues
                            from llm_explainer import generate_reliability_explanation
                            explanation = generate_reliability_explanation(sel_payload, timeout_s=15.0)
                        except Exception as ex:
                            # Module-level import or other error — show fallback
                            try:
                                from llm_explainer import generate_explanation_locally
                                explanation = generate_explanation_locally(sel_payload)
                                explanation["source"] = f"local_fallback_import_error:{type(ex).__name__}"
                            except Exception:
                                explanation = {
                                    "source": "error",
                                    "summary": f"Explanation unavailable: {type(ex).__name__}",
                                    "signal_explanation": "",
                                    "state_interpretation": "",
                                    "recommendation_for_researcher": "Check that llm_explainer.py is present in the app directory.",
                                    "research_disclaimer": "This system is a research instrument and not a certified medical device."
                                }

                    src = explanation.get("source", "unknown")
                    if src == "groq":
                        st.success("✅ Explanation generated via Groq API.")
                    elif src.startswith("local_fallback_no_key"):
                        st.info("ℹ️ GROQ_API not configured — showing deterministic local explanation.")
                    elif src.startswith("local_fallback"):
                        st.warning(f"⚠️ Groq unavailable ({src}) — showing deterministic local explanation.")
                    else:
                        st.error(f"❌ Explanation error (source: {src}). Showing best available result.")

                    # Show explanation content
                    if explanation.get("summary"):
                        st.markdown("**📝 Summary**")
                        st.write(explanation["summary"])

                    if explanation.get("signal_explanation"):
                        st.markdown("**🔍 Signal Evidence Explained**")
                        st.write(explanation["signal_explanation"])

                    if explanation.get("state_interpretation"):
                        st.markdown("**📊 State Interpretation**")
                        st.write(explanation["state_interpretation"])

                    if explanation.get("recommendation_for_researcher"):
                        st.markdown("**💡 Recommendation for Researcher**")
                        st.write(explanation["recommendation_for_researcher"])

                    # Always show disclaimer
                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid #f0a500; padding: 8px 12px; background: #fffbf0; margin-top: 12px; border-radius: 4px; font-size: 0.85em; color: #555;">
                        <b>⚠️ Research Disclaimer:</b> {explanation.get("research_disclaimer", "")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.expander("🔎 Raw Explanation JSON", expanded=False):
                        # Strip internal _verified_state before display
                        display_exp = {k: v for k, v in explanation.items() if not k.startswith("_")}
                        st.json(display_exp)

                with st.expander("📋 Reliability Payload JSON (Interface Definition)", expanded=False):
                    st.caption("Structured JSON produced by the deterministic reliability engine — sent to Groq for explanation only:")
                    st.json(sel_payload)

                st.markdown("---")
                st.subheader("📊 Full Session Reliability Table")
                st.dataframe(df_rel_summary, use_container_width=True)

                # Research disclaimer footer
                st.markdown(
                    """
                    ---
                    > **Phase 7 Research Extension Disclaimer**: The TRUST / REVIEW / ABSTAIN classifications
                    > are produced by an exploratory algorithmic reliability model trained on MIMIC-II research
                    > data. They reflect measurement consistency signals — they do **NOT** guarantee clinical
                    > accuracy of the estimated blood pressure values. This application is a research instrument
                    > and is **NOT** a certified medical device.
                    """
                )

