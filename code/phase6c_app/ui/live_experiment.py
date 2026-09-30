"""
Phase 6C / Phase 7 Live Experiment UI Component.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Renders the real-time hardware execution interface:
- Physical ESP32 + MAX30102 serial streaming
- Real-time scrolling PPG waveform with professional scientific axes & scales
- Automatic state reset on input source switching
- Real-time causal window accumulation and quality assessment
- 60-second temporal sequence progress (6 x 10s windows)
- Real inference with frozen Phase 4A CNN + Phase 4B GRU
- Real Phase 5C extreme-aware calibration
- Real Phase 7 multi-domain reliability classification (TRUST / REVIEW / ABSTAIN)
- Optional Groq explanation with local deterministic fallback
- Optional exploratory reference cuff comparison
"""

import time
import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from phase6c_app.analysis.live_engine import LiveExperimentEngine, discover_serial_ports
from phase6c_app.data_io.session_loader import load_session_from_zip
from phase6c_app.config import PROJECT_ROOT

def get_or_create_engine() -> LiveExperimentEngine:
    """Singleton getter for the frozen streaming experiment engine in session state."""
    if "live_engine" not in st.session_state or st.session_state.live_engine is None:
        st.session_state.live_engine = LiveExperimentEngine()
    return st.session_state.live_engine

def _plot_live_waveform(times: list, values: list, is_filtered: bool = True):
    """
    Render clean, minimal Apple-style PPG waveform with clear, subtle axes and scales.
    Restores the authentic physiological arterial pulse waveform with proper scientific axes and grid.
    """
    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor('#F5F5F7')
    ax.set_facecolor('#F5F5F7')

    # Subtle light grid ("the graph")
    ax.grid(True, linestyle="--", alpha=0.35, color="#d2d2d7")

    # Spines: clean bottom & left axes in subtle grey, hide top & right
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#d2d2d7')
        ax.spines[spine].set_linewidth(0.8)

    if len(values) > 1:
        if len(times) == len(values) and times[-1] > times[0]:
            t_arr = np.array(times)
        else:
            default_fs = 125.0 if is_filtered else 100.0
            t_arr = np.linspace(0.0, len(values) / default_fs, len(values))
            
        ax.plot(t_arr, values, color="#1d1d1f", linewidth=1.5)
        
        vmin, vmax = float(np.min(values)), float(np.max(values))
        pad = max(10.0 if is_filtered else 500.0, (vmax - vmin) * 0.12)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_xlim(t_arr[0], max(t_arr[-1], t_arr[0] + 0.5))
    else:
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(-600, 600) if is_filtered else ax.set_ylim(30000, 70000)
        ax.text(
            0.5, 0.5,
            "Waiting for incoming pulse samples...",
            ha="center", va="center", color="#86868b",
            transform=ax.transAxes, fontsize=12
        )

    # Clean, subtle axes styling
    ax.tick_params(axis='both', colors='#86868b', labelsize=9, length=3)
    if is_filtered:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.set_ylabel('PPG Amplitude (a.u.)', fontsize=9, color='#86868b', labelpad=4)
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.set_ylabel('Amplitude (counts)', fontsize=9, color='#86868b', labelpad=4)
        
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}s'))
    ax.set_xlabel('Time (s)', fontsize=9, color='#86868b', labelpad=4)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def _plot_three_channel_signals(win_tensor_3x1250: np.ndarray):
    """Render the 3 normalized causal derivative channels (PPG, VPG, APG) with clean axes."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.2), sharex=True)
    fig.patch.set_facecolor('#F5F5F7')
    
    t_10s = np.linspace(0.0, 10.0, 1250)
    channels = [
        (0, "#1d1d1f", "FILTERED PPG (0.5–8.0 Hz)"),
        (1, "#4F6EF7", "VPG (1st Derivative)"),
        (2, "#86868b", "APG (2nd Derivative)"),
    ]
    for ax, (ch_idx, color, label) in zip(axes, channels):
        ax.set_facecolor('#F5F5F7')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#d2d2d7')
            ax.spines[spine].set_linewidth(0.8)
        ax.grid(True, linestyle="--", alpha=0.35, color="#d2d2d7")
        ax.tick_params(axis='both', colors='#86868b', labelsize=8, length=2)
        ax.plot(t_10s, win_tensor_3x1250[ch_idx], color=color, linewidth=1.2)
        ax.text(0.01, 0.78, label, transform=ax.transAxes, fontsize=9, fontweight='500', color='#1d1d1f')
        ax.set_ylabel("Z-score", fontsize=8, color="#86868b")
        
    axes[-1].set_xlabel("Window Time (s)", fontsize=8, color="#86868b")
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}s'))
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def render_live_experiment_tab():
    """Main renderer for Live Experiment Mode."""
    engine = get_or_create_engine()

    # Header
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #86868b; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;'>Real-Time Research Mode</p>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 32px; margin-bottom: 6px;'>LIVE EXPERIMENT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; color: #333336; margin-bottom: 20px;'>Run the complete cuffless BP pipeline using an actual PPG recording.<br><span style='font-size: 13px; color: #86868b;'>Input → Signal Quality → Processing → 60 s Context → BP Estimate → Reliability</span></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Input Selection with Automatic State Reset on Switch
    # -------------------------------------------------------------------------
    input_options = [
        "Connect Hardware (ESP32)",
        "Upload Recorded Session",
        "Load Demo Session"
    ]

    if "active_input_source" not in st.session_state:
        st.session_state.active_input_source = input_options[0]

    st.markdown("<p style='font-size: 13px; font-weight: 600; color: #86868b; text-transform: uppercase; margin-bottom: 6px;'>Data Source</p>", unsafe_allow_html=True)
    
    selected_input = st.radio(
        "Select Data Source",
        input_options,
        index=input_options.index(st.session_state.active_input_source),
        horizontal=True,
        label_visibility="collapsed"
    )

    # Detect tab switch -> AUTOMATIC RESET!
    if selected_input != st.session_state.active_input_source:
        st.session_state.active_input_source = selected_input
        engine.reset()
        if engine.is_serial_connected:
            engine.disconnect_serial()
        for k in ["is_streaming_file", "stream_source_df", "stream_cursor", "live_explanation"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Input Option A: Connect Hardware
    if selected_input == "Connect Hardware (ESP32)":
        with st.container(border=True):
            st.markdown("<p style='font-size: 14px; font-weight: 600; color: #1d1d1f; margin-bottom: 8px;'>Physical Serial Acquisition (MAX30102 + ESP32)</p>", unsafe_allow_html=True)
            col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
            
            ports = discover_serial_ports()
            port_options = [p["device"] for p in ports]
            if not port_options:
                port_options = ["/dev/ttyUSB0", "/dev/ttyACM0", "COM3"]
                
            with col_p1:
                selected_port = st.selectbox("Serial Port", port_options, index=0)
            with col_p2:
                baud_rate = st.selectbox("Baud Rate", [921600, 115200, 460800, 230400], index=0)
            with col_p3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if not engine.is_serial_connected:
                    if st.button("Connect", use_container_width=True):
                        success = engine.connect_serial(selected_port, baud_rate)
                        if success:
                            st.success(f"Connected to {selected_port}")
                            st.rerun()
                        else:
                            st.error(engine.status_message)
                else:
                    if st.button("Disconnect", use_container_width=True):
                        engine.disconnect_serial()
                        st.info("Disconnected from hardware.")
                        st.rerun()

            # Connection status strip
            if engine.is_serial_connected:
                st.markdown(f"<p style='color: #2e7d32; font-size: 13px; font-weight: 500; margin-top: 8px;'>● <b>Connected:</b> {engine.connected_port} @ {baud_rate:,} baud &nbsp;|&nbsp; <b>DEVICE STREAM ACTIVE</b></p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #86868b; font-size: 13px; margin-top: 8px;'>○ Hardware disconnected. Connect your ESP32 or use recorded session.</p>", unsafe_allow_html=True)

    # Input Option B: Upload Recorded Session
    elif selected_input == "Upload Recorded Session":
        with st.container(border=True):
            st.markdown("<p style='font-size: 14px; font-weight: 600; color: #1d1d1f; margin-bottom: 8px;'>Upload Captured Physical Recording</p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Select session package (.zip) or PPG samples (.csv)", type=["zip", "csv"])
            if uploaded_file is not None:
                if st.button("Load & Stream Session", key="btn_load_uploaded"):
                    with st.spinner("Parsing recorded session..."):
                        try:
                            if uploaded_file.name.endswith(".zip"):
                                sdata = load_session_from_zip(uploaded_file)
                                df_replay = sdata.df_ppg_replay
                            else:
                                df_raw = pd.read_csv(uploaded_file)
                                df_replay = df_raw
                            
                            engine.reset()
                            st.session_state.stream_source_df = df_replay
                            st.session_state.stream_source_name = uploaded_file.name
                            st.session_state.stream_cursor = 0
                            st.session_state.is_streaming_file = True
                            st.success(f"Loaded {len(df_replay):,} samples from {uploaded_file.name}.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading session: {e}")

    # Input Option C: Load Demo Session
    elif selected_input == "Load Demo Session":
        with st.container(border=True):
            st.markdown("<p style='font-size: 14px; font-weight: 600; color: #1d1d1f; margin-bottom: 8px;'>Standard Physical Baseline Capture</p>", unsafe_allow_html=True)
            st.write("Loads the verified multi-session physical recording from Phase 6C (`04_20260930_161820.zip`).")
            if st.button("Load Baseline Recording", key="btn_load_demo_live"):
                with st.spinner("Loading baseline physical data..."):
                    zip_path = PROJECT_ROOT / "hardware" / "samples" / "04_20260930_161820.zip"
                    if not zip_path.exists():
                        zip_files = list((PROJECT_ROOT / "hardware" / "samples").glob("*.zip"))
                        zip_path = zip_files[0] if zip_files else None
                    
                    if zip_path and zip_path.exists():
                        sdata = load_session_from_zip(str(zip_path))
                        engine.reset()
                        st.session_state.stream_source_df = sdata.df_ppg_replay
                        st.session_state.stream_source_name = zip_path.name
                        st.session_state.stream_cursor = 0
                        st.session_state.is_streaming_file = True
                        st.success(f"Loaded {len(sdata.df_ppg_replay):,} samples from baseline recording.")
                        st.rerun()
                    else:
                        st.error("No sample recording found in hardware/samples/")

    # -------------------------------------------------------------------------
    # Stream Ingestion Loop (File or Hardware)
    # -------------------------------------------------------------------------
    if st.session_state.get("is_streaming_file", False) and "stream_source_df" in st.session_state:
        df_src = st.session_state.stream_source_df
        cursor = st.session_state.get("stream_cursor", 0)
        
        # Ingest in chunks of 100 samples per refresh
        chunk_sz = 100
        if cursor < len(df_src):
            chunk = df_src.iloc[cursor : min(cursor + chunk_sz, len(df_src))]
            for _, r in chunk.iterrows():
                s_idx = int(r.get("sample_index", cursor))
                exp_ts = float(r.get("expected_timestamp_ms", s_idx * 10.0))
                host_ts = float(r.get("host_timestamp_ms", exp_ts))
                ir_v = float(r.get("ir", 0.0))
                red_v = float(r.get("red", 0.0))
                engine.ingest_sample(s_idx, exp_ts, host_ts, ir_v, red_v)
            st.session_state.stream_cursor = cursor + len(chunk)
        else:
            st.session_state.is_streaming_file = False

    # If polling active hardware serial
    if engine.is_serial_connected:
        engine.poll_serial(max_lines=40)

    # -------------------------------------------------------------------------
    # Main Dashboard: Real-time PPG, Progress, Results
    # -------------------------------------------------------------------------
    st.markdown("<hr style='border: 0; border-top: 1px solid #E5E5EA; margin: 20px 0;'>", unsafe_allow_html=True)
    
    # Status Message Strip
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 10px; padding: 14px 20px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <p style="font-size: 12px; font-weight: 600; color: #86868b; text-transform: uppercase; margin-bottom: 2px;">System Status</p>
                <p style="font-size: 15px; font-weight: 500; color: #1d1d1f; margin: 0;">{engine.status_message}</p>
            </div>
            <div>
                <span class="{'trust-badge' if engine.pipeline_state=='COMPLETE' else 'neutral-badge'}">{engine.pipeline_state}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Real-Time PPG
    with st.container(border=True):
        col_hdr, col_mode = st.columns([3, 2])
        with col_hdr:
            st.markdown("<p style='font-size: 14px; font-weight: 600; color: #1d1d1f; margin-bottom: 4px;'>REAL-TIME PPG</p>", unsafe_allow_html=True)
        with col_mode:
            ppg_view = st.radio(
                "Signal View",
                ["Pulsatile PPG (Filtered)", "Raw Optical IR (ADC Counts)"],
                index=0,
                horizontal=True,
                label_visibility="collapsed"
            )

        if ppg_view == "Pulsatile PPG (Filtered)":
            _plot_live_waveform(engine.display_times, engine.display_ppg, is_filtered=True)
        else:
            _plot_live_waveform(engine.raw_display_times, engine.raw_display_ir, is_filtered=False)
        
        # Live Acquisition Metrics
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Samples Received", f"{engine.samples_received:,}")
        c_m2.metric("Acquisition Rate", f"{engine.effective_rate_hz:.1f} Hz")
        elapsed_s = engine.raw_display_times[-1] if engine.raw_display_times else 0.0
        c_m3.metric("Elapsed Time", f"{elapsed_s:.1f} s")
        qc_col = "#2e7d32" if engine.current_signal_qc == "PASS" else ("#f57f17" if engine.current_signal_qc == "WARN" else "#86868b")
        c_m4.markdown(f"<div style='text-align:center;'><p style='font-size:14px;color:#86868b;margin:0;'>Signal Quality</p><p style='font-size:22px;font-weight:600;color:{qc_col};margin:0;'>{engine.current_signal_qc}</p></div>", unsafe_allow_html=True)

        # Optional 3-Channel Processed View (PPG, VPG, APG)
        if getattr(engine, "latest_window_tensor", None) is not None:
            with st.expander("View 3-Channel Processed Waveforms (PPG, VPG, APG)", expanded=False):
                _plot_three_channel_signals(engine.latest_window_tensor)

    # -------------------------------------------------------------------------
    # Temporal Context Progress (60-second Sequence)
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: #1d1d1f; margin-bottom: 4px;'>60-SECOND TEMPORAL CONTEXT PROGRESS</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 13px; color: #86868b; margin-bottom: 16px;'>The temporal GRU requires 6 consecutive 10-second windows before generating blood pressure predictions.</p>", unsafe_allow_html=True)

        # 6 Window Boxes
        cols_w = st.columns(6)
        n_windows = len(engine.completed_windows)
        
        for w_i in range(6):
            with cols_w[w_i]:
                if w_i < n_windows:
                    w_rec = engine.completed_windows[w_i]
                    w_stat = w_rec["status"]
                    bg = "#e8f5e9" if w_stat == "PASS" else ("#fff8e1" if w_stat == "WARN" else "#ffebee")
                    fg = "#2e7d32" if w_stat == "PASS" else ("#f57f17" if w_stat == "WARN" else "#c62828")
                    txt = w_stat
                elif w_i == n_windows and engine.samples_received > 0:
                    bg = "#4F6EF7"
                    fg = "#FFFFFF"
                    txt = "COLLECTING"
                else:
                    bg = "#F5F5F7"
                    fg = "#86868b"
                    txt = "WAITING"

                st.markdown(
                    f"""
                    <div style="background: {bg}; border-radius: 8px; padding: 12px 6px; text-align: center; border: 1px solid #E5E5EA;">
                        <p style="font-size: 12px; font-weight: 600; color: {fg}; margin: 0 0 4px 0;">WIN {w_i+1:02d}</p>
                        <p style="font-size: 11px; font-weight: 600; color: {fg}; margin: 0;">{txt}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(f"<p style='text-align: center; font-size: 13px; color: #6e6e73; margin-top: 14px;'><b>Sequence Context:</b> {min(n_windows, 6)} / 6 windows ready ({min(n_windows * 10, 60)} / 60 seconds)</p>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Blood Pressure Estimation & Reliability Output (Centerpiece)
    # -------------------------------------------------------------------------
    if engine.latest_prediction is not None and engine.latest_reliability is not None:
        pred = engine.latest_prediction
        rel = engine.latest_reliability
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
        
        # BP Display Card
        with st.container(border=True):
            st.markdown("<p style='font-size: 13px; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 0.5px; text-align: center;'>Blood Pressure Estimate</p>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="bp-display">
                    {pred['calibrated_sbp']:.0f} <span style="color:#d2d2d7; font-weight:300;">/</span> {pred['calibrated_dbp']:.0f} <span class="bp-unit">mmHg</span>
                </div>
                <div class="bp-labels">
                    <span>SBP</span>
                    <span>DBP</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; gap: 30px; font-size: 13px; color: #86868b; margin-top: 10px;">
                    <span>SBP 95% Bound: [{pred['sbp_lower']:.1f} – {pred['sbp_upper']:.1f}]</span>
                    <span>DBP 95% Bound: [{pred['dbp_lower']:.1f} – {pred['dbp_upper']:.1f}]</span>
                    <span>Context: 60-second temporal history</span>
                </div>
                <p style="text-align: center; font-size: 12px; color: #86868b; margin-top: 12px;">Generated from physical PPG alone • Zero ECG input</p>
                """,
                unsafe_allow_html=True
            )

        # Phase 7 Reliability Card
        with st.container(border=True):
            state = rel["reliability_state"]
            badge_class = f"{state.lower()}-badge"
            
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 24px;">
                    <p style="font-size: 13px; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Phase 7 Prediction Reliability</p>
                    <span class="{badge_class}" style="font-size: 18px; padding: 6px 20px;">{state}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.write(f"**Signal Quality:** `{rel['signal_quality']}`")
                st.write(f"**Model Uncertainty:** `{rel['model_uncertainty']}`")
            with r_col2:
                st.write(f"**Prediction Interval:** `{rel['conformal_width']}`")
                st.write(f"**Temporal Stability:** `{rel['temporal_stability']}`")
                
            st.markdown(f"<p style='font-size: 13px; color: #6e6e73; margin-top: 12px;'><b>Risk Score:</b> <code>{rel['reliability_score']:.4f}</code> &nbsp;|&nbsp; <b>Recommendation:</b> {rel['recommendation']}</p>", unsafe_allow_html=True)

        # AI Explanation Section
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate AI Explanation", key="btn_live_explain", type="primary"):
            with st.spinner("Generating explanation..."):
                try:
                    from phase6c_app.llm_explainer import generate_reliability_explanation
                    st.session_state.live_explanation = generate_reliability_explanation(rel, timeout_s=15.0)
                except Exception:
                    from phase6c_app.llm_explainer import generate_explanation_locally
                    st.session_state.live_explanation = generate_explanation_locally(rel)

        if "live_explanation" in st.session_state and st.session_state.live_explanation:
            exp = st.session_state.live_explanation
            with st.container(border=True):
                st.markdown("### Result Explanation")
                if exp.get("summary"):
                    st.markdown(exp["summary"])
                if exp.get("signal_explanation"):
                    st.markdown(f"**Signal Evidence:** {exp['signal_explanation']}")
                if exp.get("state_interpretation"):
                    st.markdown(f"**State Assessment:** {exp['state_interpretation']}")
                if exp.get("recommendation_for_researcher"):
                    st.markdown(f"**Recommendation:** {exp['recommendation_for_researcher']}")
                disclaimer = exp.get("research_disclaimer", "This system is a research prototype and not a clinical device.")
                st.info(f"**Research Note:** {disclaimer}")
                st.caption("Generated from the structured research output")

        # ---------------------------------------------------------------------
        # Optional Exploratory Reference Cuff Comparison
        # ---------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Reference Cuff Measurement (Optional Comparison)", expanded=False):
            st.markdown("<p style='font-size: 13px; color: #6e6e73;'>Enter an external oscillometric cuff reading for exploratory comparison against the neural estimate. <b>This value is not fed into the model and does not recalibrate predictions.</b></p>", unsafe_allow_html=True)
            col_rc1, col_rc2, col_rc3 = st.columns([1, 1, 1])
            with col_rc1:
                ref_s = st.number_input("Reference SBP (mmHg)", min_value=60.0, max_value=240.0, value=120.0, step=1.0)
            with col_rc2:
                ref_d = st.number_input("Reference DBP (mmHg)", min_value=40.0, max_value=160.0, value=80.0, step=1.0)
            with col_rc3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("Compare", key="btn_compare_ref"):
                    engine.compare_reference(ref_s, ref_d)
                    
            if engine.reference_comparison:
                comp = engine.reference_comparison
                c_c1, c_c2 = st.columns(2)
                c_c1.metric("SBP Difference", f"{comp['diff_sbp']:+.1f} mmHg", delta=f"{comp['diff_sbp']:+.1f} vs Cuff", delta_color="inverse")
                c_c2.metric("DBP Difference", f"{comp['diff_dbp']:+.1f} mmHg", delta=f"{comp['diff_dbp']:+.1f} vs Cuff", delta_color="inverse")
                st.caption(comp["disclaimer"])

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Experiment Controls & Recording Export
    # -------------------------------------------------------------------------
    st.markdown("<br><hr style='border: 0; border-top: 1px solid #E5E5EA;'>", unsafe_allow_html=True)
    c_ctl1, c_ctl2, c_ctl3 = st.columns(3)
    
    with c_ctl1:
        if not engine.is_recording:
            if st.button("Start Recording", use_container_width=True):
                engine.start_recording()
                st.success("Recording started.")
                st.rerun()
        else:
            if st.button("Stop Recording", use_container_width=True):
                engine.stop_recording()
                st.info(f"Recorded {len(engine.recorded_samples):,} samples.")
                st.rerun()

    with c_ctl2:
        if engine.recorded_samples:
            csv_data = engine.export_recorded_csv()
            st.download_button(
                label="Save & Download CSV",
                data=csv_data,
                file_name=f"physical_ppg_capture_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("Save Session (No Data)", disabled=True, use_container_width=True)

    with c_ctl3:
        if st.button("Reset Experiment", use_container_width=True):
            engine.reset()
            if "is_streaming_file" in st.session_state:
                del st.session_state.is_streaming_file
            if "stream_source_df" in st.session_state:
                del st.session_state.stream_source_df
            if "stream_cursor" in st.session_state:
                del st.session_state.stream_cursor
            if "live_explanation" in st.session_state:
                del st.session_state.live_explanation
            st.rerun()

    # Rerun trigger if active streaming is ongoing
    if st.session_state.get("is_streaming_file", False) or engine.is_serial_connected:
        time.sleep(0.08)
        st.rerun()
