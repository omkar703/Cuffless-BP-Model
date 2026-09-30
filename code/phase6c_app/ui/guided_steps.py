import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from phase6c_app.ui.components import render_step_title, render_what_this_means

def _setup_plot():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor('#F5F5F7')
    ax.set_facecolor('#F5F5F7')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def render_step_01(demo_data, tech_mode):
    render_step_title(1, "THE SIGNAL", "We start with a raw optical pulse signal captured from the MAX30102 sensor.")
    
    sdata = demo_data["sdata"]
    df_raw = sdata.df_ppg_raw
    
    audit = demo_data.get("audit", {})
    df_replay = sdata.df_ppg_replay
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Plot raw waveform
        if not df_raw.empty and "ir" in df_raw.columns:
            fig, ax = _setup_plot()
            sample_data = df_raw["ir"].iloc[:500] if len(df_raw) > 500 else df_raw["ir"]
            ax.plot(sample_data.values, color="#1d1d1f", linewidth=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        elif not df_replay.empty and "ir" in df_replay.columns:
            fig, ax = _setup_plot()
            sample_data = df_replay["ir"].iloc[:500] if len(df_replay) > 500 else df_replay["ir"]
            ax.plot(sample_data.values, color="#1d1d1f", linewidth=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
    with col2:
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        st.write(f"**Session:** `{sdata.session_id}`")
        st.write(f"**Samples:** `{len(df_raw):,}`")
        rate_hz = float(audit.get("effective_rate_hz", audit.get("inferred_nominal_rate", 100.0)))
        st.write(f"**Acquisition:** `{rate_hz:.1f} Hz`")
        duration = float(df_replay["elapsed_s"].iloc[-1]) if not df_replay.empty and "elapsed_s" in df_replay else (len(df_raw) / rate_hz if rate_hz > 0 else 0.0)
        st.write(f"**Duration:** `{duration:.1f} s`")
        st.markdown("</div>", unsafe_allow_html=True)
        
    render_what_this_means("Before estimating blood pressure, the system first checks whether the optical signal contains a usable pulse waveform.")
    
    if tech_mode:
        with st.expander("Technical Details", expanded=False):
            st.json({"source": sdata.source_name, "fields": list(df_raw.columns)})

def render_step_02(demo_data, tech_mode):
    render_step_title(2, "IS THE SIGNAL USABLE?", "Can the system trust this recording enough to continue?")
    
    audit = demo_data["audit"]
    verdict = audit.get("audit_verdict", "UNKNOWN")
    
    is_pass = verdict == "PASS"
    color_class = "trust-text" if is_pass else ("review-text" if verdict == "WARN" else "abstain-text")
    
    st.markdown(f"""
    <div style="background: white; border: 1px solid #E5E5EA; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px;" class="fade-in">
        <p style="font-size: 14px; color: #86868b; text-transform: uppercase; font-weight: 600; margin-bottom: 8px;">Signal Quality</p>
        <h2 class="{color_class}" style="margin: 0;">{verdict}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("<div style='text-align:center;'><b>Pulse waveform</b><br><span style='color:#2e7d32'>PASS</span></div>", unsafe_allow_html=True)
    c2.markdown("<div style='text-align:center;'><b>Clipping</b><br><span style='color:#2e7d32'>LOW</span></div>", unsafe_allow_html=True)
    c3.markdown("<div style='text-align:center;'><b>Pulse detection</b><br><span style='color:#2e7d32'>STABLE</span></div>", unsafe_allow_html=True)
    c4.markdown("<div style='text-align:center;'><b>Signal amplitude</b><br><span style='color:#2e7d32'>WITHIN RANGE</span></div>", unsafe_allow_html=True)
    
    render_what_this_means("Good signal quality means the system has a recognizable pulse waveform to process. Poor signal quality can reduce the reliability of everything that follows.")
    
    if tech_mode:
        with st.expander("Technical Details", expanded=False):
            st.json(audit)

def render_step_03(demo_data, tech_mode):
    render_step_title(3, "TURNING THE WAVEFORM INTO FEATURES", "We preserve the pulse shape while reducing unwanted variation.")
    
    replay = demo_data["replay"]
    traces = getattr(replay, "streaming_traces", {})
    
    if traces and "ppg" in traces and len(traces["ppg"]) > 0:
        fig, axes = plt.subplots(4, 1, figsize=(10, 5.0), sharex=True)
        fig.patch.set_facecolor('#F5F5F7')
        
        channels = [
            ("raw_resampled", "#6e6e73", "RAW PPG (125 Hz resampled)"),
            ("ppg", "#1d1d1f", "FILTERED PPG (0.5–8.0 Hz bandpass)"),
            ("vpg", "#4F6EF7", "VPG (1st Derivative — velocity)"),
            ("apg", "#86868b", "APG (2nd Derivative — acceleration)"),
        ]
        
        # Display 4 seconds = 500 samples
        n_pts = min(500, len(traces["ppg"]))
        
        for ax, (ch_key, color, label) in zip(axes, channels):
            ax.set_facecolor('#F5F5F7')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            data = traces[ch_key][:n_pts]
            ax.plot(data, color=color, linewidth=1.5)
            ax.text(0.01, 0.75, label, transform=ax.transAxes, fontsize=10, fontweight='500', color='#1d1d1f')
            
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Filtered waveform representations are being processed.")
        
    render_what_this_means(
        "The system preserves the pulse shape while reducing unwanted variation and derives two additional representations of the waveform: "
        "VPG emphasizes how quickly the pulse waveform is changing, and APG highlights changes in that pulse-wave movement."
    )
    
    if tech_mode:
        with st.expander("Technical Details", expanded=False):
            st.write("- **Sampling Rate**: 125 Hz (resampled from 100 Hz acquisition)")
            st.write("- **Filter Band**: 0.5 – 8.0 Hz causal Butterworth SOS")
            st.write("- **Window Length**: 10 seconds (1,250 samples)")
            st.write("- **Normalization**: Per-window z-score normalization across all 3 channels")

def render_step_04(demo_data, tech_mode):
    render_step_title(4, "ONE BEAT IS NOT ENOUGH", "Instead of estimating blood pressure from a single 10-second segment, the temporal model looks at six consecutive segments.")
    
    st.markdown("""
    <div style="display: flex; justify-content: space-between; margin: 30px 0;" class="fade-in">
        <div style="width: 15%; height: 60px; background: #E5E5EA; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 500;">01</div>
        <div style="width: 15%; height: 60px; background: #E5E5EA; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 500;">02</div>
        <div style="width: 15%; height: 60px; background: #E5E5EA; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 500;">03</div>
        <div style="width: 15%; height: 60px; background: #E5E5EA; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 500;">04</div>
        <div style="width: 15%; height: 60px; background: #E5E5EA; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 500;">05</div>
        <div style="width: 15%; height: 60px; background: #4F6EF7; color: white; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 600;">06</div>
    </div>
    <div style="text-align: center; color: #86868b; font-weight: 500; letter-spacing: 1px; font-size: 13px; margin-top: -15px; margin-bottom: 30px;">
        ↓<br>60 SECOND CONTEXT
    </div>
    """, unsafe_allow_html=True)
    
    col_cnn, col_gru = st.columns(2)
    with col_cnn:
        st.markdown(
            """
            <div style="background: white; border: 1px solid #E5E5EA; border-radius: 10px; padding: 18px; text-align: center;">
                <p style="font-weight: 600; font-size: 16px; margin-bottom: 6px; color: #1d1d1f;">CNN Layer</p>
                <p style="font-size: 14px; color: #6e6e73; margin: 0;">Extracts spatial pulse features from each 10-second window</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_gru:
        st.markdown(
            """
            <div style="background: white; border: 1px solid #E5E5EA; border-radius: 10px; padding: 18px; text-align: center;">
                <p style="font-weight: 600; font-size: 16px; margin-bottom: 6px; color: #1d1d1f;">Temporal GRU</p>
                <p style="font-size: 14px; color: #6e6e73; margin: 0;">Learns how features evolve across the 60-second sequence</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    render_what_this_means("Recent pulse behavior provides more context than a single 10-second window. The sequence helps capture short-term changes in the pulse pattern.")

def render_step_05(demo_data, tech_mode):
    render_step_title(5, "THE BLOOD PRESSURE ESTIMATE", "The frozen neural model converts the processed pulse information into an SBP/DBP estimate.")
    
    idx = demo_data["demo_idx"]
    payload = demo_data["payloads"][idx]
    sbp = payload["prediction"]["sbp"]
    dbp = payload["prediction"]["dbp"]
    
    st.markdown(f"""
    <div style="background: white; border: 1px solid #E5E5EA; border-radius: 12px; padding: 40px; margin: 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);" class="fade-in">
        <div class="bp-display">
            {sbp:.0f} <span style="color:#d2d2d7; font-weight:300;">/</span> {dbp:.0f} <span class="bp-unit">mmHg</span>
        </div>
        <div class="bp-labels">
            <span>SBP</span>
            <span>DBP</span>
        </div>
        <hr style="border: 0; border-top: 1px solid #E5E5EA; margin: 24px 0;">
        <div style="display: flex; justify-content: center; gap: 40px; font-size: 14px; color: #86868b;">
            <span>Estimated from PPG</span>
            <span>No ECG input</span>
            <span>60-second temporal context</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    render_what_this_means("The frozen CNN extracts pulse-wave features, and the temporal GRU combines information across the recent 60-second sequence.")

def render_step_06(demo_data, tech_mode):
    render_step_title(6, "HOW MUCH SHOULD WE TRUST THIS ESTIMATE?", "A separate research layer evaluates signals associated with prediction error risk.")
    
    idx = demo_data["demo_idx"]
    payload = demo_data["payloads"][idx]
    state = payload["reliability_state"]
    
    badge_class = f"{state.lower()}-badge"
    
    st.markdown(f"""
    <div style="background: white; border: 1px solid #E5E5EA; border-radius: 12px; padding: 30px; margin-bottom: 20px;" class="fade-in">
        <div style="text-align: center; margin-bottom: 30px;">
            <p style="font-size: 13px; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Prediction Reliability</p>
            <span class="{badge_class}" style="font-size: 20px; padding: 8px 24px;">{state}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
            <div style="padding: 12px; background: #F5F5F7; border-radius: 8px; display: flex; justify-content: space-between;">
                <span style="color: #86868b; font-weight: 500;">Signal quality</span>
                <span style="font-weight: 600;">{payload['signal_quality']}</span>
            </div>
            <div style="padding: 12px; background: #F5F5F7; border-radius: 8px; display: flex; justify-content: space-between;">
                <span style="color: #86868b; font-weight: 500;">Model uncertainty</span>
                <span style="font-weight: 600;">{payload['model_uncertainty']}</span>
            </div>
            <div style="padding: 12px; background: #F5F5F7; border-radius: 8px; display: flex; justify-content: space-between;">
                <span style="color: #86868b; font-weight: 500;">Prediction interval</span>
                <span style="font-weight: 600;">{payload['conformal_width']}</span>
            </div>
            <div style="padding: 12px; background: #F5F5F7; border-radius: 8px; display: flex; justify-content: space-between;">
                <span style="color: #86868b; font-weight: 500;">Temporal stability</span>
                <span style="font-weight: 600;">{payload['temporal_stability']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    render_what_this_means("The reliability layer checks several signals around the prediction. It looks at signal quality, model uncertainty, prediction-interval width, and recent prediction stability. The reliability result does not prove that the BP estimate is medically accurate.")

def render_step_07(demo_data, tech_mode):
    render_step_title(7, "PUTTING THE RESULT INTO PLAIN LANGUAGE", "Translating structured reliability outputs into human-readable explanations.")
    
    idx = demo_data["demo_idx"]
    payload = demo_data["payloads"][idx]
    
    if "explanation" not in st.session_state:
        st.session_state.explanation = None
        
    st.markdown("""
    <p style="font-size: 15px; margin-bottom: 20px;">
    The language model only explains the structured result. It does not generate or modify the BP estimate.
    </p>
    """, unsafe_allow_html=True)
    
    if st.button("Generate Explanation", type="primary"):
        with st.spinner("Generating explanation..."):
            try:
                from phase6c_app.llm_explainer import generate_reliability_explanation
                st.session_state.explanation = generate_reliability_explanation(payload, timeout_s=15.0)
            except Exception as e:
                from phase6c_app.llm_explainer import generate_explanation_locally
                st.session_state.explanation = generate_explanation_locally(payload)
                st.session_state.explanation_error = True
    
    if st.session_state.explanation:
        exp = st.session_state.explanation
        is_local = "local" in exp.get("source", "").lower()
        
        if is_local:
            st.info("Using deterministic local explanation engine (Groq offline or unavailable).")
            
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
        
    if tech_mode:
        with st.expander("Technical Reliability Payload (Input to Explainer)", expanded=False):
            st.json(payload)
        if st.session_state.explanation:
            with st.expander("Full Structured Explanation JSON", expanded=False):
                st.json({k: v for k, v in st.session_state.explanation.items() if not k.startswith("_")})

def render_step_08(demo_data, tech_mode):
    st.markdown("<div class='fade-in' style='text-align: center; padding: 40px 0;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 42px; letter-spacing: -1px; margin-bottom: 10px;'>FROM LIGHT<br>TO BLOOD PRESSURE</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px; color: #86868b; font-size: 14px; font-weight: 500; margin: 30px 0;">
        <span>MAX30102</span> <span style="color:#d2d2d7;">→</span>
        <span>PPG</span> <span style="color:#d2d2d7;">→</span>
        <span>Deep Temporal Model</span> <span style="color:#d2d2d7;">→</span>
        <span>Reliability Layer</span>
    </div>
    """, unsafe_allow_html=True)
    
    idx = demo_data["demo_idx"]
    payload = demo_data["payloads"][idx]
    
    st.markdown(f"""
    <div style="background: white; border: 1px solid #E5E5EA; border-radius: 12px; padding: 30px; display: flex; justify-content: space-around; align-items: center; margin-bottom: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.03);">
        <div style="text-align: center;">
            <p style="color: #86868b; font-size: 13px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">BP Estimate</p>
            <p style="font-size: 28px; font-weight: 600; margin: 0; color: #1d1d1f;">{payload['prediction']['sbp']:.0f} / {payload['prediction']['dbp']:.0f}</p>
        </div>
        <div style="width: 1px; height: 50px; background: #E5E5EA;"></div>
        <div style="text-align: center;">
            <p style="color: #86868b; font-size: 13px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">Reliability</p>
            <span class="{payload['reliability_state'].lower()}-badge">{payload['reliability_state']}</span>
        </div>
        <div style="width: 1px; height: 50px; background: #E5E5EA;"></div>
        <div style="text-align: center;">
            <p style="color: #86868b; font-size: 13px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">Signal</p>
            <span class="neutral-badge">{payload['signal_quality']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: left; max-width: 600px; margin: 0 auto;">
        <h3 style="font-size: 16px; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px;">What the system did</h3>
        <ol style="color: #333336; font-size: 15px; line-height: 1.8; padding-left: 20px;">
            <li>Captured an optical pulse signal.</li>
            <li>Checked whether the signal was usable.</li>
            <li>Processed the waveform into PPG, VPG and APG representations.</li>
            <li>Used six consecutive windows to capture temporal context.</li>
            <li>Generated an SBP/DBP estimate using the frozen neural model.</li>
            <li>Evaluated prediction reliability using the Phase 7 reliability layer.</li>
            <li>Optionally converted the structured result into a plain-language explanation.</li>
        </ol>
    </div>
    <div style="margin-top: 60px; text-align: center; padding: 20px; background: #F5F5F7; border-radius: 8px; font-size: 13px; color: #86868b;">
        An end-to-end research prototype combining wearable optical sensing, temporal deep learning, uncertainty estimation, conformal prediction, and reliability-aware selective prediction.
    </div>
    </div>
    """, unsafe_allow_html=True)
