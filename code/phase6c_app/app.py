import sys
from pathlib import Path
import streamlit as st

# Ensure app package and project paths are importable
APP_DIR = Path(__file__).resolve().parent
CODE_DIR = APP_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

for p in [APP_DIR, CODE_DIR, CODE_DIR / "scripts", CODE_DIR / "phase4a"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase6c_app.ui.theme import apply_premium_theme
from phase6c_app.ui.components import render_progress_bar, render_navigation_buttons, render_research_footer
from phase6c_app.ui.pipeline import load_and_run_demo_session
from phase6c_app.ui import guided_steps
from phase6c_app.ui.live_experiment import render_live_experiment_tab

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cuffless BP Estimation — Physical Research System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_premium_theme()

# Initialize Session State
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Guided Demo"
if "current_step" not in st.session_state:
    st.session_state.current_step = -1  # -1 is Landing Page
if "tech_mode" not in st.session_state:
    st.session_state.tech_mode = False
if "demo_data" not in st.session_state:
    st.session_state.demo_data = None

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("EXPERIENCE")
    side_mode = st.radio(
        "Select Mode",
        ["Guided Demo", "Live Experiment"],
        index=0 if st.session_state.app_mode == "Guided Demo" else 1,
        label_visibility="collapsed"
    )
    if side_mode != st.session_state.app_mode:
        st.session_state.app_mode = side_mode
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("SESSION")
    if st.session_state.app_mode == "Guided Demo":
        st.write("Demonstration Walkthrough")
    else:
        st.write("Physical Experiment Mode")
    
    st.subheader("MODE")
    tech_toggle = st.toggle("Technical View", value=st.session_state.tech_mode)
    if tech_toggle != st.session_state.tech_mode:
        st.session_state.tech_mode = tech_toggle
        st.rerun()
        
    st.subheader("SYSTEM")
    st.write("Frozen research pipeline  \nPhase 6C / Phase 7")
    
    # Step navigation only in Guided Demo mode
    if st.session_state.app_mode == "Guided Demo" and st.session_state.current_step >= 0:
        st.markdown("### NAVIGATION")
        step_names = [
            "01 Input Signal",
            "02 Signal Quality",
            "03 Preprocessing",
            "04 Temporal Context",
            "05 BP Estimate",
            "06 Reliability",
            "07 Result Explanation",
            "08 Final Summary"
        ]
        chosen = st.radio(
            "Jump to step:",
            range(8),
            index=st.session_state.current_step,
            format_func=lambda i: step_names[i],
            label_visibility="collapsed"
        )
        if chosen != st.session_state.current_step:
            st.session_state.current_step = chosen
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Reset Session", use_container_width=True):
        st.session_state.current_step = -1
        if "explanation" in st.session_state:
            del st.session_state.explanation
        if "live_engine" in st.session_state and st.session_state.live_engine is not None:
            st.session_state.live_engine.reset()
        st.rerun()

# -----------------------------------------------------------------------------
# Main Layout Container
# -----------------------------------------------------------------------------
col1, main_col, col3 = st.columns([1, 10, 1])

with main_col:
    # Top-Level Mode Bar
    top_c1, top_c2 = st.columns([3, 2])
    with top_c1:
        st.markdown(
            "<p style='font-size: 13px; font-weight: 600; color: #86868b; letter-spacing: 0.5px; margin-top: 6px;'>CALIBRATION-FREE BLOOD PRESSURE RESEARCH</p>",
            unsafe_allow_html=True
        )
    with top_c2:
        top_mode = st.radio(
            "App Mode Switcher",
            ["Guided Demo", "Live Experiment"],
            index=0 if st.session_state.app_mode == "Guided Demo" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        if top_mode != st.session_state.app_mode:
            st.session_state.app_mode = top_mode
            st.rerun()

    st.markdown("<hr style='border: 0; border-top: 1px solid #E5E5EA; margin: 8px 0 24px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE: LIVE EXPERIMENT MODE
    # -------------------------------------------------------------------------
    if st.session_state.app_mode == "Live Experiment":
        render_live_experiment_tab()

    # -------------------------------------------------------------------------
    # ROUTE: GUIDED DEMO MODE
    # -------------------------------------------------------------------------
    else:
        # LANDING PAGE
        if st.session_state.current_step == -1:
            st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <h1 style="text-align: center; font-size: 42px; margin-bottom: 20px;">CALIBRATION-FREE BLOOD PRESSURE</h1>
                <p style="text-align: center; font-size: 20px; color: #333336; font-weight: 400; max-width: 600px; margin: 0 auto 50px auto;">
                    Estimate blood pressure from a photoplethysmography signal alone.<br><br>
                    <span style="font-size: 16px; color: #86868b;">A research prototype built around MAX30102 + ESP32 + temporal deep learning.</span>
                </p>
                """, 
                unsafe_allow_html=True
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                if st.button("Begin Demonstration", type="primary", use_container_width=True):
                    with st.spinner("Initializing deterministic research pipeline..."):
                        load_and_run_demo_session()
                    st.session_state.current_step = 0
                    st.rerun()
                    
            st.markdown(
                """
                <div style="margin-top: 80px; display: flex; justify-content: center; align-items: center; gap: 15px; color: #86868b; font-size: 13px; font-weight: 500;">
                    <span>MAX30102</span> <span style="color:#E5E5EA;">|</span>
                    <span>PPG</span> <span style="color:#E5E5EA;">|</span>
                    <span>Signal Processing</span> <span style="color:#E5E5EA;">|</span>
                    <span>CNN</span> <span style="color:#E5E5EA;">|</span>
                    <span>Temporal GRU</span> <span style="color:#E5E5EA;">|</span>
                    <span>BP Estimate</span> <span style="color:#E5E5EA;">|</span>
                    <span>Reliability</span>
                </div>
                <p style="text-align: center; font-size: 12px; color: #86868b; margin-top: 40px;">Research prototype — not a clinical device.</p>
                """,
                unsafe_allow_html=True
            )
            
        # GUIDED STEPS
        else:
            render_progress_bar(st.session_state.current_step)
            
            # We need the demo data to render anything
            if st.session_state.demo_data is None:
                load_and_run_demo_session()
                
            demo_data = st.session_state.demo_data
            tech_mode = st.session_state.tech_mode
            
            if st.session_state.current_step == 0:
                guided_steps.render_step_01(demo_data, tech_mode)
            elif st.session_state.current_step == 1:
                guided_steps.render_step_02(demo_data, tech_mode)
            elif st.session_state.current_step == 2:
                guided_steps.render_step_03(demo_data, tech_mode)
            elif st.session_state.current_step == 3:
                guided_steps.render_step_04(demo_data, tech_mode)
            elif st.session_state.current_step == 4:
                guided_steps.render_step_05(demo_data, tech_mode)
            elif st.session_state.current_step == 5:
                guided_steps.render_step_06(demo_data, tech_mode)
            elif st.session_state.current_step == 6:
                guided_steps.render_step_07(demo_data, tech_mode)
            elif st.session_state.current_step == 7:
                guided_steps.render_step_08(demo_data, tech_mode)
                
            render_navigation_buttons(st.session_state.current_step, 8)
            render_research_footer()

