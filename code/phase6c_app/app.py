import sys
import base64
from pathlib import Path
import streamlit as st

# Ensure app package and project paths are importable
APP_DIR = Path(__file__).resolve().parent
CODE_DIR = APP_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

for p in [PROJECT_ROOT, CODE_DIR, CODE_DIR / "phase7_reliability", APP_DIR, CODE_DIR / "scripts", CODE_DIR / "phase4a"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import phase6c_app.config  # Initializes safe_load_pickle and cross-platform module aliases

from phase6c_app.ui.theme import apply_premium_theme
from phase6c_app.ui.components import render_progress_bar, render_navigation_buttons
from phase6c_app.ui.pipeline import load_and_run_demo_session
from phase6c_app.ui import guided_steps
from phase6c_app.ui.live_experiment import render_live_experiment_tab

@st.cache_data
def get_heart_gif_b64() -> str:
    gif_path = APP_DIR / "assets" / "heart_pulse.gif"
    if gif_path.exists():
        return base64.b64encode(gif_path.read_bytes()).decode("utf-8")
    return ""

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cuffless BP Estimation — Physical Research System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
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
# Main Layout Container
# -----------------------------------------------------------------------------
col1, main_col, col3 = st.columns([1, 10, 1])

with main_col:
    # Determine if Technical View and Reset controls should appear:
    # Only in Live Experiment OR in Guided Demo after clicking "Begin Demonstration" (current_step >= 0)
    show_tech_view = (st.session_state.app_mode == "Live Experiment") or (
        st.session_state.app_mode == "Guided Demo" and st.session_state.current_step >= 0
    )

    if show_tech_view:
        top_c1, top_c2, top_c3, top_c4 = st.columns([4, 3, 2, 1.2])
    else:
        top_c1, top_c2 = st.columns([3, 2])

    with top_c1:
        st.markdown(
            "<p style='font-size: 15px; font-weight: 600; color: #86868b; letter-spacing: 0.5px; margin-top: 6px;'>CALIBRATION-FREE BLOOD PRESSURE RESEARCH</p>",
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

    if show_tech_view:
        with top_c3:
            tech_toggle = st.toggle("Technical View", value=st.session_state.tech_mode)
            if tech_toggle != st.session_state.tech_mode:
                st.session_state.tech_mode = tech_toggle
                st.rerun()
        with top_c4:
            if st.button("Reset", key="btn_top_reset", use_container_width=True):
                st.session_state.current_step = -1
                if "explanation" in st.session_state:
                    del st.session_state.explanation
                if "live_engine" in st.session_state and st.session_state.live_engine is not None:
                    st.session_state.live_engine.reset()
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
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            heart_b64 = get_heart_gif_b64()
            if heart_b64:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;" class="fade-in">
                        <img src="data:image/gif;base64,{heart_b64}" 
                             alt="Pulsatile Heart" 
                             style="width: 110px; height: auto;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown(
                """
                <h1 style="text-align: center; font-size: 46px; margin-bottom: 20px;">CALIBRATION-FREE BLOOD PRESSURE</h1>
                <p style="text-align: center; font-size: 22px; color: #333336; font-weight: 400; max-width: 650px; margin: 0 auto 50px auto;">
                    Estimate blood pressure from a photoplethysmography signal alone.<br><br>
                    <span style="font-size: 18px; color: #86868b;">Continuous blood pressure estimation using deep learning.</span>
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

