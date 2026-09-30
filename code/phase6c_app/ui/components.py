import streamlit as st

STEPS = [
    "01 INPUT",
    "02 SIGNAL",
    "03 PROCESS",
    "04 CONTEXT",
    "05 ESTIMATE",
    "06 RELIABILITY",
    "07 EXPLAIN",
    "08 SUMMARY"
]

def render_progress_bar(current_step: int):
    """Renders the top horizontal progress indicator."""
    total_steps = len(STEPS)
    
    html = '<div class="progress-container">'
    for i in range(total_steps):
        is_completed = i < current_step
        is_active = i == current_step
        
        # Dot
        dot_class = "progress-dot"
        if is_active:
            dot_class += " active"
        elif is_completed:
            dot_class += " completed"
        
        html += f'<div class="{dot_class}"></div>'
        
        # Line (don't add after the last dot)
        if i < total_steps - 1:
            line_class = "progress-line"
            if is_completed:
                line_class += " completed"
            html += f'<div class="{line_class}"></div>'
            
    html += '</div>'
    
    # Label
    label_text = STEPS[current_step] if 0 <= current_step < total_steps else ""
    html += f'<div class="progress-label-container">{label_text}</div>'
    
    st.markdown(html, unsafe_allow_html=True)

def render_step_title(step_num: int, title: str, subtitle: str = ""):
    """Renders a clean step title."""
    st.markdown(f'<div class="fade-in">', unsafe_allow_html=True)
    st.markdown(f"<p style='color: #86868b; font-size: 14px; font-weight: 600; margin-bottom: -10px;'>{step_num:02d}</p>", unsafe_allow_html=True)
    st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='font-size: 18px; color: #1d1d1f; margin-bottom: 24px;'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_what_this_means(text: str):
    """Renders the 'What this means' microcopy block."""
    html = f"""
    <div style="margin-top: 30px; margin-bottom: 15px; border-top: 1px solid #E5E5EA; padding-top: 15px;" class="fade-in">
        <p style="font-size: 12px; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">What this means</p>
        <p style="font-size: 15px; color: #333336;">{text}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_navigation_buttons(current_step: int, total_steps: int):
    """Renders Back / Continue navigation."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_step > 0:
            if st.button("← Back", key=f"btn_back_{current_step}"):
                st.session_state.current_step -= 1
                st.rerun()
                
    with col3:
        if current_step < total_steps - 1:
            if st.button("Continue →", key=f"btn_next_{current_step}", type="primary"):
                st.session_state.current_step += 1
                st.rerun()
        elif current_step == total_steps - 1:
            if st.button("Restart Demo ↺", key="btn_restart"):
                st.session_state.current_step = 0
                st.rerun()

def render_research_footer():
    st.markdown(
        """
        <div class="research-footer">
            <b>Research prototype</b> &nbsp;|&nbsp; Frozen Phase 4A–Phase 7 pipeline &nbsp;|&nbsp; PPG-only input &nbsp;|&nbsp; No ECG used by the model
        </div>
        """,
        unsafe_allow_html=True
    )
