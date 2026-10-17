import streamlit as st

def apply_premium_theme():
    """Injects Apple-style scientific product CSS theme."""
    custom_css = """
    <style>
        /* Base typography and colors */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro', system-ui, sans-serif !important;
            color: #1d1d1f;
            background-color: #F5F5F7;
        }

        /* Clean white surfaces for main app content */
        .stApp {
            background-color: #F5F5F7;
        }

        /* Standardize Streamlit blocks to look like cards */
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid #E5E5EA;
        }

        /* Typography hierarchy */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600 !important;
            color: #1d1d1f !important;
        }

        h1 {
            font-size: 44px !important;
            letter-spacing: -0.5px !important;
            margin-bottom: 0.5em !important;
        }
        
        h2 {
            font-size: 32px !important;
            margin-bottom: 0.5em !important;
        }

        h3 {
            font-size: 24px !important;
        }

        h4 {
            font-size: 20px !important;
        }

        p, div, span, li {
            font-size: 18px;
            line-height: 1.6;
            color: #333336;
        }

        .stMarkdown p {
            font-size: 18px !important;
            line-height: 1.6 !important;
        }

        .caption-text {
            font-size: 15px !important;
            color: #86868b !important;
        }

        /* Step Progress Indicator */
        .progress-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px 0 40px 0;
            width: 100%;
        }
        .progress-step {
            display: flex;
            align-items: center;
        }
        .progress-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #d2d2d7;
            margin: 0 4px;
            transition: all 0.3s ease;
        }
        .progress-dot.active {
            background-color: #4F6EF7;
            box-shadow: 0 0 0 4px rgba(79, 110, 247, 0.2);
        }
        .progress-dot.completed {
            background-color: #1d1d1f;
        }
        .progress-line {
            height: 2px;
            width: 40px;
            background-color: #E5E5EA;
            margin: 0 4px;
        }
        .progress-line.completed {
            background-color: #1d1d1f;
        }
        .progress-label-container {
            text-align: center;
            margin-top: -10px;
            margin-bottom: 20px;
            font-size: 16px;
            font-weight: 500;
            color: #86868b;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        /* Semantic Colors */
        .trust-text { color: #2e7d32; font-weight: 600; }
        .review-text { color: #f57f17; font-weight: 600; }
        .abstain-text { color: #c62828; font-weight: 600; }
        
        .trust-badge { background-color: #e8f5e9; color: #2e7d32; padding: 6px 16px; border-radius: 18px; font-weight: 600; font-size: 16px; border: 1px solid #c8e6c9;}
        .review-badge { background-color: #fff8e1; color: #f57f17; padding: 6px 16px; border-radius: 18px; font-weight: 600; font-size: 16px; border: 1px solid #ffecb3;}
        .abstain-badge { background-color: #ffebee; color: #c62828; padding: 6px 16px; border-radius: 18px; font-weight: 600; font-size: 16px; border: 1px solid #ffcdd2;}
        .neutral-badge { background-color: #f5f5f7; color: #1d1d1f; padding: 6px 16px; border-radius: 18px; font-weight: 500; font-size: 16px; border: 1px solid #d2d2d7;}

        /* BP Display */
        .bp-display {
            font-size: 80px;
            font-weight: 600;
            letter-spacing: -2px;
            color: #1d1d1f;
            text-align: center;
            margin: 20px 0;
            line-height: 1;
        }
        .bp-unit {
            font-size: 28px;
            color: #86868b;
            font-weight: 400;
            letter-spacing: 0;
            margin-left: 8px;
        }
        .bp-labels {
            display: flex;
            justify-content: center;
            gap: 60px;
            color: #86868b;
            font-size: 18px;
            font-weight: 500;
            margin-top: -10px;
            margin-bottom: 20px;
        }

        /* Subtle animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
            animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        
        @media (prefers-reduced-motion: reduce) {
            .fade-in {
                animation: none;
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Research footer */
        .research-footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Solid black button styling with white text, no animation, no text inversion */
        div.stButton > button,
        div.stButton > button:focus,
        div.stButton > button:active,
        div.stButton > button[kind="primary"],
        div.stButton > button[kind="secondary"] {
            border-radius: 20px !important;
            border: 1px solid #000000 !important;
            background-color: #000000 !important;
            color: #ffffff !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            padding: 10px 28px !important;
            transition: none !important;
            animation: none !important;
            box-shadow: none !important;
        }

        /* Ensure all text elements inside button are strictly white */
        div.stButton > button * {
            color: #ffffff !important;
            font-size: 16px !important;
            transition: none !important;
            animation: none !important;
        }

        div.stButton > button p {
            color: #ffffff !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            margin: 0 !important;
        }

        /* Hover state: dark gray background, text strictly white */
        div.stButton > button:hover {
            border-color: #222222 !important;
            background-color: #222222 !important;
            color: #ffffff !important;
            transition: none !important;
            animation: none !important;
        }

        div.stButton > button:hover * {
            color: #ffffff !important;
            font-size: 16px !important;
            transition: none !important;
            animation: none !important;
        }

        div.stButton > button:hover p {
            color: #ffffff !important;
            font-size: 16px !important;
        }

        /* Streamlit widget labels, inputs, and radios */
        label[data-testid="stWidgetLabel"] p,
        label[data-testid="stWidgetLabel"] span {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #1d1d1f !important;
        }

        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] span {
            font-size: 16px !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricLabel"] p {
            font-size: 15px !important;
            color: #86868b !important;
            font-weight: 500 !important;
        }

        /* Completely remove and hide the sidebar and any collapsed toggle control */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        div[data-testid="collapsedControl"] {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            visibility: hidden !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
