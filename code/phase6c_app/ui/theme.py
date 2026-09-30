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
            font-size: 38px !important;
            letter-spacing: -0.5px !important;
            margin-bottom: 0.5em !important;
        }
        
        h2 {
            font-size: 26px !important;
            margin-bottom: 0.5em !important;
        }

        h3 {
            font-size: 20px !important;
        }

        p, div, span, li {
            font-size: 16px;
            line-height: 1.5;
            color: #333336;
        }

        .caption-text {
            font-size: 13px !important;
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
            font-size: 13px;
            font-weight: 500;
            color: #86868b;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        /* Semantic Colors */
        .trust-text { color: #2e7d32; font-weight: 600; }
        .review-text { color: #f57f17; font-weight: 600; }
        .abstain-text { color: #c62828; font-weight: 600; }
        
        .trust-badge { background-color: #e8f5e9; color: #2e7d32; padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 14px; border: 1px solid #c8e6c9;}
        .review-badge { background-color: #fff8e1; color: #f57f17; padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 14px; border: 1px solid #ffecb3;}
        .abstain-badge { background-color: #ffebee; color: #c62828; padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 14px; border: 1px solid #ffcdd2;}
        .neutral-badge { background-color: #f5f5f7; color: #1d1d1f; padding: 4px 12px; border-radius: 16px; font-weight: 500; font-size: 14px; border: 1px solid #d2d2d7;}

        /* BP Display */
        .bp-display {
            font-size: 72px;
            font-weight: 600;
            letter-spacing: -2px;
            color: #1d1d1f;
            text-align: center;
            margin: 20px 0;
            line-height: 1;
        }
        .bp-unit {
            font-size: 24px;
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
            font-size: 16px;
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
            margin-top: 60px;
            padding: 20px;
            text-align: center;
            font-size: 13px;
            color: #86868b;
            border-top: 1px solid #E5E5EA;
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
            font-weight: 500 !important;
            padding: 8px 24px !important;
            transition: none !important;
            animation: none !important;
            box-shadow: none !important;
        }

        /* Ensure all text elements inside button are strictly white */
        div.stButton > button * {
            color: #ffffff !important;
            transition: none !important;
            animation: none !important;
        }

        div.stButton > button p {
            color: #ffffff !important;
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
            transition: none !important;
            animation: none !important;
        }

        div.stButton > button:hover p {
            color: #ffffff !important;
        }

        /* Explicit Light Apple Sidebar with High-Contrast Crisp Typography */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        div[data-testid="stSidebarUserContent"],
        div[data-testid="stSidebarHeader"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E5E5EA !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #1D1D1F !important;
            font-weight: 600 !important;
            letter-spacing: -0.2px !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {
            color: #1D1D1F !important;
        }

        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] .stCaption p {
            color: #6E6E73 !important;
            font-size: 13px !important;
        }

        /* Sidebar Toggle Control */
        section[data-testid="stSidebar"] label[data-baseweb="checkbox"] p,
        section[data-testid="stSidebar"] label[data-baseweb="checkbox"] span {
            color: #1D1D1F !important;
            font-size: 14px !important;
            font-weight: 500 !important;
        }

        /* Sidebar Radio Navigation */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 6px !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background-color: transparent !important;
            padding: 4px 10px !important;
            border-radius: 8px !important;
            transition: none !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #F5F5F7 !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            color: #1D1D1F !important;
            font-size: 14px !important;
            font-weight: 500 !important;
        }

        /* Ensure Reset Session and other buttons in sidebar remain solid black with white text */
        section[data-testid="stSidebar"] div.stButton > button {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border-radius: 20px !important;
        }

        section[data-testid="stSidebar"] div.stButton > button * {
            color: #FFFFFF !important;
        }

        section[data-testid="stSidebar"] div.stButton > button p {
            color: #FFFFFF !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
