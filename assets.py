"""
Streamlit Theme Configuration
Custom styling for medical application interface
"""

# Color Palette
COLORS = {
    # Primary colors
    'primary': '#1e3a8a',           # Dark blue
    'secondary': '#3b82f6',         # Bright blue
    'accent': '#10b981',            # Green
    'warning': '#f59e0b',           # Orange
    'danger': '#ef4444',            # Red
    
    # Tumor class colors
    'necrotic': '#ef4444',          # Red
    'edema': '#22c55e',             # Green
    'enhancing': '#eab308',         # Yellow
    'background': '#000000',         # Black
    
    # UI colors
    'text_primary': '#1f2937',      # Dark gray
    'text_secondary': '#64748b',    # Medium gray
    'background_light': '#f8fafc',  # Very light gray
    'background_white': '#ffffff',   # White
    'border': '#e2e8f0',            # Light gray
    
    # Gradients
    'gradient_1': 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
    'gradient_2': 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
    'gradient_3': 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)',
    'gradient_4': 'linear-gradient(135deg, #ec4899 0%, #f59e0b 100%)',
    'gradient_5': 'linear-gradient(135deg, #f59e0b 0%, #10b981 100%)',
}

# Typography
TYPOGRAPHY = {
    'font_family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'heading_font': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    'monospace_font': '"Fira Code", "Courier New", monospace',
    
    'font_size_xs': '0.75rem',      # 12px
    'font_size_sm': '0.875rem',     # 14px
    'font_size_base': '1rem',       # 16px
    'font_size_lg': '1.125rem',     # 18px
    'font_size_xl': '1.25rem',      # 20px
    'font_size_2xl': '1.5rem',      # 24px
    'font_size_3xl': '1.875rem',    # 30px
    'font_size_4xl': '2.25rem',     # 36px
}

# Spacing
SPACING = {
    'xs': '0.25rem',    # 4px
    'sm': '0.5rem',     # 8px
    'md': '1rem',       # 16px
    'lg': '1.5rem',     # 24px
    'xl': '2rem',       # 32px
    '2xl': '3rem',      # 48px
}

# Border Radius
RADIUS = {
    'sm': '4px',
    'md': '6px',
    'lg': '8px',
    'xl': '10px',
    'full': '9999px',
}

# Shadows
SHADOWS = {
    'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
}

# Component Styles
COMPONENT_STYLES = {
    'card': f"""
        background: {COLORS['background_white']};
        padding: {SPACING['lg']};
        border-radius: {RADIUS['lg']};
        box-shadow: {SHADOWS['md']};
        margin-bottom: {SPACING['md']};
    """,
    
    'header': f"""
        background: {COLORS['gradient_1']};
        padding: {SPACING['xl']};
        border-radius: {RADIUS['xl']};
        color: white;
        text-align: center;
        margin-bottom: {SPACING['xl']};
        box-shadow: {SHADOWS['lg']};
    """,
    
    'metric_card': f"""
        background: {COLORS['background_white']};
        padding: {SPACING['md']};
        border-radius: {RADIUS['md']};
        box-shadow: {SHADOWS['sm']};
        text-align: center;
    """,
    
    'info_box': f"""
        background: {COLORS['background_light']};
        padding: {SPACING['lg']};
        border-radius: {RADIUS['lg']};
        border-left: 4px solid {COLORS['secondary']};
    """,
}

# Streamlit Config (.streamlit/config.toml format)
STREAMLIT_CONFIG = """
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1f2937"
font = "sans serif"

[server]
maxUploadSize = 500
enableCORS = false
enableXsrfProtection = true
"""

def get_custom_css():
    """
    Generate custom CSS for Streamlit application
    
    Returns:
        str: CSS string to be used with st.markdown()
    """
    return f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: {TYPOGRAPHY['font_family']};
    }}
    
    /* Main container */
    .main {{
        background-color: {COLORS['background_light']};
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        font-family: {TYPOGRAPHY['heading_font']};
        color: {COLORS['text_primary']};
    }}
    
    h1 {{
        font-size: {TYPOGRAPHY['font_size_3xl']};
        font-weight: 700;
    }}
    
    h2 {{
        font-size: {TYPOGRAPHY['font_size_2xl']};
        font-weight: 600;
    }}
    
    h3 {{
        font-size: {TYPOGRAPHY['font_size_xl']};
        font-weight: 600;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {COLORS['gradient_1']};
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: {RADIUS['md']};
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: {SHADOWS['lg']};
    }}
    
    /* Metrics */
    .stMetric {{
        background: {COLORS['background_white']};
        padding: {SPACING['md']};
        border-radius: {RADIUS['md']};
        box-shadow: {SHADOWS['sm']};
    }}
    
    .stMetric label {{
        font-size: {TYPOGRAPHY['font_size_sm']};
        color: {COLORS['text_secondary']};
    }}
    
    .stMetric [data-testid="stMetricValue"] {{
        font-size: {TYPOGRAPHY['font_size_2xl']};
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}
    
    /* Info boxes */
    .stInfo {{
        background-color: {COLORS['background_light']};
        border-left: 4px solid {COLORS['secondary']};
        padding: {SPACING['lg']};
        border-radius: {RADIUS['md']};
    }}
    
    .stSuccess {{
        background-color: #ecfdf5;
        border-left: 4px solid {COLORS['accent']};
        padding: {SPACING['lg']};
        border-radius: {RADIUS['md']};
    }}
    
    .stWarning {{
        background-color: #fff7ed;
        border-left: 4px solid {COLORS['warning']};
        padding: {SPACING['lg']};
        border-radius: {RADIUS['md']};
    }}
    
    .stError {{
        background-color: #fef2f2;
        border-left: 4px solid {COLORS['danger']};
        padding: {SPACING['lg']};
        border-radius: {RADIUS['md']};
    }}
    
    /* Progress bars */
    .stProgress > div > div > div > div {{
        background: {COLORS['gradient_1']};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #f1f5f9;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background-color: #f1f5f9;
    }}
    
    /* Sliders */
    .stSlider > div > div > div > div {{
        background-color: {COLORS['secondary']};
    }}
    
    /* Selectbox */
    .stSelectbox > div > div {{
        border-color: {COLORS['border']};
        border-radius: {RADIUS['md']};
    }}
    
    /* Text input */
    .stTextInput > div > div {{
        border-color: {COLORS['border']};
        border-radius: {RADIUS['md']};
    }}
    
    /* File uploader */
    .stFileUploader > div {{
        border: 2px dashed {COLORS['border']};
        border-radius: {RADIUS['lg']};
        padding: {SPACING['xl']};
        background-color: {COLORS['background_light']};
    }}
    
    .stFileUploader > div:hover {{
        border-color: {COLORS['secondary']};
        background-color: #eff6ff;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: {SPACING['md']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: {RADIUS['md']} {RADIUS['md']} 0 0;
        padding: {SPACING['md']} {SPACING['lg']};
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['secondary']};
        color: white;
    }}
    
    /* Expander */
    .stExpander {{
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['md']};
        background-color: {COLORS['background_white']};
    }}
    
    /* Tables */
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    
    th {{
        background-color: {COLORS['background_light']};
        padding: {SPACING['md']};
        text-align: left;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}
    
    td {{
        padding: {SPACING['md']};
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    tr:hover {{
        background-color: {COLORS['background_light']};
    }}
    
    /* Custom classes */
    .info-card {{
        background: {COLORS['background_white']};
        padding: {SPACING['lg']};
        border-radius: {RADIUS['lg']};
        box-shadow: {SHADOWS['md']};
        margin-bottom: {SPACING['md']};
        border-left: 4px solid {COLORS['secondary']};
    }}
    
    .metric-card {{
        background: {COLORS['background_white']};
        padding: {SPACING['lg']};
        border-radius: {RADIUS['md']};
        box-shadow: {SHADOWS['sm']};
        text-align: center;
    }}
    
    .gradient-header {{
        background: {COLORS['gradient_1']};
        padding: {SPACING['xl']};
        border-radius: {RADIUS['xl']};
        color: white;
        text-align: center;
        margin-bottom: {SPACING['xl']};
        box-shadow: {SHADOWS['lg']};
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.5s ease-in-out;
    }}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {{
        h1 {{
            font-size: {TYPOGRAPHY['font_size_2xl']};
        }}
        
        h2 {{
            font-size: {TYPOGRAPHY['font_size_xl']};
        }}
        
        .stButton > button {{
            padding: 0.5rem 1rem;
        }}
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['background_light']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['secondary']};
        border-radius: {RADIUS['full']};
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['primary']};
    }}
    </style>
    """

def apply_theme():
    """
    Apply custom theme to Streamlit application
    Call this function in your main app file
    """
    import streamlit as st
    st.markdown(get_custom_css(), unsafe_allow_html=True)