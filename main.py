"""
Brain Tumor Segmentation System
Main Streamlit Application Entry Point — Multi-page App
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="NeuroSeg — Brain Tumor Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="",
)

from styles import inject_global_css, DARK_BG, SURFACE_1, SURFACE_2, BORDER
from styles import ACCENT_BLUE, ACCENT_GREEN, TEXT_PRIMARY, TEXT_DIM, TEXT_BRIGHT
from styles import FONT_BODY, FONT_MONO, ACCENT_AMBER, ACCENT_RED

inject_global_css()

# Session state defaults
_DEFAULTS = {
    "initialized": True,
    "mri_data": None,
    "segmentation_result": None,
    "preprocessing_done": False,
    "metrics": {},
    "history": [],
    "original_data": None,
    "preprocessed_data": None,
    "mri_metadata": {},
    "segmentation_probabilities": None,
    "segmentation_time": None,
    "model_type": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:1.2rem 0.5rem 1rem 0.5rem;
                    border-bottom:1px solid {BORDER};margin-bottom:0.5rem;">
            <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                        text-transform:uppercase;color:{ACCENT_BLUE};margin-bottom:0.3rem;">
                NeuroSeg
            </div>
            <div style="font-size:0.78rem;color:{TEXT_DIM};font-weight:400;">
                Brain Tumor Segmentation System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mri_ok = st.session_state.mri_data is not None
    seg_ok = st.session_state.segmentation_result is not None
    mri_dot = f"background:{ACCENT_GREEN}" if mri_ok else f"background:{TEXT_DIM}"
    seg_dot = f"background:{ACCENT_GREEN}" if seg_ok else f"background:{TEXT_DIM}"

    st.markdown(
        f"""
        <div style="padding:0.5rem 0.5rem 0.8rem 0.5rem;
                    border-bottom:1px solid {BORDER};margin-bottom:0.8rem;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <div style="width:7px;height:7px;border-radius:50%;{mri_dot};flex-shrink:0;"></div>
                <span style="font-size:0.78rem;color:{TEXT_DIM};">
                    {"MRI Loaded" if mri_ok else "No MRI"}
                </span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:7px;height:7px;border-radius:50%;{seg_dot};flex-shrink:0;"></div>
                <span style="font-size:0.78rem;color:{TEXT_DIM};">
                    {"Segmented" if seg_ok else "Not Run"}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{TEXT_DIM};padding:0 0.5rem;margin-bottom:0.4rem;">'
        f'Navigation</div>',
        unsafe_allow_html=True,
    )

    st.page_link("main.py",                       label="Home")
    st.page_link("pages/1_Dashboard.py",          label="Dashboard")
    st.page_link("pages/2_Upload_MRI.py",         label="Upload MRI")
    st.page_link("pages/3_Run_Segmentation.py",   label="Run Segmentation")
    st.page_link("pages/4_View_Results.py",       label="View Results")
    st.page_link("pages/5_Generate_Report.py",    label="Generate Report")
    st.page_link("pages/6_History.py",            label="History")
    st.page_link("pages/7_System_Info.py",        label="System Info")

    st.markdown(
    """
    <div style="
        margin-top:2rem;
        text-align:center;
        font-size:0.8rem;
        color:gray;
    ">
        <b>NeuroSeg v2.0</b> | 2026 <br>
        by <b>Ayush Singh</b>
    </div>
    """,
    unsafe_allow_html=True
)

# Hero banner
st.markdown(
    f"""
    <div style="
        background:linear-gradient(135deg,{SURFACE_1} 0%,{SURFACE_2} 100%);
        border:1px solid {BORDER};border-radius:14px;
        padding:2.5rem 2.5rem;margin-bottom:1.5rem;
        position:relative;overflow:hidden;
    ">
        <div style="position:absolute;top:-40px;right:-40px;width:220px;height:220px;
                    background:radial-gradient(circle,rgba(56,139,253,.07) 0%,transparent 70%);
                    border-radius:50%;pointer-events:none;"></div>
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;
                    text-transform:uppercase;color:{ACCENT_BLUE};margin-bottom:0.5rem;">
            AI-Powered Medical Imaging
        </div>
        <h1 style="margin:0 0 0.4rem 0;font-size:2rem;font-weight:700;
                   color:{TEXT_BRIGHT};letter-spacing:-0.03em;line-height:1.2;">
            Brain Tumor Segmentation
        </h1>
        <p style="margin:0;color:{TEXT_DIM};font-size:0.9rem;max-width:520px;line-height:1.6;">
            3D U-Net segmentation pipeline for automated detection and classification
            of brain tumor sub-regions from multi-modal MRI scans.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status cards
mri_data = st.session_state.mri_data
seg_data = st.session_state.segmentation_result
seg_time = st.session_state.segmentation_time
metrics  = st.session_state.metrics or {}

card_cfg = [
    {
        "label":  "MRI Status",
        "value":  "Loaded" if mri_data is not None else "Not Loaded",
        "sub":    (f"{mri_data.shape[0]}x{mri_data.shape[1]}x{mri_data.shape[2]} voxels"
                   if mri_data is not None else "Upload a NIfTI file to begin"),
        "accent": ACCENT_GREEN if mri_data is not None else TEXT_DIM,
    },
    {
        "label":  "Model Status",
        "value":  st.session_state.model_type or "3D U-Net",
        "sub":    "Ready for inference",
        "accent": ACCENT_BLUE,
    },
    {
        "label":  "Segmentation",
        "value":  "Completed" if seg_data is not None else "Pending",
        "sub":    (f"Tumor volume: {metrics.get('total_tumor_volume_cm3', 0):.2f} cm\u00b3"
                   if seg_data is not None else "Run segmentation to analyse"),
        "accent": ACCENT_GREEN if seg_data is not None else ACCENT_AMBER,
    },
    {
        "label":  "Inference Time",
        "value":  f"{seg_time:.2f}s" if isinstance(seg_time, (int, float)) else "N/A",
        "sub":    "Last segmentation run",
        "accent": ACCENT_BLUE,
    },
]

cols = st.columns(4)
for col, card in zip(cols, card_cfg):
    with col:
        st.markdown(
            f"""
            <div style="background:{SURFACE_1};border:1px solid {BORDER};
                        border-top:3px solid {card['accent']};border-radius:10px;
                        padding:1.2rem 1.1rem 1rem 1.1rem;height:100%;">
                <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.09em;
                            text-transform:uppercase;color:{TEXT_DIM};margin-bottom:0.6rem;">
                    {card['label']}
                </div>
                <div style="font-size:1.35rem;font-weight:600;color:{TEXT_BRIGHT};
                            font-family:{FONT_MONO};letter-spacing:-0.01em;margin-bottom:0.35rem;">
                    {card['value']}
                </div>
                <div style="font-size:0.75rem;color:{TEXT_DIM};line-height:1.4;">
                    {card['sub']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# Workflow steps
st.markdown(
    f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;'
    f'text-transform:uppercase;color:{TEXT_DIM};margin-bottom:0.8rem;">Workflow</div>',
    unsafe_allow_html=True,
)

steps = [
    ("01", "Upload MRI",       "Load NIfTI brain scan",         ACCENT_BLUE),
    ("02", "Run Segmentation", "Execute 3D U-Net pipeline",     ACCENT_GREEN),
    ("03", "View Results",     "Inspect overlay and metrics",   ACCENT_AMBER),
    ("04", "Generate Report",  "Export clinical-style report",  ACCENT_RED),
]

cols = st.columns(4)
for col, (num, title, desc, acc) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
            <div style="background:{SURFACE_1};border:1px solid {BORDER};
                        border-radius:10px;padding:1.2rem 1.1rem;">
                <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                            color:{acc};margin-bottom:0.6rem;">{num}</div>
                <div style="font-size:0.9rem;font-weight:600;color:{TEXT_BRIGHT};
                            margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.78rem;color:{TEXT_DIM};line-height:1.4;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
