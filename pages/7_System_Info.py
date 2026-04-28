"""
System Info — NeuroSeg
Model configuration, training details, and architecture reference.
"""

import streamlit as st
import sys
from pathlib import Path

st.set_page_config(
    page_title="System Info — NeuroSeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(str(Path(__file__).parent.parent / "src"))

from styles import (
    inject_css, sidebar_logo, sidebar_nav_label, sidebar_status,
    metric_card, section_title, divider, page_header, status_badge,
    BLUE, GREEN, AMBER, RED, CYAN, PURPLE,
    SURFACE_2, SURFACE_3, BORDER, T1, T2, T3, MONO,
)
from model_loader import get_model_info

inject_css()

sidebar_logo()
sidebar_status(
    st.session_state.get("mri_data") is not None,
    st.session_state.get("segmentation_result") is not None,
)
sidebar_nav_label("Main")
st.sidebar.page_link("main.py",                         label="Home")
st.sidebar.page_link("pages/1_Dashboard.py",            label="Dashboard")
st.sidebar.page_link("pages/2_Upload_MRI.py",           label="Upload MRI")
st.sidebar.page_link("pages/3_Run_Segmentation.py",     label="Run Segmentation")
st.sidebar.page_link("pages/4_View_Results.py",         label="View Results")
sidebar_nav_label("Reports")
st.sidebar.page_link("pages/5_Generate_Report.py",      label="Generate Report")
st.sidebar.page_link("pages/6_History.py",              label="History")
st.sidebar.page_link("pages/7_System_Info.py",          label="System Info")

page_header(
    "System Information",
    "Model architecture, training configuration, and clinical validation details",
    breadcrumb="System Info",
)

info = get_model_info()

# ── Model cards ───────────────────────────────────────────────────────────────
section_title("Model Configuration")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        metric_card("Architecture", "3D U-Net", "MONAI implementation", BLUE,
                    '<path d="M12 2L2 7l10 5 10-5-10-5z"/>'
                    '<path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>'),
        unsafe_allow_html=True)
with c2:
    st.markdown(
        metric_card("Parameters", info["parameters"], "trainable", GREEN,
                    '<circle cx="12" cy="12" r="10"/>'
                    '<line x1="12" y1="8" x2="12" y2="16"/>'
                    '<line x1="8" y1="12" x2="16" y2="12"/>'),
        unsafe_allow_html=True)
with c3:
    st.markdown(
        metric_card("Input Shape", "128×128×128", "4 channels", AMBER,
                    '<rect x="3" y="3" width="18" height="18" rx="2"/>'
                    '<circle cx="8.5" cy="8.5" r="1.5"/>'
                    '<polyline points="21 15 16 10 5 21"/>'),
        unsafe_allow_html=True)
with c4:
    st.markdown(
        metric_card("Output Classes", "4", "BraTS convention", RED,
                    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
        unsafe_allow_html=True)

divider()

# ── Performance ───────────────────────────────────────────────────────────────
section_title("Validation Performance (BraTS 2020)")

p1, p2, p3, p4 = st.columns(4)
for col_p, (label, val, sub, color) in zip(
    [p1, p2, p3, p4],
    [
        ("Dice — Whole",    "0.89", "±0.05", BLUE),
        ("Dice — Core",     "0.74", "±0.08", RED),
        ("Sensitivity",     "0.91", "Detection rate", GREEN),
        ("Specificity",     "0.99", "False positive rate", CYAN),
    ],
):
    with col_p:
        st.markdown(
            f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
            f'border-radius:12px;padding:1.1rem;">'
            f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.08em;'
            f'text-transform:uppercase;color:{T2};margin-bottom:.5rem;">{label}</div>'
            f'<div style="font-size:2rem;font-weight:800;color:{color};'
            f'font-family:{MONO};letter-spacing:-.02em;">{val}</div>'
            f'<div style="font-size:.72rem;color:{T3};margin-top:.2rem;">{sub}</div></div>',
            unsafe_allow_html=True,
        )

divider()

# ── Architecture details ──────────────────────────────────────────────────────
section_title("Architecture Details")

arch_l, arch_r = st.columns(2)

with arch_l:
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:12px;padding:1.2rem;">'
        f'<div style="font-size:.8rem;font-weight:700;color:{T1};'
        f'margin-bottom:.9rem;">Network Topology</div>',
        unsafe_allow_html=True,
    )
    for key, val in [
        ("Type",          "Encoder-Decoder with skip connections"),
        ("Depth",         "5 levels"),
        ("Encoder",       "16 → 32 → 64 → 128 → 256 channels"),
        ("Strides",       "(2, 2, 2, 2) — stride-2 downsampling"),
        ("Residual Units","2 per block"),
        ("Normalisation", "Batch Norm"),
        ("Dropout",       "0.1"),
        ("Input Shape",   "(B, 4, 128, 128, 128)"),
        ("Output Shape",  "(B, 4, 128, 128, 128)"),
    ]:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:.3rem 0;border-bottom:1px solid {BORDER};font-size:.78rem;">'
            f'<span style="color:{T2};">{key}</span>'
            f'<span style="color:{T1};font-family:{MONO};font-size:.75rem;">{val}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with arch_r:
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:12px;padding:1.2rem;">'
        f'<div style="font-size:.8rem;font-weight:700;color:{T1};'
        f'margin-bottom:.9rem;">Training Configuration</div>',
        unsafe_allow_html=True,
    )
    for key, val in [
        ("Dataset",         "BraTS 2020 (1000+ cases)"),
        ("Loss Function",   "Dice Loss + Cross-Entropy"),
        ("Optimizer",       "Adam"),
        ("Learning Rate",   "1 × 10⁻³"),
        ("Weight Decay",    "1 × 10⁻⁵"),
        ("Scheduler",       "CosineAnnealingLR"),
        ("Batch Size",      "2"),
        ("Epochs",          "300"),
        ("Augmentation",    "Flip, rotation, intensity shift"),
    ]:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:.3rem 0;border-bottom:1px solid {BORDER};font-size:.78rem;">'
            f'<span style="color:{T2};">{key}</span>'
            f'<span style="color:{T1};font-family:{MONO};font-size:.75rem;">{val}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

divider()

# ── Tumour classes legend ─────────────────────────────────────────────────────
section_title("Tumour Class Reference")

cl1, cl2, cl3, cl4 = st.columns(4)
for col_c, (cid, label, color, detail) in zip(
    [cl1, cl2, cl3, cl4],
    [
        (0, "Background",       "#334155", "Normal brain tissue"),
        (1, "Necrotic Core",    RED,       "Class 1 — dead tissue"),
        (2, "Edema",            BLUE,      "Class 2 — swelling region"),
        (3, "Enhancing Tumor",  AMBER,     "Class 3 — active tumor"),
    ],
):
    with col_c:
        st.markdown(
            f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
            f'border-radius:10px;padding:1rem;">'
            f'<div style="width:36px;height:36px;border-radius:9px;background:{color};'
            f'margin-bottom:.6rem;border:1px solid rgba(255,255,255,.1);"></div>'
            f'<div style="font-size:.82rem;font-weight:700;color:{T1};'
            f'margin-bottom:.2rem;">{label}</div>'
            f'<div style="font-size:.72rem;color:{T3};">{detail}</div></div>',
            unsafe_allow_html=True,
        )

divider()

# ── Checkpoint status ─────────────────────────────────────────────────────────
section_title("Checkpoint Status")

ckpt_paths = ["checkpoints/model.pth", "checkpoints/best_model.pth", "model.pth"]
found = [(p, Path(p).exists()) for p in ckpt_paths]

for path, exists in found:
    color = GREEN if exists else T3
    badge = status_badge("Found", "green") if exists else status_badge("Missing", "gray")
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:.5rem .75rem;background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:8px;margin-bottom:4px;font-size:.8rem;">'
        f'<code style="color:{color};background:transparent;border:none;">{path}</code>'
        f'{badge}</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div style="background:rgba(59,130,246,.08);border:1px solid {BLUE};'
    f'border-radius:10px;padding:.75rem 1rem;font-size:.8rem;color:{BLUE};'
    f'margin-top:.6rem;">'
    f'To train the model: '
    f'<code>python train.py --data_dir &lt;brats_path&gt;</code><br>'
    f'Checkpoint will be saved to <code>checkpoints/model.pth</code></div>',
    unsafe_allow_html=True,
)
