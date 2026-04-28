"""
Upload MRI — NeuroSeg
Premium drag-and-drop upload page with validation feedback.
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import tempfile
import os

st.set_page_config(
    page_title="Upload MRI — NeuroSeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(str(Path(__file__).parent.parent / "src"))

from styles import (
    inject_css, sidebar_logo, sidebar_nav_label, sidebar_status,
    metric_card, section_title, divider, page_header, status_badge,
    BLUE, GREEN, AMBER, RED, SURFACE_2, SURFACE_3, BORDER, BORDER_2,
    T1, T2, T3, MONO,
)
from preprocess import MRIPreprocessor, create_sample_mri
from utils import validate_nifti

inject_css()

for k, v in {
    "mri_data": None, "mri_metadata": None,
    "preprocessing_done": False, "original_data": None, "preprocessed_data": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

sidebar_logo()
sidebar_status(
    st.session_state.mri_data is not None,
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
    "Upload MRI Scan",
    "Load a brain MRI in NIfTI format (.nii / .nii.gz) for segmentation analysis",
    breadcrumb="Upload MRI",
)

# ── Upload zone ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([2.5, 1])

with left_col:
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:14px;padding:1.5rem;">'
        f'<div style="font-size:.82rem;font-weight:700;color:{T1};margin-bottom:.3rem;">'
        f'NIfTI File Upload</div>'
        f'<div style="font-size:.75rem;color:{T2};margin-bottom:.8rem;">'
        f'Accepted: .nii, .nii.gz &nbsp;·&nbsp; Max size: 500 MB</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drop your NIfTI file here, or click to browse",
        type=["nii", "gz"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:14px;padding:1.5rem;">'
        f'<div style="font-size:.82rem;font-weight:700;color:{T1};margin-bottom:.4rem;">'
        f'Demo Data</div>'
        f'<div style="font-size:.75rem;color:{T2};margin-bottom:.9rem;">'
        f'Generate a synthetic 128³ phantom with a simulated tumour region '
        f'for demonstration purposes.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Generate Sample MRI", use_container_width=True):
        with st.spinner("Synthesising 128x128x128 phantom with tumour..."):
            sample = create_sample_mri(shape=(128, 128, 128), add_tumor=True)
            st.session_state.original_data     = sample
            st.session_state.mri_data          = sample
            st.session_state.mri_metadata      = {
                "original_shape": sample.shape, "original_spacing": (1., 1., 1.),
                "source": "synthetic", "shape": sample.shape,
                "spacing": (1., 1., 1.), "dtype": str(sample.dtype),
            }
            st.session_state.preprocessing_done = True
        st.success("Sample MRI loaded successfully.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Process uploaded file ─────────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("Validating and loading NIfTI file..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            is_valid, message, metadata = validate_nifti(tmp_path)
            if is_valid:
                prep = MRIPreprocessor()
                data, affine, header = prep.load_nifti(tmp_path)
                st.session_state.original_data      = data
                st.session_state.mri_data           = data
                st.session_state.mri_metadata       = metadata
                st.session_state.preprocessing_done = False
                st.success(f"Loaded: {uploaded_file.name}")
            else:
                st.error(f"Validation failed: {message}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
        finally:
            os.unlink(tmp_path)

# ── File info ─────────────────────────────────────────────────────────────────
if st.session_state.mri_data is not None:
    divider()
    section_title("Loaded Scan Information")

    data = st.session_state.mri_data
    meta = st.session_state.mri_metadata or {}

    shape   = meta.get("shape", data.shape)
    spacing = meta.get("spacing", (1., 1., 1.))
    source  = meta.get("source", "uploaded")
    dtype   = meta.get("dtype", str(data.dtype))

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        dim = f"{shape[0]}×{shape[1]}×{shape[2]}" if shape else "N/A"
        st.markdown(metric_card("Dimensions", dim, "voxels", BLUE,
            '<rect x="3" y="3" width="18" height="18" rx="2"/>'
            '<circle cx="8.5" cy="8.5" r="1.5"/>'
            '<polyline points="21 15 16 10 5 21"/>'),
            unsafe_allow_html=True)
    with i2:
        sp = (f"{spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm"
              if spacing else "N/A")
        st.markdown(metric_card("Voxel Spacing", sp, "mm per voxel", GREEN,
            '<path d="M21 3H3v7h18V3z"/><path d="M21 14H3v7h18v-7z"/>'
            '<path d="M12 3v18"/><path d="M3 10h18"/>'),
            unsafe_allow_html=True)
    with i3:
        st.markdown(metric_card("Source", source.capitalize(), "data origin", AMBER,
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="8" x2="12" y2="12"/>'
            '<line x1="12" y1="16" x2="12.01" y2="16"/>'),
            unsafe_allow_html=True)
    with i4:
        st.markdown(metric_card("Data Type", dtype, "numpy dtype", RED,
            '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>'),
            unsafe_allow_html=True)

    divider()
    section_title("Intensity Profile")

    p1, p2, p3, p4 = st.columns(4)
    with p1: st.metric("Min Intensity",  f"{data.min():.1f}")
    with p2: st.metric("Max Intensity",  f"{data.max():.1f}")
    with p3: st.metric("Mean Intensity", f"{data.mean():.2f}")
    with p4: st.metric("Std Deviation",  f"{data.std():.2f}")

    divider()

    if st.session_state.preprocessing_done:
        st.success("Data is preprocessed and ready for segmentation.")
    else:
        st.info("Data loaded. Proceed to Run Segmentation when ready.")

    # Quick centre-slice preview
    section_title("Quick Preview (Central Axial Slice)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mid = data.shape[2] // 2
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#0f1623")
        ax.set_facecolor("#0f1623")
        ax.imshow(data[:, :, mid].T, cmap="gray", origin="lower",
                  vmin=np.percentile(data, 1), vmax=np.percentile(data, 99))
        ax.axis("off")
        plt.tight_layout(pad=0)

        _, prev_col, _ = st.columns([1, 2, 1])
        with prev_col:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception:
        pass

else:
    divider()
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:12px;padding:2.5rem;text-align:center;">'
        f'<div style="font-size:.88rem;color:{T3};">'
        f'No MRI loaded. Upload a NIfTI file or generate sample data above.</div></div>',
        unsafe_allow_html=True,
    )
