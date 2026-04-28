"""
View Results — NeuroSeg
Split slice viewer + segmentation overlay + optional 3D Plotly visualisation.
FIX: nibabel BytesIO save replaced with tempfile approach.
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import time
import tempfile
import os

st.set_page_config(
    page_title="View Results — NeuroSeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(str(Path(__file__).parent.parent / "src"))

from styles import (
    inject_css, sidebar_logo, sidebar_nav_label, sidebar_status,
    metric_card, section_title, divider, page_header, status_badge,
    BLUE, GREEN, AMBER, RED, CYAN, PURPLE,
    SURFACE_2, SURFACE_3, BORDER, BORDER_2, T1, T2, T3, MONO,
)
from visualize import (
    plot_slice, plot_overlay, plot_multiplane,
    create_3d_volume, TUMOR_COLORS, CLASS_NAMES,
)
from utils import calculate_all_metrics

inject_css()

for k, v in {
    "view_mode": "2D", "current_slice": None, "metrics": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
    "View Results",
    "Interactive MRI slice viewer with segmentation overlay and 3D rendering",
    breadcrumb="View Results",
)

if st.session_state.get("segmentation_result") is None:
    st.error("No segmentation results available. Please run segmentation first.")
    if st.button("Go to Run Segmentation"):
        st.switch_page("pages/3_Run_Segmentation.py")
    st.stop()

mri_data = st.session_state.mri_data
seg_data = st.session_state.segmentation_result

if not st.session_state.metrics:
    with st.spinner("Computing metrics…"):
        m = calculate_all_metrics(seg_data)
        m["tumor_percentage"] = float(np.sum(seg_data > 0) / seg_data.size * 100)
        st.session_state.metrics = m

metrics = st.session_state.metrics

if st.session_state.current_slice is None:
    st.session_state.current_slice = mri_data.shape[2] // 2

# ── Top metric strip ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    det = "Positive" if np.sum(seg_data > 0) > 0 else "Negative"
    col = GREEN if det == "Positive" else RED
    st.markdown(
        metric_card("Tumor Detected", det, "", col,
                    '<circle cx="12" cy="12" r="10"/>'
                    '<line x1="12" y1="8" x2="12" y2="12"/>'
                    '<line x1="12" y1="16" x2="12.01" y2="16"/>'),
        unsafe_allow_html=True)
with m2:
    cov = np.sum(seg_data > 0) / seg_data.size * 100
    st.markdown(
        metric_card("Coverage", f"{cov:.2f}%", "of scan volume", BLUE,
                    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
        unsafe_allow_html=True)
with m3:
    slices_t = np.where(np.sum(seg_data > 0, axis=(0, 1)) > 0)[0]
    sr = f"{slices_t[0]+1}–{slices_t[-1]+1}" if len(slices_t) > 0 else "N/A"
    st.markdown(
        metric_card("Tumor Slices", sr, "axial range", PURPLE,
                    '<line x1="8" y1="6" x2="21" y2="6"/>'
                    '<line x1="8" y1="12" x2="21" y2="12"/>'
                    '<line x1="8" y1="18" x2="21" y2="18"/>'
                    '<line x1="3" y1="6" x2="3.01" y2="6"/>'
                    '<line x1="3" y1="12" x2="3.01" y2="12"/>'
                    '<line x1="3" y1="18" x2="3.01" y2="18"/>'),
        unsafe_allow_html=True)
with m4:
    t2 = st.session_state.get("segmentation_time")
    tv = f"{t2:.2f}s" if isinstance(t2, (int, float)) else "N/A"
    st.markdown(
        metric_card("Inference Time", tv, "wall-clock", CYAN,
                    '<circle cx="12" cy="12" r="10"/>'
                    '<polyline points="12 6 12 12 16 14"/>'),
        unsafe_allow_html=True)

divider()

# ── View mode toggle ──────────────────────────────────────────────────────────
vm_l, vm_r, _ = st.columns([1, 1, 4])
with vm_l:
    if st.button("2D Slice Viewer", use_container_width=True,
                 type="primary" if st.session_state.view_mode == "2D" else "secondary"):
        st.session_state.view_mode = "2D"
        st.rerun()
with vm_r:
    if st.button("3D Volume", use_container_width=True,
                 type="primary" if st.session_state.view_mode == "3D" else "secondary"):
        st.session_state.view_mode = "3D"
        st.rerun()

st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

# ── 2D viewer ─────────────────────────────────────────────────────────────────
if st.session_state.view_mode == "2D":
    n_slices = mri_data.shape[2]

    sl_col, inf_col = st.columns([5, 1])
    with sl_col:
        slice_idx = st.slider(
            "Slice",
            min_value=0, max_value=n_slices - 1,
            value=st.session_state.current_slice,
            label_visibility="collapsed",
            key="view_slice_slider",
        )
        st.session_state.current_slice = slice_idx
    with inf_col:
        st.markdown(
            f'<div style="text-align:center;padding:.4rem;background:{SURFACE_2};'
            f'border:1px solid {BORDER};border-radius:8px;font-size:.78rem;'
            f'color:{T2};">Slice<br>'
            f'<span style="color:{T1};font-weight:700;font-family:{MONO};">'
            f'{slice_idx+1}/{n_slices}</span></div>',
            unsafe_allow_html=True,
        )

    # Split layout: LEFT = Original MRI, RIGHT = Overlay
    left_v, right_v = st.columns(2)

    with left_v:
        st.markdown(
            f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
            f'border-radius:12px;padding:1rem;">'
            f'<div style="font-size:.78rem;font-weight:700;color:{T1};'
            f'margin-bottom:.5rem;display:flex;align-items:center;gap:.4rem;">'
            f'<div style="width:6px;height:6px;border-radius:50%;background:{BLUE};"></div>'
            f'Original MRI</div>',
            unsafe_allow_html=True,
        )
        try:
            fig_mri, ax = plt.subplots(figsize=(5, 5))
            fig_mri.patch.set_facecolor("#0f1623")
            ax.set_facecolor("#0f1623")
            mri_s = mri_data[:, :, slice_idx]
            ax.imshow(mri_s.T, cmap="gray", origin="lower",
                      vmin=np.percentile(mri_s, 1),
                      vmax=np.percentile(mri_s, 99))
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_mri, use_container_width=True)
            plt.close(fig_mri)
        except Exception as e:
            st.error(f"MRI render error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_v:
        st.markdown(
            f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
            f'border-radius:12px;padding:1rem;">'
            f'<div style="font-size:.78rem;font-weight:700;color:{T1};'
            f'margin-bottom:.5rem;display:flex;align-items:center;gap:.4rem;">'
            f'<div style="width:6px;height:6px;border-radius:50%;background:{GREEN};"></div>'
            f'Segmentation Overlay</div>',
            unsafe_allow_html=True,
        )
        try:
            fig_ov, ax2 = plt.subplots(figsize=(5, 5))
            fig_ov.patch.set_facecolor("#0f1623")
            ax2.set_facecolor("#0f1623")
            mri_s   = mri_data[:, :, slice_idx]
            seg_s   = seg_data[:, :, slice_idx]
            ax2.imshow(mri_s.T, cmap="gray", origin="lower",
                       vmin=np.percentile(mri_s, 1),
                       vmax=np.percentile(mri_s, 99))
            # Overlay
            overlay = np.zeros((*seg_s.shape, 4))
            color_map = {
                1: [1., 0.22, 0.22, 0.75],
                2: [0.24, 0.71, 0.97, 0.55],
                3: [1., 0.85, 0.09, 0.80],
            }
            for cid, rgba in color_map.items():
                overlay[seg_s == cid] = rgba
            ax2.imshow(overlay.transpose(1, 0, 2), origin="lower")
            ax2.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_ov, use_container_width=True)
            plt.close(fig_ov)
        except Exception as e:
            st.error(f"Overlay render error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Legend
    st.markdown(
        f'<div style="display:flex;gap:.75rem;flex-wrap:wrap;margin:.6rem 0;">'
        + "".join([
            f'<div style="display:flex;align-items:center;gap:5px;">'
            f'<div style="width:10px;height:10px;border-radius:3px;background:{c};"></div>'
            f'<span style="font-size:.72rem;color:{T2};">{lbl}</span></div>'
            for c, lbl in [
                (RED, "Necrotic Core"),
                ("#3db4f7", "Edema"),
                (AMBER, "Enhancing Tumor"),
            ]
        ]) + "</div>",
        unsafe_allow_html=True,
    )

    # Multi-plane
    if st.checkbox("Show Multi-Plane (Axial · Sagittal · Coronal)", key="multiplane"):
        try:
            fig_m = plot_multiplane(mri_data, seg_data,
                                    slice_indices=(slice_idx, slice_idx, slice_idx),
                                    alpha=0.4, figsize=(16, 5))
            fig_m.patch.set_facecolor("#0f1623")
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)
        except Exception as e:
            st.error(f"Multi-plane error: {e}")

else:
    # ── 3D viewer ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:12px;padding:1rem;margin-bottom:.6rem;">'
        f'<div style="font-size:.8rem;color:{T2};">'
        f'Use mouse to rotate, scroll to zoom. Only tumour voxels are rendered.</div></div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Rendering 3D volume…"):
        try:
            fig_3d = create_3d_volume(seg_data, downsample=2)
            import plotly.graph_objects as go
            fig_3d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    bgcolor="rgba(15,22,35,1)",
                    xaxis=dict(gridcolor=BORDER, backgroundcolor="rgba(0,0,0,0)"),
                    yaxis=dict(gridcolor=BORDER, backgroundcolor="rgba(0,0,0,0)"),
                    zaxis=dict(gridcolor=BORDER, backgroundcolor="rgba(0,0,0,0)"),
                ),
                font=dict(color=T2),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T2)),
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as e:
            st.error(f"3D render error: {e}")

divider()

# ── Detailed metrics ──────────────────────────────────────────────────────────
section_title("Volumetric Analysis")

dm1, dm2, dm3, dm4 = st.columns(4)
with dm1: st.metric("Total Volume",   f"{metrics.get('total_tumor_volume_cm3',0):.3f} cm³")
with dm2: st.metric("Necrotic Core",  f"{metrics.get('volume_class_1_cm3',0):.3f} cm³")
with dm3: st.metric("Edema",          f"{metrics.get('volume_class_2_cm3',0):.3f} cm³")
with dm4: st.metric("Enhancing",      f"{metrics.get('volume_class_3_cm3',0):.3f} cm³")

dm5, dm6, dm7, dm8 = st.columns(4)
with dm5: st.metric("Surface Area",   f"{metrics.get('surface_area_mm2',0):.0f} mm²")
with dm6: st.metric("Tumor Coverage", f"{metrics.get('tumor_percentage',0):.3f}%")
with dm7:
    meta_sh = (st.session_state.mri_metadata or {}).get("shape", seg_data.shape)
    st.metric("MRI Dimensions", f"{meta_sh[0]}×{meta_sh[1]}×{meta_sh[2]}")
with dm8:
    t3 = st.session_state.get("segmentation_time")
    st.metric("Inference Time", f"{t3:.2f}s" if isinstance(t3,(int,float)) else "N/A")

divider()

# ── Downloads ─────────────────────────────────────────────────────────────────
section_title("Download Options")

dl1, dl2, dl3 = st.columns(3)

with dl1:
    buf_npz = io.BytesIO()
    np.savez_compressed(buf_npz, segmentation=seg_data, mri=mri_data)
    buf_npz.seek(0)
    st.download_button(
        "Download Segmentation (.npz)",
        data=buf_npz,
        file_name=f"segmentation_{int(time.time())}.npz",
        mime="application/octet-stream",
        use_container_width=True,
    )

with dl2:
    # FIX: nibabel cannot write to BytesIO — use tempfile instead
    try:
        import nibabel as nib
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_nii:
            seg_img = nib.Nifti1Image(seg_data.astype(np.int16), np.eye(4))
            nib.save(seg_img, tmp_nii.name)
            tmp_nii_path = tmp_nii.name
        with open(tmp_nii_path, "rb") as f:
            nii_bytes = f.read()
        os.unlink(tmp_nii_path)

        st.download_button(
            "Download Segmentation (.nii.gz)",
            data=nii_bytes,
            file_name=f"segmentation_{int(time.time())}.nii.gz",
            mime="application/gzip",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"NIfTI export error: {e}")

with dl3:
    if st.button("Export Slice as PNG", use_container_width=True):
        try:
            fig_exp, ax_exp = plt.subplots(figsize=(6, 6))
            fig_exp.patch.set_facecolor("#0f1623")
            ax_exp.set_facecolor("#0f1623")
            mri_s2 = mri_data[:, :, st.session_state.current_slice]
            seg_s2 = seg_data[:, :, st.session_state.current_slice]
            ax_exp.imshow(mri_s2.T, cmap="gray", origin="lower")
            ov2 = np.zeros((*seg_s2.shape, 4))
            for cid, rgba in {1:[1,.22,.22,.75],2:[.24,.71,.97,.55],3:[1,.85,.09,.80]}.items():
                ov2[seg_s2==cid]=rgba
            ax_exp.imshow(ov2.transpose(1,0,2),origin="lower")
            ax_exp.axis("off")
            plt.tight_layout(pad=0)
            buf_png = io.BytesIO()
            fig_exp.savefig(buf_png, format="png", dpi=150, bbox_inches="tight",
                            facecolor="#0f1623")
            buf_png.seek(0)
            plt.close(fig_exp)
            st.download_button(
                "Save PNG",
                data=buf_png,
                file_name=f"slice_{st.session_state.current_slice}_{int(time.time())}.png",
                mime="image/png",
                key="png_dl",
            )
        except Exception as e:
            st.error(f"PNG export error: {e}")
