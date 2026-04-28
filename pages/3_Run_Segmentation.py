"""
Run Segmentation — NeuroSeg
Pipeline visualisation with animated progress steps.
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import time

st.set_page_config(
    page_title="Run Segmentation — NeuroSeg",
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
from model_loader import load_model, get_model_info
from utils import calculate_all_metrics

inject_css()

for k, v in {
    "segmentation_result": None, "segmentation_probabilities": None,
    "segmentation_time": None, "metrics": {}, "model_type": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

sidebar_logo()
sidebar_status(
    st.session_state.get("mri_data") is not None,
    st.session_state.segmentation_result is not None,
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
    "Run Segmentation",
    "3D U-Net · Brain Tumor Detection & Sub-Region Classification",
    breadcrumb="Run Segmentation",
)

if st.session_state.get("mri_data") is None:
    st.error("No MRI data loaded. Please upload a scan first.")
    if st.button("Go to Upload MRI"):
        st.switch_page("pages/2_Upload_MRI.py")
    st.stop()

# ── Model detection ───────────────────────────────────────────────────────────
CKPT_PATHS  = ["checkpoints/model.pth", "checkpoints/best_model.pth", "model.pth"]
ckpt_found  = next((p for p in CKPT_PATHS if Path(p).exists()), None)
real_model  = ckpt_found is not None

# ── Status bar ────────────────────────────────────────────────────────────────
sb1, sb2, sb3, sb4 = st.columns(4)
seg_done = st.session_state.segmentation_result is not None

SCAN_SVG   = '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>'
MODEL_SVG  = '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>'
SEG_SVG    = '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
TIME_SVG   = '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'

mets = st.session_state.metrics or {}
t    = st.session_state.segmentation_time

with sb1:
    st.markdown(metric_card("MRI Data", "Ready", "Loaded in session", GREEN, SCAN_SVG),
                unsafe_allow_html=True)
with sb2:
    ml = "Real 3D U-Net" if real_model else "Mock (Demo)"
    mc = GREEN if real_model else AMBER
    st.markdown(metric_card("Model", ml, ckpt_found or "no checkpoint", mc, MODEL_SVG),
                unsafe_allow_html=True)
with sb3:
    sl = "Completed" if seg_done else "Pending"
    sc = GREEN if seg_done else T3
    v  = f"Vol: {mets.get('total_tumor_volume_cm3',0):.2f} cm³" if seg_done else "Run to analyse"
    st.markdown(metric_card("Segmentation", sl, v, sc, SEG_SVG),
                unsafe_allow_html=True)
with sb4:
    tv = f"{t:.2f} s" if isinstance(t, (int, float)) else "N/A"
    st.markdown(metric_card("Last Inference", tv, "wall-clock time", CYAN, TIME_SVG),
                unsafe_allow_html=True)

divider()

# ── Settings ──────────────────────────────────────────────────────────────────
section_title("Inference Settings")

left_set, right_set = st.columns([1.5, 1])

with left_set:
    if not real_model:
        st.markdown(
            f'<div style="background:{AMBER_DIM if False else "rgba(245,158,11,.1)"};'
            f'border:1px solid {AMBER};border-radius:10px;padding:.75rem 1rem;'
            f'font-size:.82rem;color:{AMBER};margin-bottom:.8rem;">'
            f'No trained checkpoint found. Using mock (synthetic) predictions. '
            f'Train via <code>python train.py --data_dir &lt;path&gt;</code> '
            f'and place <code>checkpoints/model.pth</code> to use the real model.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:rgba(16,185,129,.1);border:1px solid {GREEN};'
            f'border-radius:10px;padding:.75rem 1rem;font-size:.82rem;color:{GREEN};'
            f'margin-bottom:.8rem;">'
            f'Trained model found: <code>{ckpt_found}</code></div>',
            unsafe_allow_html=True,
        )

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Voxels with max-softmax probability below this → background (class 0)",
    )

with right_set:
    use_sliding = st.checkbox(
        "Sliding-Window Inference",
        value=False,
        help="Recommended for volumes >128³. Slower but more accurate.",
    )
    with st.expander("Model Architecture Details"):
        info = get_model_info()
        for k2, v2 in info.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.75rem;padding:.15rem 0;border-bottom:1px solid {BORDER};">'
                f'<span style="color:{T2};">{k2.replace("_"," ").title()}</span>'
                f'<span style="color:{T1};font-family:{MONO};">{v2}</span></div>',
                unsafe_allow_html=True,
            )

divider()

# ── Pipeline steps (visual) ───────────────────────────────────────────────────
section_title("Pipeline")

pipe_steps = [
    ("Load Model",       "Initialise 3D U-Net weights"),
    ("Prepare Input",    "Reshape & normalise to 128³"),
    ("Neural Network",   "Forward pass / sliding window"),
    ("Post-Processing",  "Remove artefacts <10 voxels"),
    ("Metrics",          "Volume, surface area, coverage"),
]
done_steps = min(5, (1 if seg_done else 0) * 5)

pcols = st.columns(5)
for i, (col, (step, detail)) in enumerate(zip(pcols, pipe_steps)):
    with col:
        step_done = i < done_steps
        dot_bg = GREEN if step_done else (BLUE if i == done_steps else SURFACE_3)
        dot_fg = "white"
        icon_inner = (
            '<polyline points="20 6 9 17 4 12"/>'
            if step_done else str(i + 1)
        )
        icon_tag = (
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" '
            f'stroke="white" stroke-width="3" stroke-linecap="round">{icon_inner}</svg>'
            if step_done else
            f'<span style="font-size:.7rem;font-weight:700;color:{"white" if i==done_steps else T3};">{i+1}</span>'
        )
        connector = (
            f'<div style="position:absolute;top:11px;right:-50%;width:100%;'
            f'height:1px;background:{"#3b82f6" if step_done else BORDER};'
            f'z-index:0;"></div>'
            if i < 4 else ""
        )
        st.markdown(
            f'<div style="text-align:center;position:relative;">'
            f'<div style="width:24px;height:24px;border-radius:50%;'
            f'background:{dot_bg};display:flex;align-items:center;'
            f'justify-content:center;margin:0 auto .5rem;position:relative;z-index:1;">'
            f'{icon_tag}</div>'
            f'<div style="font-size:.75rem;font-weight:600;color:{"#f1f5f9" if step_done else T2};'
            f'margin-bottom:.2rem;">{step}</div>'
            f'<div style="font-size:.65rem;color:{T3};line-height:1.3;">{detail}</div>'
            f'{connector}</div>',
            unsafe_allow_html=True,
        )

divider()

# ── Run button ────────────────────────────────────────────────────────────────
_, btn_c, _ = st.columns([1, 2, 1])
with btn_c:
    run_clicked = st.button(
        "Run Tumor Segmentation",
        type="primary",
        use_container_width=True,
        key="seg_btn",
    )

if run_clicked:
    prog   = st.progress(0)
    status = st.empty()

    try:
        status.markdown(
            f'<div style="font-size:.82rem;color:{T2};text-align:center;">Loading model…</div>',
            unsafe_allow_html=True)
        prog.progress(10)
        model = load_model(
            checkpoint_path=str(ckpt_found) if ckpt_found else None,
            use_mock=not real_model,
        )
        st.session_state.model_type = "Real 3D U-Net" if real_model else "Mock (Demo)"

        status.markdown(
            f'<div style="font-size:.82rem;color:{T2};text-align:center;">Preparing input…</div>',
            unsafe_allow_html=True)
        prog.progress(25)
        input_data = st.session_state.mri_data

        status.markdown(
            f'<div style="font-size:.82rem;color:{T2};text-align:center;">'
            f'Running neural network…</div>', unsafe_allow_html=True)
        prog.progress(45)
        t0 = time.time()
        if use_sliding and real_model:
            seg_out, probs = model.predict_sliding_window(input_data)
        else:
            seg_out, probs = model.predict(input_data, confidence_threshold=conf_thresh)
        infer_time = time.time() - t0
        prog.progress(70)

        status.markdown(
            f'<div style="font-size:.82rem;color:{T2};text-align:center;">'
            f'Post-processing…</div>', unsafe_allow_html=True)
        from scipy import ndimage
        for cid in [1, 2, 3]:
            mask = seg_out == cid
            labeled, n = ndimage.label(mask)
            if n > 0:
                sizes = ndimage.sum(mask, labeled, range(1, n + 1))
                for comp in np.where(np.array(sizes) < 10)[0] + 1:
                    seg_out[labeled == comp] = 0
        prog.progress(85)

        status.markdown(
            f'<div style="font-size:.82rem;color:{T2};text-align:center;">'
            f'Computing metrics…</div>', unsafe_allow_html=True)
        metrics = calculate_all_metrics(seg_out)
        metrics["tumor_percentage"] = float(np.sum(seg_out > 0) / seg_out.size * 100)
        prog.progress(100)

        st.session_state.segmentation_result        = seg_out
        st.session_state.segmentation_probabilities = probs
        st.session_state.segmentation_time          = infer_time
        st.session_state.metrics                    = metrics

        time.sleep(0.3)
        prog.empty()
        status.empty()

        st.success(
            f"Segmentation complete in {infer_time:.2f} s — "
            f"Tumor volume: {metrics.get('total_tumor_volume_cm3',0):.2f} cm³"
        )
        st.rerun()

    except Exception as exc:
        prog.empty()
        status.empty()
        st.error(f"Segmentation failed: {exc}")
        import traceback
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

# ── Results summary ───────────────────────────────────────────────────────────
if st.session_state.segmentation_result is not None:
    divider()
    section_title("Results Summary")

    seg2 = st.session_state.segmentation_result
    m2   = st.session_state.metrics or {}

    r1, r2, r3, r4 = st.columns(4)
    with r1: st.metric("Tumor Classes",   len(np.unique(seg2)) - 1)
    with r2: st.metric("Coverage",        f"{m2.get('tumor_percentage',0):.3f} %")
    with r3: st.metric("Inference",       f"{st.session_state.segmentation_time:.2f} s")
    with r4: st.metric("Total Volume",    f"{m2.get('total_tumor_volume_cm3',0):.2f} cm³")

    section_title("Sub-Region Breakdown")
    sr1, sr2, sr3 = st.columns(3)
    for col_r, (cid, name, color) in zip(
        [sr1, sr2, sr3],
        [(1, "Necrotic Core", RED), (2, "Edema", BLUE), (3, "Enhancing", AMBER)],
    ):
        with col_r:
            cnt = int(np.sum(seg2 == cid))
            pct = cnt / seg2.size * 100
            vol = m2.get(f"volume_class_{cid}_cm3", 0)
            st.markdown(
                f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
                f'border-top:2px solid {color};border-radius:10px;padding:1rem;">'
                f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.08em;'
                f'text-transform:uppercase;color:{T2};margin-bottom:.5rem;">{name}</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{T1};'
                f'font-family:{MONO};">{pct:.3f}%</div>'
                f'<div style="font-size:.72rem;color:{T2};margin-top:.25rem;">'
                f'{vol:.3f} cm³</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="background:rgba(59,130,246,.08);border:1px solid {BLUE};'
        f'border-radius:10px;padding:.75rem 1rem;font-size:.82rem;color:{BLUE};'
        f'margin-top:.8rem;">'
        f'Proceed to View Results for the segmentation overlay and 3D visualisation, '
        f'or Generate Report to export a clinical PDF.</div>',
        unsafe_allow_html=True,
    )
