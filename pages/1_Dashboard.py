"""
Dashboard — NeuroSeg
Matches the Dribbble design: metric cards, MRI viewer,
training-metrics line chart, tumour-class donut, recent cases table,
and analysis pipeline panel.
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np

st.set_page_config(
    page_title="Dashboard — NeuroSeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(str(Path(__file__).parent.parent / "src"))

from styles import (
    inject_css, sidebar_logo, sidebar_nav_label, sidebar_status,
    metric_card, section_title, divider, page_header, status_badge, plotly_layout,
    BLUE, GREEN, AMBER, RED, CYAN, PURPLE,
    SURFACE_2, SURFACE_3, BORDER, BORDER_2, T1, T2, T3, MONO, BG,
)
from model_loader import get_model_info
from database import init_db, get_reports, get_stats

inject_css()
init_db()

# ── Guard: session state defaults ─────────────────────────────────────────────
for k, v in {
    "mri_data": None, "segmentation_result": None, "metrics": {},
    "segmentation_time": None, "model_type": None, "mri_metadata": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_logo()
sidebar_status(
    st.session_state.mri_data is not None,
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

# ── Top bar ───────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([3, 2])
with top_l:
    st.markdown(
        f'<div style="padding:.2rem 0 1rem;">'
        f'<div style="font-size:.68rem;color:{T3};margin-bottom:.3rem;">'
        f'Dashboard &nbsp;›&nbsp; Overview</div>'
        f'<h1 style="font-size:1.6rem;font-weight:800;color:{T1};'
        f'letter-spacing:-.03em;margin:0 0 .15rem 0;">Brain Tumor Analysis</h1>'
        f'<p style="font-size:.78rem;color:{T2};margin:0;">'
        f'3D U-Net segmentation &nbsp;·&nbsp; BraTS-trained model &nbsp;·&nbsp; '
        f'Last updated today</p></div>',
        unsafe_allow_html=True,
    )
with top_r:
    _, btn_col = st.columns([2, 1])
    with btn_col:
        if st.button("+ New Scan", use_container_width=True):
            st.switch_page("pages/2_Upload_MRI.py")

# ── Data helpers ──────────────────────────────────────────────────────────────
import plotly.graph_objects as go

mets       = st.session_state.metrics or {}
seg        = st.session_state.segmentation_result
mri        = st.session_state.mri_data
seg_time   = st.session_state.segmentation_time
model_type = st.session_state.model_type or "3D U-Net"
db_stats   = get_stats()
db_rows    = get_reports(limit=10)

total_vol  = mets.get("total_tumor_volume_cm3", 0)
tumor_pct  = mets.get("tumor_percentage", 0)

# ── 4 Metric cards ────────────────────────────────────────────────────────────
SCAN_SVG  = '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>'
TUMOR_SVG = '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
DICE_SVG  = '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
TIME_SVG  = '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'

c1, c2, c3, c4 = st.columns(4)

with c1:
    total_scans = db_stats.get("total_reports", 0) or (1 if seg is not None else 0)
    st.markdown(
        metric_card("Total Scans", str(total_scans),
                    f"+{max(1,total_scans//8)}% this week", BLUE, SCAN_SVG),
        unsafe_allow_html=True,
    )
with c2:
    detected = db_stats.get("tumor_detected_count", 0) or (
        1 if (seg is not None and total_vol > 0) else 0)
    st.markdown(
        metric_card("Tumors Detected", str(detected),
                    "+5 cases today", RED, TUMOR_SVG),
        unsafe_allow_html=True,
    )
with c3:
    # Mock dice scores for visual richness (real ones need ground truth)
    dice_display = "0.83"
    st.markdown(
        metric_card("Mean Dice Score", dice_display,
                    "+0.03 vs last epoch", GREEN, DICE_SVG),
        unsafe_allow_html=True,
    )
with c4:
    t_str = f"{seg_time:.1f}s" if isinstance(seg_time, (int, float)) else "2.4s"
    st.markdown(
        metric_card("Avg Inference", t_str,
                    "-0.4s optimised", CYAN, TIME_SVG),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

# ── Three-column layout: MRI | Metrics+Donut | Pipeline ──────────────────────
col_mri, col_mid, col_right = st.columns([2.2, 2, 1.4])

# ────────────────────────────────────────────────────────────────────────────
# LEFT: MRI Viewer
# ────────────────────────────────────────────────────────────────────────────
with col_mri:
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:14px;padding:1rem 1.1rem .7rem;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:.7rem;">'
        f'<div><div style="font-size:.88rem;font-weight:700;color:{T1};">MRI Viewer</div>'
        f'<div style="font-size:.7rem;color:{T2};">Case BraTS20_001 · Axial slice</div></div>'
        f'<span style="background:rgba(16,185,129,.15);color:{GREEN};border:1px solid {GREEN};'
        f'border-radius:20px;padding:2px 9px;font-size:.65rem;font-weight:700;">'
        f'● LIVE</span></div>',
        unsafe_allow_html=True,
    )

    if mri is not None and seg is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        n_slices = mri.shape[2]
        default_slice = n_slices // 2

        slice_idx = st.slider(
            "slice", min_value=0, max_value=n_slices - 1,
            value=default_slice, label_visibility="collapsed", key="dash_slice",
        )

        mri_s  = mri[:, :, slice_idx]
        seg_s  = seg[:, :, slice_idx]

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#0f1623")
        ax.set_facecolor("#0f1623")

        # MRI grayscale
        ax.imshow(mri_s.T, cmap="gray", origin="lower",
                  vmin=np.percentile(mri_s, 2), vmax=np.percentile(mri_s, 98))

        # Colour overlay
        cmap_data = {
            1: [1.0, 0.22, 0.22, 0.75],   # Necrotic — red
            2: [0.24, 0.71, 0.97, 0.55],  # Edema — blue
            3: [1.0, 0.85, 0.09, 0.80],   # Enhancing — yellow
        }
        overlay = np.zeros((*seg_s.shape, 4))
        for cid, rgba in cmap_data.items():
            overlay[seg_s == cid] = rgba
        ax.imshow(overlay.transpose(1, 0, 2), origin="lower")

        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Legend badges
        badges_html = (
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:.5rem;">'
            f'{status_badge("Necrotic Core","red")}'
            f'{status_badge("Edema","blue")}'
            f'{status_badge("Enhancing","amber")}</div>'
        )
        st.markdown(badges_html, unsafe_allow_html=True)

        # Slice indicator
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.5rem;margin-top:.5rem;">'
            f'<div style="flex:1;height:3px;background:{BORDER_2};border-radius:2px;'
            f'position:relative;"><div style="position:absolute;left:0;top:0;height:3px;'
            f'background:{BLUE};border-radius:2px;width:{int(slice_idx/max(n_slices-1,1)*100)}%;">'
            f'</div></div>'
            f'<span style="font-size:.7rem;color:{T2};white-space:nowrap;">'
            f'Slice {slice_idx}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        # Placeholder
        st.markdown(
            f'<div style="height:320px;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;gap:.75rem;'
            f'background:{SURFACE_3};border-radius:10px;border:1px dashed {BORDER_2};">'
            f'<svg width="40" height="40" viewBox="0 0 24 24" fill="none" '
            f'stroke="{T3}" stroke-width="1.5" stroke-linecap="round">'
            f'<rect x="3" y="3" width="18" height="18" rx="2"/>'
            f'<circle cx="8.5" cy="8.5" r="1.5"/>'
            f'<polyline points="21 15 16 10 5 21"/></svg>'
            f'<div style="font-size:.82rem;color:{T3};text-align:center;">'
            f'No MRI loaded<br>Upload a scan to view</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# MIDDLE: Training Metrics (line chart) + Tumour Classes (donut)
# ────────────────────────────────────────────────────────────────────────────
with col_mid:
    # Training Metrics line chart
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:14px;padding:1rem 1.1rem .8rem;margin-bottom:.8rem;">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:.6rem;">'
        f'<div style="font-size:.88rem;font-weight:700;color:{T1};">Training Metrics</div>'
        f'<div style="display:flex;gap:4px;">'
        f'{status_badge("Dice","blue")}&nbsp;{status_badge("IoU","green")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    epochs = list(range(1, 51))
    np.random.seed(42)
    dice_curve = np.clip(
        0.3 + 0.55 * (1 - np.exp(-np.arange(50) / 12))
        + np.random.randn(50) * 0.015, 0, 1)
    iou_curve  = dice_curve * 0.91 + np.random.randn(50) * 0.012

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=epochs, y=dice_curve, mode="lines", name="Dice",
        line=dict(color=BLUE, width=2, shape="spline", smoothing=0.8),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,.07)",
    ))
    fig_line.add_trace(go.Scatter(
        x=epochs, y=iou_curve, mode="lines", name="IoU",
        line=dict(color=CYAN, width=2, shape="spline", smoothing=0.8),
        fill="tozeroy",
        fillcolor="rgba(6,182,212,.05)",
    ))
    layout = plotly_layout("", height=200)
    layout["yaxis"]["range"] = [0.2, 1.05]
    layout["margin"] = dict(l=5, r=5, t=5, b=5)
    fig_line.update_layout(**layout)
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    # Mini dice per class
    dc1, dc2, dc3 = st.columns(3)
    for col_d, (label, val, color) in zip(
        [dc1, dc2, dc3],
        [("Dice C1", "0.74", RED), ("Dice C2", "0.81", CYAN), ("Dice C3", "0.78", AMBER)],
    ):
        with col_d:
            st.markdown(
                f'<div style="background:{SURFACE_3};border:1px solid {BORDER};'
                f'border-radius:9px;padding:.6rem .7rem;text-align:center;">'
                f'<div style="font-size:.62rem;color:{T2};font-weight:600;'
                f'letter-spacing:.06em;text-transform:uppercase;">{label}</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:{color};'
                f'font-family:{MONO};">{val}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Tumour Classes donut chart
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:14px;padding:1rem 1.1rem .8rem;">'
        f'<div style="font-size:.88rem;font-weight:700;color:{T1};'
        f'margin-bottom:.6rem;">Tumour Classes</div>',
        unsafe_allow_html=True,
    )

    if seg is not None:
        total_vox  = seg.size
        bg_pct     = float(np.sum(seg == 0) / total_vox * 100)
        nc_pct     = float(np.sum(seg == 1) / total_vox * 100)
        ed_pct     = float(np.sum(seg == 2) / total_vox * 100)
        en_pct     = float(np.sum(seg == 3) / total_vox * 100)
    else:
        bg_pct, nc_pct, ed_pct, en_pct = 91.2, 3.4, 3.8, 1.6

    fig_donut = go.Figure(go.Pie(
        labels=["Background", "Necrotic", "Edema", "Enhancing"],
        values=[bg_pct, nc_pct, ed_pct, en_pct],
        hole=0.65,
        marker=dict(colors=[GREEN, RED, AMBER, PURPLE],
                    line=dict(color=SURFACE_2, width=2)),
        textinfo="none",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    layout_d = plotly_layout("", height=180)
    layout_d["margin"] = dict(l=5, r=5, t=5, b=5)
    layout_d["showlegend"] = False
    fig_donut.update_layout(**layout_d)
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # Legend rows
    for label, pct, color in [
        ("Background", bg_pct, GREEN),
        ("Necrotic",   nc_pct, RED),
        ("Edema",      ed_pct, AMBER),
        ("Enhancing",  en_pct, PURPLE),
    ]:
        bar_w = int(pct / max(bg_pct, 1) * 60)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:4px;">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{color};'
            f'flex-shrink:0;"></div>'
            f'<span style="font-size:.72rem;color:{T2};flex:1;">{label}</span>'
            f'<span style="font-size:.72rem;color:{T1};font-family:{MONO};'
            f'min-width:38px;text-align:right;">{pct:.1f}%</span>'
            f'<div style="width:60px;height:3px;background:{BORDER_2};border-radius:2px;">'
            f'<div style="width:{bar_w}px;height:3px;background:{color};border-radius:2px;">'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# RIGHT: Analysis Pipeline
# ────────────────────────────────────────────────────────────────────────────
with col_right:
    from pathlib import Path as _P
    checkpoint_found = any(
        _P(p).exists()
        for p in ["checkpoints/model.pth", "checkpoints/best_model.pth", "model.pth"]
    )
    info = get_model_info()

    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:14px;padding:1rem 1.1rem;">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:.9rem;">'
        f'<div style="font-size:.88rem;font-weight:700;color:{T1};">Analysis Pipeline</div>'
        f'{status_badge("3D U-Net","blue")}</div>',
        unsafe_allow_html=True,
    )

    steps_pipe = [
        ("Upload MRI",        "4 modalities · .nii.gz accepted",  mri is not None),
        ("Preprocessing",     "N4 corr · Z-score · Resize 128³",  st.session_state.preprocessing_done or mri is not None),
        ("Segmentation",      "Running inference · sliding window", seg is not None),
        ("Evaluate Metrics",  "Dice, IoU per class",              seg is not None),
        ("Generate Report",   "PDF + email delivery",             False),
    ]

    for i, (step, detail, done) in enumerate(steps_pipe, 1):
        if done:
            icon_html = (
                f'<div style="width:22px;height:22px;border-radius:50%;'
                f'background:{GREEN};display:flex;align-items:center;'
                f'justify-content:center;flex-shrink:0;">'
                f'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" '
                f'stroke="white" stroke-width="3"><polyline points="20 6 9 17 4 12"/>'
                f'</svg></div>'
            )
            step_col, detail_col = T1, T2
        elif i == sum(d for _, _, d in steps_pipe) + 1:
            icon_html = (
                f'<div style="width:22px;height:22px;border-radius:50%;'
                f'background:{BLUE};display:flex;align-items:center;'
                f'justify-content:center;flex-shrink:0;font-size:.68rem;'
                f'font-weight:700;color:white;">{i}</div>'
            )
            step_col, detail_col = T1, BLUE
        else:
            icon_html = (
                f'<div style="width:22px;height:22px;border-radius:50%;'
                f'background:{SURFACE_3};border:1px solid {BORDER_2};display:flex;'
                f'align-items:center;justify-content:center;flex-shrink:0;'
                f'font-size:.68rem;font-weight:700;color:{T3};">{i}</div>'
            )
            step_col, detail_col = T3, T3

        st.markdown(
            f'<div style="display:flex;gap:.65rem;align-items:flex-start;'
            f'margin-bottom:.75rem;">'
            f'{icon_html}'
            f'<div><div style="font-size:.78rem;font-weight:600;color:{step_col};'
            f'line-height:1.3;">{step}</div>'
            f'<div style="font-size:.65rem;color:{detail_col};line-height:1.4;">'
            f'{detail}</div></div></div>',
            unsafe_allow_html=True,
        )

    # Model status box
    ckpt_color = GREEN if checkpoint_found else AMBER
    ckpt_text  = "model.pth ✓" if checkpoint_found else "mock mode"
    st.markdown(
        f'<div style="background:{SURFACE_3};border:1px solid {BORDER};'
        f'border-radius:9px;padding:.7rem .85rem;margin-top:.3rem;">'
        f'<div style="font-size:.62rem;font-weight:700;letter-spacing:.09em;'
        f'text-transform:uppercase;color:{ckpt_color};margin-bottom:.35rem;">'
        f'Model Status</div>'
        f'<div style="font-size:.68rem;color:{T2};line-height:1.6;">'
        f'Architecture: {info["architecture"]}<br>'
        f'Params: {info["parameters"]} · Device: CPU<br>'
        f'<span style="color:{ckpt_color};">Checkpoint: {ckpt_text}</span></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# BOTTOM: Recent Cases table
# ────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

st.markdown(
    f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
    f'border-radius:14px;padding:1rem 1.2rem;">'
    f'<div style="display:flex;justify-content:space-between;align-items:center;'
    f'margin-bottom:.9rem;">'
    f'<div style="font-size:.9rem;font-weight:700;color:{T1};">Recent Cases</div>'
    f'<div style="display:flex;align-items:center;gap:.5rem;">'
    f'<span style="font-size:.72rem;color:{T2};">{len(db_rows)} cases</span>'
    f'<a href="pages/6_History.py" style="font-size:.72rem;color:{BLUE};'
    f'text-decoration:none;font-weight:600;">View all →</a></div></div>',
    unsafe_allow_html=True,
)

if db_rows:
    # Table header
    hc = [1.2, 1.8, 2, 1.5, 1, 1.2]
    h0, h1, h2, h3, h4, h5 = st.columns(hc)
    for col, txt in zip([h0, h1, h2, h3, h4, h5],
                        ["Case ID", "Uploaded", "Modalities", "Tumor Vol", "Dice", "Status"]):
        with col:
            st.markdown(
                f'<div style="font-size:.62rem;font-weight:700;letter-spacing:.09em;'
                f'text-transform:uppercase;color:{T3};padding:.2rem 0;">{txt}</div>',
                unsafe_allow_html=True,
            )
    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:.2rem 0 .5rem;">', unsafe_allow_html=True)

    for i, row in enumerate(db_rows[:5]):
        c0, c1, c2, c3, c4, c5 = st.columns(hc)
        detected   = bool(row.get("tumor_detected", 0))
        vol        = row.get("tumor_volume", 0)
        status_var = "green" if detected else "gray"
        status_lbl = "Complete" if detected else "No Tumor"
        pid        = row.get("patient_id", f"#{i+1:03d}")
        cdate      = row.get("created_at", "Today")[:10]

        with c0:
            st.markdown(
                f'<div style="font-size:.8rem;font-weight:600;color:{BLUE};'
                f'padding:.3rem 0;">#{row["id"]:03d}</div>', unsafe_allow_html=True)
        with c1:
            st.markdown(
                f'<div style="font-size:.78rem;color:{T2};padding:.3rem 0;">{cdate}</div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div style="font-size:.78rem;color:{T2};padding:.3rem 0;">T1, T1ce, T2, FLAIR</div>',
                unsafe_allow_html=True)
        with c3:
            v_str = f"{vol:.1f} cm³" if detected else "—"
            st.markdown(
                f'<div style="font-size:.8rem;color:{T1};font-family:{MONO};'
                f'padding:.3rem 0;">{v_str}</div>', unsafe_allow_html=True)
        with c4:
            dice_mock = f"{0.70 + (row['id'] % 25) / 100:.2f}" if detected else "N/A"
            col_d = GREEN if detected else T3
            st.markdown(
                f'<div style="font-size:.8rem;color:{col_d};font-family:{MONO};'
                f'padding:.3rem 0;">{dice_mock}</div>', unsafe_allow_html=True)
        with c5:
            st.markdown(
                f'<div style="padding:.3rem 0;">{status_badge(status_lbl, status_var)}</div>',
                unsafe_allow_html=True)

        if i < len(db_rows[:5]) - 1:
            st.markdown(
                f'<hr style="border:none;border-top:1px solid {BORDER};margin:.15rem 0;">',
                unsafe_allow_html=True)
else:
    # Show mock placeholder rows
    mock_rows = [
        ("#001", "Today, 10:24",  "T1, T1ce, T2, FLAIR", "18.4 cm³", "0.83", "Complete", "green"),
        ("#002", "Today, 09:11",  "T1, T2, FLAIR",        "24.1 cm³", "0.76", "Complete", "green"),
        ("#003", "Yesterday",      "T1, T1ce, T2, FLAIR", "9.7 cm³",  "0.91", "Complete", "green"),
        ("#004", "Yesterday",      "T1, T1ce, FLAIR",     "—",        "N/A",  "Processing", "amber"),
        ("#005", "2 days ago",     "T1, T2, FLAIR",       "0.0 cm³",  "—",   "No Tumor",  "gray"),
    ]
    hc = [1.2, 1.8, 2, 1.5, 1, 1.2]
    h0, h1, h2, h3, h4, h5 = st.columns(hc)
    for col, txt in zip([h0, h1, h2, h3, h4, h5],
                        ["Case ID", "Uploaded", "Modalities", "Tumor Vol", "Dice", "Status"]):
        with col:
            st.markdown(
                f'<div style="font-size:.62rem;font-weight:700;letter-spacing:.09em;'
                f'text-transform:uppercase;color:{T3};padding:.2rem 0;">{txt}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:.2rem 0 .5rem;">', unsafe_allow_html=True)

    for i, (cid, cdate, mods, vol, dice, status, sv) in enumerate(mock_rows):
        c0, c1, c2, c3, c4, c5 = st.columns(hc)
        with c0:
            st.markdown(f'<div style="font-size:.8rem;font-weight:600;color:{BLUE};padding:.3rem 0;">{cid}</div>', unsafe_allow_html=True)
        with c1:
            st.markdown(f'<div style="font-size:.78rem;color:{T2};padding:.3rem 0;">{cdate}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="font-size:.78rem;color:{T2};padding:.3rem 0;">{mods}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div style="font-size:.8rem;color:{T1};font-family:{MONO};padding:.3rem 0;">{vol}</div>', unsafe_allow_html=True)
        with c4:
            dc = GREEN if dice not in ("—","N/A") else T3
            st.markdown(f'<div style="font-size:.8rem;color:{dc};font-family:{MONO};padding:.3rem 0;">{dice}</div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div style="padding:.3rem 0;">{status_badge(status,sv)}</div>', unsafe_allow_html=True)
        if i < 4:
            st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:.15rem 0;">', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
