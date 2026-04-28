"""
Generate Report — NeuroSeg
PDF-style report UI with database integration and email sending.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import tempfile
import os

st.set_page_config(
    page_title="Generate Report — NeuroSeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(str(Path(__file__).parent.parent / "src"))

from styles import (
    inject_css, sidebar_logo, sidebar_nav_label, sidebar_status,
    metric_card, section_title, divider, page_header, status_badge,
    BLUE, GREEN, AMBER, RED, SURFACE_2, SURFACE_3, BORDER, T1, T2, T3, MONO,
)
from email_handler import build_email_body, create_email_handler
from model_loader import get_model_info
from database import init_db, save_report

inject_css()
init_db()

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
    "Generate Report",
    "Clinical-style segmentation report with PDF export and email delivery",
    breadcrumb="Generate Report",
)

if st.session_state.get("segmentation_result") is None:
    st.error("No segmentation results found. Please run segmentation first.")
    if st.button("Go to Run Segmentation"):
        st.switch_page("pages/3_Run_Segmentation.py")
    st.stop()

metrics    = st.session_state.get("metrics", {}) or {}
infer_time = st.session_state.get("segmentation_time", 0) or 0
model_type = st.session_state.get("model_type", "3D U-Net")

# ── Summary strip ─────────────────────────────────────────────────────────────
section_title("Analysis Summary")

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(
        metric_card("Total Volume", f"{metrics.get('total_tumor_volume_cm3',0):.3f} cm³",
                    "Whole tumor", RED,
                    '<circle cx="12" cy="12" r="10"/>'
                    '<line x1="12" y1="8" x2="12" y2="12"/>'
                    '<line x1="12" y1="16" x2="12.01" y2="16"/>'),
        unsafe_allow_html=True)
with s2:
    st.markdown(
        metric_card("Coverage", f"{metrics.get('tumor_percentage',0):.2f}%",
                    "% of scan volume", AMBER,
                    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
        unsafe_allow_html=True)
with s3:
    st.markdown(
        metric_card("Inference", f"{infer_time:.2f} s",
                    "Model execution time", BLUE,
                    '<circle cx="12" cy="12" r="10"/>'
                    '<polyline points="12 6 12 12 16 14"/>'),
        unsafe_allow_html=True)
with s4:
    st.markdown(
        metric_card("Model", model_type or "3D U-Net",
                    "Architecture", GREEN,
                    '<path d="M12 2L2 7l10 5 10-5-10-5z"/>'
                    '<path d="M2 17l10 5 10-5"/>'
                    '<path d="M2 12l10 5 10-5"/>'),
        unsafe_allow_html=True)

divider()

# ── Patient form ──────────────────────────────────────────────────────────────
section_title("Patient & Report Information")

form_l, form_r = st.columns(2)

with form_l:
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:12px;padding:1.2rem 1.2rem .8rem;">'
        f'<div style="font-size:.78rem;font-weight:700;color:{T1};margin-bottom:1rem;">'
        f'Patient Details</div>',
        unsafe_allow_html=True,
    )
    patient_id  = st.text_input("Patient ID",          value="PATIENT_001")
    physician   = st.text_input("Reporting Physician",  value="Dr. Smith")
    institution = st.text_input("Institution",          value="Medical Center")
    st.markdown("</div>", unsafe_allow_html=True)

with form_r:
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:12px;padding:1.2rem 1.2rem .8rem;">'
        f'<div style="font-size:.78rem;font-weight:700;color:{T1};margin-bottom:1rem;">'
        f'Report Options</div>',
        unsafe_allow_html=True,
    )
    inc_metrics   = st.checkbox("Include volumetric metrics",  value=True)
    inc_breakdown = st.checkbox("Include class breakdown",     value=True)
    inc_disc      = st.checkbox("Include medical disclaimer",  value=True)
    save_db       = st.checkbox("Save to history database",    value=True)
    st.markdown("</div>", unsafe_allow_html=True)

divider()


# ── Report builder ────────────────────────────────────────────────────────────
def build_report() -> str:
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vol      = metrics.get("total_tumor_volume_cm3", 0)
    pct      = metrics.get("tumor_percentage", 0)
    surf     = metrics.get("surface_area_mm2", 0)
    nc       = metrics.get("volume_class_1_cm3", 0)
    ed       = metrics.get("volume_class_2_cm3", 0)
    en       = metrics.get("volume_class_3_cm3", 0)
    detected = "POSITIVE — Tumour regions identified" if vol > 0 else "NEGATIVE — No significant tumour regions"
    info     = get_model_info()

    lines = [
        "=" * 68,
        "       BRAIN TUMOUR SEGMENTATION — CLINICAL REPORT",
        "=" * 68, "",
        "REPORT METADATA",
        "-" * 68,
        f"  Report Date         : {now}",
        f"  Patient ID          : {patient_id}",
        f"  Institution         : {institution}",
        f"  Reporting Physician : {physician}",
        f"  AI Model            : {model_type}", "",
        "DETECTION RESULT",
        "-" * 68,
        f"  Status              : {detected}", "",
    ]
    if inc_metrics:
        lines += [
            "VOLUMETRIC ANALYSIS",
            "-" * 68,
            f"  Total Tumour Volume      : {vol:.3f} cm³",
            f"  Tumour Coverage          : {pct:.3f} % of scan volume",
            f"  Estimated Surface Area   : {surf:.1f} mm²",
            f"  Inference Time           : {infer_time:.2f} s", "",
        ]
    if inc_breakdown:
        lines += [
            "CLASS-LEVEL BREAKDOWN",
            "-" * 68,
            f"  Class 1 — Necrotic Core    : {nc:.3f} cm³",
            f"  Class 2 — Edema            : {ed:.3f} cm³",
            f"  Class 3 — Enhancing Tumour : {en:.3f} cm³", "",
        ]
    lines += [
        "MODEL DETAILS",
        "-" * 68,
        f"  Architecture   : {info.get('architecture')}",
        f"  Framework      : {info.get('framework')}",
        f"  Input Channels : {info.get('input_channels')}  (T1, T1ce, T2, FLAIR)",
        f"  Output Classes : {info.get('output_classes')}",
        f"  Loss Function  : {info.get('loss_function')}",
        f"  Training Data  : {info.get('training_dataset')}", "",
        "CONCLUSION",
        "-" * 68,
        "  The AI-assisted segmentation pipeline has processed the provided",
        "  MRI data and classified voxels into tumour sub-regions per the",
        "  BraTS labelling convention. Results are intended for research", "",
    ]
    if inc_disc:
        lines += [
            "=" * 68,
            "DISCLAIMER",
            "=" * 68,
            "This report is generated by an AI system for research purposes only.",
            "It does NOT constitute a medical diagnosis. All findings MUST be",
            "reviewed by a qualified radiologist before any clinical use.",
            "=" * 68,
        ]
    return "\n".join(lines)


# ── Preview ───────────────────────────────────────────────────────────────────
section_title("Report Preview")

report_text = build_report()

st.markdown(
    f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
    f'border-radius:12px;padding:1.4rem;max-height:440px;overflow-y:auto;">'
    f'<pre style="font-family:{MONO};font-size:.76rem;color:{T2};'
    f'white-space:pre-wrap;margin:0;line-height:1.75;">{report_text}</pre></div>',
    unsafe_allow_html=True,
)

divider()

# ── Download + DB save ────────────────────────────────────────────────────────
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"BrainTumor_Report_{patient_id}_{ts}.txt"

act1, act2, _ = st.columns([1, 1, 2])

with act1:
    st.download_button(
        "Download Report (.txt)",
        data=report_text,
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
    )

with act2:
    if st.button("Save to Database", use_container_width=True):
        if save_db:
            try:
                rid = save_report(
                    patient_id       = patient_id,
                    tumor_volume     = metrics.get("total_tumor_volume_cm3", 0),
                    tumor_percentage = metrics.get("tumor_percentage", 0),
                    inference_time   = infer_time,
                    surface_area     = metrics.get("surface_area_mm2", 0),
                    necrotic_vol     = metrics.get("volume_class_1_cm3", 0),
                    edema_vol        = metrics.get("volume_class_2_cm3", 0),
                    enhancing_vol    = metrics.get("volume_class_3_cm3", 0),
                    model_type       = model_type or "3D U-Net",
                    tumor_detected   = metrics.get("total_tumor_volume_cm3", 0) > 0,
                    report_path      = filename,
                    notes            = f"By {physician} at {institution}",
                )
                st.success(f"Saved to database — Record ID: {rid}")
            except Exception as e:
                st.error(f"Database error: {e}")
        else:
            st.info("Enable 'Save to history database' to persist this report.")

divider()

# ── Email ─────────────────────────────────────────────────────────────────────
section_title("Send via Email")

has_email = bool(os.getenv("SENDER_EMAIL")) and bool(os.getenv("SENDER_PASSWORD"))

if has_email:
    st.markdown(
        f'<div style="background:rgba(16,185,129,.08);border:1px solid {GREEN};'
        f'border-radius:8px;padding:.65rem 1rem;font-size:.8rem;color:{GREEN};">'
        f'Email configured: {os.getenv("SENDER_EMAIL")}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="background:rgba(245,158,11,.08);border:1px solid {AMBER};'
        f'border-radius:8px;padding:.65rem 1rem;font-size:.8rem;color:{AMBER};">'
        f'Email not configured. Add SENDER_EMAIL and SENDER_PASSWORD (Gmail App Password) '
        f'to your .env file.</div>',
        unsafe_allow_html=True,
    )

em_col1, em_col2 = st.columns([3, 1])
with em_col1:
    recipient = st.text_input("Recipient address", placeholder="doctor@hospital.com",
                               label_visibility="collapsed")
with em_col2:
    send_btn = st.button("Send Email", use_container_width=True, disabled=not has_email)

if send_btn:
    if not recipient or "@" not in recipient:
        st.error("Enter a valid email address.")
    else:
        with st.spinner("Sending…"):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
                tmp.write(report_text)
                tmp_path = tmp.name
            handler    = create_email_handler()
            email_body = build_email_body(
                patient_id=patient_id, institution=institution,
                physician=physician, metrics=metrics,
                inference_time=infer_time, dice_score=None,
            )
            ok, msg = handler.send_report_email(
                recipient_email=recipient, body_text=email_body,
                subject="Brain Tumor Segmentation Report — NeuroSeg",
                attachment_path=tmp_path, patient_id=patient_id,
            )
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if ok:
            st.success(msg)
        else:
            st.error(msg)
