"""
History — NeuroSeg
Persistent SQLite records + in-session history.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

st.set_page_config(
    page_title="History — NeuroSeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(str(Path(__file__).parent.parent / "src"))

from styles import (
    inject_css, sidebar_logo, sidebar_nav_label, sidebar_status,
    metric_card, section_title, divider, page_header, status_badge,
    BLUE, GREEN, AMBER, RED, SURFACE_2, SURFACE_3, BORDER, T1, T2, T3, MONO,
)
from database import init_db, get_reports, get_stats, delete_report

inject_css()
init_db()

if "history" not in st.session_state:
    st.session_state.history = []

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
    "History",
    "Persistent scan records and cross-session statistics",
    breadcrumb="History",
)

db_stats = get_stats()
db_rows  = get_reports()

# ── DB stats strip ────────────────────────────────────────────────────────────
section_title("Database Overview")

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(
        metric_card("Total Reports", str(db_stats["total_reports"]),
                    "Across all sessions", BLUE,
                    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                    '<polyline points="14 2 14 8 20 8"/>'),
        unsafe_allow_html=True)
with s2:
    st.markdown(
        metric_card("Tumors Detected", str(db_stats["tumor_detected_count"]),
                    "Positive findings", RED,
                    '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'),
        unsafe_allow_html=True)
with s3:
    st.markdown(
        metric_card("Avg Volume", f"{db_stats['avg_tumor_volume']:.3f} cm³",
                    "Mean tumor volume", AMBER,
                    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
        unsafe_allow_html=True)
with s4:
    st.markdown(
        metric_card("Avg Inference", f"{db_stats['avg_inference_time']:.2f}s",
                    "Mean processing time", GREEN,
                    '<circle cx="12" cy="12" r="10"/>'
                    '<polyline points="12 6 12 12 16 14"/>'),
        unsafe_allow_html=True)

divider()

# ── Records table ─────────────────────────────────────────────────────────────
section_title("Persistent Records")

if db_rows:
    disp = []
    for r in db_rows:
        disp.append({
            "ID":           r["id"],
            "Patient":      r["patient_id"],
            "Date":         r["created_at"],
            "Detected":     "Yes" if r["tumor_detected"] else "No",
            "Volume (cm³)": f"{r['tumor_volume']:.3f}",
            "Coverage (%)": f"{r['tumor_percentage']:.2f}",
            "Inference (s)":f"{r['inference_time']:.2f}",
            "Model":        r["model_type"],
        })
    df = pd.DataFrame(disp)
    st.dataframe(df, use_container_width=True, hide_index=True)

    divider()

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.download_button(
            "Export as CSV",
            data=df.to_csv(index=False),
            file_name=f"neuroseg_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True,
        )
    with ex2:
        st.download_button(
            "Export as JSON",
            data=json.dumps(db_rows, indent=2, default=str),
            file_name=f"neuroseg_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json", use_container_width=True,
        )
    with ex3:
        with st.expander("Delete a Record"):
            del_id = st.number_input("Record ID", min_value=1, step=1,
                                     label_visibility="collapsed")
            if st.button("Delete", use_container_width=True):
                if delete_report(int(del_id)):
                    st.success(f"Record {del_id} deleted.")
                    st.rerun()
                else:
                    st.error(f"Record {del_id} not found.")

    # Detail view
    divider()
    section_title("Record Detail")
    detail_id = st.selectbox(
        "Select ID",
        options=[r["id"] for r in db_rows],
        format_func=lambda x: f"#{x} — {next((r['patient_id'] for r in db_rows if r['id']==x),'')}",
        label_visibility="collapsed",
    )
    chosen = next((r for r in db_rows if r["id"] == detail_id), None)
    if chosen:
        d1, d2, d3 = st.columns(3)
        for col_d, (lbl, key, color) in zip(
            [d1, d2, d3],
            [("Necrotic Core","necrotic_vol",RED),
             ("Edema","edema_vol",AMBER),
             ("Enhancing","enhancing_vol",BLUE)],
        ):
            with col_d:
                st.markdown(
                    f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
                    f'border-top:2px solid {color};border-radius:10px;padding:1rem;">'
                    f'<div style="font-size:.65rem;font-weight:700;letter-spacing:.08em;'
                    f'text-transform:uppercase;color:{T2};margin-bottom:.5rem;">{lbl}</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;color:{T1};'
                    f'font-family:{MONO};">{chosen.get(key,0):.3f} cm³</div></div>',
                    unsafe_allow_html=True,
                )
        if chosen.get("notes"):
            st.markdown(
                f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
                f'border-radius:8px;padding:.75rem 1rem;font-size:.8rem;color:{T2};'
                f'margin-top:.6rem;"><strong style="color:{T1};">Notes:</strong> '
                f'{chosen["notes"]}</div>',
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:12px;padding:2.5rem;text-align:center;">'
        f'<div style="font-size:.85rem;color:{T3};">'
        f'No database records yet. Generate and save a report to build history.</div></div>',
        unsafe_allow_html=True,
    )

divider()

# ── Session history ───────────────────────────────────────────────────────────
section_title("Current Session (In-Memory)")

# Auto-append
if (st.session_state.get("segmentation_result") is not None
        and st.session_state.get("mri_data") is not None):
    entry = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tumor_volume":   (st.session_state.metrics or {}).get("total_tumor_volume_cm3", 0),
        "tumor_detected": (st.session_state.metrics or {}).get("total_tumor_volume_cm3", 0) > 0,
        "processing_time": st.session_state.get("segmentation_time", 0),
    }
    hist = st.session_state.history
    if not hist or hist[-1].get("timestamp") != entry["timestamp"]:
        st.session_state.history.append(entry)

hist = st.session_state.history
if hist:
    h_disp = []
    for i, e in enumerate(hist, 1):
        h_disp.append({
            "#":         i,
            "Date":      e["timestamp"],
            "Detected":  "Yes" if e["tumor_detected"] else "No",
            "Volume":    f"{e['tumor_volume']:.3f} cm³",
            "Time (s)":  f"{e['processing_time']:.2f}",
        })
    st.dataframe(pd.DataFrame(h_disp), use_container_width=True, hide_index=True)
    if st.button("Clear Session History"):
        st.session_state.history = []
        st.rerun()
else:
    st.markdown(
        f'<div style="color:{T3};font-size:.82rem;padding:.4rem 0;">'
        f'No in-session entries yet.</div>',
        unsafe_allow_html=True,
    )
