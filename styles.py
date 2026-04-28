"""
styles.py — NeuroSeg Premium Dark Theme
Complete CSS + HTML component library matching the Dribbble design.
"""

import streamlit as st

# ── Design tokens ──────────────────────────────────────────────────────────────
BG         = "#080d1a"
SURFACE_1  = "#0f1623"
SURFACE_2  = "#141e2e"
SURFACE_3  = "#1a2640"
BORDER     = "#1e2d45"
BORDER_2   = "#243450"

BLUE       = "#3b82f6"
BLUE_DIM   = "rgba(59,130,246,.14)"
BLUE_GLOW  = "rgba(59,130,246,.22)"
CYAN       = "#06b6d4"
GREEN      = "#10b981"
GREEN_DIM  = "rgba(16,185,129,.14)"
AMBER      = "#f59e0b"
AMBER_DIM  = "rgba(245,158,11,.14)"
RED        = "#ef4444"
RED_DIM    = "rgba(239,68,68,.14)"
PURPLE     = "#8b5cf6"
PURPLE_DIM = "rgba(139,92,246,.14)"

T1   = "#f1f5f9"
T2   = "#94a3b8"
T3   = "#475569"
FONT = "'Inter','IBM Plex Sans','Segoe UI',system-ui,sans-serif"
MONO = "'JetBrains Mono','IBM Plex Mono','SF Mono',monospace"

GLOBAL_CSS = """

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,.block-container{
  background:#080d1a!important;color:#f1f5f9!important;
  font-family:'Inter','IBM Plex Sans','Segoe UI',system-ui,sans-serif!important}

.block-container{padding:1.25rem 1.5rem 2rem!important;max-width:1440px!important}

[data-testid="stSidebar"]{
  background:#0f1623!important;border-right:1px solid #1e2d45!important;min-width:228px!important}
[data-testid="stSidebar"] *{font-family:'Inter','IBM Plex Sans',system-ui,sans-serif!important}

[data-testid="stSidebarNav"] a{
  color:#94a3b8!important;font-size:.84rem!important;font-weight:500!important;
  padding:.42rem .75rem!important;border-radius:7px!important;display:flex!important;
  align-items:center!important;gap:8px!important;text-decoration:none!important;
  transition:all .15s!important;margin:1px 0!important}
[data-testid="stSidebarNav"] a:hover{background:#1a2640!important;color:#f1f5f9!important}
[data-testid="stSidebarNav"] a[aria-current="page"]{
  background:rgba(59,130,246,.14)!important;color:#3b82f6!important;
  border-left:2px solid #3b82f6!important}

h1,h2,h3,h4,h5,h6{color:#f1f5f9!important;font-family:'Inter',system-ui,sans-serif!important;letter-spacing:-.025em!important}

[data-testid="stMetric"]{
  background:#141e2e!important;border:1px solid #1e2d45!important;
  border-radius:12px!important;padding:1.1rem 1.2rem!important}
[data-testid="stMetric"] label{
  color:#94a3b8!important;font-size:.68rem!important;font-weight:700!important;
  letter-spacing:.09em!important;text-transform:uppercase!important}
[data-testid="stMetricValue"]{
  color:#f1f5f9!important;font-size:1.65rem!important;font-weight:700!important;
  font-family:'JetBrains Mono','IBM Plex Mono',monospace!important;letter-spacing:-.02em!important}

.stButton>button{
  background:#3b82f6!important;color:#fff!important;border:none!important;
  border-radius:8px!important;padding:.5rem 1.4rem!important;font-weight:600!important;
  font-size:.84rem!important;transition:opacity .15s,transform .1s!important;
  box-shadow:0 1px 2px rgba(0,0,0,.4)!important}
.stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important;
  box-shadow:0 4px 12px rgba(59,130,246,.22)!important}
.stButton>button:active{transform:translateY(0)!important}

[data-testid="stDownloadButton"]>button{
  background:#1a2640!important;color:#94a3b8!important;border:1px solid #1e2d45!important}
[data-testid="stDownloadButton"]>button:hover{border-color:#3b82f6!important;color:#3b82f6!important}

.stTextInput input,.stNumberInput input,.stTextArea textarea,[data-baseweb="select"]>div{
  background:#141e2e!important;color:#f1f5f9!important;border:1px solid #1e2d45!important;
  border-radius:8px!important;font-family:'Inter',system-ui,sans-serif!important;font-size:.84rem!important}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{
  border-color:#3b82f6!important;box-shadow:0 0 0 3px rgba(59,130,246,.14)!important;outline:none!important}
.stTextInput label,.stTextArea label,.stSelectbox label,.stNumberInput label{
  color:#94a3b8!important;font-size:.68rem!important;font-weight:700!important;
  letter-spacing:.07em!important;text-transform:uppercase!important}

[data-testid="stFileUploader"]{
  background:#141e2e!important;border:2px dashed #243450!important;
  border-radius:12px!important;padding:1rem!important;transition:border-color .2s!important}
[data-testid="stFileUploader"]:hover{border-color:#3b82f6!important}
[data-testid="stFileUploader"] label,[data-testid="stFileUploadDropzone"] span{color:#94a3b8!important}

[data-testid="stSlider"] label{
  color:#94a3b8!important;font-size:.68rem!important;font-weight:700!important;
  letter-spacing:.07em!important;text-transform:uppercase!important}
[data-testid="stSlider"]>div>div>div>div{background:#3b82f6!important}
[data-testid="stSlider"]>div>div>div{background:#243450!important}

.stProgress>div>div>div>div{
  background:linear-gradient(90deg,#3b82f6 0%,#06b6d4 100%)!important;border-radius:4px!important}
.stProgress>div>div>div{background:#1a2640!important;border-radius:4px!important}

[data-testid="stAlert"]{border-radius:10px!important;font-size:.84rem!important}

[data-testid="stDataFrame"]{border:1px solid #1e2d45!important;border-radius:12px!important;overflow:hidden!important}

[data-baseweb="tab-list"]{
  background:#141e2e!important;border-radius:10px!important;
  padding:4px!important;gap:2px!important;border:1px solid #1e2d45!important}
[data-baseweb="tab"]{
  background:transparent!important;color:#94a3b8!important;
  border-radius:7px!important;font-size:.82rem!important;font-weight:500!important;
  padding:.42rem 1rem!important;border:none!important}
[aria-selected="true"][data-baseweb="tab"]{background:#1a2640!important;color:#f1f5f9!important}

[data-testid="stExpander"]{
  background:#141e2e!important;border:1px solid #1e2d45!important;border-radius:10px!important}
[data-testid="stExpander"] summary{
  color:#f1f5f9!important;font-size:.84rem!important;font-weight:600!important;padding:.75rem 1rem!important}

code,pre{
  background:#141e2e!important;color:#10b981!important;border-radius:6px!important;
  font-family:'JetBrains Mono',monospace!important;font-size:.78rem!important;border:1px solid #1e2d45!important}

::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#080d1a}
::-webkit-scrollbar-thumb{background:#243450;border-radius:10px}

hr{border:none!important;border-top:1px solid #1e2d45!important;margin:.8rem 0!important}

.stCaption,[data-testid="stCaptionContainer"]{color:#475569!important;font-size:.75rem!important}

[data-testid="stPageLink"] a{
  color:#94a3b8!important;font-weight:500!important;font-size:.84rem!important;
  text-decoration:none!important;padding:.4rem .75rem!important;
  display:block!important;border-radius:7px!important;transition:background .15s!important}
[data-testid="stPageLink"] a:hover{background:#1a2640!important;color:#f1f5f9!important}

[data-testid="stCheckbox"] label{color:#94a3b8!important;font-size:.84rem!important}
.stRadio label{color:#94a3b8!important;font-size:.84rem!important}

[data-testid="stToolbar"]{display:none!important}
header[data-testid="stHeader"]{background:transparent!important}

[data-testid="column"]{padding:0 .35rem!important}
[data-testid="stSidebarNav"] {
    display: none !important;
}
</style>
"""


def inject_css() -> None:
    """Inject global dark-theme CSS. Must be called AFTER st.set_page_config."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Component helpers ──────────────────────────────────────────────────────────

def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def metric_card(label: str, value: str, delta: str = "",
                accent: str = BLUE, icon_svg: str = "") -> str:
    delta_html = ""
    if delta:
        col = GREEN if delta.startswith("+") else (RED if delta.startswith("-") else T2)
        delta_html = (f'<div style="font-size:.72rem;font-weight:600;color:{col};'
                      f'margin-top:.35rem;">{delta}</div>')
    icon_html = ""
    if icon_svg:
        icon_html = (
            f'<div style="width:36px;height:36px;border-radius:9px;'
            f'background:rgba({_hex_rgb(accent)},.14);display:flex;'
            f'align-items:center;justify-content:center;flex-shrink:0;">'
            f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
            f'stroke="{accent}" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{icon_svg}</svg></div>'
        )
    return (
        f'<div style="background:{SURFACE_2};border:1px solid {BORDER};'
        f'border-radius:14px;padding:1.2rem;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;right:0;width:80px;height:80px;'
        f'background:radial-gradient(circle,rgba({_hex_rgb(accent)},.07) 0%,transparent 70%);'
        f'pointer-events:none;"></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:flex-start;margin-bottom:.8rem;">'
        f'<span style="font-size:.68rem;font-weight:700;letter-spacing:.09em;'
        f'text-transform:uppercase;color:{T2};">{label}</span>'
        f'{icon_html}</div>'
        f'<div style="font-size:1.8rem;font-weight:700;color:{T1};'
        f'font-family:{MONO};letter-spacing:-.02em;line-height:1.1;">{value}</div>'
        f'{delta_html}</div>'
    )


def status_badge(text: str, variant: str = "blue") -> str:
    c = {
        "blue": (BLUE, BLUE_DIM), "green": (GREEN, GREEN_DIM),
        "amber": (AMBER, AMBER_DIM), "red": (RED, RED_DIM),
        "gray": (T3, SURFACE_3), "purple": (PURPLE, PURPLE_DIM),
    }.get(variant, (BLUE, BLUE_DIM))
    return (
        f'<span style="background:{c[1]};color:{c[0]};border:1px solid {c[0]};'
        f'border-radius:20px;padding:2px 9px;font-size:.68rem;font-weight:700;'
        f'letter-spacing:.04em;white-space:nowrap;">{text}</span>'
    )


def section_title(text: str) -> None:
    st.markdown(
        f'<p style="font-size:.68rem;font-weight:700;letter-spacing:.1em;'
        f'text-transform:uppercase;color:{T3};margin:1rem 0 .6rem 0;">{text}</p>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {BORDER};margin:.8rem 0;">',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", breadcrumb: str = "") -> None:
    bc = (f'<div style="font-size:.68rem;color:{T3};margin-bottom:.4rem;'
          f'letter-spacing:.04em;">Dashboard &nbsp;›&nbsp; {breadcrumb}</div>'
          if breadcrumb else "")
    st.markdown(
        f'<div style="padding:0 0 1.1rem 0;border-bottom:1px solid {BORDER};'
        f'margin-bottom:1.2rem;">{bc}'
        f'<h1 style="font-size:1.55rem;font-weight:700;color:{T1};'
        f'letter-spacing:-.03em;margin:0 0 .2rem 0;">{title}</h1>'
        f'<p style="font-size:.84rem;color:{T2};margin:0;">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def sidebar_logo() -> None:
    st.sidebar.markdown(
        f'<div style="padding:1rem .75rem .8rem;border-bottom:1px solid {BORDER};'
        f'margin-bottom:.4rem;display:flex;align-items:center;gap:10px;">'
        f'<div style="width:32px;height:32px;border-radius:9px;flex-shrink:0;'
        f'background:linear-gradient(135deg,{BLUE} 0%,{CYAN} 100%);'
        f'display:flex;align-items:center;justify-content:center;">'
        f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" '
        f'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8'
        f'a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg></div>'
        f'<div><div style="font-size:.9rem;font-weight:700;color:{T1};">NeuroSeg</div>'
        f'<div style="font-size:.62rem;color:{T3};">Brain Tumor Analysis</div></div></div>',
        unsafe_allow_html=True,
    )


def sidebar_nav_label(text: str) -> None:
    st.sidebar.markdown(
        f'<p style="font-size:.6rem;font-weight:700;letter-spacing:.12em;'
        f'text-transform:uppercase;color:{T3};padding:.5rem .75rem .15rem;">{text}</p>',
        unsafe_allow_html=True,
    )


def sidebar_status(mri_loaded: bool, seg_done: bool) -> None:
    def dot(ok: bool) -> str:
        return (f'<span style="width:6px;height:6px;border-radius:50%;'
                f'background:{"#10b981" if ok else "#475569"};display:inline-block;'
                f'margin-right:6px;flex-shrink:0;"></span>')
    st.sidebar.markdown(
        f'<div style="margin:.3rem .75rem .5rem;padding:.55rem .75rem;'
        f'background:{SURFACE_3};border-radius:8px;border:1px solid {BORDER};">'
        f'<div style="display:flex;align-items:center;font-size:.73rem;'
        f'color:{T2};margin-bottom:4px;">{dot(mri_loaded)} MRI '
        f'{"Loaded" if mri_loaded else "Not Loaded"}</div>'
        f'<div style="display:flex;align-items:center;font-size:.73rem;color:{T2};">'
        f'{dot(seg_done)} Segmentation {"Complete" if seg_done else "Pending"}</div></div>',
        unsafe_allow_html=True,
    )
# 🔧 Backward compatibility aliases

def inject_global_css():
    """Alias for compatibility with main.py"""
    inject_css()

# Color aliases
DARK_BG = BG

ACCENT_BLUE = BLUE
ACCENT_GREEN = GREEN
TEXT_PRIMARY = T1
TEXT_DIM = T2
TEXT_BRIGHT = T1

FONT_BODY = FONT
FONT_MONO = MONO

ACCENT_AMBER = AMBER
ACCENT_RED = RED

def plotly_layout(title: str = "", height: int = 300) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=T1, size=12, family=FONT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T2, family=FONT, size=10),
        height=height,
        margin=dict(l=8, r=8, t=28 if title else 8, b=8),
        xaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False,
                   linecolor=BORDER, tickfont=dict(color=T3, size=9)),
        yaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False,
                   linecolor=BORDER, tickfont=dict(color=T3, size=9)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
                    font=dict(color=T2, size=9)),
        hoverlabel=dict(bgcolor=SURFACE_3, bordercolor=BORDER,
                        font=dict(color=T1, family=FONT, size=10)),
    )
