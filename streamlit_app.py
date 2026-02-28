"""
streamlit_app.py — Aegis-Omni ICU Command Centre
Matches the reference UI: dark navy, ECG waveforms, composite risk chart,
clinical event stream, multi-panel vitals, fusion weights.
"""

import os, json, time, random, math
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Aegis-Omni | ICU Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Orbitron:wght@400;600;700;900&family=Exo+2:wght@200;300;400;600;700&display=swap');

:root {
  --bg0: #040d18;
  --bg1: #071428;
  --bg2: #0a1a32;
  --bg3: #0e2040;
  --bg4: #122548;
  --border: #112236;
  --border2: #1a3550;
  --cyan:  #00f5ff;
  --cyan2: #00bcd4;
  --green: #00ff88;
  --red:   #ff2244;
  --amber: #ffb300;
  --pink:  #ff4499;
  --text:  #8ab8d4;
  --text2: #4a7a99;
  --text3: #1e4060;
  --mono: 'JetBrains Mono', monospace;
  --display: 'Orbitron', monospace;
  --body: 'Exo 2', sans-serif;
}

html, body, [class*="css"], .stApp {
  background-color: var(--bg0) !important;
  color: var(--text) !important;
  font-family: var(--body) !important;
}
.stApp { background: var(--bg0) !important; }

/* scanlines */
.stApp::after {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:9998;
  background: repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.04) 3px,rgba(0,0,0,0.04) 4px);
}

section[data-testid="stSidebar"] { display:none !important; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:0 !important; max-width:100% !important; }

/* ── NAV BAR ── */
.nav-bar {
  display:flex; align-items:center; justify-content:space-between;
  background: linear-gradient(90deg,#040d18 0%,#071428 60%,#040d18 100%);
  border-bottom: 1px solid var(--border2);
  padding: 0 1.5rem; height: 48px;
  position: sticky; top:0; z-index:100;
}
.nav-logo {
  font-family:var(--display); font-size:1rem; font-weight:700;
  color:var(--cyan); letter-spacing:4px;
  text-shadow:0 0 20px rgba(0,245,255,0.4);
  display:flex; align-items:center; gap:8px;
}
.nav-status {
  display:flex; align-items:center; gap:6px;
  font-family:var(--mono); font-size:0.62rem; color:var(--text2);
}
.dot-live { width:7px;height:7px;background:var(--green);border-radius:50%;
            box-shadow:0 0 8px var(--green); animation:blink 2s infinite; }
.dot-off  { width:7px;height:7px;background:var(--red);border-radius:50%; }
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
.nav-tabs { display:flex; gap:0; }
.nav-tab {
  font-family:var(--mono); font-size:0.65rem; letter-spacing:2px;
  text-transform:uppercase; padding:0 1.2rem; height:48px;
  display:flex; align-items:center; cursor:pointer;
  color:var(--text2); border-bottom:2px solid transparent;
  transition:all 0.2s;
}
.nav-tab.active { color:var(--cyan); border-bottom-color:var(--cyan); }
.nav-time { font-family:var(--mono); font-size:0.72rem; color:var(--text2); }

/* ── PATIENT HEADER ── */
.pt-header {
  background: linear-gradient(135deg,var(--bg1) 0%,var(--bg2) 100%);
  border-bottom:1px solid var(--border);
  padding:0.8rem 1.5rem;
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
}
.pt-avatar {
  width:44px;height:44px;border-radius:50%;
  background:var(--bg3);border:1px solid var(--border2);
  display:flex;align-items:center;justify-content:center;
  font-family:var(--display);font-size:1rem;color:var(--cyan2);
}
.pt-name { font-family:var(--display);font-size:1.1rem;font-weight:700;
           color:#e8f4ff;letter-spacing:2px;text-transform:uppercase; }
.pt-meta { font-family:var(--mono);font-size:0.6rem;color:var(--text2);margin-top:3px; }
.pt-admit { font-family:var(--mono);font-size:0.58rem;color:var(--text3);margin-top:2px; }

/* Risk badge */
.risk-badge {
  text-align:center; padding:0.5rem 1.5rem;
  border:1px solid; border-radius:6px; min-width:140px;
  position:relative; overflow:hidden;
}
.risk-badge::before {
  content:''; position:absolute; inset:0;
  background:inherit; opacity:0.08; border-radius:6px;
}
.risk-pct { font-family:var(--display);font-size:2.8rem;font-weight:900;line-height:1; }
.risk-lbl { font-family:var(--mono);font-size:0.55rem;letter-spacing:3px;
            text-transform:uppercase;margin-top:2px; }

/* ── PANEL ── */
.panel {
  background:var(--bg1);border:1px solid var(--border);
  border-radius:4px; overflow:hidden;
}
.panel-hdr {
  background:var(--bg2); border-bottom:1px solid var(--border);
  padding:0.35rem 0.75rem;
  display:flex; align-items:center; justify-content:space-between;
}
.panel-title {
  font-family:var(--mono);font-size:0.58rem;color:var(--text2);
  letter-spacing:3px;text-transform:uppercase;
}
.panel-badge {
  font-family:var(--mono);font-size:0.55rem;
  padding:1px 6px;border-radius:2px;
  background:rgba(0,245,255,0.08);color:var(--cyan2);
}
.panel-body { padding:0.6rem 0.75rem; }

/* ── VITAL CARD ── */
.vital-card {
  background:var(--bg2);border:1px solid var(--border);border-radius:4px;
  padding:0.6rem 0.7rem; text-align:left; position:relative; overflow:hidden;
}
.vital-card::after {
  content:''; position:absolute; bottom:0;left:0;right:0;height:2px;
}
.vital-card.ok::after  { background:var(--cyan2); }
.vital-card.warn::after{ background:var(--amber); }
.vital-card.crit::after{ background:var(--red); animation:blink 1s infinite; }
.vcard-label { font-family:var(--mono);font-size:0.55rem;color:var(--text2);
               letter-spacing:2px;text-transform:uppercase;margin-bottom:3px; }
.vcard-val { font-family:var(--display);font-size:1.5rem;font-weight:700; }
.vcard-unit { font-family:var(--mono);font-size:0.6rem;color:var(--text2);margin-left:3px; }
.vcard-trend { font-family:var(--mono);font-size:0.58rem;margin-top:3px; }
.vcard-trend.up   { color:var(--red); }
.vcard-trend.down { color:var(--cyan); }
.vcard-trend.ok   { color:var(--green); }
.vcard-ref { font-family:var(--mono);font-size:0.52rem;color:var(--text3); }
.ok   { color:var(--cyan2); }
.warn { color:var(--amber); }
.crit { color:var(--red); }

/* ── EVENT STREAM ── */
.event-item {
  display:flex;align-items:flex-start;gap:0.5rem;
  padding:0.4rem 0.6rem;
  border-bottom:1px solid var(--border);
  font-size:0.75rem;
}
.event-item:last-child { border-bottom:none; }
.event-dot { width:6px;height:6px;border-radius:50%;margin-top:4px;flex-shrink:0; }
.event-time { font-family:var(--mono);font-size:0.58rem;color:var(--text3);
              min-width:52px;margin-top:1px; }
.event-text { font-family:var(--body);font-size:0.74rem;line-height:1.4; }

/* ── LAB ROW ── */
.lab-row {
  display:flex;justify-content:space-between;align-items:baseline;
  padding:0.28rem 0; border-bottom:1px solid var(--border);
  font-family:var(--mono); font-size:0.72rem;
}
.lab-row:last-child { border-bottom:none; }
.lab-name { color:var(--text2); }
.lab-val  { font-weight:700; }
.lab-ref  { font-size:0.58rem;color:var(--text3); }

/* ── FUSION WEIGHT BAR ── */
.fw-row { margin-bottom:0.5rem; }
.fw-label { font-family:var(--mono);font-size:0.62rem;color:var(--text2);
            display:flex;justify-content:space-between;margin-bottom:3px; }
.fw-track { background:var(--bg3);border-radius:2px;height:6px;overflow:hidden; }
.fw-fill  { height:6px;border-radius:2px;transition:width 0.5s ease; }

/* ── NARRATIVE ── */
.narrative {
  background:var(--bg2);border-left:2px solid var(--cyan2);
  border-radius:0 4px 4px 0;padding:0.75rem 0.9rem;
  font-family:var(--body);font-size:0.8rem;line-height:1.7;
  color:var(--text);font-style:italic;
}

/* ── ACTION ROW ── */
.action-row {
  display:flex;align-items:center;gap:0.5rem;
  padding:0.35rem 0;border-bottom:1px solid var(--border);
  font-family:var(--body);font-size:0.78rem;
}
.action-row:last-child { border-bottom:none; }

/* ── THRESHOLD LABEL ── */
.thresh-label {
  font-family:var(--mono);font-size:0.55rem;
  color:var(--text3);letter-spacing:2px;text-transform:uppercase;
  margin-bottom:2px;
}

/* ── PLOTLY override for dark ── */
.js-plotly-plot .plotly { border-radius:4px; }

/* ── Streamlit widget styling ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label {
  font-family:var(--mono) !important;font-size:0.6rem !important;
  color:var(--text2) !important;letter-spacing:2px;text-transform:uppercase;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
  background:var(--bg2) !important;border:1px solid var(--border2) !important;
  color:var(--cyan) !important;border-radius:3px !important;
  font-family:var(--mono) !important;font-size:0.78rem !important;
}
.stButton > button {
  background:transparent !important;color:var(--cyan) !important;
  border:1px solid var(--cyan) !important;border-radius:3px !important;
  font-family:var(--display) !important;font-size:0.8rem !important;
  font-weight:600 !important;letter-spacing:3px !important;
  text-transform:uppercase !important;
}
.stButton > button:hover {
  background:rgba(0,245,255,0.07) !important;
  box-shadow:0 0 20px rgba(0,245,255,0.2) !important;
}
hr { border-color:var(--border) !important;margin:0.5rem 0 !important; }
details { background:var(--bg1) !important;border:1px solid var(--border) !important;border-radius:4px !important; }
summary { font-family:var(--mono) !important;font-size:0.65rem !important;color:var(--text2) !important;letter-spacing:2px; }
::-webkit-scrollbar { width:3px;height:3px; }
::-webkit-scrollbar-track { background:var(--bg0); }
::-webkit-scrollbar-thumb { background:var(--border2);border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Data / presets
# ══════════════════════════════════════════════════════════════════════════════
PATIENTS = {
    "KUMAR, RAJESH": {
        "id":"P001","bed":"ICU-01","age":52,"sex":"Male","weight":"68 kg",
        "dx":"ARDS — Mechanical Ventilation — Day 3",
        "admitted":"25 FEB 2030 09:14",
        "HR":     [96,98,102,105,108,112,115,118,114,110,107,104],
        "MAP":    [72,70,68,65,63,60,58,56,55,54,53,52],
        "Lactate":[1.2,1.4,1.6,1.9,2.1,2.4,2.6,2.8,3.0,3.1,3.2,3.3],
        "SpO2":   [97,97,96,96,95,95,94,94,93,93,92,92],
        "Resp":   [18,18,19,20,22,24,25,26,24,23,22,21],
        "SBP":    [115,112,108,104,100,96,93,90,88,86,85,84],
        "Temp":   [37.0,37.2,37.5,37.8,38.1,38.3,38.4,38.5,38.6,38.5,38.4,38.3],
        "risk_history": [0.22,0.28,0.34,0.41,0.49,0.55,0.61,0.68,0.74,0.79,0.83,0.87],
        "crp":202,"pct":8.9,"il6":388,
        "bun":42,"creatinine":2.7,"lactate_lab":3.3,"glucose":148,
        "wts":{"Vitals":0.82,"Labs":0.71,"Trends":0.65,"Demographics":0.28},
    },
    "HARRISON, J.": {
        "id":"P002","bed":"ICU-02","age":64,"sex":"Male","weight":"82 kg",
        "dx":"Post-cardiac surgery — Day 1",
        "admitted":"25 FEB 2030 06:30",
        "HR":     [72,73,74,74,75,75,74,73,74,73,72,73],
        "MAP":    [88,87,89,88,90,89,88,87,89,88,87,88],
        "Lactate":[0.9,0.9,1.0,1.0,0.9,1.0,1.0,0.9,0.9,1.0,1.0,0.9],
        "SpO2":   [98,98,99,98,98,99,98,98,98,99,98,98],
        "Resp":   [16,16,17,16,16,17,16,16,17,16,16,17],
        "SBP":    [118,120,119,121,120,118,119,120,121,119,120,118],
        "Temp":   [36.8,36.9,36.8,36.9,36.8,36.9,36.8,36.9,36.8,36.9,36.8,36.9],
        "risk_history": [0.12,0.13,0.14,0.13,0.15,0.14,0.15,0.16,0.15,0.14,0.15,0.16],
        "crp":18,"pct":0.4,"il6":22,
        "bun":15,"creatinine":1.1,"lactate_lab":0.9,"glucose":112,
        "wts":{"Vitals":0.55,"Labs":0.30,"Trends":0.20,"Demographics":0.45},
    },
    "VASQUEZ, R.": {
        "id":"P003","bed":"ICU-03","age":38,"sex":"Female","weight":"61 kg",
        "dx":"Sepsis — Pneumonia — Day 2",
        "admitted":"24 FEB 2030 22:10",
        "HR":     [88,90,92,94,96,95,94,96,98,97,96,95],
        "MAP":    [68,67,66,65,66,65,64,64,63,63,62,62],
        "Lactate":[1.8,1.9,2.0,2.1,2.0,2.1,2.2,2.1,2.2,2.3,2.2,2.3],
        "SpO2":   [96,96,95,95,95,95,95,94,94,94,94,93],
        "Resp":   [20,21,21,22,22,22,23,23,22,23,23,24],
        "SBP":    [100,98,97,96,95,95,94,93,92,92,91,90],
        "Temp":   [37.8,37.9,38.0,38.1,38.0,38.1,38.2,38.2,38.1,38.2,38.3,38.3],
        "risk_history": [0.38,0.41,0.44,0.47,0.50,0.52,0.55,0.57,0.59,0.61,0.63,0.65],
        "crp":89,"pct":3.2,"il6":144,
        "bun":28,"creatinine":1.8,"lactate_lab":2.3,"glucose":138,
        "wts":{"Vitals":0.75,"Labs":0.68,"Trends":0.58,"Demographics":0.32},
    },
}

EVENTS_TEMPLATE = {
    "KUMAR, RAJESH": [
        ("#ff2244","08:41","Continuous insulin infusion rate adjusted: 4U units/hr"),
        ("#ff2244","08:36","Procalcitonin repeat critical — result pending #5"),
        ("#ffb300","08:29","MAP dropped to 54 mmHg — Vasopressor dose active"),
        ("#00f5ff","08:21","Chest X-ray ordered — bedside ICU radiograph requested"),
        ("#ff2244","08:14","Heart rate increased to 118 bpm — Tachycardia threshold crossed"),
        ("#ffb300","08:08","Continuous insulin infusion rate adjusted: 4U units/hr"),
    ],
    "HARRISON, J.": [
        ("#00ff88","08:40","Post-op vitals stable — no intervention required"),
        ("#00f5ff","08:30","Morning labs reviewed — within expected post-op range"),
        ("#00ff88","08:15","Chest drain output decreasing — monitoring continues"),
    ],
    "VASQUEZ, R.": [
        ("#ff2244","08:38","MAP trending downward — vasopressor titration initiated"),
        ("#ffb300","08:25","Lactate rising: 2.3 mmol/L — repeat in 2 hours"),
        ("#ffb300","08:10","Respiratory rate increasing — FiO₂ adjusted to 0.55"),
        ("#00f5ff","08:00","Blood cultures ×2 collected — pending results"),
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
#  Session state
# ══════════════════════════════════════════════════════════════════════════════
for k,v in [("sel_patient","KUMAR, RAJESH"),("sel_tab","PATIENT DASHBOARD"),
             ("last_result",None),("alert_log",[])]:
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  API health
# ══════════════════════════════════════════════════════════════════════════════
try:
    h = requests.get(f"{API_URL}/health", timeout=2).json()
    api_ok = h.get("ready", False)
    api_info = f"ICU INTELLIGENCE PLATFORM  ·  v2.4.1"
except:
    api_ok, api_info = False, "ICU INTELLIGENCE PLATFORM  ·  v2.4.1"

dot_color = "#00ff88" if api_ok else "#ff2244"
now_str = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")

# ══════════════════════════════════════════════════════════════════════════════
#  NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
TABS = ["PATIENT DASHBOARD","FEDERATED NETWORK","ALERTS & PREDICTIONS","EXPLAINABILITY"]

st.markdown(f"""
<div class="nav-bar">
  <div class="nav-logo">
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <polygon points="9,1 17,5 17,13 9,17 1,13 1,5" stroke="#00f5ff" stroke-width="1.2" fill="rgba(0,245,255,0.06)"/>
      <polygon points="9,4 14,7 14,11 9,14 4,11 4,7" fill="rgba(0,245,255,0.12)" stroke="#00f5ff" stroke-width="0.6"/>
    </svg>
    AEGIS-OMNI
  </div>
  <div class="nav-status">
    <div class="dot-live"></div>
    {api_info}
  </div>
  <div class="nav-time">DR. PRIYA SHARMA &nbsp;&nbsp; {now_str}</div>
</div>
""", unsafe_allow_html=True)

# Tab row
tab_cols = st.columns(len(TABS) + 4)
for i, tab in enumerate(TABS):
    with tab_cols[i+1]:
        active = "active" if st.session_state.sel_tab == tab else ""
        if st.button(tab, key=f"tab_{i}"):
            st.session_state.sel_tab = tab
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PATIENT SELECTOR ROW
# ══════════════════════════════════════════════════════════════════════════════
sel_cols = st.columns([1,1,1,3])
for i, (pname, pdata) in enumerate(PATIENTS.items()):
    risk = pdata["risk_history"][-1]
    tier_color = "#ff2244" if risk>=0.80 else "#ffb300" if risk>=0.65 else "#00f5ff"
    active_bg = "background:rgba(0,245,255,0.06);border-color:rgba(0,245,255,0.3);" if st.session_state.sel_patient==pname else ""
    with sel_cols[i]:
        st.markdown(f"""
        <div style="border:1px solid var(--border);border-radius:4px;padding:0.4rem 0.6rem;
                    cursor:pointer;{active_bg}margin-bottom:0.3rem">
          <div style="font-family:var(--display);font-size:0.7rem;color:#e8f4ff;letter-spacing:1px">{pname}</div>
          <div style="font-family:var(--mono);font-size:0.55rem;color:var(--text2)">{pdata['bed']}  ·  {pdata['dx'][:28]}...</div>
          <div style="font-family:var(--display);font-size:1rem;font-weight:700;color:{tier_color};
                      text-shadow:0 0 12px {tier_color}88;margin-top:2px">{risk*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"Select", key=f"sel_{i}", use_container_width=True):
            st.session_state.sel_patient = pname
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  ACTIVE PATIENT
# ══════════════════════════════════════════════════════════════════════════════
P = PATIENTS[st.session_state.sel_patient]
risk = P["risk_history"][-1]
tier = "CRITICAL RISK" if risk>=0.80 else "HIGH RISK" if risk>=0.65 else "MODERATE RISK"
tc   = "#ff2244" if risk>=0.80 else "#ffb300" if risk>=0.65 else "#00f5ff"

# ══════════════════════════════════════════════════════════════════════════════
#  PATIENT HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="pt-header">
  <div style="display:flex;align-items:center;gap:0.8rem">
    <div class="pt-avatar">{P['id'][-2:]}</div>
    <div>
      <div class="pt-name">{st.session_state.sel_patient}</div>
      <div class="pt-meta">
        Age {P['age']}  ·  {P['sex']}  ·  {P['weight']}  ·  {P['bed']}  &nbsp;|&nbsp;  {P['dx']}
      </div>
      <div class="pt-admit">Admitted: {P['admitted']}  &nbsp;·&nbsp;  ICU Region: ARDS  &nbsp;·&nbsp;  Bed ID: {P['bed']}</div>
    </div>
  </div>
  <div class="risk-badge" style="border-color:{tc};background:rgba(0,0,0,0)">
    <div class="risk-pct" style="color:{tc};text-shadow:0 0 30px {tc}88">{risk*100:.0f}%</div>
    <div class="risk-lbl" style="color:{tc}">{tier}</div>
    <div style="font-family:var(--mono);font-size:0.52rem;color:var(--text3);margin-top:3px">PREDICTED RISK · 6H</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: PATIENT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.sel_tab == "PATIENT DASHBOARD":

    # ── ROW 1: Vitals ─────────────────────────────────────────────────────────
    def vcard(label, val, unit, lo, hi, trend_val, trend_dir, ref):
        st_cls = "crit" if (val<lo*0.9 or val>hi*1.1) else "warn" if (val<lo or val>hi) else "ok"
        color  = "var(--red)" if st_cls=="crit" else "var(--amber)" if st_cls=="warn" else "var(--cyan2)"
        arrow  = "▲" if trend_dir=="up" else "▼"
        tr_cls = "up" if trend_dir=="up" else "down"
        return f"""
        <div class="vital-card {st_cls}">
          <div class="vcard-label">{label}</div>
          <div style="display:flex;align-items:baseline;gap:3px">
            <span class="vcard-val" style="color:{color}">{val:.0f}</span>
            <span class="vcard-unit">{unit}</span>
          </div>
          <div class="vcard-trend {tr_cls}">{arrow} {abs(trend_val):.1f} from baseline</div>
          <div class="vcard-ref">{ref}</div>
        </div>"""

    HR_   = P["HR"][-1];  MAP_  = P["MAP"][-1]
    LAC_  = P["Lactate"][-1]; SPO_  = P["SpO2"][-1]
    RESP_ = P["Resp"][-1]

    HR_tr  = P["HR"][-1]   - P["HR"][0]
    MAP_tr = P["MAP"][-1]  - P["MAP"][0]
    LAC_tr = P["Lactate"][-1]-P["Lactate"][0]
    SPO_tr = P["SpO2"][-1] - P["SpO2"][0]

    v1,v2,v3,v4,v5 = st.columns(5)
    v1.markdown(vcard("HEART RATE",     HR_,  "bpm",   60,100, HR_tr,  "up" if HR_tr>0 else "down",  "Normal: 60–100"), unsafe_allow_html=True)
    v2.markdown(vcard("MAP (MEAN ART)", MAP_, "mmHg",  65,100, MAP_tr, "up" if MAP_tr>0 else "down", "Target: ≥65 mmHg"), unsafe_allow_html=True)
    v3.markdown(vcard("SpO₂",           SPO_, "%",     95,100, SPO_tr, "up" if SPO_tr>0 else "down", "Target: ≥95%"), unsafe_allow_html=True)
    v4.markdown(vcard("RESP RATE",      RESP_,"bpm",   12,20,  RESP_-18,"up" if RESP_>18 else "down","Normal: 12–20"), unsafe_allow_html=True)
    v5.markdown(vcard("LACTATE",        LAC_, "mmol/L",0.5,2.0,LAC_tr,"up" if LAC_tr>0 else "down", "Normal: 0.5–2.0"), unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── ROW 2: ECG + PPG ─────────────────────────────────────────────────────
    ecg_col, ppg_col = st.columns(2)

    def make_ecg(n=200):
        t = np.linspace(0, 4*np.pi, n)
        sig = np.zeros(n)
        for i in range(0, n, 40):
            if i+35 < n:
                seg = np.linspace(0,1,35)
                sig[i:i+5]   += np.linspace(0, 0.1, 5)
                sig[i+5:i+8] += np.array([0.15, 0.08, 0.05])
                sig[i+8]     += 1.8
                sig[i+9]     -= 0.9
                sig[i+10:i+15] += np.linspace(0.2, 0, 5)
                sig[i+15:i+25] += 0.12*np.sin(np.linspace(0,np.pi,10))
        sig += np.random.normal(0,0.02,n)
        return sig

    def make_ppg(n=200):
        t = np.linspace(0,8*np.pi,n)
        return 0.5*np.sin(t) + 0.15*np.sin(2*t) + 0.05*np.sin(3*t) + np.random.normal(0,0.02,n)

    ecg_x = list(range(200))

    with ecg_col:
        st.markdown("""
        <div class="panel-hdr" style="background:var(--bg2);border-bottom:1px solid var(--border)">
          <span class="panel-title">ECG WAVEFORM — 50HMZ</span>
          <span class="panel-badge">LIVE</span>
        </div>""", unsafe_allow_html=True)
        ecg_sig = make_ecg()
        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(
            x=ecg_x, y=ecg_sig,
            line=dict(color="#00ff88", width=1.4),
            fill="tozeroy", fillcolor="rgba(0,255,136,0.04)",
            mode="lines", hoverinfo="skip",
        ))
        # reference grid lines
        for yv in [-0.5, 0, 0.5, 1.0, 1.5]:
            fig_ecg.add_hline(y=yv, line_color="#0e2040", line_width=0.6)
        fig_ecg.update_layout(
            height=110, margin=dict(t=4,b=4,l=4,r=4),
            paper_bgcolor="#071428", plot_bgcolor="#071428",
            xaxis=dict(showgrid=False,showticklabels=True,tickfont=dict(family="JetBrains Mono",size=7,color="#1e4060"),
                       showline=False,zeroline=False,
                       tickvals=[0,50,100,150,200],
                       ticktext=["0s","0.5s","1.0s","1.5s","2.0s"]),
            yaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_ecg, use_container_width=True, config={"displayModeBar":False})

    with ppg_col:
        st.markdown("""
        <div class="panel-hdr" style="background:var(--bg2);border-bottom:1px solid var(--border)">
          <span class="panel-title">PPG SIGNAL — 100HZ</span>
          <span class="panel-badge">LIVE</span>
        </div>""", unsafe_allow_html=True)
        ppg_sig = make_ppg()
        fig_ppg = go.Figure()
        # dual-color PPG — cyan baseline, pink peaks
        fig_ppg.add_trace(go.Scatter(
            x=ecg_x, y=ppg_sig,
            line=dict(color="#00bcd4", width=1.4),
            fill="tozeroy", fillcolor="rgba(0,188,212,0.05)",
            mode="lines", hoverinfo="skip", name="PPG",
        ))
        # overlay high segments in pink
        ppg_hi = ppg_sig.copy()
        ppg_hi[ppg_hi < 0.3] = np.nan
        fig_ppg.add_trace(go.Scatter(
            x=ecg_x, y=ppg_hi,
            line=dict(color="#ff4499", width=1.6),
            mode="lines", hoverinfo="skip", name="Peak",
        ))
        fig_ppg.update_layout(
            height=110, margin=dict(t=4,b=4,l=4,r=4),
            paper_bgcolor="#071428", plot_bgcolor="#071428",
            xaxis=dict(showgrid=False,showticklabels=True,tickfont=dict(family="JetBrains Mono",size=7,color="#1e4060"),
                       showline=False,zeroline=False,
                       tickvals=[0,50,100,150,200],
                       ticktext=["0s","0.5s","1.0s","1.5s","2.0s"]),
            yaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_ppg, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: ALERTS & PREDICTIONS  (matches screenshot 2)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.sel_tab == "ALERTS & PREDICTIONS":

    # ── ROW 1: Composite Risk Score + Event Stream ────────────────────────────
    risk_col, event_col = st.columns([1.6, 1])

    with risk_col:
        st.markdown("""
        <div class="panel-hdr">
          <span class="panel-title">COMPOSITE RISK SCORE — 24H</span>
          <span class="panel-badge">MODEL OUTPUT</span>
        </div>""", unsafe_allow_html=True)

        # Build 24h risk history with realistic progression
        base = P["risk_history"]
        t_axis = []
        r_axis = []
        now = datetime.now()
        for i in range(24):
            t_axis.append((now - timedelta(hours=23-i)).strftime("%H:%M"))
            if i < 12:
                r_axis.append(base[i] + np.random.normal(0, 0.01))
            else:
                last = r_axis[-1]
                delta = np.random.normal(0.004, 0.008)
                r_axis.append(float(np.clip(last + delta, 0.05, 0.97)))

        threshold = 0.62
        fig_risk = go.Figure()

        # Fill zones
        fig_risk.add_hrect(y0=0, y1=threshold, fillcolor="rgba(0,188,212,0.04)",
                           line_width=0, layer="below")
        fig_risk.add_hrect(y0=threshold, y1=1.0, fillcolor="rgba(255,34,68,0.04)",
                           line_width=0, layer="below")

        # Area fill under curve
        fig_risk.add_trace(go.Scatter(
            x=t_axis, y=r_axis,
            fill="tozeroy", fillcolor="rgba(0,188,212,0.06)",
            line=dict(color="#00bcd4", width=0), mode="lines",
            hoverinfo="skip", showlegend=False,
        ))
        # Main line
        fig_risk.add_trace(go.Scatter(
            x=t_axis, y=r_axis,
            line=dict(color="#00f5ff", width=2),
            mode="lines",
            hovertemplate="<b>%{x}</b><br>Risk: %{y:.1%}<extra></extra>",
            name="Risk Score",
        ))
        # Threshold line
        fig_risk.add_hline(
            y=threshold, line_dash="dot", line_color="#ff2244", line_width=1,
            annotation_text=f"Alert Threshold ({threshold:.0%})",
            annotation_font=dict(size=9, color="#ff2244", family="JetBrains Mono"),
            annotation_position="top right",
        )
        # Current risk marker
        fig_risk.add_trace(go.Scatter(
            x=[t_axis[-1]], y=[r_axis[-1]],
            mode="markers",
            marker=dict(color=tc, size=10,
                        line=dict(color=tc, width=2),
                        symbol="circle",
                        ),
            hovertemplate=f"<b>NOW</b><br>Risk: {r_axis[-1]:.1%}<extra></extra>",
            showlegend=False,
        ))

        fig_risk.update_layout(
            height=220, margin=dict(t=8,b=30,l=8,r=8),
            paper_bgcolor="#071428", plot_bgcolor="#071428",
            xaxis=dict(color="#1e4060", gridcolor="#0e2040",
                       tickfont=dict(family="JetBrains Mono",size=8),
                       showgrid=True, zeroline=False,
                       nticks=8),
            yaxis=dict(color="#1e4060", gridcolor="#0e2040",
                       tickfont=dict(family="JetBrains Mono",size=8),
                       tickformat=".0%", range=[0,1],
                       showgrid=True, zeroline=False),
            showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar":False})

    with event_col:
        st.markdown("""
        <div class="panel-hdr">
          <span class="panel-title">CLINICAL EVENT STREAM</span>
          <span class="panel-badge">LIVE</span>
        </div>""", unsafe_allow_html=True)

        events = EVENTS_TEMPLATE.get(st.session_state.sel_patient, [])
        html = ""
        for color, etime, text in events:
            html += f"""
            <div class="event-item">
              <div class="event-dot" style="background:{color};box-shadow:0 0 6px {color}88"></div>
              <div class="event-time">{etime}</div>
              <div class="event-text" style="color:{'#e8f4ff' if color=='#ff2244' else '#8ab8d4'}">{text}</div>
            </div>"""
        st.markdown(f'<div style="background:var(--bg1);border:1px solid var(--border);border-radius:4px;max-height:230px;overflow-y:auto">{html}</div>',
                    unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── ROW 2: Inflammatory + Renal + Fusion Weights ─────────────────────────
    inf_col, ren_col, fw_col = st.columns([1,1,1])

    with inf_col:
        st.markdown("""
        <div class="panel-hdr">
          <span class="panel-title">INFLAMMATORY MARKERS</span>
        </div>""", unsafe_allow_html=True)

        def lab_row(name, val, unit, ref, abnormal):
            color = "var(--red)" if abnormal == "high" else "var(--amber)" if abnormal == "mid" else "var(--cyan2)"
            arrow = " ↑" if abnormal == "high" else " ↑" if abnormal == "mid" else ""
            return f"""<div class="lab-row">
              <span class="lab-name">{name}</span>
              <span>
                <span class="lab-val" style="color:{color}">{val}{arrow}</span>
                <span class="lab-ref"> {unit} · {ref}</span>
              </span>
            </div>"""

        crp = P["crp"]; pct = P["pct"]; il6 = P["il6"]
        html  = lab_row("CRP",           f"{crp:.1f}", "mg/L",  "< 5",  "high" if crp>100 else "mid" if crp>20 else "")
        html += lab_row("Procalcitonin", f"{pct:.1f}", "ng/mL", "< 0.5","high" if pct>2 else "mid" if pct>0.5 else "")
        html += lab_row("IL-6",          f"{il6:.0f}", "pg/mL", "< 7",  "high" if il6>100 else "mid" if il6>20 else "")
        st.markdown(f'<div style="background:var(--bg1);border:1px solid var(--border);border-radius:4px;padding:0.1rem 0.2rem">{html}</div>',
                    unsafe_allow_html=True)

    with ren_col:
        st.markdown("""
        <div class="panel-hdr">
          <span class="panel-title">RENAL / METABOLIC</span>
        </div>""", unsafe_allow_html=True)

        bun = P["bun"]; cr = P["creatinine"]; lac = P["lactate_lab"]; glu = P["glucose"]
        html  = lab_row("BUN",       f"{bun}",   "mg/dL",  "7–20",   "high" if bun>25 else "")
        html += lab_row("Creatinine",f"{cr:.1f}", "mg/dL",  "0.6–1.2","high" if cr>1.5 else "mid" if cr>1.2 else "")
        html += lab_row("Lactate",   f"{lac:.1f}","mmol/L", "0.5–2.0","high" if lac>2 else "mid" if lac>1.5 else "")
        html += lab_row("Glucose",   f"{glu}",   "mg/dL",  "70–140", "mid" if glu>140 else "")
        st.markdown(f'<div style="background:var(--bg1);border:1px solid var(--border);border-radius:4px;padding:0.1rem 0.2rem">{html}</div>',
                    unsafe_allow_html=True)

    with fw_col:
        st.markdown("""
        <div class="panel-hdr">
          <span class="panel-title">MULTI-MODAL FUSION WEIGHTS</span>
        </div>""", unsafe_allow_html=True)

        wt_colors = {"Vitals":"#00f5ff","Labs":"#00ff88","Trends":"#ffb300","Demographics":"#ff4499"}
        html = '<div style="background:var(--bg1);border:1px solid var(--border);border-radius:4px;padding:0.6rem 0.75rem">'
        for mod, wt in P["wts"].items():
            color = wt_colors[mod]
            html += f"""
            <div class="fw-row">
              <div class="fw-label">
                <span style="color:var(--text2)">{mod}</span>
                <span style="color:{color}">{wt:.2f}</span>
              </div>
              <div class="fw-track">
                <div class="fw-fill" style="width:{wt*100:.0f}%;background:linear-gradient(90deg,{color}88,{color})"></div>
              </div>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── ROW 3: Run prediction + AI narrative ─────────────────────────────────
    st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text2);letter-spacing:3px;text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:4px;margin-bottom:8px">AEGIS-OMNI PREDICTION ENGINE</div>', unsafe_allow_html=True)
    pred_btn_col, narr_col = st.columns([1,3])

    with pred_btn_col:
        if st.button("⬡  RUN PREDICTION", use_container_width=True):
            payload = {
                "patient_id": P["id"],
                "window": {
                    "HR":P["HR"],"MAP":P["MAP"],"Lactate":P["Lactate"],
                    "SpO2":P["SpO2"],"Resp":P["Resp"],"SBP":P["SBP"],"Temp":P["Temp"],
                },
                "vitals_snapshot": {"HR":P["HR"][-1],"MAP":P["MAP"][-1],"Lactate":P["Lactate"][-1]},
            }
            with st.spinner(""):
                try:
                    r = requests.post(f"{API_URL}/predict",json=payload,timeout=20).json()
                except:
                    r_val = P["risk_history"][-1]
                    tier_str = "CRITICAL" if r_val>=0.80 else "WARN" if r_val>=0.65 else "WATCH"
                    r = {
                        "patient_id":P["id"],"risk_score":round(r_val,4),
                        "risk_percent":f"{r_val:.1%}","confidence":0.86,
                        "alert_tier":tier_str,"alert":r_val>=0.62,"threshold":0.62,
                        "raw_prob":round(r_val*0.92,4),
                        "top_features":["Lactate","MAP","shock_index","HR","Resp_slope2h","SpO2","lactate_map_ratio","Temp"],
                        "shap_values":[0.31,0.24,0.18,0.12,0.09,0.08,0.07,0.05],
                        "ai_narrative":(
                            f"Aegis-Omni assigns {'HIGH' if r_val>=0.65 else 'MODERATE'} risk ({r_val:.1%}) "
                            f"for septic shock within 6 hours. Lactate trending to {P['Lactate'][-1]:.1f} mmol/L "
                            f"with MAP at {P['MAP'][-1]:.0f} mmHg indicates progressive haemodynamic compromise. "
                            f"Immediate vasopressor review and repeat lactate in 2 hours recommended."
                        ),
                    }
            st.session_state.last_result = r

        # Model info under button
        st.markdown(f"""
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:4px;
                    padding:0.5rem 0.6rem;margin-top:0.5rem;font-family:var(--mono);font-size:0.62rem;line-height:1.9;color:var(--text2)">
          <span style="color:var(--text3)">ARCH  </span> BiLSTM+Attention<br>
          <span style="color:var(--text3)">MODEL </span> AegisLSTM v1.0<br>
          <span style="color:var(--text3)">HORIZ </span> 6 hours<br>
          <span style="color:var(--text3)">FDR   </span> &lt; 5%<br>
          <span style="color:var(--text3)">LLM   </span> Llama-3.1-8B
        </div>""", unsafe_allow_html=True)

    with narr_col:
        r = st.session_state.get("last_result")
        if r:
            rt    = r["alert_tier"]
            rtc   = "#ff2244" if rt=="CRITICAL" else "#ffb300" if rt=="WARN" else "#00f5ff"
            conf  = r.get("confidence",0.85)
            thr   = r.get("threshold",0.62)

            # Header strip
            st.markdown(f"""
            <div style="display:flex;gap:1rem;align-items:center;margin-bottom:0.5rem">
              <div style="background:rgba(0,0,0,0);border:1px solid {rtc};border-radius:3px;padding:0.2rem 0.8rem;
                          font-family:var(--display);font-size:1.4rem;font-weight:700;
                          color:{rtc};text-shadow:0 0 20px {rtc}88">{r['risk_percent']}</div>
              <div style="font-family:var(--mono);font-size:0.65rem;color:{rtc};letter-spacing:2px">{rt}</div>
              <div style="font-family:var(--mono);font-size:0.62rem;color:var(--text2)">
                Confidence: <span style="color:var(--cyan)">{conf:.0%}</span>  &nbsp;·&nbsp;
                Threshold: <span style="color:var(--text)">{thr:.3f}</span>  &nbsp;·&nbsp;
                Alert: <span style="color:{'var(--red)' if r.get('alert') else 'var(--green)'}">{'⚡ FIRED' if r.get('alert') else '✓ CLEAR'}</span>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="narrative">{r["ai_narrative"]}</div>', unsafe_allow_html=True)

            # Action items
            ACTIONS = {
                "CRITICAL":[("🚨","Immediate physician bedside assessment","var(--red)"),
                            ("💉","Vasopressors if MAP < 65 mmHg","var(--red)"),
                            ("🩸","Blood cultures × 2 — before antibiotics","var(--amber)"),
                            ("💊","Broad-spectrum antibiotics within 1 hour","var(--amber)")],
                "WARN":    [("⚠️","Physician notification within 30 min","var(--amber)"),
                            ("🩸","Blood cultures + lactate STAT","var(--amber)"),
                            ("💧","500 mL crystalloid bolus","var(--text)")],
                "WATCH":   [("✅","Continue q1h monitoring","var(--cyan2)"),
                            ("📝","Reassess if MAP or lactate worsens","var(--text)")],
            }
            act_html = '<div style="margin-top:0.5rem">'
            for icon, text, color in ACTIONS.get(rt,[]):
                act_html += f'<div class="action-row"><span>{icon}</span><span style="color:{color}">{text}</span></div>'
            act_html += "</div>"
            st.markdown(act_html, unsafe_allow_html=True)

            # SHAP bar
            st.markdown('<div style="font-family:var(--mono);font-size:0.55rem;color:var(--text3);letter-spacing:2px;text-transform:uppercase;margin-top:0.6rem;margin-bottom:4px">Feature Importance</div>', unsafe_allow_html=True)
            feats = r["top_features"][:6]; shaps = r["shap_values"][:6]
            bar_colors = [tc if v>0.15 else "#ff6644" if v>0.08 else "#00aabb" for v in shaps]
            fig_shap = go.Figure(go.Bar(
                x=shaps[::-1], y=feats[::-1], orientation="h",
                marker=dict(color=bar_colors[::-1],opacity=0.9,line=dict(color="#0e2040",width=1)),
                text=[f"{v:.3f}" for v in shaps[::-1]],
                textposition="inside", textfont=dict(family="JetBrains Mono",size=8,color="#03080f"),
            ))
            fig_shap.update_layout(
                height=180, margin=dict(t=2,b=2,l=2,r=2),
                paper_bgcolor="#071428", plot_bgcolor="#071428",
                xaxis=dict(color="#1e4060",gridcolor="#0e2040",tickfont=dict(family="JetBrains Mono",size=8)),
                yaxis=dict(color="#4a7a99",gridcolor="#0e2040",tickfont=dict(family="JetBrains Mono",size=9)),
            )
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar":False})

        else:
            st.markdown("""
            <div style="background:var(--bg2);border:1px solid var(--border);border-radius:4px;
                        padding:2rem;text-align:center;color:var(--text3);
                        font-family:var(--mono);font-size:0.7rem;letter-spacing:2px">
              CLICK  ⬡  RUN PREDICTION  TO GENERATE RISK ASSESSMENT
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: FEDERATED NETWORK
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.sel_tab == "FEDERATED NETWORK":
    st.markdown("""
    <div class="panel-hdr">
      <span class="panel-title">FEDERATED LEARNING NETWORK STATUS</span>
      <span class="panel-badge">3 NODES ACTIVE</span>
    </div>""", unsafe_allow_html=True)

    node_cols = st.columns(3)
    NODES = [
        ("HOSPITAL NODE 001","HOSP_001","Last sync: 2 min ago","rounds:12","14,220 samples","AUROC: 0.891","#00ff88"),
        ("HOSPITAL NODE 002","HOSP_002","Last sync: 5 min ago","rounds:12","11,840 samples","AUROC: 0.878","#00ff88"),
        ("HOSPITAL NODE 003","HOSP_003","Last sync: 9 min ago","rounds:12","9,612 samples", "AUROC: 0.882","#ffb300"),
    ]
    for col, (name,nid,sync,rounds,samples,auroc,color) in zip(node_cols,NODES):
        with col:
            st.markdown(f"""
            <div style="background:var(--bg1);border:1px solid var(--border);border-radius:4px;padding:1rem">
              <div style="font-family:var(--display);font-size:0.75rem;color:{color};letter-spacing:2px;font-weight:700">{name}</div>
              <div style="font-family:var(--mono);font-size:0.6rem;color:var(--text2);margin-top:4px">{nid}</div>
              <hr/>
              <div style="font-family:var(--mono);font-size:0.65rem;line-height:2;color:var(--text)">
                {sync}<br>{rounds}  ·  {samples}<br>
                <span style="color:{color}">{auroc}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # FL round progress chart
    st.markdown('<div style="font-family:var(--mono);font-size:0.58rem;color:var(--text2);letter-spacing:3px;border-bottom:1px solid var(--border);padding-bottom:4px;margin-bottom:8px">FEDERATED AVERAGING — AUROC PER ROUND</div>', unsafe_allow_html=True)
    rounds = list(range(1,13))
    n1 = [0.71,0.75,0.78,0.81,0.83,0.85,0.86,0.87,0.88,0.89,0.89,0.891]
    n2 = [0.69,0.73,0.76,0.79,0.81,0.83,0.85,0.86,0.87,0.87,0.878,0.878]
    n3 = [0.70,0.74,0.77,0.80,0.82,0.84,0.85,0.86,0.87,0.88,0.881,0.882]
    fig_fl = go.Figure()
    for vals, name, color in [(n1,"Node 001","#00f5ff"),(n2,"Node 002","#00ff88"),(n3,"Node 003","#ffb300")]:
        fig_fl.add_trace(go.Scatter(x=rounds,y=vals,name=name,
                                     line=dict(color=color,width=2),mode="lines+markers",
                                     marker=dict(size=5,color=color)))
    fig_fl.update_layout(
        height=240, margin=dict(t=8,b=30,l=8,r=8),
        paper_bgcolor="#071428",plot_bgcolor="#071428",
        xaxis=dict(title="FL Round",color="#1e4060",gridcolor="#0e2040",
                   tickfont=dict(family="JetBrains Mono",size=9)),
        yaxis=dict(title="AUROC",color="#1e4060",gridcolor="#0e2040",
                   tickfont=dict(family="JetBrains Mono",size=9),range=[0.65,0.95]),
        legend=dict(font=dict(family="JetBrains Mono",size=9,color="#4a7a99"),
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_fl, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.sel_tab == "EXPLAINABILITY":
    r = st.session_state.get("last_result")
    if not r:
        st.markdown("""
        <div style="text-align:center;padding:3rem;font-family:var(--mono);font-size:0.7rem;
                    color:var(--text3);letter-spacing:2px">
          RUN A PREDICTION FIRST FROM THE ALERTS & PREDICTIONS TAB
        </div>""", unsafe_allow_html=True)
    else:
        xp1, xp2 = st.columns(2)
        with xp1:
            st.markdown('<div class="panel-hdr"><span class="panel-title">ATTENTION HEATMAP — TIME WEIGHTS</span></div>', unsafe_allow_html=True)
            hours = [f"-{11-i}h" for i in range(12)]
            attn  = np.random.dirichlet(np.ones(12)*2)
            attn[-3:] *= 2.5; attn /= attn.sum()
            fig_attn = go.Figure(go.Bar(
                x=hours, y=attn,
                marker=dict(
                    color=attn,
                    colorscale=[[0,"#071428"],[0.5,"#00bcd4"],[1.0,"#ff2244"]],
                    showscale=False,
                ),
            ))
            fig_attn.update_layout(
                height=200,margin=dict(t=8,b=30,l=8,r=8),
                paper_bgcolor="#071428",plot_bgcolor="#071428",
                xaxis=dict(color="#1e4060",tickfont=dict(family="JetBrains Mono",size=9)),
                yaxis=dict(color="#1e4060",tickfont=dict(family="JetBrains Mono",size=9),title="Attention"),
                bargap=0.1,
            )
            st.plotly_chart(fig_attn,use_container_width=True,config={"displayModeBar":False})

        with xp2:
            st.markdown('<div class="panel-hdr"><span class="panel-title">SHAP — FEATURE CONTRIBUTIONS</span></div>', unsafe_allow_html=True)
            feats = r["top_features"][:8]; shaps = r["shap_values"][:8]
            colors = [tc if v>0.15 else "#ff6644" if v>0.08 else "#00aabb" for v in shaps]
            fig_s = go.Figure(go.Bar(
                x=shaps[::-1],y=feats[::-1],orientation="h",
                marker=dict(color=colors[::-1],opacity=0.9),
                text=[f"{v:.3f}" for v in shaps[::-1]],
                textposition="inside",textfont=dict(family="JetBrains Mono",size=9,color="#03080f"),
            ))
            fig_s.update_layout(
                height=200,margin=dict(t=8,b=30,l=8,r=8),
                paper_bgcolor="#071428",plot_bgcolor="#071428",
                xaxis=dict(color="#1e4060",gridcolor="#0e2040",tickfont=dict(family="JetBrains Mono",size=9)),
                yaxis=dict(color="#4a7a99",gridcolor="#0e2040",tickfont=dict(family="JetBrains Mono",size=9)),
            )
            st.plotly_chart(fig_s,use_container_width=True,config={"displayModeBar":False})

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-hdr"><span class="panel-title">AI CLINICAL NARRATIVE — FEATHERLESS LLM</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="narrative" style="font-size:0.88rem">{r["ai_narrative"]}</div>', unsafe_allow_html=True)

        with st.expander("⬡  Raw model output"):
            st.json(r)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:0.8rem 0 0.3rem;border-top:1px solid var(--border);
            margin-top:0.5rem;font-family:var(--mono);font-size:0.55rem;
            color:var(--text3);letter-spacing:3px">
  AEGIS-OMNI v1.0  ·  HACKATHON PROTOTYPE  ·  NOT FOR CLINICAL USE WITHOUT FORMAL VALIDATION
</div>""", unsafe_allow_html=True)