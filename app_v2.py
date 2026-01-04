# test redeploy...
import random
from dataclasses import dataclass
from typing import Dict
from datetime import datetime
import time
import json
import streamlit as st
import pandas as pd
import altair as alt
from datetime import date
import math
import neurokit2 as nk 
import numpy as np


import os
import base64

from Simulation.Psimulation import build_patient as build_patient



# R_alart.py  (complete, deploy-friendly)
# NEWS2 scoring (0–3) for: bp_sys, hr, rr, temp (NO bp_dia).
# getOverallLevel(): per-parameter level based on SINGLE subscore:
#       3 -> "severe", 2 -> "moderate", 0/1 -> "ok"
# getAllscore(): total score + NEWS2 band: high/medium/low-medium/low
# updateAlerts(): plays alarm when HIGH (total >= 7) with cooldown to avoid spam
#
# IMPORTANT:
# 1) Put your mp3 in: <same folder as this file>/assets/deep2.mp3
# 2) Browser may block autoplay. This code also offers a manual "Play alarm" button.

import os
import time
import base64
from typing import Dict, List, Any, Optional

import streamlit as st
import streamlit.components.v1 as components


def get_mode():
    #test panel: /?mode=test
    if hasattr(st, "query_params"):
        v = st.query_params.get("mode", "")
        return v if isinstance(v, str) else (v[0] if v else "")
    if hasattr(st, "experimental_get_query_params"):
        return st.experimental_get_query_params().get("mode", [""])[0]
    return ""


# Alarm audio configuration
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(THIS_DIR, "assets", "deep2.mp3")

MIME_TYPE = "audio/mpeg"
ALARM_VOLUME = 0.8
ALARM_COOLDOWN_SEC = 8 


@st.cache_data
def _load_audio_b64(path: str) -> Optional[str]:
    """Read mp3 once and cache as base64 (faster + avoids re-embedding every rerun)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _play_alarm_html(audio_b64: str, volume: float = 0.8, loop: bool = False) -> str:
    """HTML+JS that attempts to play audio. Autoplay may be blocked by browser."""
    loop_js = "true" if loop else "false"
    return f"""
    <audio id="invisible_alarm" autoplay style="display:none;">
      <source src="data:{MIME_TYPE};base64,{audio_b64}" type="{MIME_TYPE}">
    </audio>
    <script>
      const audio = document.getElementById('invisible_alarm');
      if (audio) {{
        audio.volume = {volume};
        audio.loop = {loop_js};
        const p = audio.play();
        if (p !== undefined) {{
          p.catch(err => console.error("Alarm audio blocked:", err));
        }}
      }}
    </script>
    """


def triggerAudio(container, *, loop: bool = False) -> None:
    """Try to play alarm sound."""
    audio_b64 = _load_audio_b64(ALARM_PATH)
    if not audio_b64:
        with container:
            st.warning(f"Alarm audio file not found! Check path: {ALARM_PATH}")
        return

    html = _play_alarm_html(audio_b64, volume=ALARM_VOLUME, loop=loop)
    with container:
        components.html(html, height=0, width=0, scrolling=False)


def showManualAlarmButton(container, label: str = "Play alarm sound") -> None:
    """
    Optional manual button. Useful because browsers often block autoplay.
    When clicked, it will play once (user interaction helps unlock audio).
    """
    with container:
        if st.button(label, key="manual_alarm_btn"):
            triggerAudio(container)

# NEWS2 scoring helpers
def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def score_rr(rr: Any) -> int:
    rr = _to_float(rr)
    if rr is None:
        return 0
    if rr <= 8:
        return 3
    if 9 <= rr <= 11:
        return 1
    if 12 <= rr <= 20:
        return 0
    if 21 <= rr <= 24:
        return 2
    return 3  # rr >= 25

def score_hr(hr: Any) -> int:
    hr = _to_float(hr)
    if hr is None:
        return 0
    if hr <= 40:
        return 3
    if 41 <= hr <= 50:
        return 1
    if 51 <= hr <= 90:
        return 0
    if 91 <= hr <= 110:
        return 1
    if 111 <= hr <= 130:
        return 2
    return 3  # hr >= 131

def score_bp_sys(bp_sys: Any) -> int:
    bp_sys = _to_float(bp_sys)
    if bp_sys is None:
        return 0
    if bp_sys <= 90:
        return 3
    if 91 <= bp_sys <= 100:
        return 2
    if 101 <= bp_sys <= 110:
        return 1
    if 111 <= bp_sys <= 219:
        return 0
    return 3  # bp_sys >= 220

def score_temp(temp: Any) -> int:
    temp = _to_float(temp)
    if temp is None:
        return 0
    if temp <= 35.0:
        return 3
    if 35.1 <= temp <= 36.0:
        return 1
    if 36.1 <= temp <= 38.0:
        return 0
    if 38.1 <= temp <= 39.0:
        return 1
    return 2  # temp >= 39.1

def compute_subscores(vitals: Any) -> Dict[str, int]:
    """
    vitals can be an object with attributes: rr/hr/bp_sys/temp
    e.g., vitals.rr, vitals.hr, vitals.bp_sys, vitals.temp
    """
    return {
        "rr": score_rr(getattr(vitals, "rr", None)),
        "hr": score_hr(getattr(vitals, "hr", None)),
        "bp_sys": score_bp_sys(getattr(vitals, "bp_sys", None)),
        "temp": score_temp(getattr(vitals, "temp", None)),
    }

def getVitalScores(vitals: Any) -> Dict[str, int]:
    """Return NEWS2 subscores (0–3) for each vital used: rr, hr, bp_sys, temp."""
    return compute_subscores(vitals)

def getOverallLevel(vitals: Any, *args, **kwargs) -> Dict[str, str]:
    subs = compute_subscores(vitals)

    def _lvl(s: int) -> str:
        if s == 3:
            return "severe"
        if s == 2:
            return "moderate"
        return "ok"  # 0 or 1

    return {k: _lvl(v) for k, v in subs.items()}

def getAllscore(vitals: Any) -> Dict[str, Any]:
    """
    Total NEWS2-like score (only rr/hr/bp_sys/temp in this simplified version)
    + banding:
      - high: total >= 7
      - medium: total 5-6
      - low-medium: any single parameter = 3 (red) but total < 5
      - low: otherwise
    """
    subs = compute_subscores(vitals)
    total = sum(int(x) for x in subs.values())
    has_red = any(s == 3 for s in subs.values())

    if total >= 7:
        risk_level = "high"
        trigger = "aggregate_score_7_or_more"
    elif 5 <= total <= 6:
        risk_level = "medium"
        trigger = "aggregate_score_5_to_6"
    elif has_red:
        risk_level = "low-medium"
        trigger = "red_score_single_parameter_3"
    else:
        risk_level = "low"
        trigger = "aggregate_score_0_to_4"

    return {
        "total_score": total,
        "risk_level": risk_level,
        "trigger": trigger,
        "subscores": subs,  # handy for debugging/UI
    }

def updateAlerts(patients: Dict, audio_container) -> None:
    """
    Triggers alarm when NEWS2 band is HIGH (total_score >= 7) for any patient.
    Uses cooldown to avoid repeated replay on Streamlit reruns.

    patients example:
      patients = {
        "p001": {"vitals": vitals_obj},
        "p002": {"vitals": vitals_obj},
      }
    """
    severePatients: List[str] = []

    for pid, p in patients.items():
        res = getAllscore(p["vitals"])
        if res["risk_level"] == "high":
            severePatients.append(pid)

    st.session_state["severe_patients_list"] = severePatients

    now = time.time()
    last = float(st.session_state.get("last_alarm_ts", 0.0))

    if severePatients:
        if (now - last) >= ALARM_COOLDOWN_SEC:
            st.session_state["last_alarm_ts"] = now
            triggerAudio(audio_container, loop=False)

        showManualAlarmButton(audio_container, label="Play alarm (if autoplay is blocked)")
    else:
        with audio_container:
            st.empty()
        st.session_state["last_alarm_ts"] = 0.0


FS = 250           # sampling rate used in Psimulation
STEP_SAMPLES = FS  # advance 1 simulated second each rerun


@dataclass
class Ranges:
    min: float
    max: float
    moderate_band: float

@dataclass
class Vitals:
    hr: float # bpm
    temp: float # °C
    rr: float # /min
    bp_sys: float # mmHg
    bp_dia: float # mmHg



st.set_page_config(page_title="Remote Monitor", layout="wide")

st.markdown(
    """
<style>
[data-testid="stHeader"] {
    display: none;
}

.block-container {
    padding-top: 0.5rem;
}

.rm-card {
  border-radius:18px; 
  padding:0; 
  overflow:hidden; 
  background:#ffffff;
  border:1px solid rgba(0,0,0,0.08);
  transition: box-shadow .15s ease, transform .05s ease;
}
.rm-card:hover { 
    box-shadow:0 8px 20px rgba(2,132,199,.12), 0 2px 8px rgba(2,132,199,.08); 
}

.rm-id-header {
  display:flex; 
  align-items:center; 
  justify-content:space-between;
  padding:10px 12px; 
  color:white;
}

.rm-head-normal { background:linear-gradient(90deg,#0ea5e9 0%, #38bdf8 100%); }
.rm-head-danger { background:linear-gradient(90deg,#ff0000 0%, #f87171 100%); }

.rm-id-left {
    display:flex; 
    align-items:center; 
    gap:10px;
}
.rm-avatar {
  width:36px; height:36px; border-radius:50%; background:rgba(255,255,255,.25);
  display:grid; place-items:center; font-weight:800; letter-spacing:.5px;
}
.rm-name {font-weight:700; line-height:1.05;}
.rm-sub {font-size:11px; opacity:.9}

.rm-id-body {padding:10px 12px;}
.rm-v {border:1px solid rgba(0,0,0,.06); border-radius:10px; padding:8px; background:#f8fafc; margin-bottom:8px;}
.rm-v .lab {font-size:11px; opacity:.65;}
.rm-v .val {font-size:18px; font-weight:800;}

.val-ok  { color:#065f46; }
.val-mod { color:#ffb400; }
.val-sev { color:#ff0000; }

</style>
""",
    unsafe_allow_html=True,
)


DEFAULT_RANGES: Dict[str, Ranges] = {
    "hr": Ranges(50, 110, 10),
    "temp": Ranges(36.0, 37.8, 0.3),
    "rr": Ranges(10, 20, 2),
    "bp_sys": Ranges(90, 140, 10),
    "bp_dia": Ranges(60, 90, 5),
}

def vital_level(value, r):
    if value < r.min - r.moderate_band or value > r.max + r.moderate_band:
        return "severe"
    if value < r.min or value > r.max:
        return "moderate"
    return "ok"

def overall_level(levels):
    if "severe" in levels.values(): return "severe"
    if "moderate" in levels.values(): return "moderate"
    return "ok"

def val_class(level):
    return {"ok":"val-ok", "moderate":"val-mod", "severe":"val-sev"}[level]

def initials(name):
    parts = [p for p in name.split() if p]
    if not parts: return "?"
    if len(parts)==1: return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def severity_rank(lvl):
    return {"ok": 0, "moderate": 1, "severe": 2}[lvl]


#Patient data
N_PATIENTS = 10
NAMES = [
    "A B10", "B 10", "C", "D", "E", "F", "G", "H", "I", "J"
]

if "patients" not in st.session_state:
    st.session_state.patients = {}
    
    for i in range(1, N_PATIENTS+1):
        pid = f"p{i:02d}"
        name = NAMES[i-1]
        st.session_state.patients[pid] = {
            "id": pid,
            "name": name,
            "vitals": Vitals(hr=100+random.randint(-2,2), temp=36.8, rr=10, bp_sys=118, bp_dia=78),

            "hr_history": [],
            "minute_logs": [],
            "last_log_min": None,

            "ecg_live": [],          
            "ecg_phase": 0.0,        
            "ecg_samples": [],     
            "last_ecg_min": None,

            "daily_reports": {},  # { "YYYY-MM-DD": [event, event, ...] }
            "last_report_key": None,
        }

@st.cache_data(show_spinner=False)
def cached_build_patient(i: int):
    return build_patient(i)


if "sim_data" not in st.session_state:
    st.session_state.sim_data = {}
    st.session_state.sim_idx = {}

    for i, pid in enumerate(st.session_state.patients.keys(), start=1):
        sim_dict = cached_build_patient(i)
        st.session_state.sim_data[pid] = sim_dict
        st.session_state.sim_idx[pid] = 0

if "scenario" not in st.session_state:
    st.session_state.scenario = {}

def apply_scenario_override(pid: str, v):

    sc = st.session_state.scenario.get(pid)
    if not sc:
        return v

    until = sc.get("until")
    if until is not None and time.time() > float(until):
        st.session_state.scenario.pop(pid, None)
        return v

    ov = sc.get("overrides", {}) or {}
    if "hr" in ov: v.hr = float(ov["hr"])
    if "rr" in ov: v.rr = float(ov["rr"])
    if "temp" in ov: v.temp = float(ov["temp"])
    if "bp_sys" in ov: v.bp_sys = float(ov["bp_sys"])
    if "bp_dia" in ov: v.bp_dia = float(ov["bp_dia"])
    return v


def emergency_test_panel_page():
    st.title("Emergency Test Panel")
    st.caption("Use this page to inject emergency scenarios into the live simulation.")

    pids = list(st.session_state.patients.keys())
    test_pid = st.selectbox(
        "Target patient",
        pids,
        format_func=lambda pid: f"{st.session_state.patients[pid]['name']} ({pid})"
    )

    duration = st.slider("Duration (seconds)", 5, 300, 30, step=5)

    presets = {
        "Hypertensive crisis (stroke-like)": {"bp_sys": 230, "bp_dia": 120, "rr": 22, "hr": 115, "temp": 37.0},
        "Sepsis-like": {"temp": 39.4, "rr": 26, "hr": 135, "bp_sys": 95, "bp_dia": 60},
        "Respiratory failure": {"rr": 30, "hr": 120, "temp": 37.2, "bp_sys": 115, "bp_dia": 75},
        "Bradycardia collapse": {"hr": 38, "rr": 10, "temp": 36.5, "bp_sys": 85, "bp_dia": 55},
        "Return to normal": None,
    }

    preset_name = st.selectbox("Preset scenario", list(presets.keys()))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply", use_container_width=True):
            if presets[preset_name] is None:
                st.session_state.scenario.pop(test_pid, None)
            else:
                st.session_state.scenario[test_pid] = {
                    "name": preset_name,
                    "until": time.time() + duration,
                    "overrides": presets[preset_name],
                }
            st.success("Applied.")

    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.scenario.pop(test_pid, None)
            st.success("Cleared.")

    active = st.session_state.scenario.get(test_pid)
    if active:
        remaining = int(active["until"] - time.time()) if active.get("until") else None
        st.info(
            f"Active: {active['name']} | Remaining: {remaining}s"
            if remaining is not None
            else f"Active: {active['name']}"
        )
    else:
        st.caption("No active scenario for selected patient.")


# Simulation
def get_live_vitals(patient_id):
    p = st.session_state.patients[patient_id]
    v = p["vitals"]

    sim = st.session_state.sim_data.get(patient_id)
    idx = st.session_state.sim_idx.get(patient_id, 0)

    if not sim:
        return v

    hr_arr     = sim.get("HeartRate_250Hz")
    temp_arr   = sim.get("Temp")
    rr_arr     = sim.get("RespRate")
    bp_sys_arr = sim.get("BP_sys")
    bp_dia_arr = sim.get("BP_dia")

    if hr_arr is None or len(hr_arr) == 0:
        return v

    n = len(hr_arr)
    i = idx % n

    v.hr = float(hr_arr[i])

    if temp_arr is not None and len(temp_arr) > i:
        v.temp = float(temp_arr[i])
    if rr_arr is not None and len(rr_arr) > i:
        v.rr = float(rr_arr[i])
    if bp_sys_arr is not None and len(bp_sys_arr) > i:
        v.bp_sys = float(bp_sys_arr[i])
    if bp_dia_arr is not None and len(bp_dia_arr) > i:
        v.bp_dia = float(bp_dia_arr[i])

    st.session_state.sim_idx[patient_id] = (idx + STEP_SAMPLES) % n
    v = apply_scenario_override(patient_id, v)

    return v


def update_all_patients():
    for pid in st.session_state.patients.keys():
        get_live_vitals(pid)      
        maybe_log_minute(pid)   


def update_daily_report(pid: str, now: datetime) -> None:
    """
    Append an event to per-patient daily report if:
      - total_score >= 5 OR any single parameter score == 3
    """

    p = st.session_state.patients[pid]
    v = p["vitals"]

    # Build a per-minute unique key (avoid duplicates due to rerun)
    minute_key = now.strftime("%Y-%m-%d %H:%M")
    if p.get("last_report_key") == minute_key:
        return
    p["last_report_key"] = minute_key

    # NEWS2 scoring from your embedded R_alart functions
    news = getAllscore(v)  # includes subscores + total_score + risk_level + trigger
    total = int(news.get("total_score", 0))
    subs = news.get("subscores", {}) or {}

    single_param_3 = any(int(s) == 3 for s in subs.values())

    # Only record if meets your criteria
    if not (total >= 5 or single_param_3):
        return

    day_key = now.strftime("%Y-%m-%d")
    if "daily_reports" not in p:
        p["daily_reports"] = {}

    if day_key not in p["daily_reports"]:
        p["daily_reports"][day_key] = []

    event = {
        "time": minute_key,
        "total_score": total,
        "risk_level": news.get("risk_level"),
        "trigger": news.get("trigger"),
        "subscores": subs,  # rr/hr/bp_sys/temp
        # Optional: snapshot of vitals at that minute (useful for clinical context)
        "vitals": {
            "hr": float(getattr(v, "hr", 0)),
            "rr": float(getattr(v, "rr", 0)),
            "temp": float(getattr(v, "temp", 0)),
            "bp_sys": float(getattr(v, "bp_sys", 0)),
        }
    }

    p["daily_reports"][day_key].append(event)

    # Optional retention (keep last 7 days only)
    keys = sorted(p["daily_reports"].keys())
    if len(keys) > 7:
        for k in keys[:-7]:
            del p["daily_reports"][k]


def maybe_log_minute(pid):
    p = st.session_state.patients[pid]
    v = p["vitals"]
    now = datetime.now()

    p["hr_history"].append((now.strftime("%H:%M:%S"), int(v.hr)))
    p["hr_history"] = p["hr_history"][-3600:] 

    if p["last_log_min"] != now.minute:
        p["last_log_min"] = now.minute

        sim = st.session_state.sim_data.get(pid)
        idx = st.session_state.sim_idx.get(pid, 0)

        ecg_wave = []
        if sim and "ECG_single_250Hz" in sim:
            fs = 250  # Hz
            window_seconds = 4.0   
            window_n = int(window_seconds * fs)

            start = max(0, idx - window_n)
            wave = sim["ECG_single_250Hz"][start:idx]
            ecg_wave = wave.tolist() 

        entry = {
            "t": now.strftime("%Y-%m-%d %H:%M"),
            "hr": int(v.hr),
            "temp": round(v.temp, 1),
            "rr": int(v.rr),
            "bp_sys": int(v.bp_sys),
            "bp_dia": int(v.bp_dia),
            "ecg": ecg_wave,  
        }

        p["minute_logs"].append(entry)
        p["minute_logs"] = p["minute_logs"][-10:]  
        update_daily_report(pid, now)


def render_hr_trend(p):
    if not p["hr_history"]:
        st.info("No HR data yet.")
        return

    rows = p["hr_history"]
    times, hrs = [], []
    today_str = date.today().strftime("%Y-%m-%d")
    for t, h in rows:
        if isinstance(t, datetime):
            times.append(t)
        else:
            times.append(pd.to_datetime(f"{today_str} {t}"))
        hrs.append(float(h))

    df = pd.DataFrame({"Time": pd.to_datetime(times), "HR": hrs}).sort_values("Time")

    now_floor = pd.Timestamp.now().floor("min")
    window_start = now_floor - pd.Timedelta(minutes=30)

    df = df[(df["Time"] >= window_start) & (df["Time"] <= now_floor)]

    if df.empty:
        st.info("No data in the last 30 minutes yet.")
        return

    base = alt.Chart(df).encode(
        x=alt.X(
            "Time:T",
            axis=alt.Axis(format="%H:%M", title=None),
            scale=alt.Scale(domain=[window_start, now_floor])  # fixed 30-min domain
        ),
        y=alt.Y("HR:Q", title="Heart Rate (bpm)", scale=alt.Scale(domain=[0, 200]))
    ).properties(height=220)

    # Hover-only tooltips, small filled dot
    hover = alt.selection_point(fields=["Time"], on="mouseover", nearest=True, empty="none")

    line = base.mark_line(interpolate="monotone", strokeWidth=2)
    selectors = base.mark_point(opacity=0).add_params(hover)
    hover_dots = base.mark_point(size=24, filled=True).transform_filter(hover).encode(
        tooltip=[
            alt.Tooltip("Time:T", title="Time", format="%H:%M:%S"),
            alt.Tooltip("HR:Q",   title="HR (bpm)", format=".1f"),
        ]
    )
    vrule = alt.Chart(df).mark_rule(strokeDash=[4,3]).encode(x="Time:T").transform_filter(hover)

    st.altair_chart((line + selectors + vrule + hover_dots).interactive(), use_container_width=True)


def eod_summary(pid):
    p = st.session_state.patients[pid]
    logs = p["minute_logs"]
    hrs = [float(x["hr"]) for x in logs] if logs else []
    return {
        "patient": p["name"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "minHR": min(hrs) if hrs else None,
        "maxHR": max(hrs) if hrs else None,
        "avgHR": round(sum(hrs)/len(hrs), 1) if hrs else None,
        "totalLogs": len(logs),
        "ecgSamples": len(p.get("ecg_samples", []))
    }


def render_ecg_sparkline(pid: str):
    sim = st.session_state.sim_data.get(pid)
    if not sim or "ECG_single_250Hz" not in sim:
        st.caption("ECG loading…")
        return

    arr = sim["ECG_single_250Hz"]
    idx = st.session_state.sim_idx.get(pid, 0)
    fs = 250
    window_n = 300  # ~1.2 seconds of ECG at 250 Hz

    if idx == 0:
        st.caption("ECG loading…")
        return

    start = max(0, idx - window_n)
    wave = arr[start:idx]

    df = pd.DataFrame({
        "i": np.arange(len(wave)),
        "ecg": wave,
    })

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("i:Q", title=None, axis=None),
            y=alt.Y("ecg:Q", title=None),
        )
        .properties(height=60)
    )
    st.altair_chart(chart, use_container_width=True)


def render_single_ecg(pid: str, window_seconds: float = 60.0):
    sim = st.session_state.sim_data.get(pid)
    if not sim or "ECG_single_250Hz" not in sim:
        st.caption("No simulated single-lead ECG available.")
        return

    arr = sim["ECG_single_250Hz"]
    fs = FS  # 250 Hz

    idx = st.session_state.sim_idx.get(pid, 0)
    if idx == 0:
        st.caption("Waiting for ECG history…")
        return

    window_n = int(window_seconds * fs)
    start = max(0, idx - window_n)

    wave = arr[start:idx]
    if len(wave) == 0:
        st.caption("No ECG samples in window.")
        return

    # time axis in seconds, aligned to simulation time
    df = pd.DataFrame({
        "t": np.arange(start, start + len(wave)) / fs,
        "ecg": wave,
    })

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Time (s)"),
            y=alt.Y("ecg:Q", title="ECG (a.u.)"),
        )
        .properties(height=160)
    )

    st.altair_chart(chart, use_container_width=True)


def render_12lead_ecg(pid: str, selected_leads: list[str], window_seconds: float = 10.0):
    sim = st.session_state.sim_data.get(pid)
    if not sim:
        st.caption("No simulated 12-lead ECG available.")
        return

    available = [lead for lead in selected_leads if lead in sim]
    if not available:
        st.caption("Selected leads are not available in simulation data.")
        return

    fs = 250
    idx = st.session_state.sim_idx.get(pid, 0)
    if idx == 0:
        st.caption("Waiting for ECG history…")
        return

    window_n = int(window_seconds * fs)
    start = max(0, idx - window_n)

    rows = []
    for lead in available:
        wave = sim[lead][start:idx]
        for i, v in enumerate(wave):
            rows.append({
                "t": (start + i) / fs,
                "ecg": float(v),
                "lead": lead,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        st.caption("No ECG samples in window.")
        return

    if len(available) == 1:
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("t:Q", title="Time (s)"),
                y=alt.Y("ecg:Q", title="ECG (a.u.)"),
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)
        return

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Time (s)"),
            y=alt.Y("ecg:Q", title=None),
            facet=alt.Facet("lead:N", columns=1),
        )
        .properties(height=90)
    )

    st.altair_chart(chart, use_container_width=True)


def navbar():
    left, right = st.columns([4,1])

    with left:
        st.markdown("### Remote Monitor")
        st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with right:
  
        st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

        pids = list(st.session_state.patients.keys())
        if "active" not in st.session_state:
            st.session_state.active = pids[0]
        idx = pids.index(st.session_state.active)

        selected_pid = st.selectbox(
            "View detail for:",
            pids,
            index=idx,
            format_func=lambda pid: f"{st.session_state.patients[pid]['name']} ({pid})"
        )
        st.session_state.active = selected_pid

# Grid Page
def render_grid():

    cols = st.columns(5)
    i = 0
    for pid, p in st.session_state.patients.items():
        v = st.session_state.patients[pid]["vitals"]

        # per-vital levels
        lv = {
            "hr": vital_level(v.hr, DEFAULT_RANGES["hr"]),
            "temp": vital_level(v.temp, DEFAULT_RANGES["temp"]),
            "rr": vital_level(v.rr, DEFAULT_RANGES["rr"]),
            "bp_sys": vital_level(v.bp_sys, DEFAULT_RANGES["bp_sys"]),
            "bp_dia": vital_level(v.bp_dia, DEFAULT_RANGES["bp_dia"]),
        }
        ov = overall_level(lv)  # overall card level

        with cols[i % 5]:
            st.markdown("<div class='rm-card'>", unsafe_allow_html=True)

            head_cls = (
                "rm-id-header rm-head-danger"
                if ov == "severe"
                else "rm-id-header rm-head-normal"
            )

            st.markdown(
                f"""
                <div class="{head_cls}">
                    <div class="rm-id-left" style="width:100%;">
                        <div class="rm-avatar">{initials(p['name'])}</div>
                        <div class="rm-header-left">
                            <div class="rm-name">{p['name']}</div>
                            <div class="rm-sub">ID: {pid}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            #Body
            st.markdown("<div class='rm-id-body'>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(
                    f"<div class='rm-v'><div class='lab'>HR</div>"
                    f"<div class='val {val_class(lv['hr'])}'>{int(v.hr)} bpm</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rm-v'><div class='lab'>Temp</div>"
                    f"<div class='val {val_class(lv['temp'])}'>{v.temp:.1f} °C</div></div>",
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    f"<div class='rm-v'><div class='lab'>RR</div>"
                    f"<div class='val {val_class(lv['rr'])}'>{int(v.rr)} /min</div></div>",
                    unsafe_allow_html=True,
                )
                bp_lvl = (
                    lv["bp_sys"]
                    if severity_rank(lv["bp_sys"])
                    >= severity_rank(lv["bp_dia"])
                    else lv["bp_dia"]
                )
                st.markdown(
                    f"<div class='rm-v'><div class='lab'>BP</div>"
                    f"<div class='val {val_class(bp_lvl)}'>{int(v.bp_sys)}/{int(v.bp_dia)} mmHg</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div style='margin-top:-6px;'>"
                "<div style='font-size:11px; opacity:.65; margin-bottom:2px;'>ECG</div>",
                unsafe_allow_html=True,
            )
            render_ecg_sparkline(pid)
            st.markdown("</div>", unsafe_allow_html=True)  # close ECG wrapper
            st.markdown("</div>", unsafe_allow_html=True)  # end body
            st.markdown("</div>", unsafe_allow_html=True)  # end card

        i += 1

# Detail Page
def render_detail(pid: str):
    p = st.session_state.patients[pid]
    v = st.session_state.patients[pid]["vitals"]


    lv = {
        "hr": vital_level(v.hr, DEFAULT_RANGES["hr"]),
        "temp": vital_level(v.temp, DEFAULT_RANGES["temp"]),
        "rr": vital_level(v.rr, DEFAULT_RANGES["rr"]),
        "bp_sys": vital_level(v.bp_sys, DEFAULT_RANGES["bp_sys"]),
        "bp_dia": vital_level(v.bp_dia, DEFAULT_RANGES["bp_dia"]),
    }
    ov = overall_level(lv)

    head_cls = "rm-id-header rm-head-danger" if ov == "severe" else "rm-id-header rm-head-normal"

    st.markdown(
        f"""
        <div class="rm-card" style="margin-bottom:10px">
        <div class="{head_cls}">
            <div class="rm-id-left">
            <div class="rm-avatar">{initials(p['name'])}</div>
            <div>
                <div class="rm-name">{p['name']}</div>
                <div class="rm-sub">ID: {pid}</div>
            </div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
)

    #st.markdown(f"### {p['name']} — Detail")

    # Banner
    news = getAllscore(v)  # uses rr/hr/bp_sys/temp only
    total = int(news.get("total_score", 0))
    subs = news.get("subscores", {}) or {}
    single_param_3 = any(int(s) == 3 for s in subs.values())

    if total >= 7:
        st.error(
        f"Total score ≥ 7 (emergency response threshold): Activate an emergency response with continuous "
        f"vital-sign monitoring and consider transfer to HDU or ICU."
        f"\n\n(NEWS2 total = {total}; subscores: {subs})"
    )
    elif total >= 5:
        st.error(
            f"Total score ≥ 5 (urgent response threshold): Immediately inform the medical team and request urgent "
            f"clinical assessment in a monitored care environment."
            f"\n\n(NEWS2 total = {total}; subscores: {subs})"
        )
    elif single_param_3:
        st.warning(
            f"Single parameter score = 3: The registered nurse must inform the medical team to review the patient "
            f"and consider escalation of care."
            f"\n\n(NEWS2 total = {total}; subscores: {subs})"
        )
    elif 1 <= total <= 4:
        st.info(
            f"Total score 1–4: Inform the registered nurse to assess the patient and decide whether increased "
            f"monitoring or escalation of care is required."
            f"\n\n(NEWS2 total = {total}; subscores: {subs})"
        )
    else:  # total == 0
        st.success(
            f"Total score = 0: Continue routine monitoring."
            f"\n\n(NEWS2 total = {total}; subscores: {subs})"
        )


    # Vitals now
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("HR (bpm)", f"{int(v.hr)}")
    c2.metric("RR (/min)", f"{int(v.rr)}")
    c3.metric("Temp (°C)", f"{v.temp:.1f}")
    c4.metric("BP Sys (mmHg)", f"{int(v.bp_sys)}")
    c5.metric("BP Dia (mmHg)", f"{int(v.bp_dia)}")

    
    # History + logs + report
    a, b = st.columns([2, 1])

    with a:
        st.markdown("**Heart Rate History (last 30 min)**")
        if p["hr_history"]:
            render_hr_trend(p)
        else:
            st.caption("(history will populate over time)")
        
        st.markdown("**Simulated Single-lead ECG (ECG_single_250Hz)**")
        render_single_ecg(pid, window_seconds=60.0)

        st.markdown("**Simulated 12-lead ECG history (select leads)**")
        all_leads = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]


        sel_key = f"ecg_leads_{pid}"
        if sel_key not in st.session_state:
            st.session_state[sel_key] = []  # default lead

        selected_leads = st.multiselect(
            "Leads to display",
            options=all_leads,
            default=st.session_state[sel_key],
            key=sel_key,
        )
        render_12lead_ecg(pid, selected_leads=selected_leads, window_seconds=60.0)

    with b:
        logs = p["minute_logs"]

        if not logs:
            st.markdown("**Recent Logs**")
            st.caption("(logs appear each minute)")
            return

        # find log at specific time (same day only)
        all_dates = {row["t"].split()[0] for row in logs}
        latest_date_str = max(all_dates)  # most recent date as string, e.g. "2025-11-13"

        logs_same_date = [row for row in logs if row["t"].startswith(latest_date_str)]

        if not logs_same_date:
            st.caption(f"No logs for {latest_date_str}.")
            return

        st.markdown(f"**Recent Logs on {latest_date_str} (latest 3)**")

        recent = list(reversed(logs_same_date))[:3]

        for idx, row in enumerate(recent):
            st.markdown(
                f"**{row['t']}**  "
                f"HR {row['hr']} | RR {row['rr']} | "
                f"T {row['temp']}°C | BP {row['bp_sys']}/{row['bp_dia']}"
            )

            wave = row.get("ecg")
            if wave:
                df_ecg = pd.DataFrame({
                    "i": list(range(len(wave))),
                    "ecg": wave,
                })
                ecg_chart = (
                    alt.Chart(df_ecg)
                    .mark_line()
                    .encode(
                        x=alt.X("i:Q", title=None, axis=None),
                        y=alt.Y(
                            "ecg:Q",
                            title=None,
                            axis=None,
                            scale=alt.Scale(domain=[-1.5, 2.0]),
                        ),
                    )
                    .properties(height=80)
                )
                st.altair_chart(ecg_chart, use_container_width=True)
            else:
                st.caption("No ECG snapshot stored for this log.")

            if idx < len(recent) - 1:
                st.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(f"**View other logs on {latest_date_str}**")

        hour_options = sorted({row["t"][11:13] for row in logs_same_date})

        col_h, col_m = st.columns(2)
        with col_h:
            selected_hour = st.selectbox(
                "Hour",
                hour_options,
                key=f"hour_selector_{pid}",
            )

        minute_options = sorted({
            row["t"][14:16]
            for row in logs_same_date
            if row["t"].startswith(f"{latest_date_str} {selected_hour}:")
        })

        with col_m:
            selected_minute = st.selectbox(
                "Minute",
                minute_options,
                key=f"minute_selector_{pid}",
            )

        target_ts = f"{latest_date_str} {selected_hour}:{selected_minute}"
        selected_log = next((row for row in logs_same_date if row["t"] == target_ts), None)

        if selected_log:
            st.markdown(
                f"**Log at {selected_log['t']}**  "
                f"HR {selected_log['hr']} | RR {selected_log['rr']} | "
                f"T {selected_log['temp']}°C | BP {selected_log['bp_sys']}/{selected_log['bp_dia']}"
            )

            wave = selected_log.get("ecg")
            if wave:
                df_ecg_sel = pd.DataFrame({
                    "i": list(range(len(wave))),
                    "ecg": wave,
                })
                ecg_chart_sel = (
                    alt.Chart(df_ecg_sel)
                    .mark_line()
                    .encode(
                        x=alt.X("i:Q", title=None, axis=None),
                        y=alt.Y(
                            "ecg:Q",
                            title=None,
                            axis=None,
                            scale=alt.Scale(domain=[-1.5, 2.0]),
                        ),
                    )
                    .properties(height=120)
                )
                st.altair_chart(ecg_chart_sel, use_container_width=True)
            else:
                st.caption("No ECG snapshot stored for this log.")
        else:
            st.info("No log found for that time.")


    summary = eod_summary(pid)
    d1, d2 = st.columns(2)
    with d1:
        json_payload = {
            "summary": summary,
            "logs": p["minute_logs"],
            "ecg_samples": p["ecg_samples"],  
        }
        st.download_button(
            "Download EOD JSON",
            data=json.dumps(json_payload, indent=2),
            file_name=f"{p['name'].replace(' ','_')}_EOD_{summary['date']}.json",
            mime="application/json",
            key=f"json_{pid}",
        )
    with d2:
        if p["minute_logs"]:
            lines = ["time,hr,rr,temp,bp_sys,bp_dia"]
            for row in p["minute_logs"]:
                lines.append(f"{row['t']},{row['hr']},{row['rr']},{row['temp']},{row['bp_sys']},{row['bp_dia']}")
            csv_data = "\n".join(lines)
        else:
            csv_data = "time,hr,rr,temp,bp_sys,bp_dia"
        st.download_button(
            "Download Logs CSV",
            data=csv_data,
            file_name=f"{p['name'].replace(' ','_')}_logs_{summary['date']}.csv",
            mime="text/csv",
            key=f"csv_{pid}",
        )


    # Daily Report
    # st.session_state.patients[pid]["daily_reports"][YYYY-MM-DD] = [event...]
    today_key = datetime.now().strftime("%Y-%m-%d")
    events_today = (p.get("daily_reports", {}) or {}).get(today_key, [])

    st.markdown("---")
    st.markdown(f"**Daily Report (Events) — {today_key}**")

    if not events_today:
        st.caption("No events recorded today (no Total score ≥ 5 or single-parameter score = 3).")
    else:
        # show latest first
        for ev in reversed(events_today[-20:]):
            subs_ev = ev.get("subscores", {}) or {}
            st.markdown(
                f"**{ev.get('time','')}**  |  "
                f"Total: **{ev.get('total_score','')}**  |  "
                f"RR={subs_ev.get('rr')}, HR={subs_ev.get('hr')}, "
                f"BPsys={subs_ev.get('bp_sys')}, Temp={subs_ev.get('temp')}  |  "
                f"Trigger: {ev.get('trigger')}"
            )

        report_payload = {
            "patient_id": pid,
            "patient_name": p.get("name"),
            "date": today_key,
            "events": events_today,
        }

        st.download_button(
            "Download Daily Report (JSON)",
            data=json.dumps(report_payload, indent=2),
            file_name=f"{p.get('name','patient').replace(' ','_')}_DailyReport_{today_key}.json",
            mime="application/json",
            key=f"daily_json_{pid}",
        )

        lines = ["time,total_score,risk_level,trigger,rr,hr,bp_sys,temp"]
        for ev in events_today:
            subs_ev = ev.get("subscores", {}) or {}
            lines.append(
                f"{ev.get('time','')},"
                f"{ev.get('total_score','')},"
                f"{ev.get('risk_level','')},"
                f"{ev.get('trigger','')},"
                f"{subs_ev.get('rr','')},"
                f"{subs_ev.get('hr','')},"
                f"{subs_ev.get('bp_sys','')},"
                f"{subs_ev.get('temp','')}"
            )

        st.download_button(
            "Download Daily Report (CSV)",
            data="\n".join(lines),
            file_name=f"{p.get('name','patient').replace(' ','_')}_DailyReport_{today_key}.csv",
            mime="text/csv",
            key=f"daily_csv_{pid}",
        )

mode = get_mode()
if mode == "test":
    emergency_test_panel_page()
    st.stop()

nav_ph = st.container()
with nav_ph:
    navbar()

audio_ph = st.container()
grid_ph = st.container()
detail_ph = st.container()

update_all_patients()


updateAlerts(
    st.session_state.patients,
    audio_ph
)

with grid_ph:
    render_grid()

with detail_ph:
    active = st.session_state.active or "p01"
    render_detail(active)

# auto refresh
time.sleep(1)
if hasattr(st, "rerun"):
    st.rerun()
elif hasattr(st, "experimental_rerun"):
    st.experimental_rerun()
