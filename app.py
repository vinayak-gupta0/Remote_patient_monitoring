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

from alart import updateAlerts

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
            "vitals": Vitals(hr=72+random.randint(-2,2), temp=36.8, rr=14, bp_sys=118, bp_dia=78),
            "hr_history": [], # for history
            "minute_logs": [], # for history
            "last_log_min": None,
            "ecg_live": [],          
            "ecg_phase": 0.0,        
            "ecg_samples": [],     
            "last_ecg_min": None,
        }

# Sample Simulation
def get_live_vitals(patient_id) :
    p = st.session_state.patients[patient_id]
    v = p["vitals"]
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    v.hr     = clamp(v.hr     + (random.random()-0.5)*2.0, 35, 200)
    v.temp   = clamp(v.temp   + (random.random()-0.5)*0.03, 34.5, 40.5)
    v.rr     = clamp(v.rr     + (random.random()-0.5)*0.6, 6, 40)
    v.bp_sys = clamp(v.bp_sys + (random.random()-0.5)*2.0, 80, 200)
    v.bp_dia = clamp(v.bp_dia + (random.random()-0.5)*1.2, 50, 120)
    return v

def update_all_patients():
    """Update vitals + logs for all patients once per rerun."""
    for pid, p in st.session_state.patients.items():
        get_live_vitals(pid)
        maybe_log_minute(pid)
        update_ecg_for_patient(p)
        maybe_sample_ecg(pid)



def maybe_log_minute(pid):
    p = st.session_state.patients[pid]
    v = p["vitals"]
    now = datetime.now()
    #HR history
    p["hr_history"].append((now.strftime("%H:%M:%S"), int(v.hr)))
    p["hr_history"] = p["hr_history"][-3600:] # Keep only the last 120 entries
    # minute logs
    if p["last_log_min"] != now.minute:
        p["last_log_min"] = now.minute
        ecg_wave = p["ecg_live"][-300:].copy()
        entry = {
            "t": now.strftime("%Y-%m-%d %H:%M"),
            "hr": int(v.hr),
            "temp": round(v.temp,1),
            "rr": int(v.rr),
            "bp_sys": int(v.bp_sys),
            "bp_dia": int(v.bp_dia),
            "ecg": ecg_wave, 
        }
        p["minute_logs"].append(entry)

        p["ecg_samples"].append({
            "t": entry["t"],
            "wave": ecg_wave,
        })
        p["minute_logs"] = p["minute_logs"][-10:]
        p["ecg_samples"] = p["ecg_samples"][-10:]


def render_hr_trend(p):
    if not p["hr_history"]:
        st.info("No HR data yet.")
        return

    # Build dataframe from history (accept datetime or time-string)
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

    # Fixed 30-min window anchored to current minute
    now_floor = pd.Timestamp.now().floor("min")
    window_start = now_floor - pd.Timedelta(minutes=30)

    # Keep only points inside window
    df = df[(df["Time"] >= window_start) & (df["Time"] <= now_floor)]

    if df.empty:
        st.info("No data in the last 30 minutes yet.")
        return

    # Altair chart: fixed axes
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

def update_ecg_for_patient(p, points_per_tick=30):
    phase = p["ecg_phase"]
    ecg = p["ecg_live"]

    for i in range(points_per_tick):
        t = phase + i * 0.02  
        baseline = 0.1 * math.sin(2 * math.pi * 1.0 * t)
        x = (t % 1.0) - 0.1
        qrs = math.exp(-(x * x) / 0.0005) * 1.2
        value = baseline + qrs
        ecg.append(value)

    # keep last ~300 points for the live trace
    p["ecg_live"] = ecg[-300:]
    p["ecg_phase"] = phase + points_per_tick * 0.02

def maybe_sample_ecg(pid):
    p = st.session_state.patients[pid]
    now = datetime.now()

    # once per minute
    if p["last_ecg_min"] != now.minute:
        p["last_ecg_min"] = now.minute
        wave = p["ecg_live"][-300:]  # same as previous one
        p["ecg_samples"].append({
            "t": now.strftime("%Y-%m-%d %H:%M"),
            "wave": wave,
        })

def render_ecg_sparkline(p):
    ecg = p["ecg_live"]
    if not ecg:
        st.caption("ECG loading…")
        return
    df = pd.DataFrame({
        "i": list(range(len(ecg))),
        "ecg": ecg,
    })
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("i:Q", title=None, axis=None),
            y=alt.Y("ecg:Q", title=None, scale=alt.Scale(domain=[-1.5, 2.0])),
        )
        .properties(height=60)
    )
    st.altair_chart(chart, use_container_width=True)

def render_ecg_history_10min(p):
    samples = p.get("ecg_samples", [])
    if not samples:
        st.caption("ECG history will appear after a few minutes.")
        return

    # take up to 10 latest snapshots (≈ last 10 minutes)
    recent = samples[-10:]

    rows = []
    x_idx = 0
    for snap in recent:
        wave = snap.get("wave", [])
        for v in wave:
            rows.append({"x": x_idx, "ecg": v})
            x_idx += 1

    if not rows:
        st.caption("ECG history will appear after a few minutes.")
        return

    df_hist = pd.DataFrame(rows)

    chart = (
        alt.Chart(df_hist)
        .mark_line()
        .encode(
            x=alt.X("x:Q", title=None, axis=None),
            y=alt.Y(
                "ecg:Q",
                title=None,
                axis=None,
                scale=alt.Scale(domain=[-1.5, 2.0]),
            ),
        )
        .properties(height=140)
    )

    st.altair_chart(chart, use_container_width=True)
    st.caption("ECG snapshots over the last ~10 minutes")


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
            render_ecg_sparkline(p)
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
    if ov == "severe":
        st.error("CRITICAL deviation detected")
    elif ov == "moderate":
        st.warning("Moderate deviation detected")
    else:
        st.success("All vitals within range")

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
        
        st.markdown("**ECG History (last 10 min)**")
        render_ecg_history_10min(p)

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


nav_ph = st.container()
with nav_ph:
    navbar()

update_all_patients()
updateAlerts(st.session_state.patients,vital_level,DEFAULT_RANGES)
grid_ph = st.container()
detail_ph = st.container()

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

