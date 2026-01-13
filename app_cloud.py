"""
Remote Monitor (for Tsuru deployment)

What this app does:
- Simulates vitals (HR, Temp, RR, BP) for 5 patients.
- Occasionally injects “shock” events that push 1–2 vitals to extreme values.
- Synthesizes a lightweight ECG waveform per patient (plus occasional artifacts).
- Renders:
  1) A 5-card grid overview (each card shows vitals + ECG status)
  2) A detail view for the selected patient (live ECG, HR trend, recent logs, download)

What really makes it faster: 
1) Reduced rerun frequency to a realistic REFRESH_MS for low-CPU allotment.
2) Kept all state in st.session_state so we do incremental updates instead of re-generating everything.
3) Limited history sizes:
   - hr_history is capped (rolling window) instead of growing forever.
    - minute_logs capped to last N entries.
    - ecg_samples capped to last N snapshots.
 4) Made ECG cheaper:
    - Low ECG sampling rate (ECG_FS) suitable for cloud.
   - Keep only a short live buffer (ECG_WINDOW_SEC) instead of large arrays.
   - Store minute snapshots as a downsampled fixed length (ECG_SNAPSHOT_POINTS).
    - Render live ECG only for the currently selected patient (avoids 5 charts every rerun).
5) Avoided expensive conversions on every rerun:
    - Store datetime objects directly in hr_history rather than strings that need parsing later.
 6) Avoided heavy dependencies and features that increased memory/CPU (removed CSV download; kept JSON only).
 7) Kept UI lightweight:
    - “Recent logs” display uses small HTML cards (no ECG charts inside logs).
    - Charts are used only where they add value (HR trend + selected patient ECG).


"""

import random
from dataclasses import dataclass
from typing import Dict
from datetime import datetime
import json
import streamlit as st
import pandas as pd
import altair as alt
import math
from streamlit_autorefresh import st_autorefresh


# -----------------------------------------------------------------------------
# Refresh / "real-time" loop
# -----------------------------------------------------------------------------
# Streamlit reruns the script top-to-bottom each refresh.
# This autorefresh triggers that rerun every REFRESH_MS milliseconds.
REFRESH_MS = 4000
st_autorefresh(interval=REFRESH_MS, key="refresh")


# -----------------------------------------------------------------------------
# Extreme event ("shock") simulation config
# -----------------------------------------------------------------------------
# Probability per refresh tick that a patient starts a shock episode.
EXTREME_EVENT_PROB_PER_TICK = 0.003

# Shock length in seconds.
EXTREME_EVENT_DURATION_SEC_RANGE = (45, 180)

# If a shock is triggered, chance that a second vital is also forced extreme.
EXTREME_SECOND_VITAL_PROB = 0.35

# Pulls vitals back toward baseline each tick (simple mean reversion).
MEAN_REVERT_STRENGTH = 0.03

# For each vital: (low extreme range), (high extreme range)
EXTREME_TARGETS = {
    "hr": ((30, 45), (170, 200)),
    "temp": ((34.5, 35.6), (39.2, 40.5)),
    "rr": ((6, 9), (28, 40)),
    "bp_sys": ((80, 92), (175, 200)),
    "bp_dia": ((50, 62), (105, 120)),
}


# -----------------------------------------------------------------------------
# ECG synthesis config
# -----------------------------------------------------------------------------
# Low sampling rate to keep CPU light. Each refresh adds points_per_tick samples.
ECG_FS = 30  # samples per second

# Keep only the last ECG_WINDOW_SEC seconds in memory/display.
ECG_WINDOW_SEC = 6
ECG_LIVE_MAXLEN = int(ECG_FS * ECG_WINDOW_SEC)

# When logging once per minute, store a small downsampled ECG snapshot.
ECG_SNAPSHOT_POINTS = 90

# Thresholds used to label ECG status based on min/max amplitude in the window.
ECG_OK_MIN, ECG_OK_MAX = -0.8, 1.6
ECG_MOD_MIN, ECG_MOD_MAX = -1.0, 2.0

# Random artifact bursts: inject higher noise for a short number of samples.
ECG_ARTIFACT_PROB_PER_TICK = 0.01
ECG_ARTIFACT_SAMPLES_RANGE = (30, 120)
ECG_ARTIFACT_NOISE_AMPLITUDE = 0.30


# -----------------------------------------------------------------------------
# Small data structures
# -----------------------------------------------------------------------------
@dataclass
class Ranges:
    """
    Range definition for each vital.
    - min/max define the OK band
    - moderate_band expands the "severe" thresholds outside min/max
    """
    min: float
    max: float
    moderate_band: float


@dataclass
class Vitals:
    """
    Current vital values for a patient.
    These are mutated in-place by the simulation.
    """
    hr: float
    temp: float
    rr: float
    bp_sys: float
    bp_dia: float


# -----------------------------------------------------------------------------
# Streamlit page setup + CSS (UI styling)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Remote Monitor", layout="wide")

# Custom CSS to style patient cards, headers, and log cards.
# This is purely presentation.
st.markdown(
    """
<style>
[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 0.5rem; }

.rm-card {
  border-radius:18px; padding:0; overflow:hidden; background:#ffffff;
  border:1px solid rgba(0,0,0,0.08);
  transition: box-shadow .15s ease, transform .05s ease;
}
.rm-card:hover { box-shadow:0 8px 20px rgba(2,132,199,.12), 0 2px 8px rgba(2,132,199,.08); }

.rm-id-header {
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 12px; color:white;
}
.rm-head-normal { background:linear-gradient(90deg,#0ea5e9 0%, #38bdf8 100%); }
.rm-head-danger { background:linear-gradient(90deg,#ff0000 0%, #f87171 100%); }

.rm-id-left { display:flex; align-items:center; gap:10px; }
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

.rm-mini { font-size:11px; opacity:.75; margin-top:6px; }
.rm-mini-ok { color:#065f46; font-weight:700; }
.rm-mini-mod { color:#ffb400; font-weight:700; }
.rm-mini-sev { color:#ff0000; font-weight:800; }

/* Recent logs cards */
.log-wrap { margin-top: 6px; }
.log-card {
  border:1px solid rgba(0,0,0,0.08);
  background:#ffffff;
  border-radius:14px;
  padding:10px 12px;
  margin:10px 0;
  color:#0f172a;
}
.log-card * { color:#0f172a; }

.log-top {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom:8px;
}
.log-time { font-weight:900; font-size:13px; color:#0f172a; }

.log-level-ok { color:#065f46 !important; font-weight:900; }
.log-level-mod { color:#ffb400 !important; font-weight:900; }
.log-level-sev { color:#ff0000 !important; font-weight:900; }

.pills { display:flex; flex-wrap:wrap; gap:8px; }
.pill {
  background:#f8fafc;
  border:1px solid rgba(0,0,0,0.06);
  border-radius:999px;
  padding:6px 10px;
  font-size:12px;
  color:#0f172a;
}
.pill b { font-weight:900; color:#0f172a; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# "Normal" vital ranges used for status labeling
# -----------------------------------------------------------------------------
DEFAULT_RANGES: Dict[str, Ranges] = {
    "hr": Ranges(50, 110, 10),
    "temp": Ranges(36.0, 37.8, 0.3),
    "rr": Ranges(10, 20, 2),
    "bp_sys": Ranges(90, 140, 10),
    "bp_dia": Ranges(60, 90, 5),
}


# -----------------------------------------------------------------------------
# Utility functions for status labels / CSS class selection
# -----------------------------------------------------------------------------
def vital_level(value, r: Ranges) -> str:
    """
    Returns "ok" / "moderate" / "severe" based on a value and its range.
    - OK: within [min, max]
    - Moderate: slightly outside min/max
    - Severe: outside (min - band) or (max + band)
    """
    if value < r.min - r.moderate_band or value > r.max + r.moderate_band:
        return "severe"
    if value < r.min or value > r.max:
        return "moderate"
    return "ok"


def overall_level(levels: Dict[str, str]) -> str:
    """
    Collapse per-vital status into an overall status.
    Any severe beats moderate, any moderate beats ok.
    """
    if "severe" in levels.values():
        return "severe"
    if "moderate" in levels.values():
        return "moderate"
    return "ok"


def val_class(level: str) -> str:
    """Maps status to a CSS class for coloring text."""
    return {"ok": "val-ok", "moderate": "val-mod", "severe": "val-sev"}[level]


def initials(name: str) -> str:
    """Used for the avatar circle on patient cards."""
    parts = [p for p in name.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def severity_rank(lvl: str) -> int:
    """Convenient numeric rank for comparisons (e.g., BP sys vs BP dia)."""
    return {"ok": 0, "moderate": 1, "severe": 2}[lvl]


def _clamp(x: float, lo: float, hi: float) -> float:
    """Keeps a number within [lo, hi]."""
    return max(lo, min(hi, x))


def _baseline_for(vital_name: str) -> float:
    """Baseline = midpoint of the default range."""
    r = DEFAULT_RANGES[vital_name]
    return (r.min + r.max) / 2.0


# -----------------------------------------------------------------------------
# Shock event helpers
# -----------------------------------------------------------------------------
def _pick_extreme_target(vital_name: str) -> float:
    """
    Chooses a random target from either the low extreme range or the high extreme range.
    """
    low_range, high_range = EXTREME_TARGETS[vital_name]
    if random.random() < 0.5:
        lo, hi = low_range
    else:
        lo, hi = high_range
    return random.uniform(lo, hi)


def _maybe_start_shock(p: dict) -> None:
    """
    With small probability, start a shock on the patient:
    - Set p["shock"] = {"until_ts": ..., "targets": {...}}
    - targets contains 1 or 2 vitals to be driven extreme.
    """
    if random.random() >= EXTREME_EVENT_PROB_PER_TICK:
        return

    now = datetime.now()
    duration = random.randint(
        EXTREME_EVENT_DURATION_SEC_RANGE[0],
        EXTREME_EVENT_DURATION_SEC_RANGE[1],
    )
    until = now.timestamp() + duration

    vitals_list = ["hr", "temp", "rr", "bp_sys", "bp_dia"]
    primary = random.choice(vitals_list)

    targets = {primary: _pick_extreme_target(primary)}

    if random.random() < EXTREME_SECOND_VITAL_PROB:
        remaining = [v for v in vitals_list if v != primary]
        secondary = random.choice(remaining)
        targets[secondary] = _pick_extreme_target(secondary)

    p["shock"] = {"until_ts": until, "targets": targets}


# -----------------------------------------------------------------------------
# ECG synthesis (cheap, stylized waveform, not physiological accuracy)
# -----------------------------------------------------------------------------
def _gauss(x: float, mu: float, sigma: float) -> float:
    """Simple Gaussian bump used to approximate P, QRS, T waves."""
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def synth_ecg_sample(phase_norm: float) -> float:
    """
    One ECG sample given a beat phase in [0, 1).
    The waveform is a sum of gaussian-shaped components.
    """
    p = 0.10 * _gauss(phase_norm, 0.18, 0.035)
    q = -0.15 * _gauss(phase_norm, 0.38, 0.015)
    r = 1.20 * _gauss(phase_norm, 0.40, 0.012)
    s = -0.25 * _gauss(phase_norm, 0.43, 0.018)
    t = 0.35 * _gauss(phase_norm, 0.70, 0.060)
    return p + q + r + s + t


def ecg_level_from_wave(ecg_live) -> str:
    """
    Label ECG window status by amplitude range.
    Uses min/max over the live window.
    """
    if not ecg_live:
        return "ok"

    mn = min(ecg_live)
    mx = max(ecg_live)

    if mn < ECG_MOD_MIN or mx > ECG_MOD_MAX:
        return "severe"
    if mn < ECG_OK_MIN or mx > ECG_OK_MAX:
        return "moderate"
    return "ok"


# -----------------------------------------------------------------------------
# Patient initialization (session_state persists across reruns)
# -----------------------------------------------------------------------------
N_PATIENTS = 5
NAMES = ["A", "B", "C", "D", "E"]

if "patients" not in st.session_state:
    st.session_state.patients = {}
    for i in range(1, N_PATIENTS + 1):
        pid = f"p{i:02d}"
        name = NAMES[i - 1]

        # Each patient dict holds:
        # - vitals: current values
        # - histories / logs: for charts + download
        # - ecg buffers: live rolling window + snapshots
        # - shock state: None or dict with until_ts/targets
        st.session_state.patients[pid] = {
            "id": pid,
            "name": name,
            "vitals": Vitals(
                hr=72 + random.randint(-2, 2),
                temp=36.8,
                rr=14,
                bp_sys=118,
                bp_dia=78,
            ),
            "hr_history": [],      # list[(datetime, int hr)]
            "minute_logs": [],     # list[dict] of vitals+ecg snapshot per minute
            "last_log_min": None,  # last minute number used for logging
            "ecg_live": [],        # rolling ECG window
            "ecg_phase_sec": 0.0,  # beat phase accumulator (seconds)
            "ecg_baseline_phase": 0.0,  # baseline wander phase accumulator
            "ecg_samples": [],     # list of ECG snapshots
            "shock": None,         # active shock state
            "ecg_artifact": None,  # active artifact state
        }


# -----------------------------------------------------------------------------
# Vitals simulation
# -----------------------------------------------------------------------------
def get_live_vitals(patient_id: str) -> Vitals:
    """
    Updates and returns the patient's vitals in-place.

    Cases:
    1) If a shock is active: pull affected vitals toward a target more aggressively.
    2) If normal: random walk + mean reversion to baseline.
    3) Maybe start a new shock at end of tick.
    """
    p = st.session_state.patients[patient_id]
    v = p["vitals"]
    now_ts = datetime.now().timestamp()

    # Clear shock once time is up.
    shock = p.get("shock")
    if shock and now_ts >= shock.get("until_ts", 0):
        p["shock"] = None
        shock = None

    # Shock mode: drift toward target(s) with jitter.
    if shock:
        targets = shock.get("targets", {})
        jitter = {"hr": 2.5, "temp": 0.06, "rr": 1.2, "bp_sys": 3.0, "bp_dia": 2.0}

        for key in ["hr", "temp", "rr", "bp_sys", "bp_dia"]:
            current = getattr(v, key)

            if key in targets:
                target = targets[key]
                pulled = current + 0.35 * (target - current)
                new_val = pulled + (random.random() - 0.5) * jitter[key]
            else:
                step_scale = {"hr": 1.0, "temp": 0.02, "rr": 0.4, "bp_sys": 1.0, "bp_dia": 0.6}[key]
                new_val = current + (random.random() - 0.5) * step_scale

            bounds = {
                "hr": (35, 200),
                "temp": (34.5, 40.5),
                "rr": (6, 40),
                "bp_sys": (80, 200),
                "bp_dia": (50, 120),
            }[key]
            setattr(v, key, _clamp(new_val, bounds[0], bounds[1]))

        return v

    # Normal mode: mild random walk.
    v.hr = _clamp(v.hr + (random.random() - 0.5) * 2.0, 35, 200)
    v.temp = _clamp(v.temp + (random.random() - 0.5) * 0.03, 34.5, 40.5)
    v.rr = _clamp(v.rr + (random.random() - 0.5) * 0.6, 6, 40)
    v.bp_sys = _clamp(v.bp_sys + (random.random() - 0.5) * 2.0, 80, 200)
    v.bp_dia = _clamp(v.bp_dia + (random.random() - 0.5) * 1.2, 50, 120)

    # Mean reversion back to default midpoint.
    v.hr = v.hr + MEAN_REVERT_STRENGTH * (_baseline_for("hr") - v.hr)
    v.temp = v.temp + MEAN_REVERT_STRENGTH * (_baseline_for("temp") - v.temp)
    v.rr = v.rr + MEAN_REVERT_STRENGTH * (_baseline_for("rr") - v.rr)
    v.bp_sys = v.bp_sys + MEAN_REVERT_STRENGTH * (_baseline_for("bp_sys") - v.bp_sys)
    v.bp_dia = v.bp_dia + MEAN_REVERT_STRENGTH * (_baseline_for("bp_dia") - v.bp_dia)

    # Possibly begin a shock episode.
    _maybe_start_shock(p)

    # This block forces target values immediately if a shock just started.
    # (Means the patient jumps straight into the extreme state on that tick.)
    if p.get("shock"):
        targets = p["shock"]["targets"]
        for key, target in targets.items():
            setattr(v, key, float(target))

    return v


# -----------------------------------------------------------------------------
# ECG update per patient
# -----------------------------------------------------------------------------
def update_ecg_for_patient(p: dict, hr_bpm: float) -> None:
    """
    Adds ECG samples to p["ecg_live"].

    - points_per_tick is based on refresh rate and ECG_FS
    - rr_sec sets the beat length based on HR
    - adds baseline wander + small noise
    - sometimes injects a short burst artifact
    """
    art = p.get("ecg_artifact")
    if not art and random.random() < ECG_ARTIFACT_PROB_PER_TICK:
        p["ecg_artifact"] = {
            "remaining": random.randint(ECG_ARTIFACT_SAMPLES_RANGE[0], ECG_ARTIFACT_SAMPLES_RANGE[1])
        }

    dt = 1.0 / ECG_FS
    points_per_tick = max(1, int((REFRESH_MS / 1000.0) * ECG_FS))

    # Convert HR to RR interval (seconds per beat), clamped to keep stable.
    rr_sec = max(0.35, min(2.0, 60.0 / max(30.0, float(hr_bpm))))

    phase_sec = p["ecg_phase_sec"]
    baseline_phase = p["ecg_baseline_phase"]
    ecg_live = p["ecg_live"]

    for _ in range(points_per_tick):
        phase_sec += dt
        beat_pos = (phase_sec % rr_sec) / rr_sec  # normalized 0..1 position within beat

        value = synth_ecg_sample(beat_pos)

        # Baseline wander (slow sine)
        baseline_phase += dt
        value += 0.05 * math.sin(2.0 * math.pi * 0.33 * baseline_phase)

        # Small measurement noise
        value += (random.random() - 0.5) * 0.03

        # Artifact burst noise
        art = p.get("ecg_artifact")
        if art and art.get("remaining", 0) > 0:
            value += (random.random() - 0.5) * 2.0 * ECG_ARTIFACT_NOISE_AMPLITUDE
            art["remaining"] -= 1
            if art["remaining"] <= 0:
                p["ecg_artifact"] = None

        ecg_live.append(value)

    # Keep only the last ECG_LIVE_MAXLEN samples.
    if len(ecg_live) > ECG_LIVE_MAXLEN:
        p["ecg_live"] = ecg_live[-ECG_LIVE_MAXLEN:]
    else:
        p["ecg_live"] = ecg_live

    p["ecg_phase_sec"] = phase_sec
    p["ecg_baseline_phase"] = baseline_phase


# -----------------------------------------------------------------------------
# Logging once per minute per patient
# -----------------------------------------------------------------------------
def maybe_log_minute(pid: str) -> None:
    """
    Adds one log entry per minute.
    Stores:
    - timestamp
    - vitals snapshot
    - downsampled ECG snapshot
    Also keeps HR history for trend plot.
    """
    p = st.session_state.patients[pid]
    v = p["vitals"]
    now = datetime.now()

    # HR history is used for the 30-minute trend chart.
    p["hr_history"].append((now, int(v.hr)))
    p["hr_history"] = p["hr_history"][-360:]  # cap memory

    # Only log when the minute changes.
    if p["last_log_min"] != now.minute:
        p["last_log_min"] = now.minute

        # Downsample ECG for storage.
        live = p["ecg_live"]
        if live:
            step = max(1, len(live) // ECG_SNAPSHOT_POINTS)
            ecg_wave = live[::step][-ECG_SNAPSHOT_POINTS:]
        else:
            ecg_wave = []

        entry = {
            "t": now.strftime("%Y-%m-%d %H:%M"),
            "hr": int(v.hr),
            "temp": round(v.temp, 1),
            "rr": int(v.rr),
            "bp_sys": int(v.bp_sys),
            "bp_dia": int(v.bp_dia),
            "ecg": ecg_wave,
        }

        # Keep only the most recent 10 logs/snapshots.
        p["minute_logs"].append(entry)
        p["minute_logs"] = p["minute_logs"][-10:]

        p["ecg_samples"].append({"t": entry["t"], "wave": ecg_wave})
        p["ecg_samples"] = p["ecg_samples"][-10:]


# -----------------------------------------------------------------------------
# Global update: called once per refresh to step all patients forward
# -----------------------------------------------------------------------------
def update_all_patients() -> None:
    for pid, p in st.session_state.patients.items():
        v = get_live_vitals(pid)
        update_ecg_for_patient(p, hr_bpm=v.hr)
        maybe_log_minute(pid)


# -----------------------------------------------------------------------------
# Chart rendering helpers
# -----------------------------------------------------------------------------
def render_hr_trend(p: dict) -> None:
    """
    Shows HR over last 30 minutes using Altair.
    """
    if not p["hr_history"]:
        st.info("No HR data yet.")
        return

    rows = p["hr_history"]
    times = [t for t, _ in rows]
    hrs = [float(h) for _, h in rows]

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
            scale=alt.Scale(domain=[window_start, now_floor]),
        ),
        y=alt.Y("HR:Q", title="Heart Rate (bpm)", scale=alt.Scale(domain=[0, 200])),
    ).properties(height=220)

    hover = alt.selection_point(fields=["Time"], on="mouseover", nearest=True, empty="none")
    line = base.mark_line(interpolate="monotone", strokeWidth=2)
    selectors = base.mark_point(opacity=0).add_params(hover)
    hover_dots = base.mark_point(size=24, filled=True).transform_filter(hover).encode(
        tooltip=[
            alt.Tooltip("Time:T", title="Time", format="%H:%M:%S"),
            alt.Tooltip("HR:Q", title="HR (bpm)", format=".1f"),
        ]
    )
    vrule = alt.Chart(df).mark_rule(strokeDash=[4, 3]).encode(x="Time:T").transform_filter(hover)

    st.altair_chart((line + selectors + vrule + hover_dots), use_container_width=True)


def render_ecg_live(p: dict) -> None:
    """
    Shows the rolling ECG window for the selected patient.
    """
    ecg = p.get("ecg_live", [])
    if not ecg:
        st.caption("ECG loading...")
        return

    df = pd.DataFrame({"i": list(range(len(ecg))), "ecg": ecg})
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("i:Q", title=None, axis=None),
            y=alt.Y("ecg:Q", title=None, scale=alt.Scale(domain=[-1.5, 2.2])),
        )
        .properties(height=140)
    )
    st.altair_chart(chart, use_container_width=True)


def render_ecg_history_10min(p: dict) -> None:
    """
    Shows concatenated snapshots from the last ~10 minutes.
    This is a quick "trend" view rather than an aligned beat view.
    """
    samples = p.get("ecg_samples", [])
    if not samples:
        st.caption("ECG history will appear after a few minutes.")
        return

    recent = samples[-10:]
    rows = []
    x_idx = 0
    for snap in recent:
        wave = snap.get("wave", [])
        for val in wave:
            rows.append({"x": x_idx, "ecg": val})
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
            y=alt.Y("ecg:Q", title=None, axis=None, scale=alt.Scale(domain=[-1.5, 2.2])),
        )
        .properties(height=140)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("ECG snapshots over the last ~10 minutes")


# -----------------------------------------------------------------------------
# Top navigation bar (title + patient selector)
# -----------------------------------------------------------------------------
def navbar() -> None:
    left, right = st.columns([4, 1])
    with left:
        st.markdown("### Remote Monitor")
        st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with right:
        pids = list(st.session_state.patients.keys())
        if "active" not in st.session_state:
            st.session_state.active = pids[0]
        idx = pids.index(st.session_state.active)
        selected_pid = st.selectbox(
            "View detail for:",
            pids,
            index=idx,
            format_func=lambda pid: f"{st.session_state.patients[pid]['name']} ({pid})",
        )
        st.session_state.active = selected_pid


# -----------------------------------------------------------------------------
# Grid overview (5 cards)
# -----------------------------------------------------------------------------
def render_grid() -> None:
    """
    Renders patient cards across 5 columns.
    Each card shows vitals + ECG status. The active card also shows live ECG.
    """
    cols = st.columns(5)
    i = 0
    active_pid = st.session_state.get("active")

    for pid, p in st.session_state.patients.items():
        v = p["vitals"]
        ecg_lvl = ecg_level_from_wave(p.get("ecg_live", []))

        # Per-vital status + ECG status
        lv = {
            "hr": vital_level(v.hr, DEFAULT_RANGES["hr"]),
            "temp": vital_level(v.temp, DEFAULT_RANGES["temp"]),
            "rr": vital_level(v.rr, DEFAULT_RANGES["rr"]),
            "bp_sys": vital_level(v.bp_sys, DEFAULT_RANGES["bp_sys"]),
            "bp_dia": vital_level(v.bp_dia, DEFAULT_RANGES["bp_dia"]),
            "ecg": ecg_lvl,
        }
        ov = overall_level(lv)

        with cols[i % 5]:
            st.markdown("<div class='rm-card'>", unsafe_allow_html=True)

            # Header color turns red if overall severe.
            head_cls = "rm-id-header rm-head-danger" if ov == "severe" else "rm-id-header rm-head-normal"

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

            # Body: vitals in 2 columns
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

                # BP: color uses the worse of sys/dia.
                bp_lvl = lv["bp_sys"] if severity_rank(lv["bp_sys"]) >= severity_rank(lv["bp_dia"]) else lv["bp_dia"]
                st.markdown(
                    f"<div class='rm-v'><div class='lab'>BP</div>"
                    f"<div class='val {val_class(bp_lvl)}'>{int(v.bp_sys)}/{int(v.bp_dia)} mmHg</div></div>",
                    unsafe_allow_html=True,
                )

            # ECG status line
            cls = {"ok": "rm-mini-ok", "moderate": "rm-mini-mod", "severe": "rm-mini-sev"}[ecg_lvl]
            st.markdown(
                f"<div class='rm-mini'>ECG status: <span class='{cls}'>{ecg_lvl.upper()}</span></div>",
                unsafe_allow_html=True,
            )

            # Only show live ECG on the active patient card to reduce clutter.
            if pid == active_pid:
                st.markdown(
                    "<div style='margin-top:8px; font-size:11px; opacity:.65;'>Live ECG</div>",
                    unsafe_allow_html=True,
                )
                render_ecg_live(p)

            st.markdown("</div>", unsafe_allow_html=True)  # end body
            st.markdown("</div>", unsafe_allow_html=True)  # end card

        i += 1


# -----------------------------------------------------------------------------
# Recent log helpers (pretty log cards)
# -----------------------------------------------------------------------------
def _log_level_from_row(row: dict) -> str:
    """
    Determine log row severity from vital thresholds (ECG isn't included in log severity here).
    """
    lv = {
        "hr": vital_level(float(row["hr"]), DEFAULT_RANGES["hr"]),
        "temp": vital_level(float(row["temp"]), DEFAULT_RANGES["temp"]),
        "rr": vital_level(float(row["rr"]), DEFAULT_RANGES["rr"]),
        "bp_sys": vital_level(float(row["bp_sys"]), DEFAULT_RANGES["bp_sys"]),
        "bp_dia": vital_level(float(row["bp_dia"]), DEFAULT_RANGES["bp_dia"]),
    }
    return overall_level(lv)


def _render_recent_logs_pretty(logs_same_date) -> None:
    """
    Renders the last 3 log entries for the selected date as styled HTML cards.
    """
    recent = list(reversed(logs_same_date))[:3]

    for row in recent:
        lvl = _log_level_from_row(row)
        lvl_cls = {"ok": "log-level-ok", "moderate": "log-level-mod", "severe": "log-level-sev"}[lvl]
        ts = row["t"]

        html = f"""
        <div class="log-card">
          <div class="log-top">
            <div class="log-time">{ts}</div>
            <div class="{lvl_cls}">{lvl.upper()}</div>
          </div>
          <div class="pills">
            <div class="pill">HR <b>{row['hr']}</b> bpm</div>
            <div class="pill">RR <b>{row['rr']}</b> /min</div>
            <div class="pill">Temp <b>{row['temp']}</b> °C</div>
            <div class="pill">BP <b>{row['bp_sys']}/{row['bp_dia']}</b> mmHg</div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Detail page for selected patient
# -----------------------------------------------------------------------------
def render_detail(pid: str) -> None:
    """
    Patient detail panel:
    - summary header + overall severity banner
    - vitals metrics row
    - live ECG
    - HR history (30 min) + ECG history (~10 min)
    - recent logs (latest 3)
    - download JSON for EOD summary/logs/ECG snapshots
    """
    p = st.session_state.patients[pid]
    v = p["vitals"]
    ecg_lvl = ecg_level_from_wave(p.get("ecg_live", []))

    lv = {
        "hr": vital_level(v.hr, DEFAULT_RANGES["hr"]),
        "temp": vital_level(v.temp, DEFAULT_RANGES["temp"]),
        "rr": vital_level(v.rr, DEFAULT_RANGES["rr"]),
        "bp_sys": vital_level(v.bp_sys, DEFAULT_RANGES["bp_sys"]),
        "bp_dia": vital_level(v.bp_dia, DEFAULT_RANGES["bp_dia"]),
        "ecg": ecg_lvl,
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

    if ov == "severe":
        st.error("CRITICAL deviation detected (vitals or ECG)")
    elif ov == "moderate":
        st.warning("Moderate deviation detected (vitals or ECG)")
    else:
        st.success("All vitals and ECG within range")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("HR (bpm)", f"{int(v.hr)}")
    c2.metric("RR (/min)", f"{int(v.rr)}")
    c3.metric("Temp (°C)", f"{v.temp:.1f}")
    c4.metric("BP Sys (mmHg)", f"{int(v.bp_sys)}")
    c5.metric("BP Dia (mmHg)", f"{int(v.bp_dia)}")

    st.markdown("**Live ECG**")
    render_ecg_live(p)

    a, b = st.columns([2, 1])
    with a:
        st.markdown("**Heart Rate History (last 30 min)**")
        render_hr_trend(p)

        st.markdown("**ECG History (last 10 min)**")
        render_ecg_history_10min(p)

    with b:
        logs = p["minute_logs"]
        if not logs:
            st.markdown("**Recent Logs**")
            st.caption("(logs appear each minute)")
            return

        # Show only logs from the latest date in the buffer.
        all_dates = {row["t"].split()[0] for row in logs}
        latest_date_str = max(all_dates)
        logs_same_date = [row for row in logs if row["t"].startswith(latest_date_str)]

        if not logs_same_date:
            st.caption(f"No logs for {latest_date_str}.")
            return

        st.markdown(f"**Recent Logs on {latest_date_str} (latest 3)**")
        _render_recent_logs_pretty(logs_same_date)

    # Download EOD JSON (summary + logs + ECG snapshots)
    summary = eod_summary(pid)
    d1, _ = st.columns(2)
    with d1:
        json_payload = {"summary": summary, "logs": p["minute_logs"], "ecg_samples": p["ecg_samples"]}
        st.download_button(
            "Download EOD JSON",
            data=json.dumps(json_payload, indent=2),
            file_name=f"{p['name'].replace(' ', '_')}_EOD_{summary['date']}.json",
            mime="application/json",
            key=f"json_{pid}",
        )


def eod_summary(pid: str) -> dict:
    """
    Quick summary computed from the minute logs.
    """
    p = st.session_state.patients[pid]
    logs = p["minute_logs"]
    hrs = [float(x["hr"]) for x in logs] if logs else []
    return {
        "patient": p["name"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "minHR": min(hrs) if hrs else None,
        "maxHR": max(hrs) if hrs else None,
        "avgHR": round(sum(hrs) / len(hrs), 1) if hrs else None,
        "totalLogs": len(logs),
        "ecgSamples": len(p.get("ecg_samples", [])),
    }


# -----------------------------------------------------------------------------
# Main app flow (runs every refresh)
# -----------------------------------------------------------------------------
nav_ph = st.container()
with nav_ph:
    navbar()

# Step the simulation forward.
update_all_patients()

# Render UI sections.
grid_ph = st.container()
detail_ph = st.container()

with grid_ph:
    render_grid()

with detail_ph:
    active = st.session_state.active or "p01"
    render_detail(active)
