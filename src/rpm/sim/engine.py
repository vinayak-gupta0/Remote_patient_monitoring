from __future__ import annotations
import time
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st

from src.rpm.models.app_types import STEP_SAMPLES
from src.rpm.sim.db import insert_vitals
from src.rpm.scoring.news2 import getAllscore
from src.rpm.sim.Psimulation import build_patient as build_patient


@st.cache_data(show_spinner=False)
def _cached_build_patient(i: int):
    return build_patient(i)


def init_sim() -> None:
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = {}
        st.session_state.sim_idx = {}
        for i, pid in enumerate(st.session_state.patients.keys(), start=1):
            sim_dict = _cached_build_patient(i)
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
    if "hr" in ov:
        v.hr = float(ov["hr"])
    if "rr" in ov:
        v.rr = float(ov["rr"])
    if "temp" in ov:
        v.temp = float(ov["temp"])
    if "bp_sys" in ov:
        v.bp_sys = float(ov["bp_sys"])
    if "bp_dia" in ov:
        v.bp_dia = float(ov["bp_dia"])
    return v


def get_live_vitals(patient_id: str):
    p = st.session_state.patients[patient_id]
    v = p["vitals"]

    sim = st.session_state.sim_data.get(patient_id)
    idx = st.session_state.sim_idx.get(patient_id, 0)

    if not sim:
        return v

    hr_arr = sim.get("HeartRate_250Hz")
    temp_arr = sim.get("Temp")
    rr_arr = sim.get("RespRate")
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

    now = time.time()
    last = p.get("last_db_write", 0.0)
    if now - last >= 60:
        insert_vitals(patient_id, v.hr, v.rr, v.temp, v.bp_sys, v.bp_dia)
        p["last_db_write"] = now
    return v


def update_all_patients() -> None:
    for pid in st.session_state.patients.keys():
        get_live_vitals(pid)
        maybe_log_minute(pid)


def update_daily_report(pid: str, now: datetime) -> None:
    p = st.session_state.patients[pid]
    v = p["vitals"]

    minute_key = now.strftime("%Y-%m-%d %H:%M")
    if p.get("last_report_key") == minute_key:
        return
    p["last_report_key"] = minute_key

    news = getAllscore(v)
    total = int(news.get("total_score", 0))
    subs = news.get("subscores", {}) or {}

    single_param_3 = any(int(s) == 3 for s in subs.values())
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
        "subscores": subs,
        "vitals": {
            "hr": float(getattr(v, "hr", 0)),
            "rr": float(getattr(v, "rr", 0)),
            "temp": float(getattr(v, "temp", 0)),
            "bp_sys": float(getattr(v, "bp_sys", 0)),
        },
    }

    p["daily_reports"][day_key].append(event)

    keys = sorted(p["daily_reports"].keys())
    if len(keys) > 7:
        for k in keys[:-7]:
            del p["daily_reports"][k]


def maybe_log_minute(pid: str) -> None:
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
            fs = 250
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
