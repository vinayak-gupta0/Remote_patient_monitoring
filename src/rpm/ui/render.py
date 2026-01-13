from __future__ import annotations
from typing import Callable, Dict
from datetime import datetime

import streamlit as st

from src.rpm.models.app_types import Ranges
from src.rpm.config.ranges import DEFAULT_RANGES
from src.rpm.scoring.severity import vital_level, overall_level, severity_rank
from src.rpm.scoring.news2 import getAllscore
from src.rpm.ui.helpers import val_class, initials

# UI-specific ranges imported from config


# severity_rank imported from scoring


def navbar() -> None:
    left, right = st.columns([4, 1])

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
            format_func=lambda pid: f"{st.session_state.patients[pid]['name']} ({pid})",
        )
        st.session_state.active = selected_pid


def render_grid(render_ecg_sparkline: Callable[[str], None]) -> None:
    cols = st.columns(5)
    i = 0
    for pid, p in st.session_state.patients.items():
        v = st.session_state.patients[pid]["vitals"]

        lv = {
            "hr": vital_level(v.hr, DEFAULT_RANGES["hr"]),
            "temp": vital_level(v.temp, DEFAULT_RANGES["temp"]),
            "rr": vital_level(v.rr, DEFAULT_RANGES["rr"]),
            "bp_sys": vital_level(v.bp_sys, DEFAULT_RANGES["bp_sys"]),
            "bp_dia": vital_level(v.bp_dia, DEFAULT_RANGES["bp_dia"]),
        }
        ov = overall_level(lv)

        with cols[i % 5]:
            st.markdown("<div class='rm-card'>", unsafe_allow_html=True)

            head_cls = (
                "rm-id-header rm-head-danger" if ov == "severe" else "rm-id-header rm-head-normal"
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
                    if severity_rank(lv["bp_sys"]) >= severity_rank(lv["bp_dia"]) else lv["bp_dia"]
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
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        i += 1


def render_detail(
    pid: str,
    render_single_ecg: Callable[[str, float], None],
    render_12lead_ecg: Callable[[str, list[str], float], None],
    render_hr_trend: Callable[[dict], None],
) -> None:
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

    news = getAllscore(v)
    total = int(news.get("total_score", 0))
    subs = news.get("subscores", {}) or {}

    if total >= 7:
        st.error(
            f"Total score ≥ 7 (emergency response threshold): Activate an emergency response with continuous "
            f"vital-sign monitoring and consider transfer to HDU or ICU.\n\n"
            f"(NEWS2 total = {total}; subscores: {subs})"
        )
    elif total >= 5:
        st.error(
            f"Total score 5–6: Urgent review by a clinician (core team or equivalent).\n\n"
            f"(NEWS2 total = {total}; subscores: {subs})"
        )
    elif any(int(s) == 3 for s in subs.values()):
        st.warning(
            f"Any single 3 (red score): Urgent review.\n\n(NEWS2 subscores: {subs})"
        )
    else:
        st.info(
            f"Low risk (0–4): Continue routine monitoring.\n\n(NEWS2 subscores: {subs})"
        )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Heart Rate History (last 30 min)**")
        render_hr_trend(p)

        st.markdown("**Single-lead ECG (ECG_single_250Hz)**")
        render_single_ecg(pid, window_seconds=60.0)

        st.markdown("**12-lead ECG history (select leads)**")
        render_12lead_ecg(pid, ["I", "II", "III", "aVR", "aVL", "aVF"], window_seconds=10.0)

    with right:
        # Additional detail or logs handled by app_v2.0.py
        pass
