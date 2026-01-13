from __future__ import annotations

from datetime import datetime, date
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from src.rpm.models.app_types import FS


def render_hr_trend(p: dict) -> None:
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

    base = (
        alt.Chart(df)
        .encode(
            x=alt.X(
                "Time:T",
                axis=alt.Axis(format="%H:%M", title=None),
                scale=alt.Scale(domain=[window_start, now_floor]),
            ),
            y=alt.Y("HR:Q", title="Heart Rate (bpm)", scale=alt.Scale(domain=[0, 200])),
        )
        .properties(height=220)
    )

    hover = alt.selection_point(fields=["Time"], on="mouseover", nearest=True, empty="none")

    line = base.mark_line(interpolate="monotone", strokeWidth=2)
    selectors = base.mark_point(opacity=0).add_params(hover)
    hover_dots = (
        base.mark_point(size=24, filled=True)
        .transform_filter(hover)
        .encode(
            tooltip=[
                alt.Tooltip("Time:T", title="Time", format="%H:%M:%S"),
                alt.Tooltip("HR:Q", title="HR (bpm)", format=".1f"),
            ]
        )
    )
    vrule = alt.Chart(df).mark_rule(strokeDash=[4, 3]).encode(x="Time:T").transform_filter(hover)

    st.altair_chart((line + selectors + vrule + hover_dots).interactive(), use_container_width=True)


def render_ecg_sparkline(pid: str) -> None:
    sim = st.session_state.sim_data.get(pid)
    if not sim or "ECG_single_250Hz" not in sim:
        st.caption("ECG loading…")
        return

    arr = sim["ECG_single_250Hz"]
    idx = st.session_state.sim_idx.get(pid, 0)
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


def render_single_ecg(pid: str, window_seconds: float = 60.0) -> None:
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


def render_12lead_ecg(pid: str, selected_leads: list[str], window_seconds: float = 10.0) -> None:
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
