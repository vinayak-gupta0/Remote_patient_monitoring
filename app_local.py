"""Remote Patient Monitoring Streamlit app (final prototype with local database).

This file is the Streamlit *entrypoint*.

What this app does
------------------
- Simulates vitals (HR, Temp, RR, BP) for 5 patients.
- Occasionally injects “shock” events that push 1–2 vitals to extreme values.
- Synthesizes a lightweight ECG waveform per patient (plus occasional artifacts).
- Renders:
  1) A 10-card grid overview (each card shows vitals + ECG status)
  2) A detail view for the selected patient (live ECG, HR trend, recent logs, download)

How it runs
-----------
Prefer running via Streamlit:
    `python -m streamlit run app_v2.0.py`

"""

import os
import sys
import time

import streamlit as st

# Ensure the project root (folder containing `src/`) is on sys.path.
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rpm.alerts.alerts import updateAlerts
from src.rpm.ui import styles as ui_styles
from src.rpm.ui import render as ui_render
from src.rpm.ui.charts import (
    render_hr_trend,
    render_ecg_sparkline,
    render_single_ecg,
    render_12lead_ecg,
)
from src.rpm.sim.engine import update_all_patients
from src.rpm.state.init import ensure_init
from src.rpm.ui.panels.test_panel import emergency_test_panel


def main() -> None:
    st.set_page_config(page_title="Remote Monitor", layout="wide", initial_sidebar_state="expanded")

    ui_styles.inject()

    # Initialize session state and simulation
    ensure_init()

    @st.cache_data(show_spinner=False)
    def _noop_cache_marker() -> int:
        return 1

    emergency_test_panel()

    nav_ph = st.container()
    with nav_ph:
        ui_render.navbar()

    audio_ph = st.container()
    grid_ph = st.container()
    detail_ph = st.container()

    update_all_patients()

    updateAlerts(st.session_state.patients, audio_ph)

    with grid_ph:
        ui_render.render_grid(lambda pid: render_ecg_sparkline(pid))

    with detail_ph:
        active = st.session_state.active or "p01"
        ui_render.render_detail(active, render_single_ecg, render_12lead_ecg, render_hr_trend)

    # Auto refresh
    time.sleep(1)
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


if __name__ == "__main__":
    main()
