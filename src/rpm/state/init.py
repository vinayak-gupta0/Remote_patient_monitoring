from __future__ import annotations

import random
import streamlit as st

from src.rpm.models.app_types import Vitals, N_PATIENTS, NAMES
from src.rpm.sim.db import upsert_patient
from src.rpm.sim.engine import init_sim


def ensure_init() -> None:
    """Initialize Streamlit session state for patients and simulation.

    Creates `st.session_state.patients` with default vitals and metadata if missing,
    upserts patients into the database, and initializes the simulation data.
    """
    if "patients" in st.session_state:
        return

    st.session_state.patients = {}

    for i in range(1, N_PATIENTS + 1):
        pid = f"p{i:02d}"
        name = NAMES[i - 1]
        upsert_patient(pid, name)
        st.session_state.patients[pid] = {
            "id": pid,
            "name": name,
            "vitals": Vitals(
                hr=100 + random.randint(-2, 2),
                temp=36.8,
                rr=10,
                bp_sys=118,
                bp_dia=78,
            ),
            "hr_history": [],
            "minute_logs": [],
            "last_log_min": None,
            "ecg_live": [],
            "ecg_phase": 0.0,
            "ecg_samples": [],
            "last_ecg_min": None,
            "daily_reports": {},  # { "YYYY-MM-DD": [event, ...] }
            "last_report_key": None,
        }

    # Initialize simulation data and scenario state
    init_sim()
