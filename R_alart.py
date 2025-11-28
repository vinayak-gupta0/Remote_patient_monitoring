# alart.py

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List
import os
import base64
import time

VITAL_RANGES = {
    #Systolic BP (mmHg)
    "bp_sys": {
        "normal": (90, 139),
        "warning": (140, 179),
        # critical: >= 180 or < 90
        "critical_low": 90,
        "critical_high": 180,
    },
    # Diastolic BP (mmHg)
    "bp_dia": {
        "normal": (60, 89),
        "warning": (90, 119),
        # critical: >= 120
        "critical_low": None,
        "critical_high": 120,
    },
    #Heart Rate (bpm)
    "hr": {
        "normal": (60, 99),
        "warning": (100, 129),
        # critical: >= 130 or <= 40
        "critical_low": 40,
        "critical_high": 130,
    },
    #Respiratory Rate (breaths/min)
    "rr": {
        "normal": (12, 19),
        "warning": (20, 29),
        # critical: >= 30 or <= 8
        "critical_low": 8,
        "critical_high": 30,
    },
    # Temperature
    "temp": {
        "normal": (36.0, 37.4),
        "warning": (37.5, 38.4),
        # critical: >= 38.5 or < 35.0
        "critical_low": 35.0,
        "critical_high": 38.5,
    },
}

def vitalLevelFunc(value: float, cfg: Dict) -> str:
    if value is None:
        return "ok"

    try:
        v = float(value)
    except (TypeError, ValueError):
        return "ok"

    normal_low, normal_high = cfg["normal"]
    warn_low, warn_high = cfg["warning"]
    crit_low = cfg.get("critical_low", None)
    crit_high = cfg.get("critical_high", None)

    if crit_low is not None and v <= crit_low:
        return "severe"
    if crit_high is not None and v >= crit_high:
        return "severe"

    if warn_low <= v <= warn_high:
        return "moderate"

    if normal_low <= v <= normal_high:
        return "ok"

    return "moderate"


def triggerAudio(container):
    file_path = "/Users/ricky/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Computer Science & Programming/Y3 Software Tutorial/Tutorial_1/Remote_patient_monitoring/deep2.mp3"

    if not os.path.exists(file_path):
        with container:
            st.warning("Alarm audio file not found! Check the path: deep2.mp3")
        return

    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        mime_type = "audio/mp3"

    audio_js = f"""
    <audio id="invisible_alarm" controls autoplay style="display:none;">
      <source src="data:{mime_type};base64,{audio_b64}" type="{mime_type}">
      Your browser does not support the audio element.
    </audio>
    <script>
      const audio = document.getElementById('invisible_alarm');
      if (audio) {{
        audio.volume = 0.8;
        audio.loop = false;
        const playPromise = audio.play();
        if (playPromise !== undefined) {{
          playPromise.then(_ => {{
            // started OK
          }}).catch(error => {{
            console.error("ALARM AUDIO BLOCKED. Reason:", error.name);
          }});
        }}
      }}
    </script>
    """
    unique_key = f"alarm_trigger_{time.time()}"

    with container:
        components.html(audio_js, height=0, width=0, scrolling=False)


def getOverallLevel(vitals, vitalLevelFunc, ranges):

    lv = {
        "hr":     vitalLevelFunc(getattr(vitals, "hr", None),      ranges["hr"]),
        "temp":   vitalLevelFunc(getattr(vitals, "temp", None),    ranges["temp"]),
        "rr":     vitalLevelFunc(getattr(vitals, "rr", None),      ranges["rr"]),
        "bp_sys": vitalLevelFunc(getattr(vitals, "bp_sys", None),  ranges["bp_sys"]),
        "bp_dia": vitalLevelFunc(getattr(vitals, "bp_dia", None),  ranges["bp_dia"]),
    }

    if "severe" in lv.values():
        return "severe"
    if "moderate" in lv.values():
        return "moderate"
    return "ok"


def updateAlerts(patients: Dict, vitalLevelFunc, rangesDict, audio_container):
    severePatients: List[str] = []

    for pid, p in patients.items():
        ov = getOverallLevel(p["vitals"], vitalLevelFunc, rangesDict)
        if ov == "severe":
            severePatients.append(pid)

    anySevere = len(severePatients) > 0

    prev_severe_list = st.session_state.get("severe_patients_list", [])
    st.session_state["severe_patients_list"] = severePatients

    if anySevere:
        triggerAudio(audio_container)
    else:

        with audio_container:
            st.empty()
