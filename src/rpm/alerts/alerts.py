# updateAlerts(): plays alarm when HIGH (total >= 7) with cooldown to avoid spam
#
# IMPORTANT:
# 1) Put your mp3 in: <same folder as this file>/beep.mp3
# 2) Browser may block autoplay. This code also offers a manual "Play alarm" button.

import os
import time
import base64
from typing import Dict, List, Any, Optional

import streamlit as st
import streamlit.components.v1 as components

from src.rpm.scoring.news2 import getAllscore

# Alarm audio configuration
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(THIS_DIR, "beep.mp3")

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
    <audio id=\"invisible_alarm\" autoplay style=\"display:none;\">
      <source src=\"data:{MIME_TYPE};base64,{audio_b64}\" type=\"{MIME_TYPE}\">
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


def updateAlerts(patients: Dict[str, Dict[str, Any]], audio_container) -> None:
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
