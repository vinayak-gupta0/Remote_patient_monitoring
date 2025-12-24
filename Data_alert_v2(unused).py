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


# -----------------------------
# Alarm audio configuration
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(THIS_DIR, "assets", "deep2.mp3")

MIME_TYPE = "audio/mpeg"          # correct MIME for mp3
ALARM_VOLUME = 0.8
ALARM_COOLDOWN_SEC = 8            # prevent repeated alarm on Streamlit reruns


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


# -----------------------------
# NEWS2 scoring helpers
# -----------------------------
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


# -----------------------------
# Alert updater
# -----------------------------
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
        # cooldown gate
        if (now - last) >= ALARM_COOLDOWN_SEC:
            st.session_state["last_alarm_ts"] = now
            triggerAudio(audio_container, loop=False)

        # Optional: show manual play button in case autoplay is blocked
        showManualAlarmButton(audio_container, label="Play alarm (if autoplay is blocked)")
    else:
        # clear audio container + reset timer
        with audio_container:
            st.empty()
        st.session_state["last_alarm_ts"] = 0.0
