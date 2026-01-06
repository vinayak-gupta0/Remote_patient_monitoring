import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List, Any, Tuple
import os
import base64
import time

AUDIO_FILE_PATH = "audio/beep.mp3"  
ALARM_COOLDOWN_SEC = 15             


@st.cache_data(show_spinner=False)
def _load_audio_b64(file_path: str) -> Tuple[str, str]:

    if not os.path.exists(file_path):
        return ("", "")

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")


    mime_type = "audio/mpeg"
    return (mime_type, audio_b64)


def triggerAudio(container, *, loop: bool = False, volume: float = 0.8):

    mime_type, audio_b64 = _load_audio_b64(AUDIO_FILE_PATH)

    if not mime_type or not audio_b64:
        with container:
            st.warning(f"Alarm audio file not found! Check: {AUDIO_FILE_PATH}")
        return

    uid = str(int(time.time() * 1000))

    audio_html = f"""
    <audio id="alarm_{uid}" autoplay style="display:none;">
      <source src="data:{mime_type};base64,{audio_b64}" type="{mime_type}">
    </audio>
    <script>
      (function(){{
        const audio = document.getElementById("alarm_{uid}");
        if (!audio) return;
        audio.volume = {float(volume)};
        audio.loop = {str(loop).lower()};

        const p = audio.play();
        if (p !== undefined) {{
          p.catch(err => {{
            console.log("ALARM AUDIO BLOCKED:", err && err.name ? err.name : err);
          }});
        }}
      }})();
    </script>
    """

    with container:
        components.html(audio_html, height=0, width=0, scrolling=False)


def showManualAlarmButton(container, label: str = "Play alarm (if autoplay is blocked)"):

    with container:
        if st.button(label, key="manual_alarm_btn"):
            triggerAudio(container, loop=False)


def _should_alarm(res: Dict[str, Any]) -> bool:
    risk = res.get("risk_level", "")
    subs = res.get("subscores", {}) or {}

    is_high = (risk == "high")
    has_red = any(int(v) == 3 for v in subs.values() if v is not None)

    return is_high or has_red


def updateAlerts(patients: Dict[str, Dict[str, Any]], audio_container):

    alarmPatients: List[str] = []
    alarmReasons: Dict[str, str] = {}

    for pid, p in patients.items():
        vitals = p.get("vitals", None)
        if vitals is None:
            continue

        res = getAllscore(vitals)

        if _should_alarm(res):
            alarmPatients.append(pid)

            subs = res.get("subscores", {}) or {}
            has_red = any(int(v) == 3 for v in subs.values() if v is not None)
            if res.get("risk_level") == "high":
                alarmReasons[pid] = f"HIGH (total={res.get('total_score')})"
            elif has_red:
                alarmReasons[pid] = f"RED subscore (total={res.get('total_score')})"
            else:
                alarmReasons[pid] = "ALARM"

    st.session_state["alarm_patients_list"] = alarmPatients
    st.session_state["alarm_reasons"] = alarmReasons

    now = time.time()
    last = float(st.session_state.get("last_alarm_ts", 0.0))

    if alarmPatients:
        if (now - last) >= ALARM_COOLDOWN_SEC:
            st.session_state["last_alarm_ts"] = now
            triggerAudio(audio_container, loop=False)

        showManualAlarmButton(audio_container, label="Play alarm (if autoplay is blocked)")

    else:
        with audio_container:
            st.empty()
        st.session_state["last_alarm_ts"] = 0.0
