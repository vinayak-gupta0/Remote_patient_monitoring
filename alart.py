import streamlit as st
import streamlit.components.v1 as components
from typing import Dict,List
import os
import base64
import time

def triggerAudio(container):
    file_path = "/Users/ricky/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Computer Science & Programming/Y3 Software Tutorial/Tutorial_1/Remote_patient_monitoring/deep2.mp3"

    if not os.path.exists(file_path):
        with container:
            st.warning("Alarm audio file not found! Check the path: audio/beep.mp3")
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
        audio.volume = 0.8; // Set volume
        audio.loop = false;
        
        // Use the Promise returned by play() to catch the specific error
        const playPromise = audio.play();

        if (playPromise !== undefined) {{
            playPromise.then(_ => {{
                // Audio playback started successfully (no message necessary)
            }}).catch(error => {{
                // THIS WILL CATCH THE BROWSER BLOCKING THE AUDIO
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
    "hr": vitalLevelFunc(vitals.hr, ranges["hr"]),
    "temp": vitalLevelFunc(vitals.temp, ranges["temp"]),
    "rr": vitalLevelFunc(vitals.rr, ranges["rr"]),
    "bp_sys": vitalLevelFunc(vitals.bp_sys, ranges["bp_sys"]),
    "bp_dia": vitalLevelFunc(vitals.bp_dia, ranges["bp_dia"]),
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
    
    anySevere = len(severePatients)>0
    # prevSevere is no longer just a boolean; it's the last list of severe patients
    prev_severe_list = st.session_state.get("severe_patients_list", [])

    st.session_state["severe_patients_list"] = severePatients
    
    # Condition to trigger audio:
    # 1. Any patient is severe AND 
    # 2. (Either it's the first time OR the list of severe patients has changed OR it's been a few seconds)
    # A simple check: if any severe patients exist, re-render the audio component to try to play the sound.
    # The Streamlit auto-rerun loop will make this attempt to play the audio every second.
    if anySevere:
        triggerAudio(audio_container) 
    else:
        # Clear the container by rendering nothing when safe
        with audio_container:
            st.empty()