import streamlit as st
import streamlit.components.v1 as components
from typing import Dict,List

def triggerAudio():
    beep ="""

    <script>
    const AudioContext = window.AutioContext || window.webkitAudioContext;
    const ctx = new AudioContext();
    if (ctx.state === "suspended) ctx.resume();

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    osc.frequency.value = 880;

    osc.connect (gain);
    gain.connect(ctx.destination);

    osc.start();
    gain.gain.setValueAtTime (1,ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime +1.0);
    osc.stop(ctx.current +1.1)
    </script>
    """
    components.html(beep, height =0, width=0)

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

def updateAlarts( patients: Dict, vitalLevelFunc, rangesDict):
    severePatients: List[str] = []

    for pid, p in patients.items():
        ov = getOverallLevel(p["vitals"], vitalLevelFunc, rangesDict)
        if ov == "servere":
            severePatients.append(pid)
    
    anySevere = len(severePatients)>0
    prevSevere = st.session_state.get("alarm_activate", False)

    st.session_state["severe_patients"] = severePatients
    
    if anySevere and not prevSevere:
        triggerAudio()
        st.session_state["alarm_active"] = True
    elif not anySevere and prevSevere:
        st.session_state["alarm_active"] = False