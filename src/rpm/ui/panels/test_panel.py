import streamlit as st
import time

def emergency_test_panel():
    with st.sidebar:
        st.markdown("## Test Panel")

        pids = list(st.session_state.patients.keys())
        test_pid = st.selectbox(
            "Target patient",
            pids,
            format_func=lambda pid: f"{st.session_state.patients[pid]['name']} ({pid})"
        )

        duration = st.slider("Duration (seconds)", 5, 300, 30, step=5)

        presets = {
            "Stroke": {"bp_sys": 230, "bp_dia": 120, "rr": 22, "hr": 115, "temp": 37.0},
            "Sepsis-like": {"temp": 39.4, "rr": 26, "hr": 135, "bp_sys": 95, "bp_dia": 60},
            "Respiratory failure": {"rr": 30, "hr": 120, "temp": 37.2, "bp_sys": 115, "bp_dia": 75},
            "Bradycardia collapse": {"hr": 38, "rr": 10, "temp": 36.5, "bp_sys": 85, "bp_dia": 55},
        }

        preset_name = st.selectbox("Preset scenario", list(presets.keys()))

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Apply", use_container_width=True):
                if presets[preset_name] is None:
                    st.session_state.scenario.pop(test_pid, None)
                else:
                    st.session_state.scenario[test_pid] = {
                        "name": preset_name,
                        "until": time.time() + duration,
                        "overrides": presets[preset_name],
                    }

        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.scenario.pop(test_pid, None)

        # Show current scenario status
        active = st.session_state.scenario.get(test_pid)
        if active:
            remaining = int(active["until"] - time.time()) if active.get("until") else None
            st.info(f"Active: {active['name']}\n\nRemaining: {remaining}s" if remaining is not None else f"Active: {active['name']}")
        else:
            st.caption("No active scenario for selected patient.")