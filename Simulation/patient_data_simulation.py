import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_patient(random_state):
    """Returns a dictionary with simulated ECG data,
                                 heart rate,
                                  xxx for a patient."""

    # 12-lead ECG for detailed platform
    ecg12 = nk.ecg_simulate(duration=60,
                            method="multileads",
                            sampling_rate=250,
                            random_state=random_state)
    df_ecg12 = pd.DataFrame(ecg12)

    # single-lead ECG for general platform
    ecg1 = nk.ecg_simulate(duration=60,
                           sampling_rate=250,
                           random_state=random_state)
    signals, info = nk.ecg_process(ecg1, sampling_rate=250)

    df_signals = signals[['ECG_Rate', 'ECG_Clean']]
    df_signals.columns = ['HeartRate_250Hz', 'ECG_single_250Hz']

    df_all = pd.concat([df_signals, df_ecg12], axis=1)

    return {col: df_all[col].values for col in df_all.columns}

for i in range(1, 3):
    globals()[f"patient{i}"] = build_patient(i)

print(patient2)