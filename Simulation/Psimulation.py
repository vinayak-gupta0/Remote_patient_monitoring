import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_patient(random_state):
    """Returns a dictionary with simulated ECG data,
                                 heart rate,
                                 and basic vital signs for a patient."""

    ecg12 = nk.ecg_simulate(duration=60,
                            method="multileads",
                            sampling_rate=250,
                            random_state=random_state)
    df_ecg12 = pd.DataFrame(ecg12)

    ecg1 = nk.ecg_simulate(duration=60,
                           sampling_rate=250,
                           random_state=random_state)
    signals, info = nk.ecg_process(ecg1, sampling_rate=250)

    df_signals = signals[['ECG_Rate', 'ECG_Clean']]
    df_signals.columns = ['HeartRate_250Hz', 'ECG_single_250Hz']

    df_all = pd.concat([df_signals, df_ecg12], axis=1)

    # =========
    n_samples = len(df_all)
    rng = np.random.default_rng(seed=random_state)

    temp = rng.normal(loc=36.8, scale=0.2, size=n_samples)
    df_all["Temp"] = temp

    # 呼吸频率 Respiratory Rate (breaths/min)：16 ± 1.5
    rr = rng.normal(loc=16, scale=1.5, size=n_samples)
    df_all["RespRate"] = rr

    sbp = rng.normal(loc=120, scale=8, size=n_samples)
    dbp = rng.normal(loc=80, scale=5, size=n_samples)
    df_all["BP_sys"] = sbp
    df_all["BP_dia"] = dbp

    return {col: df_all[col].values for col in df_all.columns}


for i in range(1, 3):
    globals()[f"patient{i}"] = build_patient(i)

# print(patient2)
