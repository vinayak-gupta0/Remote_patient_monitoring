from dataclasses import dataclass

# Sampling constants used across app and simulation
FS: int = 250            # sampling rate used in Psimulation (Hz)
STEP_SAMPLES: int = FS   # advance 1 simulated second each rerun


# Patient data defaults
N_PATIENTS: int = 10
NAMES = [
    "A B10", "B 10", "C", "D", "E", "F", "G", "H", "I", "J"
]


@dataclass
class Ranges:
    min: float
    max: float
    moderate_band: float


@dataclass
class Vitals:
    hr: float      # bpm
    temp: float    # °C
    rr: float      # /min
    bp_sys: float  # mmHg
    bp_dia: float  # mmHg
