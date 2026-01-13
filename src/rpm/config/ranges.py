from __future__ import annotations

from typing import Dict

from src.rpm.models.app_types import Ranges


DEFAULT_RANGES: Dict[str, Ranges] = {
    "hr": Ranges(50, 110, 10),
    "temp": Ranges(36.0, 37.8, 0.3),
    "rr": Ranges(10, 20, 2),
    "bp_sys": Ranges(90, 140, 10),
    "bp_dia": Ranges(60, 90, 5),
}
