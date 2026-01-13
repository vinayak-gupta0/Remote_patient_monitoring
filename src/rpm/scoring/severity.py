from __future__ import annotations
from typing import Dict

from src.rpm.models.app_types import Ranges


def vital_level(value: float, r: Ranges) -> str:
    if value < r.min - r.moderate_band or value > r.max + r.moderate_band:
        return "severe"
    if value < r.min or value > r.max:
        return "moderate"
    return "ok"


def overall_level(levels: Dict[str, str]) -> str:
    if "severe" in levels.values():
        return "severe"
    if "moderate" in levels.values():
        return "moderate"
    return "ok"


def severity_rank(lvl: str) -> int:
    return {"ok": 0, "moderate": 1, "severe": 2}[lvl]
