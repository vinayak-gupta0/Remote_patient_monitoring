# NEWS2 scoring (0–3) for: bp_sys, hr, rr, temp (NO bp_dia).
# getOverallLevel(): per-parameter level based on SINGLE subscore:
#       3 -> "severe", 2 -> "moderate", 0/1 -> "ok"
# getAllscore(): total score + NEWS2 band: high/medium/low-medium/low

from typing import Any, Dict


def _to_float(x: Any) -> float | None:
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
        "subscores": subs,
    }
