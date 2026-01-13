import pytest
import types

import Prototypes.app_v2 as m

def vitals(rr=16, hr=70, bp_sys=120, temp=37.0):
    """Create a minimal vitals-like object with required attributes."""
    return types.SimpleNamespace(rr=rr, hr=hr, bp_sys=bp_sys, temp=temp)


def assert_all_keys(d, keys):
    for k in keys:
        assert k in d, f"Missing key: {k}"


# 1) Unit scoring
@pytest.mark.parametrize("x", [None, "abc", {}, [], object()])
@pytest.mark.parametrize("fn", [m.score_rr, m.score_hr, m.score_bp_sys, m.score_temp])
def test_scoring_bad_inputs_return_0(fn, x):
    assert fn(x) == 0


@pytest.mark.parametrize("rr, expected", [
    (8, 3),
    (9, 1), (11, 1),
    (12, 0), (20, 0),
    (21, 2), (24, 2),
    (25, 3),
])
def test_score_rr_boundaries(rr, expected):
    assert m.score_rr(rr) == expected


@pytest.mark.parametrize("hr, expected", [
    (40, 3),
    (41, 1), (50, 1),
    (51, 0), (90, 0),
    (91, 1), (110, 1),
    (111, 2), (130, 2),
    (131, 3),
])
def test_score_hr_boundaries(hr, expected):
    assert m.score_hr(hr) == expected


@pytest.mark.parametrize("bp_sys, expected", [
    (90, 3),
    (91, 2), (100, 2),
    (101, 1), (110, 1),
    (111, 0), (219, 0),
    (220, 3),
])
def test_score_bp_sys_boundaries(bp_sys, expected):
    assert m.score_bp_sys(bp_sys) == expected


@pytest.mark.parametrize("temp, expected", [
    (35.0, 3),
    (35.1, 1), (36.0, 1),
    (36.1, 0), (38.0, 0),
    (38.1, 1), (39.0, 1),
    (39.1, 2),
])
def test_score_temp_boundaries(temp, expected):
    assert m.score_temp(temp) == expected

# 2) Subscores aggregation

def test_compute_subscores_and_getVitalScores_same():
    v = vitals(rr=25, hr=70, bp_sys=120, temp=37.0)
    subs1 = m.compute_subscores(v)
    subs2 = m.getVitalScores(v)

    assert subs1 == subs2
    assert_all_keys(subs1, ["rr", "hr", "bp_sys", "temp"])
    assert subs1["rr"] == 3
    assert subs1["hr"] == 0


def test_compute_subscores_missing_attrs_default_to_0():
    # vitals object missing some attributes = score 0
    v = types.SimpleNamespace(rr=16)  # only rr exists
    subs = m.compute_subscores(v)
    assert_all_keys(subs, ["rr", "hr", "bp_sys", "temp"])
    assert subs["rr"] in (0, 1, 2, 3)
    assert subs["hr"] == 0
    assert subs["bp_sys"] == 0
    assert subs["temp"] == 0


# 3) Per-parameter level mapping
@pytest.mark.parametrize("subscore, expected_level", [
    (0, "ok"),
    (1, "ok"),
    (2, "moderate"),
    (3, "severe"),
])
def test_getOverallLevel_mapping(subscore, expected_level):
    rr_value_for_score = {
        0: 16,   # 12-20 -> 0
        1: 10,   # 9-11 -> 1 (use 10)
        2: 22,   # 21-24 -> 2
        3: 8,    # <=8 -> 3
    }[subscore]

    v = vitals(rr=rr_value_for_score, hr=70, bp_sys=120, temp=37.0)
    levels = m.getOverallLevel(v)

    assert_all_keys(levels, ["rr", "hr", "bp_sys", "temp"])
    assert levels["rr"] == expected_level
    assert levels["hr"] == "ok"
    assert levels["bp_sys"] == "ok"
    assert levels["temp"] == "ok"

# 4) Total scoring + trigger

def test_getAllscore_output_schema():
    v = vitals()
    res = m.getAllscore(v)

    assert_all_keys(res, ["total_score", "risk_level", "trigger", "subscores"])
    assert isinstance(res["total_score"], int)
    assert isinstance(res["subscores"], dict)
    assert_all_keys(res["subscores"], ["rr", "hr", "bp_sys", "temp"])


@pytest.mark.parametrize("v, expected_level, expected_trigger", [
    # HIGH: total >= 7
    (vitals(rr=25, hr=131, bp_sys=90, temp=35.0), "high", "aggregate_score_7_or_more"),
    # MEDIUM: total 5-6
    (vitals(rr=21, hr=111, bp_sys=100, temp=37.0), "medium", "aggregate_score_5_to_6"),
    # LOW-MEDIUM: any single 3 but total < 5
    (vitals(rr=8, hr=70, bp_sys=120, temp=37.0), "low-medium", "red_score_single_parameter_3"),
    # LOW: otherwise (0-4 and no single 3)
    (vitals(rr=16, hr=70, bp_sys=120, temp=37.0), "low", "aggregate_score_0_to_4"),
])
def test_getAllscore_banding_and_trigger(v, expected_level, expected_trigger):
    res = m.getAllscore(v)

    assert res["risk_level"] == expected_level
    assert res["trigger"] == expected_trigger

    # total score consistency check
    subs = res["subscores"]
    assert res["total_score"] == sum(int(x) for x in subs.values())


def test_getAllscore_medium_range_is_5_or_6():
    v5 = vitals(rr=21, hr=91, bp_sys=101, temp=37.0)   # 2 + 1 + 1 + 0 = 4 (not 5) -> adjust
    v6 = vitals(rr=21, hr=111, bp_sys=100, temp=37.0)
    r6 = m.getAllscore(v6)
    assert r6["total_score"] == 6
    assert r6["risk_level"] == "medium"

    # Make total=5: rr=21(2), hr=91(1), bp_sys=100(2), temp=37(0) => 5
    v5 = vitals(rr=21, hr=91, bp_sys=100, temp=37.0)
    r5 = m.getAllscore(v5)
    assert r5["total_score"] == 5
    assert r5["risk_level"] == "medium"
