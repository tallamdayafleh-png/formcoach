import json

import numpy as np

from cv.config import default_thresholds
from cv.form import DepthResult, RepForm, ValgusResult
from cv.reps import Rep, RepResult
from cv.schema import PoseSequence, VideoMeta
from cv.summary import build_summary, summarize
from cv.view import ViewResult

TH = default_thresholds()
SEQ = PoseSequence(VideoMeta("clip.mov", 30, 100, 100, 0), "m", [])


def _rep(n: int) -> Rep:
    return Rep(n, 0, 1, 2, 0.0, 1.0, 2.0, 90.0, 90.0, 70.0)


def _result(view: str, n_reps: int, usable: float = 0.95) -> RepResult:
    empty = np.array([])
    side = "both" if view == "front" else "left"
    reps = [_rep(i + 1) for i in range(n_reps)]
    return RepResult(
        ViewResult(view, 0.1), side, "x", reps, usable, empty, empty, empty
    )


def test_side_summary_flags_shallow_rep_and_skips_valgus():
    forms = [
        RepForm(1, DepthResult(15.0, True), None),
        RepForm(2, DepthResult(40.0, False), None),
    ]
    s = build_summary(SEQ, _result("side", 2), forms, TH)
    assert s["camera_view"] == "side"
    assert s["rep_count"] == 2
    assert s["checks"]["depth"] == {"status": "assessed"}
    assert s["checks"]["knee_valgus"]["status"] == "not_assessed"
    assert "front view" in s["checks"]["knee_valgus"]["reason"]
    assert s["reps"][1]["depth"] == {"thigh_deg": 40.0, "status": "above_parallel"}
    assert s["flags"] == [
        {"rep": 2, "joint": "hip", "issue": "above_parallel",
         "thigh_deg": 40.0, "severity": "performance"},
    ]  # fmt: skip


def test_front_summary_flags_valgus_and_skips_depth():
    forms = [
        RepForm(1, None, ValgusResult(0.95, False)),
        RepForm(2, None, ValgusResult(0.74, True)),
    ]
    s = build_summary(SEQ, _result("front", 2), forms, TH)
    assert s["checks"]["depth"]["status"] == "not_assessed"
    assert s["checks"]["knee_valgus"] == {"status": "assessed"}
    assert "min_knee_deg" not in s["reps"][0]  # 2D knee angle is meaningless here
    assert s["flags"][0]["issue"] == "valgus"
    assert s["flags"][0]["severity"] == "safety"


def test_safety_flags_come_first():
    forms = [
        RepForm(1, DepthResult(40.0, False), ValgusResult(0.9, False)),
        RepForm(2, DepthResult(40.0, False), ValgusResult(0.7, True)),
    ]
    # rep 2's valgus (safety) must be listed before rep 1's depth flag
    s = build_summary(SEQ, _result("side", 2), forms, TH)
    assert [(f["rep"], f["severity"]) for f in s["flags"]] == [
        (2, "safety"),
        (1, "performance"),
        (2, "performance"),
    ]


def test_low_confidence_when_tracking_poor_or_angled_or_no_reps():
    ok = build_summary(SEQ, _result("side", 1), [RepForm(1, None, None)], TH)
    assert ok["tracking"]["confidence"] == "high"
    poor = build_summary(
        SEQ, _result("side", 1, usable=0.5), [RepForm(1, None, None)], TH
    )
    assert poor["tracking"]["confidence"] == "low"
    angled = build_summary(SEQ, _result("angled", 1), [RepForm(1, None, None)], TH)
    assert angled["tracking"]["confidence"] == "low"
    assert angled["checks"]["knee_valgus"]["status"] == "not_assessed"
    none = build_summary(SEQ, _result("side", 0), [], TH)
    assert none["tracking"]["confidence"] == "low"


def test_empty_sequence_summarizes_to_zero_reps():
    s = summarize(SEQ)
    assert s["rep_count"] == 0
    assert s["flags"] == []
    json.dumps(s)  # serializable: no NaN, no numpy types
