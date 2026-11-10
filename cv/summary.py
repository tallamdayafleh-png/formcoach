"""Build the pose-summary JSON: the only thing the LLM coach ever sees.

Every check states whether it was assessed and, if not, why, so the coach
can be honest about uncertainty (coach principle 5) instead of guessing.
Flags carry a severity so safety issues are mentioned first (principle 4).
"""

from __future__ import annotations

from typing import Any

from cv.config import FormThresholds, default_thresholds
from cv.form import RepForm, assess_rep, form_signals
from cv.reps import RepResult, detect_reps
from cv.schema import PoseSequence
from cv.view import ANGLED, FRONT, SIDE

SUMMARY_SCHEMA_VERSION = 1
SEVERITY_ORDER = {"safety": 0, "performance": 1}

_NOT_ASSESSED = {
    ("depth", FRONT): "front view: depth needs a side view",
    ("knee_valgus", SIDE): "side view: knee cave needs a front view",
    ("knee_valgus", ANGLED): "angled view: knee cave needs a front view",
}


def assess_all(seq: PoseSequence, reps: RepResult, th: FormThresholds) -> list[RepForm]:
    signals = form_signals(seq, reps.side, th.tracking)
    return [
        assess_rep(r.number, r.start_frame, r.end_frame, reps.view.view, signals, th)
        for r in reps.reps
    ]


def _r(x: float, nd: int = 1) -> float | None:
    return None if x != x else round(x, nd)  # NaN -> None (JSON null)


def build_summary(
    seq: PoseSequence,
    reps: RepResult,
    forms: list[RepForm],
    th: FormThresholds,
) -> dict[str, Any]:
    view = reps.view.view
    low = (
        reps.usable_fraction < th.tracking.low_confidence_below
        or view == ANGLED
        or not reps.reps
    )
    checks: dict[str, Any] = {}
    for check in ("depth", "knee_valgus"):
        reason = _NOT_ASSESSED.get((check, view))
        checks[check] = (
            {"status": "not_assessed", "reason": reason}
            if reason
            else {"status": "assessed"}
        )

    rep_entries: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    for rep, form in zip(reps.reps, forms, strict=True):
        entry: dict[str, Any] = {
            "rep": rep.number,
            "start_s": round(rep.start_s, 2),
            "bottom_s": round(rep.bottom_s, 2),
            "end_s": round(rep.end_s, 2),
        }
        if view != FRONT:
            entry["min_knee_deg"] = _r(rep.min_knee_deg, 0)
            entry["min_hip_deg"] = _r(rep.min_hip_deg, 0)
        if form.depth is not None:
            status = "parallel_or_below" if form.depth.hit else "above_parallel"
            entry["depth"] = {"thigh_deg": _r(form.depth.thigh_deg), "status": status}
            if not form.depth.hit:
                flags.append({
                    "rep": rep.number, "joint": "hip", "issue": "above_parallel",
                    "thigh_deg": _r(form.depth.thigh_deg), "severity": "performance",
                })  # fmt: skip
        if form.valgus is not None:
            ratio = _r(form.valgus.knee_hip_width_ratio, 2)
            status = "knees_caving" if form.valgus.caving else "ok"
            entry["valgus"] = {"knee_hip_width_ratio": ratio, "status": status}
            if form.valgus.caving:
                flags.append({
                    "rep": rep.number, "joint": "knees", "issue": "valgus",
                    "knee_hip_width_ratio": ratio, "severity": "safety",
                })  # fmt: skip
        rep_entries.append(entry)

    flags.sort(key=lambda f: (SEVERITY_ORDER[f["severity"]], f["rep"]))
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "exercise": "squat",
        "source": seq.meta.source,
        "camera_view": view,
        "leg_used": reps.side,
        "tracking": {
            "usable_fraction": round(reps.usable_fraction, 2),
            "confidence": "low" if low else "high",
        },
        "checks": checks,
        "rep_count": len(reps.reps),
        "reps": rep_entries,
        "flags": flags,
    }


def summarize(seq: PoseSequence, th: FormThresholds | None = None) -> dict[str, Any]:
    """Keypoints -> reps -> form checks -> pose summary."""
    th = th or default_thresholds()
    reps = detect_reps(seq, th)
    return build_summary(seq, reps, assess_all(seq, reps, th), th)
