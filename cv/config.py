"""Load the shared form/rep thresholds from config/form_thresholds.json.

The same JSON file is read by the browser in live mode, so these numbers
must never be duplicated as constants in Python or TypeScript. Loading is
strict: an unknown or missing key raises, so a typo like "dwon" can't
silently fall back to some default and change how reps are counted.

Units:
- squat.rep.side:  knee angle in degrees (180 = straight leg)
- squat.rep.front: hip height as a fraction of recent standing height (1 = standing)
- squat.depth.max_thigh_deg: thigh angle from horizontal at the bottom, using
  MediaPipe's hip point (the hip joint, which sits above the hip crease, so a
  parallel squat reads roughly 10-25 degrees, not 0)
- squat.valgus.min_knee_hip_width_ratio: knee-to-knee width / hip width
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "form_thresholds.json"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RepThresholds:
    down: float  # signal must drop below this to count as "down"
    up: float  # ...then rise above this to finish the rep
    min_drop: float  # signal must fall at least this much from standing
    min_rep_s: float  # reps shorter than this are noise


@dataclass(frozen=True)
class DepthThresholds:
    max_thigh_deg: float  # bottom thigh angle at or below this = depth hit


@dataclass(frozen=True)
class ValgusThresholds:
    min_knee_hip_width_ratio: float  # below this at any point in a rep = caving


@dataclass(frozen=True)
class SquatThresholds:
    rep_side: RepThresholds
    rep_front: RepThresholds
    depth: DepthThresholds
    valgus: ValgusThresholds


@dataclass(frozen=True)
class ViewThresholds:
    side_max_shoulder_ratio: float  # shoulder width / torso length
    front_min_shoulder_ratio: float
    classify_s: float  # decide the view from this many seconds at the start


@dataclass(frozen=True)
class TrackingThresholds:
    visibility_min: float  # landmarks below this visibility are ignored
    max_gap_s: float  # hold the last value through gaps up to this long
    smooth_tau_s: float  # EMA time constant; ~ the smoothing lag in seconds
    standing_ref_s: float  # look-back window for the standing hip height
    low_confidence_below: float  # usable-frame fraction below this = low


@dataclass(frozen=True)
class FormThresholds:
    squat: SquatThresholds
    view: ViewThresholds
    tracking: TrackingThresholds


def _obj(data: object, where: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be an object")
    return data


def _keys(data: dict[str, Any], expected: set[str], where: str) -> None:
    missing = sorted(expected - data.keys())
    unknown = sorted(data.keys() - expected)
    if missing or unknown:
        # A typo shows up as both, and naming the unknown key points at it.
        problems = []
        if unknown:
            problems.append(f"unknown keys {unknown}")
        if missing:
            problems.append(f"missing keys {missing}")
        raise ValueError(f"{where} has " + " and ".join(problems))


def _section(cls: type[T], data: object, where: str) -> T:
    data = _obj(data, where)
    _keys(data, {f.name for f in fields(cls)}, where)
    for key, value in data.items():
        # bool is a subclass of int in Python, so reject it explicitly
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}.{key} must be a number, got {value!r}")
        if value < 0:
            raise ValueError(f"{where}.{key} must be >= 0, got {value}")
    return cls(**{k: float(v) for k, v in data.items()})


def _rep(data: object, where: str, max_up: float) -> RepThresholds:
    rep = _section(RepThresholds, data, where)
    if not 0 < rep.down < rep.up <= max_up:
        raise ValueError(
            f"{where} needs 0 < down < up <= {max_up}, got down={rep.down}, up={rep.up}"
        )
    return rep


def parse_thresholds(data: dict[str, Any]) -> FormThresholds:
    """Validate a parsed config dict and convert it to typed thresholds."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported thresholds schema_version {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    _keys(data, {"schema_version", "squat", "view", "tracking"}, "config")
    squat = _obj(data["squat"], "squat")
    _keys(squat, {"rep", "depth", "valgus"}, "squat")
    rep = _obj(squat["rep"], "squat.rep")
    _keys(rep, {"side", "front"}, "squat.rep")

    squat_th = SquatThresholds(
        rep_side=_rep(rep["side"], "squat.rep.side", max_up=180),
        rep_front=_rep(rep["front"], "squat.rep.front", max_up=1),
        depth=_section(DepthThresholds, squat["depth"], "squat.depth"),
        valgus=_section(ValgusThresholds, squat["valgus"], "squat.valgus"),
    )
    view = _section(ViewThresholds, data["view"], "view")
    if view.side_max_shoulder_ratio >= view.front_min_shoulder_ratio:
        raise ValueError(
            "view needs side_max_shoulder_ratio < front_min_shoulder_ratio"
        )
    tracking = _section(TrackingThresholds, data["tracking"], "tracking")
    for name in ("visibility_min", "low_confidence_below"):
        if getattr(tracking, name) > 1:
            raise ValueError(f"tracking.{name} must be between 0 and 1")
    return FormThresholds(squat=squat_th, view=view, tracking=tracking)


def load_thresholds(path: str | Path = CONFIG_PATH) -> FormThresholds:
    return parse_thresholds(json.loads(Path(path).read_text()))


@cache
def default_thresholds() -> FormThresholds:
    """The repo's config file, loaded once per process."""
    return load_thresholds(CONFIG_PATH)
