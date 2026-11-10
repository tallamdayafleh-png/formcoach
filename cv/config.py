"""Load the shared form/rep thresholds from config/form_thresholds.json.

The same JSON file is read by the browser in live mode, so these numbers
must never be duplicated as constants in Python or TypeScript. Loading is
strict: an unknown or missing key raises, so a typo like "dwon_deg" can't
silently fall back to some default and change how reps are counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "form_thresholds.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RepThresholds:
    down_deg: float  # knee angle must drop below this to count as "down"
    up_deg: float  # ...then rise above this to finish the rep
    min_rep_s: float  # reps shorter than this are noise
    min_drop_deg: float  # knee must bend at least this much from standing


@dataclass(frozen=True)
class TrackingThresholds:
    visibility_min: float  # landmarks below this visibility are ignored
    max_gap_s: float  # hold the last value through gaps up to this long
    smooth_tau_s: float  # EMA time constant; ~ the smoothing lag in seconds


@dataclass(frozen=True)
class FormThresholds:
    squat_rep: RepThresholds
    tracking: TrackingThresholds


def _section(cls: type[T], data: object, where: str) -> T:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be an object")
    expected = {f.name for f in fields(cls)}
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
    for key, value in data.items():
        # bool is a subclass of int in Python, so reject it explicitly
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}.{key} must be a number, got {value!r}")
        if value < 0:
            raise ValueError(f"{where}.{key} must be >= 0, got {value}")
    return cls(**{k: float(v) for k, v in data.items()})


def parse_thresholds(data: dict[str, Any]) -> FormThresholds:
    """Validate a parsed config dict and convert it to typed thresholds."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported thresholds schema_version {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    squat = data.get("squat")
    if not isinstance(squat, dict) or "rep" not in squat:
        raise ValueError("config is missing squat.rep")
    rep = _section(RepThresholds, squat["rep"], "squat.rep")
    tracking = _section(TrackingThresholds, data.get("tracking"), "tracking")

    if not 0 < rep.down_deg < rep.up_deg <= 180:
        raise ValueError(
            "squat.rep needs 0 < down_deg < up_deg <= 180, got "
            f"down_deg={rep.down_deg}, up_deg={rep.up_deg}"
        )
    if tracking.visibility_min > 1:
        raise ValueError("tracking.visibility_min must be between 0 and 1")
    return FormThresholds(squat_rep=rep, tracking=tracking)


def load_thresholds(path: str | Path = CONFIG_PATH) -> FormThresholds:
    return parse_thresholds(json.loads(Path(path).read_text()))


@cache
def default_thresholds() -> FormThresholds:
    """The repo's config file, loaded once per process."""
    return load_thresholds(CONFIG_PATH)
