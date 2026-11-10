"""Per-rep form checks: depth (side view) and knee valgus (front view).

Depth uses the thigh angle from horizontal, not the knee angle, because the
knee angle also depends on how far the shins lean forward:

    hip ●               thigh_deg > 0   hip above knee (above parallel)
         ╲
          ● knee
    hip ●────● knee      thigh_deg = 0   level
    image y points down, so the sign comes from knee_y - hip_y.

Valgus uses knee-to-knee width divided by hip width in a front view: knees
caving in pull that ratio down.

    knees OK   hip ●───────● hip        caving   hip ●───────● hip
                   │       │                          ╲     ╱
              knee ●       ● knee                knee  ●───●  knee
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from cv.config import FormThresholds, TrackingThresholds
from cv.filters import landmark_track, smooth_series
from cv.landmarks import LANDMARK_INDEX
from cv.schema import PoseSequence


def thigh_angle_deg(hip: ArrayLike, knee: ArrayLike) -> np.ndarray:
    """Thigh angle from horizontal in degrees, positive when hip is above knee.

    Inputs are pixel (x, y) points with y pointing down, shape (..., 2).
    90 = vertical thigh (standing), 0 = level, negative = hip below knee.
    """
    hip, knee = np.asarray(hip, dtype=float), np.asarray(knee, dtype=float)
    dx = np.abs(knee[..., 0] - hip[..., 0])
    dy = knee[..., 1] - hip[..., 1]
    angle = np.degrees(np.arctan2(dy, dx))
    return np.where((dx > 0) | (dy != 0), angle, np.nan)


def knee_hip_width_ratio(
    left_knee: ArrayLike,
    right_knee: ArrayLike,
    left_hip: ArrayLike,
    right_hip: ArrayLike,
) -> np.ndarray:
    """Horizontal knee-to-knee distance / hip-to-hip distance (front view)."""
    lk, rk, lh, rh = (
        np.asarray(p, dtype=float) for p in (left_knee, right_knee, left_hip, right_hip)
    )
    knees = np.abs(lk[..., 0] - rk[..., 0])
    hips = np.abs(lh[..., 0] - rh[..., 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(hips > 0, knees / hips, np.nan)


@dataclass
class FormSignals:
    """Smoothed per-frame form signals (NaN where not measurable)."""

    thigh_deg: np.ndarray  # side/angled view, leg nearest the camera
    knee_width_ratio: np.ndarray  # front view


def form_signals(seq: PoseSequence, side: str, trk: TrackingThresholds) -> FormSignals:
    t = np.array([f.timestamp_ms for f in seq.frames], dtype=float)

    def track(name: str) -> np.ndarray:
        return landmark_track(seq, LANDMARK_INDEX[name], trk.visibility_min)

    def smooth(x: np.ndarray) -> np.ndarray:
        return smooth_series(x, t, trk.smooth_tau_s, trk.max_gap_s)

    near = side if side in ("left", "right") else "left"
    thigh = thigh_angle_deg(track(f"{near}_hip"), track(f"{near}_knee"))
    width = knee_hip_width_ratio(
        track("left_knee"), track("right_knee"), track("left_hip"), track("right_hip")
    )
    return FormSignals(thigh_deg=smooth(thigh), knee_width_ratio=smooth(width))


class DepthTracker:
    """Live 'depth hit' detector: fires once per rep when the thigh crosses
    the threshold. Call reset() when a new rep starts."""

    def __init__(self, max_thigh_deg: float) -> None:
        self.max_thigh_deg = max_thigh_deg
        self.hit = False
        self.min_thigh_deg = float("inf")

    def reset(self) -> None:
        self.hit = False
        self.min_thigh_deg = float("inf")

    def update(self, thigh_deg: float) -> bool:
        """Return True only on the frame depth is first reached this rep."""
        if not np.isfinite(thigh_deg):
            return False
        self.min_thigh_deg = min(self.min_thigh_deg, thigh_deg)
        if not self.hit and thigh_deg <= self.max_thigh_deg:
            self.hit = True
            return True
        return False


@dataclass(frozen=True)
class DepthResult:
    thigh_deg: float  # smallest thigh angle during the rep
    hit: bool


@dataclass(frozen=True)
class ValgusResult:
    knee_hip_width_ratio: float  # smallest ratio during the rep
    caving: bool


@dataclass(frozen=True)
class RepForm:
    rep: int
    depth: DepthResult | None  # None = not assessed in this view
    valgus: ValgusResult | None


def _window_min(x: np.ndarray, start: int, end: int) -> float:
    w = x[start : end + 1]
    w = w[np.isfinite(w)]
    return float(w.min()) if w.size else float("nan")


def assess_rep(
    number: int,
    start_frame: int,
    end_frame: int,
    view: str,
    signals: FormSignals,
    th: FormThresholds,
) -> RepForm:
    """Check one rep. Uses only frames up to end_frame, so it is available
    the moment the rep completes, in batch and live alike."""
    depth = valgus = None
    if view in ("side", "angled"):
        thigh = _window_min(signals.thigh_deg, start_frame, end_frame)
        if np.isfinite(thigh):
            depth = DepthResult(thigh, thigh <= th.squat.depth.max_thigh_deg)
    if view == "front":
        ratio = _window_min(signals.knee_width_ratio, start_frame, end_frame)
        if np.isfinite(ratio):
            caving = ratio < th.squat.valgus.min_knee_hip_width_ratio
            valgus = ValgusResult(ratio, caving)
    return RepForm(rep=number, depth=depth, valgus=valgus)
