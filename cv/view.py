"""Classify the camera view (side / front / angled) from the first seconds.

From the side the shoulders overlap, so the horizontal distance between them
is small compared with torso length. From the front they're about a torso
apart:

    side:  shoulder width / torso length ~ 0.1      front: ~ 0.7
            ●                                        ●───────●
            │                                            │
            ●                                        ●───●

The decision is made once, from the first `classify_s` seconds of detected
frames, which is also what live mode can do before the first rep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv.config import ViewThresholds
from cv.filters import landmark_track
from cv.landmarks import LANDMARK_INDEX
from cv.schema import PoseSequence

SIDE, FRONT, ANGLED = "side", "front", "angled"


@dataclass(frozen=True)
class ViewResult:
    view: str  # "side", "front" or "angled"
    shoulder_ratio: float  # median shoulder width / torso length used to decide


def shoulder_ratio_series(seq: PoseSequence) -> np.ndarray:
    """Per-frame shoulder width / torso length, in pixels. NaN if no pose.

    Visibility is not filtered here: in a side view the far shoulder is
    hidden by definition, and its estimated position is exactly what tells
    us the shoulders overlap.
    """
    pts = {
        name: landmark_track(seq, LANDMARK_INDEX[name], min_visibility=0.0)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    }
    width = np.abs(pts["left_shoulder"][:, 0] - pts["right_shoulder"][:, 0])
    mid_sh = (pts["left_shoulder"] + pts["right_shoulder"]) / 2
    mid_hip = (pts["left_hip"] + pts["right_hip"]) / 2
    torso = np.linalg.norm(mid_sh - mid_hip, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(torso > 0, width / torso, np.nan)


def classify_ratio(ratio: float, th: ViewThresholds) -> str:
    if not np.isfinite(ratio):
        return ANGLED  # no information: don't claim either view
    if ratio <= th.side_max_shoulder_ratio:
        return SIDE
    if ratio >= th.front_min_shoulder_ratio:
        return FRONT
    return ANGLED


def classify_view(seq: PoseSequence, th: ViewThresholds) -> ViewResult:
    ratios = shoulder_ratio_series(seq)
    t = np.array([f.timestamp_ms for f in seq.frames], dtype=float)
    finite = np.isfinite(ratios)
    if not finite.any():
        return ViewResult(ANGLED, float("nan"))
    t0 = t[finite][0]
    window = finite & (t <= t0 + th.classify_s * 1000)
    ratio = float(np.median(ratios[window]))
    return ViewResult(classify_ratio(ratio, th), ratio)
