"""Squat rep detection (roadmap item 3), for side and front camera views.

All causal (past frames only) so live mode can run the same logic:

    keypoints ─► camera view (first seconds) ─► rep signal per frame ─► RepCounter

The rep signal depends on the view, because from the front the knee bends
toward the camera and its 2D angle barely changes:

    side / angled: knee angle in degrees, leg nearest the camera
    front:         hip height above the ankles / recent standing hip height

RepCounter uses two thresholds (hysteresis) so jitter around a single line
can't count extra reps:

    up   ───────╮           ╭────── must rise above to finish a rep
                │           │
    down ───────┼───────────┼────── must drop below to count as "down"
                ╰── bottom ─╯

    python -m cv.reps out/squat_keypoints.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from cv.angles import joint_angle
from cv.config import FormThresholds, RepThresholds, default_thresholds
from cv.filters import RunningMax, landmark_track, smooth_series
from cv.landmarks import LANDMARK_INDEX
from cv.schema import PoseSequence
from cv.view import FRONT, ViewResult, classify_view

SIDES = ("left", "right")


def _joints(side: str) -> tuple[int, int, int, int]:
    """(shoulder, hip, knee, ankle) landmark indices for one side."""
    return tuple(  # type: ignore[return-value]
        LANDMARK_INDEX[f"{side}_{j}"] for j in ("shoulder", "hip", "knee", "ankle")
    )


@dataclass(frozen=True)
class RepSpan:
    start_frame: int  # last frame standing before the descent
    bottom_frame: int  # lowest (smoothed) signal value
    end_frame: int  # first frame back above `up`


class _State(Enum):
    UNKNOWN = "unknown"  # no trustworthy standing position seen yet
    STANDING = "standing"
    DOWN = "down"


class RepCounter:
    """Frame-by-frame rep state machine. Feed it a smoothed rep signal where
    larger = more upright (knee angle, or relative hip height).

    A rep is STANDING -> DOWN -> STANDING. Losing tracking (NaN) cancels any
    rep in progress and waits for the lifter to be seen standing again, so a
    rep is never counted from guessed data.
    """

    def __init__(self, cfg: RepThresholds) -> None:
        self.cfg = cfg
        self.state = _State.UNKNOWN
        self._start = (0, 0.0)  # frame, t_ms of the last frame above `up`
        self._peak = 0.0  # highest value while standing: the drop reference
        self._bottom = (0, 0.0, math.inf)  # frame, t_ms, value

    def _stand(self, frame: int, t_ms: float, value: float) -> None:
        self.state = _State.STANDING
        self._start = (frame, t_ms)
        self._peak = value

    def update(self, frame: int, t_ms: float, value: float) -> RepSpan | None:
        """Process one frame; return a RepSpan on the frame a rep completes."""
        cfg = self.cfg
        if not math.isfinite(value):
            self.state = _State.UNKNOWN
            return None

        if self.state is _State.UNKNOWN:
            if value > cfg.up:
                self._stand(frame, t_ms, value)
        elif self.state is _State.STANDING:
            if value > cfg.up:
                self._start = (frame, t_ms)
                self._peak = max(self._peak, value)
            elif value < cfg.down:
                self.state = _State.DOWN
                self._bottom = (frame, t_ms, value)
        elif self.state is _State.DOWN:
            if value < self._bottom[2]:
                self._bottom = (frame, t_ms, value)
            if value > cfg.up:
                start_f, start_t = self._start
                duration_s = (t_ms - start_t) / 1000.0
                drop = self._peak - self._bottom[2]
                bottom_f = self._bottom[0]
                self._stand(frame, t_ms, value)
                if duration_s >= cfg.min_rep_s and drop >= cfg.min_drop:
                    return RepSpan(start_f, bottom_f, frame)
        return None


@dataclass(frozen=True)
class Rep:
    number: int  # 1-based, as a lifter would say it
    start_frame: int
    bottom_frame: int
    end_frame: int
    start_s: float
    bottom_s: float
    end_s: float
    bottom_signal: float  # rep signal at the bottom (deg or hip-height ratio)
    min_knee_deg: float  # side view only; NaN in front view
    min_hip_deg: float  # side view only; NaN in front view


@dataclass
class RepResult:
    view: ViewResult
    side: str  # leg used: "left"/"right" in side view, "both" in front view
    signal_name: str  # "knee_deg" or "hip_height_ratio"
    reps: list[Rep]
    usable_fraction: float  # frames with a trustworthy rep signal
    signal: np.ndarray = field(repr=False)  # smoothed rep signal per frame
    knee_deg: np.ndarray = field(repr=False)  # smoothed, NaN in front view
    hip_deg: np.ndarray = field(repr=False)

    def reps_completed_by(self, frame: int) -> int:
        """How many reps had finished by this frame (what a live counter shows)."""
        return sum(1 for r in self.reps if r.end_frame <= frame)

    def last_rep_by(self, frame: int) -> Rep | None:
        done = [r for r in self.reps if r.end_frame <= frame]
        return done[-1] if done else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view.view,
            "side": self.side,
            "signal": self.signal_name,
            "rep_count": len(self.reps),
            "usable_fraction": round(self.usable_fraction, 3),
            "reps": [
                {
                    "number": r.number,
                    "start_s": round(r.start_s, 3),
                    "bottom_s": round(r.bottom_s, 3),
                    "end_s": round(r.end_s, 3),
                    "bottom_signal": round(r.bottom_signal, 3),
                }
                for r in self.reps
            ],
        }


def choose_side(seq: PoseSequence) -> str:
    """Pick the leg whose hip, knee and ankle are most visible on average.

    In a side view the far leg is hidden behind the near one and tracks
    badly, so averaging both legs would only add noise.
    """
    best, best_score = SIDES[0], -1.0
    for side in SIDES:
        _, hip, knee, ankle = _joints(side)
        scores = [
            (f.landmarks[hip].visibility + f.landmarks[knee].visibility
             + f.landmarks[ankle].visibility) / 3
            for f in seq.frames
            if f.landmarks is not None
        ]  # fmt: skip
        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score:
            best, best_score = side, score
    return best


def _mean_of_finite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise mean of a and b, using whichever is finite; NaN if neither."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    out = np.where(fa, a, b)
    both = fa & fb
    out[both] = (a[both] + b[both]) / 2
    return out


def hip_height_ratio(
    seq: PoseSequence, t_ms: np.ndarray, th: FormThresholds
) -> np.ndarray:
    """Hip height above the ankles / highest hip height in the last few seconds.

    1.0 = standing as tall as recently seen; a deep squat is ~0.6. Pixel
    based, so walking toward the camera makes you bigger, not lower, and
    can't look like a rep. The reference uses past frames only.
    """
    trk = th.tracking

    def y(name: str) -> np.ndarray:
        return landmark_track(seq, LANDMARK_INDEX[name], trk.visibility_min)[:, 1]

    hip_y = _mean_of_finite(y("left_hip"), y("right_hip"))
    ankle_y = _mean_of_finite(y("left_ankle"), y("right_ankle"))
    height = smooth_series(ankle_y - hip_y, t_ms, trk.smooth_tau_s, trk.max_gap_s)
    ref = RunningMax(trk.standing_ref_s)
    out = np.full_like(height, np.nan)
    for i, (t, h) in enumerate(zip(t_ms, height, strict=True)):
        r = ref.update(float(t), float(h))
        if math.isfinite(h) and math.isfinite(r) and r > 0:
            out[i] = h / r
    return out


def detect_reps(
    seq: PoseSequence,
    thresholds: FormThresholds | None = None,
    view: ViewResult | None = None,
) -> RepResult:
    th = thresholds or default_thresholds()
    trk = th.tracking
    view = view or classify_view(seq, th.view)
    t_ms = np.array([f.timestamp_ms for f in seq.frames], dtype=float)
    n = len(t_ms)

    def smooth(x: np.ndarray) -> np.ndarray:
        return smooth_series(x, t_ms, trk.smooth_tau_s, trk.max_gap_s)

    if view.view == FRONT:
        side = "both"
        signal = hip_height_ratio(seq, t_ms, th)
        raw_usable = np.isfinite(signal)
        knee_s = hip_s = np.full(n, np.nan)
        cfg, signal_name = th.squat.rep_front, "hip_height_ratio"
    else:
        side = choose_side(seq)
        shoulder, hip, knee, ankle = _joints(side)
        pts = {
            j: landmark_track(seq, j, trk.visibility_min)
            for j in (shoulder, hip, knee, ankle)
        }
        knee_raw = joint_angle(pts[hip], pts[knee], pts[ankle])
        raw_usable = np.isfinite(knee_raw)
        knee_s = smooth(knee_raw)
        hip_s = smooth(joint_angle(pts[shoulder], pts[hip], pts[knee]))
        signal = knee_s
        cfg, signal_name = th.squat.rep_side, "knee_deg"

    counter = RepCounter(cfg)
    reps: list[Rep] = []
    for i, (t, value) in enumerate(zip(t_ms, signal, strict=True)):
        span = counter.update(i, float(t), float(value))
        if span is None:
            continue
        hip_window = hip_s[span.start_frame : span.end_frame + 1]
        finite_hip = hip_window[np.isfinite(hip_window)]
        reps.append(
            Rep(
                number=len(reps) + 1,
                start_frame=span.start_frame,
                bottom_frame=span.bottom_frame,
                end_frame=span.end_frame,
                start_s=t_ms[span.start_frame] / 1000,
                bottom_s=t_ms[span.bottom_frame] / 1000,
                end_s=t_ms[span.end_frame] / 1000,
                bottom_signal=float(signal[span.bottom_frame]),
                min_knee_deg=float(knee_s[span.bottom_frame]),
                min_hip_deg=float(finite_hip.min()) if finite_hip.size else math.nan,
            )
        )

    return RepResult(
        view=view,
        side=side,
        signal_name=signal_name,
        reps=reps,
        usable_fraction=float(raw_usable.mean()) if n else 0.0,
        signal=signal,
        knee_deg=knee_s,
        hip_deg=hip_s,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count squat reps from keypoints.")
    parser.add_argument("keypoints", type=Path, help="keypoints JSON from cv.extract")
    parser.add_argument("--json", type=Path, help="also write the result as JSON")
    args = parser.parse_args(argv)

    result = detect_reps(PoseSequence.load(args.keypoints))
    print(
        f"{len(result.reps)} reps  ({result.view.view} view, leg: {result.side}, "
        f"signal: {result.signal_name}, usable frames: {result.usable_fraction:.0%})"
    )
    for r in result.reps:
        print(
            f"  rep {r.number}: {r.start_s:5.2f}s -> bottom {r.bottom_s:5.2f}s "
            f"-> {r.end_s:5.2f}s   {result.signal_name} at bottom {r.bottom_signal:.2f}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
