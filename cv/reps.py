"""Squat rep detection from the knee angle (roadmap item 3).

Pipeline, all causal (past frames only) so live mode can run the same code:

    keypoints ─► pick camera-facing leg ─► knee/hip angle per frame
              ─► drop low-visibility frames ─► CausalSmoother ─► RepCounter

RepCounter uses two thresholds (hysteresis) so jitter around a single line
can't count extra reps:

    up_deg   ───────╮           ╭────── must rise above to finish a rep
                    │           │
    down_deg ───────┼───────────┼────── must drop below to count as "down"
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
from cv.filters import landmark_track, smooth_series
from cv.landmarks import LANDMARK_INDEX
from cv.schema import PoseSequence

SIDES = ("left", "right")


def _joints(side: str) -> tuple[int, int, int, int]:
    """(shoulder, hip, knee, ankle) landmark indices for one side."""
    return tuple(  # type: ignore[return-value]
        LANDMARK_INDEX[f"{side}_{j}"] for j in ("shoulder", "hip", "knee", "ankle")
    )


@dataclass(frozen=True)
class RepSpan:
    start_frame: int  # last frame standing before the descent
    bottom_frame: int  # smallest (smoothed) knee angle
    end_frame: int  # first frame back above up_deg


class _State(Enum):
    UNKNOWN = "unknown"  # no trustworthy standing position seen yet
    STANDING = "standing"
    DOWN = "down"


class RepCounter:
    """Frame-by-frame rep state machine. Feed it smoothed knee angles.

    A rep is STANDING -> DOWN -> STANDING. Losing tracking (NaN) cancels any
    rep in progress and waits for the lifter to be seen standing again, so a
    rep is never counted from guessed data.
    """

    def __init__(self, cfg: RepThresholds) -> None:
        self.cfg = cfg
        self.state = _State.UNKNOWN
        self._start = (0, 0.0)  # frame, t_ms of the last frame above up_deg
        self._peak = 0.0  # highest angle while standing: the drop reference
        self._bottom = (0, 0.0, math.inf)  # frame, t_ms, angle

    def _stand(self, frame: int, t_ms: float, angle: float) -> None:
        self.state = _State.STANDING
        self._start = (frame, t_ms)
        self._peak = angle

    def update(self, frame: int, t_ms: float, angle: float) -> RepSpan | None:
        """Process one frame; return a RepSpan on the frame a rep completes."""
        cfg = self.cfg
        if not math.isfinite(angle):
            self.state = _State.UNKNOWN
            return None

        if self.state is _State.UNKNOWN:
            if angle > cfg.up_deg:
                self._stand(frame, t_ms, angle)
        elif self.state is _State.STANDING:
            if angle > cfg.up_deg:
                self._start = (frame, t_ms)
                self._peak = max(self._peak, angle)
            elif angle < cfg.down_deg:
                self.state = _State.DOWN
                self._bottom = (frame, t_ms, angle)
        elif self.state is _State.DOWN:
            if angle < self._bottom[2]:
                self._bottom = (frame, t_ms, angle)
            if angle > cfg.up_deg:
                start_f, start_t = self._start
                duration_s = (t_ms - start_t) / 1000.0
                drop = self._peak - self._bottom[2]
                bottom_f = self._bottom[0]
                self._stand(frame, t_ms, angle)
                if duration_s >= cfg.min_rep_s and drop >= cfg.min_drop_deg:
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
    min_knee_deg: float  # smoothed knee angle at the bottom
    min_hip_deg: float  # smallest smoothed hip angle during the rep


@dataclass
class RepResult:
    side: str  # leg used: the one more visible to the camera
    reps: list[Rep]
    usable_fraction: float  # frames with a trustworthy knee angle
    knee_deg: np.ndarray = field(repr=False)  # smoothed, per frame
    hip_deg: np.ndarray = field(repr=False)

    def reps_completed_by(self, frame: int) -> int:
        """How many reps had finished by this frame (what a live counter shows)."""
        return sum(1 for r in self.reps if r.end_frame <= frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "rep_count": len(self.reps),
            "usable_fraction": round(self.usable_fraction, 3),
            "reps": [
                {
                    "number": r.number,
                    "start_frame": r.start_frame,
                    "bottom_frame": r.bottom_frame,
                    "end_frame": r.end_frame,
                    "start_s": round(r.start_s, 3),
                    "bottom_s": round(r.bottom_s, 3),
                    "end_s": round(r.end_s, 3),
                    "min_knee_deg": round(r.min_knee_deg, 1),
                    "min_hip_deg": _round_or_none(r.min_hip_deg),
                }
                for r in self.reps
            ],
        }


def _round_or_none(x: float) -> float | None:
    return None if not math.isfinite(x) else round(x, 1)


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


def detect_reps(
    seq: PoseSequence, thresholds: FormThresholds | None = None
) -> RepResult:
    th = thresholds or default_thresholds()
    trk = th.tracking
    side = choose_side(seq)
    shoulder, hip, knee, ankle = _joints(side)
    t_ms = np.array([f.timestamp_ms for f in seq.frames], dtype=float)

    pts = {
        j: landmark_track(seq, j, trk.visibility_min)
        for j in (shoulder, hip, knee, ankle)
    }
    knee_raw = joint_angle(pts[hip], pts[knee], pts[ankle])
    hip_raw = joint_angle(pts[shoulder], pts[hip], pts[knee])
    knee_s = smooth_series(knee_raw, t_ms, trk.smooth_tau_s, trk.max_gap_s)
    hip_s = smooth_series(hip_raw, t_ms, trk.smooth_tau_s, trk.max_gap_s)

    counter = RepCounter(th.squat_rep)
    reps: list[Rep] = []
    for i, (t, angle) in enumerate(zip(t_ms, knee_s, strict=True)):
        span = counter.update(i, float(t), float(angle))
        if span is None:
            continue
        hip_window = hip_s[span.start_frame : span.end_frame + 1]
        finite = hip_window[np.isfinite(hip_window)]
        reps.append(
            Rep(
                number=len(reps) + 1,
                start_frame=span.start_frame,
                bottom_frame=span.bottom_frame,
                end_frame=span.end_frame,
                start_s=t_ms[span.start_frame] / 1000,
                bottom_s=t_ms[span.bottom_frame] / 1000,
                end_s=t_ms[span.end_frame] / 1000,
                min_knee_deg=float(knee_s[span.bottom_frame]),
                min_hip_deg=float(finite.min()) if finite.size else math.nan,
            )
        )

    usable = float(np.isfinite(knee_raw).mean()) if len(knee_raw) else 0.0
    return RepResult(
        side=side, reps=reps, usable_fraction=usable, knee_deg=knee_s, hip_deg=hip_s
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count squat reps from keypoints.")
    parser.add_argument("keypoints", type=Path, help="keypoints JSON from cv.extract")
    parser.add_argument("--json", type=Path, help="also write the result as JSON")
    args = parser.parse_args(argv)

    result = detect_reps(PoseSequence.load(args.keypoints))
    print(
        f"{len(result.reps)} reps  (leg: {result.side}, "
        f"usable frames: {result.usable_fraction:.0%})"
    )
    for r in result.reps:
        hip = "--" if not math.isfinite(r.min_hip_deg) else f"{r.min_hip_deg:.0f}"
        print(
            f"  rep {r.number}: {r.start_s:5.2f}s -> bottom {r.bottom_s:5.2f}s "
            f"-> {r.end_s:5.2f}s   knee {r.min_knee_deg:.0f} deg, hip {hip} deg"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
