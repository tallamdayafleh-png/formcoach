"""Turn keypoints into clean per-frame signals, using past frames only.

Everything here is causal: the value for frame t depends only on frames up
to t. That is what live mode in the browser can do, so the batch pipeline
uses the exact same logic and the two never disagree.

    raw angle   x_t ──► CausalSmoother ──► y_t = y_(t-1) + α·(x_t − y_(t-1))
                                            α = 1 − exp(−Δt / tau)

Using a time constant tau (seconds) instead of a fixed α makes the smoothing
identical at 15 fps (webcam) and 30 fps (phone), and on variable frame rate.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import ArrayLike

from cv.schema import PoseSequence


class CausalSmoother:
    """Exponential moving average that also bridges short tracking gaps.

    - A valid sample updates the average.
    - A missing sample (NaN) repeats the last output, as long as the last
      valid sample is at most max_gap_s old. Past that, output is NaN and the
      filter resets, so it restarts cleanly instead of averaging stale data.
    """

    def __init__(self, tau_s: float, max_gap_s: float) -> None:
        if tau_s < 0 or max_gap_s < 0:
            raise ValueError("tau_s and max_gap_s must be >= 0")
        self._tau_ms = tau_s * 1000.0
        self._max_gap_ms = max_gap_s * 1000.0
        self._y: float | None = None
        self._t: float | None = None  # time of the last valid sample

    def reset(self) -> None:
        self._y = None
        self._t = None

    def update(self, t_ms: float, x: float) -> float:
        if self._t is not None and t_ms - self._t > self._max_gap_ms:
            self.reset()
        if not math.isfinite(x):
            return math.nan if self._y is None else self._y
        if self._y is None or self._t is None or self._tau_ms == 0:
            self._y = x
        else:
            alpha = 1.0 - math.exp(-max(t_ms - self._t, 0.0) / self._tau_ms)
            self._y += alpha * (x - self._y)
        self._t = t_ms
        return self._y


class RunningMax:
    """Largest value seen in the last window_s seconds (past frames only).

    Uses a monotonic queue, so each update is O(1) on average: values that
    can never be the maximum again (older and smaller) are dropped.
    """

    def __init__(self, window_s: float) -> None:
        self._window_ms = window_s * 1000.0
        self._q: deque[tuple[float, float]] = deque()  # (t_ms, value)

    def update(self, t_ms: float, x: float) -> float:
        if math.isfinite(x):
            while self._q and self._q[-1][1] <= x:
                self._q.pop()
            self._q.append((t_ms, x))
        while self._q and self._q[0][0] < t_ms - self._window_ms:
            self._q.popleft()
        return self._q[0][1] if self._q else math.nan


def smooth_series(
    values: ArrayLike, t_ms: ArrayLike, tau_s: float, max_gap_s: float
) -> np.ndarray:
    """Run CausalSmoother over a whole series (same result as live mode)."""
    smoother = CausalSmoother(tau_s, max_gap_s)
    return np.array(
        [smoother.update(float(t), float(x)) for t, x in zip(t_ms, values, strict=True)]
    )


def landmark_track(seq: PoseSequence, index: int, min_visibility: float) -> np.ndarray:
    """Pixel (x, y) of one landmark on every frame, shape (n_frames, 2).

    Rows are NaN where no person was detected or the landmark's visibility is
    below min_visibility. Coordinates are scaled to pixels so that angles are
    correct on non-square (e.g. portrait) frames.
    """
    w, h = seq.meta.width, seq.meta.height
    out = np.full((len(seq.frames), 2), np.nan)
    for i, frame in enumerate(seq.frames):
        if frame.landmarks is None:
            continue
        lm = frame.landmarks[index]
        if lm.visibility >= min_visibility:
            out[i] = (lm.x * w, lm.y * h)
    return out
