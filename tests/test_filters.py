import math

import numpy as np
import pytest

from cv.filters import CausalSmoother, landmark_track, smooth_series
from cv.schema import FramePose, Landmark, PoseSequence, VideoMeta


def test_first_sample_passes_through():
    assert CausalSmoother(0.1, 0.3).update(0.0, 120.0) == 120.0


def test_constant_input_unchanged():
    t = np.arange(30) * 33.3
    out = smooth_series(np.full(30, 150.0), t, 0.1, 0.3)
    np.testing.assert_allclose(out, 150.0)


def test_step_response_matches_time_constant():
    # After one tau, an EMA has covered 1 - 1/e (~63%) of a step.
    s = CausalSmoother(tau_s=0.1, max_gap_s=1.0)
    s.update(0.0, 0.0)
    assert s.update(100.0, 100.0) == pytest.approx(100 * (1 - math.exp(-1)))


def test_same_smoothing_at_different_frame_rates():
    # Same signal sampled at 15 fps and 30 fps: outputs agree at shared times.
    def run(fps):
        t = np.arange(0, 1000, 1000 / fps)
        x = np.where(t >= 200, 100.0, 0.0)
        return dict(zip(t.round(3), smooth_series(x, t, 0.1, 0.3), strict=True))

    slow, fast = run(15), run(30)
    for t in (400.0, 600.0, 800.0):
        assert slow[t] == pytest.approx(fast[t], abs=3.0)


def test_is_causal():
    # Changing a future sample must not change any earlier output.
    t = np.arange(20) * 33.3
    x = np.linspace(170, 90, 20)
    base = smooth_series(x, t, 0.1, 0.3)
    x2 = x.copy()
    x2[15] = 0.0
    changed = smooth_series(x2, t, 0.1, 0.3)
    np.testing.assert_array_equal(base[:15], changed[:15])


def test_short_gap_holds_last_value():
    t = np.arange(6) * 33.3
    x = np.array([100, 100, np.nan, np.nan, 100, 100], dtype=float)
    out = smooth_series(x, t, 0.1, 0.3)
    np.testing.assert_allclose(out, 100.0)


def test_long_gap_gives_nan_and_resets():
    t = np.arange(20) * 33.3  # 0.3 s gap limit ~= 9 frames
    x = np.full(20, np.nan)
    x[0] = 100.0
    x[19] = 50.0
    out = smooth_series(x, t, 0.1, 0.3)
    assert np.isnan(out[15])
    assert out[19] == 50.0  # restarted, not averaged with stale 100


def test_leading_nan_stays_nan():
    out = smooth_series([np.nan, 5.0], [0.0, 33.0], 0.1, 0.3)
    assert np.isnan(out[0])
    assert out[1] == 5.0


def test_landmark_track_scales_to_pixels_and_masks():
    lms = [Landmark(0.5, 0.25, 0, 0.9, 1) for _ in range(33)]
    low = [Landmark(0.5, 0.25, 0, 0.1, 1) for _ in range(33)]
    seq = PoseSequence(
        VideoMeta("v", 30, width=200, height=400, frame_count=3),
        "m",
        [
            FramePose(0, 0, lms, None),
            FramePose(1, 33, None, None),
            FramePose(2, 66, low, None),
        ],
    )
    track = landmark_track(seq, 25, min_visibility=0.5)
    np.testing.assert_allclose(track[0], [100, 100])  # portrait: y * 400
    assert np.isnan(track[1]).all()  # no pose
    assert np.isnan(track[2]).all()  # low visibility
