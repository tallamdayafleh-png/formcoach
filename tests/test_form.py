import numpy as np
import pytest

from cv.config import default_thresholds
from cv.form import (
    DepthTracker,
    FormSignals,
    assess_rep,
    knee_hip_width_ratio,
    thigh_angle_deg,
)

TH = default_thresholds()


# --- thigh angle (image y points down) ---------------------------------------


@pytest.mark.parametrize(
    ("hip", "knee", "expected"),
    [
        ((0, 0), (10, 0), 0.0),  # level
        ((0, 0), (10, 10), 45.0),  # hip above knee
        ((0, 0), (0, 10), 90.0),  # standing, vertical thigh
        ((0, 10), (10, 0), -45.0),  # hip below knee
    ],
)
def test_thigh_angle(hip, knee, expected):
    assert thigh_angle_deg(hip, knee) == pytest.approx(expected)


def test_thigh_angle_ignores_facing_direction():
    # lifter facing left or right gives the same angle
    assert thigh_angle_deg((0, 0), (10, 5)) == pytest.approx(
        thigh_angle_deg((0, 0), (-10, 5))
    )


def test_thigh_angle_same_point_is_nan():
    assert np.isnan(thigh_angle_deg((3, 3), (3, 3)))


# --- knee/hip width ratio (front view) ---------------------------------------


def test_width_ratio_knees_over_hips():
    # hips 20 px apart; knees 20 apart -> 1.0, knees 12 apart -> 0.6
    hips = ((60, 50), (40, 50))
    assert knee_hip_width_ratio((60, 80), (40, 80), *hips) == pytest.approx(1.0)
    assert knee_hip_width_ratio((56, 80), (44, 80), *hips) == pytest.approx(0.6)


def test_width_ratio_zero_hip_width_is_nan():
    assert np.isnan(knee_hip_width_ratio((1, 0), (0, 0), (5, 0), (5, 0)))


# --- DepthTracker (live "depth hit") -----------------------------------------


def test_depth_tracker_fires_once_per_rep():
    tracker = DepthTracker(max_thigh_deg=28)
    fired = [tracker.update(x) for x in [80, 50, 30, 25, 15, 26, 60, 85]]
    assert fired == [False, False, False, True, False, False, False, False]
    assert tracker.min_thigh_deg == 15
    tracker.reset()
    assert [tracker.update(x) for x in [80, 20]] == [False, True]


def test_depth_tracker_ignores_nan():
    tracker = DepthTracker(28)
    assert tracker.update(float("nan")) is False
    assert tracker.hit is False


# --- assess_rep --------------------------------------------------------------


def _signals(thigh: list[float], width: list[float]) -> FormSignals:
    return FormSignals(np.array(thigh, dtype=float), np.array(width, dtype=float))


def test_side_view_checks_depth_only():
    sig = _signals([80, 40, 20, 40, 80], [0.5] * 5)
    form = assess_rep(1, 0, 4, "side", sig, TH)
    assert form.depth.hit
    assert form.depth.thigh_deg == 20
    assert form.valgus is None  # not assessed from the side


def test_shallow_rep_flagged():
    sig = _signals([80, 50, 40, 50, 80], [1.0] * 5)
    assert assess_rep(1, 0, 4, "side", sig, TH).depth.hit is False


def test_front_view_checks_valgus_only():
    sig = _signals([np.nan] * 5, [1.0, 0.9, 0.7, 0.9, 1.0])
    form = assess_rep(1, 0, 4, "front", sig, TH)
    assert form.depth is None
    assert form.valgus.caving
    assert form.valgus.knee_hip_width_ratio == pytest.approx(0.7)


def test_front_view_knees_out_ok():
    sig = _signals([np.nan] * 5, [1.0, 1.1, 1.2, 1.1, 1.0])
    assert assess_rep(1, 0, 4, "front", sig, TH).valgus.caving is False


def test_only_frames_inside_the_rep_count():
    # a cave-in after the rep ended belongs to no rep
    sig = _signals([np.nan] * 6, [1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    assert assess_rep(1, 0, 4, "front", sig, TH).valgus.caving is False


def test_no_data_means_not_assessed():
    sig = _signals([np.nan] * 3, [np.nan] * 3)
    form = assess_rep(1, 0, 2, "side", sig, TH)
    assert form.depth is None
