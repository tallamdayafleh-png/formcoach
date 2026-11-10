import numpy as np
import pytest

from cv.config import ViewThresholds
from cv.landmarks import LANDMARK_INDEX
from cv.schema import FramePose, Landmark, PoseSequence, VideoMeta
from cv.view import ANGLED, FRONT, SIDE, classify_ratio, classify_view

TH = ViewThresholds(
    side_max_shoulder_ratio=0.3, front_min_shoulder_ratio=0.5, classify_s=2.0
)


def _frame(i: int, shoulder_half_width: float) -> FramePose:
    """Torso 0.2 tall (normalized) in a square frame; shoulders centred at x=0.5."""
    lms = [Landmark(0.5, 0.5, 0, 0.9, 1) for _ in range(33)]
    w = shoulder_half_width
    lms[LANDMARK_INDEX["left_shoulder"]] = Landmark(0.5 + w, 0.3, 0, 0.9, 1)
    lms[LANDMARK_INDEX["right_shoulder"]] = Landmark(0.5 - w, 0.3, 0, 0.9, 1)
    lms[LANDMARK_INDEX["left_hip"]] = Landmark(0.55, 0.5, 0, 0.9, 1)
    lms[LANDMARK_INDEX["right_hip"]] = Landmark(0.45, 0.5, 0, 0.9, 1)
    return FramePose(i, i * 100.0, lms, None)


def _seq(frames: list[FramePose]) -> PoseSequence:
    return PoseSequence(VideoMeta("v", 10, 100, 100, len(frames)), "m", frames)


@pytest.mark.parametrize(
    ("ratio", "view"),
    [(0.1, SIDE), (0.3, SIDE), (0.4, ANGLED), (0.5, FRONT), (0.8, FRONT)],
)
def test_classify_ratio(ratio, view):
    assert classify_ratio(ratio, TH) == view


def test_nan_ratio_is_angled():
    assert classify_ratio(float("nan"), TH) == ANGLED


@pytest.mark.parametrize(("half_width", "view"), [(0.01, SIDE), (0.07, FRONT)])
def test_classify_view(half_width, view):
    # torso length 0.2: half width 0.01 -> ratio 0.1, 0.07 -> ratio 0.7
    result = classify_view(_seq([_frame(i, half_width) for i in range(10)]), TH)
    assert result.view == view


def test_only_first_seconds_decide():
    # side for the first 2 s, then the lifter turns to face the camera
    frames = [_frame(i, 0.01 if i <= 20 else 0.07) for i in range(60)]
    assert classify_view(_seq(frames), TH).view == SIDE


def test_no_pose_is_angled():
    frames = [FramePose(i, i * 100.0, None, None) for i in range(5)]
    result = classify_view(_seq(frames), TH)
    assert result.view == ANGLED
    assert np.isnan(result.shoulder_ratio)
