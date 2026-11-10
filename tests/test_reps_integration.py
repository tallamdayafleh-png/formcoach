"""Rep detection on the real clip. Labeled ground truth: 5 reps."""

import pytest

from cv.reps import detect_reps
from tests.conftest import needs_clip

pytestmark = needs_clip


def test_squat_clip_has_five_reps(squat_sequence):
    result = detect_reps(squat_sequence)
    assert len(result.reps) == 5
    assert result.side == "right"  # real leg facing the camera (video is mirrored)
    # bottoms observed by eye in the overlay, about 100 frames apart
    for rep, expected in zip(result.reps, [40, 140, 240, 340, 440], strict=True):
        assert rep.bottom_frame == pytest.approx(expected, abs=15)
        assert rep.min_knee_deg < 100


def test_walking_to_camera_adds_no_rep(squat_sequence):
    result = detect_reps(squat_sequence)
    assert result.reps[-1].end_frame < 500  # nothing counted in the last ~1.3 s
