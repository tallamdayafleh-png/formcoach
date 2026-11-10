import json

import pytest

from cv.landmarks import LANDMARK_INDEX
from cv.schema import (
    SCHEMA_VERSION,
    FramePose,
    Landmark,
    PoseSequence,
    VideoMeta,
    parse_mirrored,
    with_mirrored,
)


def _lms(value: float) -> list[Landmark]:
    return [Landmark(value, value, value, 0.9, 0.95) for _ in range(33)]


def _sequence() -> PoseSequence:
    meta = VideoMeta(
        source="clip.mov", fps=29.97, width=720, height=1280, frame_count=3
    )
    frames = [
        FramePose(0, 0.0, _lms(0.25), _lms(0.1)),
        FramePose(1, 33.4, None, None),  # no person detected
        FramePose(2, 66.7, _lms(0.5), _lms(0.2)),
    ]
    return PoseSequence(meta=meta, model="pose_landmarker_full", frames=frames)


def test_round_trip_through_file(tmp_path):
    seq = _sequence()
    path = tmp_path / "nested" / "kp.json"
    seq.save(path)
    loaded = PoseSequence.load(path)
    assert loaded == seq


def test_undetected_frame_stays_null(tmp_path):
    path = tmp_path / "kp.json"
    _sequence().save(path)
    raw = json.loads(path.read_text())
    assert raw["frames"][1]["landmarks"] is None
    assert raw["frames"][1]["world_landmarks"] is None
    assert PoseSequence.load(path).frames[1].detected is False


def test_file_has_version_and_format():
    data = _sequence().to_dict()
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["landmark_format"] == ["x", "y", "z", "visibility", "presence"]
    assert len(data["frames"][0]["landmarks"][0]) == 5


def test_wrong_version_rejected():
    data = _sequence().to_dict()
    data["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        PoseSequence.from_dict(data)


def test_version_1_file_loads_as_not_mirrored():
    data = _sequence().to_dict()
    data["schema_version"] = 1
    del data["meta"]["mirrored"]
    assert PoseSequence.from_dict(data).meta.mirrored is False


def test_mirrored_flag_round_trips(tmp_path):
    seq = with_mirrored(_sequence(), True)
    path = tmp_path / "kp.json"
    seq.save(path)
    assert PoseSequence.load(path).meta.mirrored is True


def _labelled() -> PoseSequence:
    """Each landmark's x equals its own index, so swaps are easy to see."""
    lms = [Landmark(float(i), 0, 0, 0.9, 1) for i in range(33)]
    meta = VideoMeta(source="v", fps=30, width=10, height=10, frame_count=2)
    frames = [FramePose(0, 0.0, lms, lms), FramePose(1, 33.0, None, None)]
    return PoseSequence(meta=meta, model="m", frames=frames)


def test_with_mirrored_swaps_left_and_right_labels():
    seq = with_mirrored(_labelled(), True)
    lms = seq.frames[0].landmarks
    assert seq.meta.mirrored is True
    # the point labelled left_knee (25) is now what was labelled right_knee (26)
    assert lms[LANDMARK_INDEX["left_knee"]].x == LANDMARK_INDEX["right_knee"]
    assert lms[LANDMARK_INDEX["right_knee"]].x == LANDMARK_INDEX["left_knee"]
    assert lms[LANDMARK_INDEX["nose"]].x == LANDMARK_INDEX["nose"]
    assert seq.frames[0].world_landmarks == lms
    assert seq.frames[1].landmarks is None  # no-pose frames untouched


def test_with_mirrored_is_reversible_and_idempotent():
    seq = _labelled()
    once = with_mirrored(seq, True)
    assert with_mirrored(once, True) is once  # already mirrored: no change
    assert with_mirrored(once, False) == seq  # swapping back restores labels


@pytest.mark.parametrize(
    ("choice", "value"), [("auto", None), ("yes", True), ("no", False)]
)
def test_parse_mirrored(choice, value):
    assert parse_mirrored(choice) is value


def test_values_rounded():
    lm = Landmark(0.123456789, 0.5, -0.0000012, 1.0, 1.0)
    assert lm.to_list() == [0.12346, 0.5, -0.0, 1.0, 1.0]


def test_detection_rate():
    assert _sequence().detection_rate == pytest.approx(2 / 3)
    empty = PoseSequence(_sequence().meta, "m", [])
    assert empty.detection_rate == 0.0
