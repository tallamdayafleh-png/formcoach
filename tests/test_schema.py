import json

import pytest

from cv.schema import (
    SCHEMA_VERSION,
    FramePose,
    Landmark,
    PoseSequence,
    VideoMeta,
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


def test_values_rounded():
    lm = Landmark(0.123456789, 0.5, -0.0000012, 1.0, 1.0)
    assert lm.to_list() == [0.12346, 0.5, -0.0, 1.0, 1.0]


def test_detection_rate():
    assert _sequence().detection_rate == pytest.approx(2 / 3)
    empty = PoseSequence(_sequence().meta, "m", [])
    assert empty.detection_rate == 0.0
