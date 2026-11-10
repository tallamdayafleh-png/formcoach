from mediapipe.tasks.python import vision

from cv.landmarks import (
    BODY_CONNECTIONS,
    LANDMARK_INDEX,
    LANDMARK_NAMES,
    NUM_LANDMARKS,
)


def test_names_match_mediapipe_order():
    # Our hard-coded list must match MediaPipe's index order exactly, or every
    # joint downstream would be mislabeled.
    mp_names = [lm.name.lower() for lm in vision.PoseLandmark]
    assert list(LANDMARK_NAMES) == mp_names
    assert NUM_LANDMARKS == 33


def test_index_lookup():
    assert LANDMARK_INDEX["left_hip"] == 23
    assert LANDMARK_INDEX["left_knee"] == 25
    assert LANDMARK_INDEX["right_ankle"] == 28


def test_connections_are_valid_body_joints():
    for a, b in BODY_CONNECTIONS:
        assert a != b
        # body joints start at left_shoulder (11); no face points
        assert 11 <= a < NUM_LANDMARKS
        assert 11 <= b < NUM_LANDMARKS
    assert len(set(BODY_CONNECTIONS)) == len(BODY_CONNECTIONS)
