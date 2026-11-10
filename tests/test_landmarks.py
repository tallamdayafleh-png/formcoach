from mediapipe.tasks.python import vision

from cv.landmarks import (
    BODY_CONNECTIONS,
    LANDMARK_INDEX,
    LANDMARK_NAMES,
    MIRROR_INDEX,
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


def test_mirror_swaps_left_and_right():
    assert MIRROR_INDEX[LANDMARK_INDEX["left_knee"]] == LANDMARK_INDEX["right_knee"]
    assert MIRROR_INDEX[LANDMARK_INDEX["right_hip"]] == LANDMARK_INDEX["left_hip"]
    assert MIRROR_INDEX[LANDMARK_INDEX["mouth_left"]] == LANDMARK_INDEX["mouth_right"]
    assert MIRROR_INDEX[LANDMARK_INDEX["nose"]] == LANDMARK_INDEX["nose"]


def test_mirror_is_its_own_inverse():
    # swapping twice must give back the original labels
    assert all(MIRROR_INDEX[MIRROR_INDEX[i]] == i for i in range(NUM_LANDMARKS))
    assert sorted(MIRROR_INDEX) == list(range(NUM_LANDMARKS))


def test_connections_are_valid_body_joints():
    for a, b in BODY_CONNECTIONS:
        assert a != b
        # body joints start at left_shoulder (11); no face points
        assert 11 <= a < NUM_LANDMARKS
        assert 11 <= b < NUM_LANDMARKS
    assert len(set(BODY_CONNECTIONS)) == len(BODY_CONNECTIONS)
