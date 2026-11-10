"""MediaPipe Pose landmark names and the body connections we draw.

MediaPipe outputs 33 landmarks in a fixed order. We keep our own copy of the
names and connections instead of importing them from mediapipe so that the
rest of the pipeline (rep detection, form checks, overlay) does not depend on
mediapipe's API, which has changed between major versions.
"""

LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

NUM_LANDMARKS = len(LANDMARK_NAMES)

# name -> index, e.g. LANDMARK_INDEX["left_knee"] == 25
LANDMARK_INDEX: dict[str, int] = {name: i for i, name in enumerate(LANDMARK_NAMES)}

_BODY_CONNECTION_NAMES: tuple[tuple[str, str], ...] = (
    # torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # feet
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
)

# Bones drawn by the overlay, as (start_index, end_index). Face and fingers are
# left out: they are noisy and irrelevant to squat form.
BODY_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (LANDMARK_INDEX[a], LANDMARK_INDEX[b]) for a, b in _BODY_CONNECTION_NAMES
)
