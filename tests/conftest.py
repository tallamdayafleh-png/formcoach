import pytest

from cv.models import DEFAULT_POSE_MODEL, REPO_ROOT

SQUAT_CLIP = REPO_ROOT / "data" / "squat.mov"

needs_clip = pytest.mark.skipif(
    not (SQUAT_CLIP.is_file() and DEFAULT_POSE_MODEL.is_file()),
    reason="needs data/squat.mov and models/pose_landmarker_full.task",
)


@pytest.fixture(scope="session")
def squat_sequence():
    """Full-clip keypoints for data/squat.mov, extracted once per test run."""
    from cv.pose import extract_keypoints

    return extract_keypoints(SQUAT_CLIP)
