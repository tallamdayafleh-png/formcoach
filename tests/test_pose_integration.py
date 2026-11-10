"""End-to-end check on a real clip. Skipped if the clip or model is absent."""

from itertools import pairwise
from pathlib import Path

import pytest

from cv.models import DEFAULT_POSE_MODEL, REPO_ROOT
from cv.pose import extract_keypoints

CLIP = REPO_ROOT / "data" / "squat.mov"

pytestmark = pytest.mark.skipif(
    not (CLIP.is_file() and DEFAULT_POSE_MODEL.is_file()),
    reason="needs data/squat.mov and models/pose_landmarker_full.task",
)


def test_extracts_pose_from_real_clip(tmp_path: Path):
    seq = extract_keypoints(CLIP, max_frames=30)
    assert seq.meta.frame_count == 30
    assert seq.meta.height > seq.meta.width  # portrait, rotation applied
    assert seq.detection_rate > 0.9
    first = next(f for f in seq.frames if f.detected)
    assert len(first.landmarks) == 33
    assert len(first.world_landmarks) == 33
    assert all(0.0 <= lm.visibility <= 1.0 for lm in first.landmarks)
    ts = [f.timestamp_ms for f in seq.frames]
    assert all(b > a for a, b in pairwise(ts))
