"""Shared fixtures for the real-clip tests.

Running the pose model on all three clips takes ~2.5 minutes, so keypoints
are cached in out/<clip>_keypoints.json. A cached file is only reused if it
was written after both the clip and the model file, so a re-recorded clip or
a new model always triggers a fresh extraction.
"""

from pathlib import Path

import pytest

from cv.models import DEFAULT_POSE_MODEL, REPO_ROOT
from cv.schema import PoseSequence, with_mirrored

DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "out"

# Hand-labelled ground truth for the local test clips. All three were filmed
# on an iPhone front camera with "Mirror Front Camera" on, so they're mirror
# images; "leg" is the lifter's real leg facing the camera.
CLIPS = {
    "squat": {
        "file": "squat.mov",
        "view": "side",
        "leg": "right",
        "reps": 5,
        "bad": {},
    },
    "squat2": {
        "file": "squat2.mov",
        "view": "side",
        "leg": "right",
        "reps": 5,
        "bad": {2: "above_parallel", 3: "above_parallel"},
    },
    "frontview": {
        "file": "frontview.mov",
        "view": "front",
        "leg": "both",
        "reps": 5,
        "bad": {2: "valgus", 3: "valgus"},
    },
}
MIRRORED = True


def clip_available(name: str) -> bool:
    return (DATA / CLIPS[name]["file"]).is_file() and DEFAULT_POSE_MODEL.is_file()


SQUAT_CLIP = DATA / CLIPS["squat"]["file"]
needs_clip = pytest.mark.skipif(
    not clip_available("squat"),
    reason="needs data/squat.mov and models/pose_landmarker_full.task",
)


def load_keypoints(name: str) -> PoseSequence:
    clip = DATA / CLIPS[name]["file"]
    cached = OUT / f"{Path(CLIPS[name]['file']).stem}_keypoints.json"
    newest_input = max(clip.stat().st_mtime, DEFAULT_POSE_MODEL.stat().st_mtime)
    if cached.is_file() and cached.stat().st_mtime > newest_input:
        try:
            seq = PoseSequence.load(cached)
            if seq.meta.source.lower() == clip.name.lower():
                # labels only; no need to re-run the model
                return with_mirrored(seq, MIRRORED)
        except (ValueError, KeyError):
            pass  # old schema or corrupt: re-extract
    from cv.pose import extract_keypoints

    seq = extract_keypoints(clip, mirrored=MIRRORED)
    seq.save(cached)
    return seq


@pytest.fixture(scope="session")
def squat_sequence() -> PoseSequence:
    return load_keypoints("squat")
