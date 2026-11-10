"""Form checks on the real clips, against hand-labelled ground truth.

Each clip is skipped if it isn't in data/ (clips are gitignored).
"""

import pytest

from cv.summary import summarize
from cv.video import is_front_camera
from tests.conftest import CLIPS, DATA, clip_available, load_keypoints


@pytest.mark.parametrize("name", list(CLIPS))
def test_clips_detected_as_front_camera(name):
    if not clip_available(name):
        pytest.skip(f"needs data/{CLIPS[name]['file']}")
    # so mirrored="auto" un-mirrors them without being told
    assert is_front_camera(DATA / CLIPS[name]["file"]) is True


@pytest.mark.parametrize("name", list(CLIPS))
def test_summary_matches_labels(name):
    if not clip_available(name):
        pytest.skip(f"needs data/{CLIPS[name]['file']}")
    label = CLIPS[name]
    s = summarize(load_keypoints(name))

    assert s["camera_view"] == label["view"]
    assert s["leg_used"] == label["leg"]  # real leg, after un-mirroring
    assert s["rep_count"] == label["reps"]
    flagged = {f["rep"]: f["issue"] for f in s["flags"]}
    assert flagged == label["bad"]
    assert s["tracking"]["confidence"] == "high"
