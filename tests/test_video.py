import math
from itertools import pairwise

import cv2
import numpy as np
import pytest

from cv.pose import resize_max_side
from cv.video import fix_timestamp, iter_frames, read_info


@pytest.fixture
def synthetic_video(tmp_path):
    """10 frames of 64x48 video at 10 fps; frame i is filled with value i*20."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


# --- fix_timestamp -----------------------------------------------------------


def test_first_frame_uses_decoder_time():
    assert fix_timestamp(12.5, None, 33.3) == 12.5


@pytest.mark.parametrize("raw", [None, math.nan, -5.0])
def test_first_frame_invalid_becomes_zero(raw):
    assert fix_timestamp(raw, None, 33.3) == 0.0


def test_increasing_time_kept():
    assert fix_timestamp(40.0, 33.0, 33.3) == 40.0


@pytest.mark.parametrize("raw", [33.0, 20.0, None, math.inf])
def test_stalled_or_invalid_time_advances_one_frame(raw):
    assert fix_timestamp(raw, 33.0, 33.3) == pytest.approx(66.3)


# --- reading -----------------------------------------------------------------


def test_read_info(synthetic_video):
    info = read_info(synthetic_video)
    assert info.fps == pytest.approx(10.0)
    assert (info.width, info.height) == (64, 48)


def test_iter_frames(synthetic_video):
    frames = list(iter_frames(synthetic_video))
    assert [f.index for f in frames] == list(range(10))
    assert all(f.image.shape == (48, 64, 3) for f in frames)
    ts = [f.timestamp_ms for f in frames]
    assert ts[0] == pytest.approx(0.0)
    assert all(b > a for a, b in pairwise(ts))
    assert ts[-1] == pytest.approx(900.0, abs=1.0)
    # frames come out in order: brightness rises with index (JPEG is lossy)
    means = [f.image.mean() for f in frames]
    assert means == sorted(means)


def test_iter_frames_max_frames(synthetic_video):
    assert len(list(iter_frames(synthetic_video, max_frames=3))) == 3


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_info(tmp_path / "nope.mov")


# --- resize_max_side ---------------------------------------------------------


def test_resize_portrait_4k():
    img = np.zeros((3840, 2160, 3), dtype=np.uint8)
    assert resize_max_side(img, 1280).shape == (1280, 720, 3)


def test_resize_leaves_small_image_alone():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert resize_max_side(img, 1280) is img
