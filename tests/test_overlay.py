import cv2
import numpy as np
import pytest

from cv.landmarks import LANDMARK_INDEX
from cv.overlay import draw_skeleton, render_overlay, to_pixel
from cv.schema import FramePose, Landmark, PoseSequence, VideoMeta

W, H = 200, 400


def _pose(visibility: float = 0.9) -> list[Landmark]:
    """33 landmarks spread across the frame, all with the given visibility."""
    return [
        Landmark(0.2 + 0.02 * i, 0.1 + 0.025 * i, 0.0, visibility, 1.0)
        for i in range(33)
    ]


def _blank() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def test_to_pixel():
    assert to_pixel(Landmark(0.5, 0.25, 0, 1, 1), W, H) == (100, 100)
    assert to_pixel(Landmark(0.0, 1.0, 0, 1, 1), W, H) == (0, 400)


def test_no_pose_leaves_frame_untouched():
    img = _blank()
    out = draw_skeleton(img, None)
    assert out is img
    assert not img.any()


def test_visible_pose_draws_joints():
    img = draw_skeleton(_blank(), _pose(0.9))
    x, y = to_pixel(_pose()[LANDMARK_INDEX["left_knee"]], W, H)
    assert img[y, x].any()


def test_low_visibility_draws_no_bones():
    lms = _pose(0.1)
    img = draw_skeleton(_blank(), lms)
    # midpoint of the left thigh would be covered by a bone if one were drawn
    hip = to_pixel(lms[LANDMARK_INDEX["left_hip"]], W, H)
    knee = to_pixel(lms[LANDMARK_INDEX["left_knee"]], W, H)
    mx, my = (hip[0] + knee[0]) // 2, (hip[1] + knee[1]) // 2
    assert not img[my, mx].any()
    # but the joint itself is still marked (hollow, grey)
    assert img.any()


def test_face_points_not_drawn():
    lms = _pose(0.9)
    img = draw_skeleton(_blank(), lms)
    x, y = to_pixel(lms[LANDMARK_INDEX["nose"]], W, H)
    assert not img[y, x].any()


# --- render_overlay ----------------------------------------------------------


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (W, H))
    for _ in range(5):
        writer.write(_blank())
    writer.release()
    return path


def _seq(n: int) -> PoseSequence:
    frames = [
        FramePose(i, i * 100.0, None if i == 2 else _pose(), None) for i in range(n)
    ]
    meta = VideoMeta(source="clip.avi", fps=10.0, width=W, height=H, frame_count=n)
    return PoseSequence(meta=meta, model="test", frames=frames)


def test_render_overlay_writes_every_frame(clip, tmp_path):
    out = render_overlay(clip, _seq(5), tmp_path / "out" / "overlay.mp4")
    cap = cv2.VideoCapture(str(out))
    count = 0
    while cap.read()[0]:
        count += 1
    assert count == 5
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == W
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == H
    cap.release()


def test_render_overlay_from_json_file(clip, tmp_path):
    kp = tmp_path / "kp.json"
    _seq(5).save(kp)
    assert render_overlay(clip, kp, tmp_path / "o.mp4").is_file()


@pytest.mark.parametrize("n", [3, 8])
def test_render_overlay_rejects_mismatched_keypoints(clip, tmp_path, n):
    with pytest.raises(ValueError, match="this video"):
        render_overlay(clip, _seq(n), tmp_path / "o.mp4")
