import cv2
import numpy as np
import pytest

from cv.form import DepthResult, FormSignals, RepForm, ValgusResult
from cv.landmarks import LANDMARK_INDEX
from cv.overlay import (
    GREEN,
    RED,
    HudContext,
    draw_skeleton,
    hud_lines,
    render_overlay,
    to_pixel,
)
from cv.reps import Rep, RepResult
from cv.schema import FramePose, Landmark, PoseSequence, VideoMeta
from cv.view import ViewResult

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


# --- HUD ---------------------------------------------------------------------


def _rep(number: int, end_frame: int) -> Rep:
    return Rep(number, end_frame - 20, end_frame - 10, end_frame, 0, 0, 0, 90, 90, 70)


def _ctx(view: str = "side") -> HudContext:
    knee = np.array([170.0, 120.0, np.nan, 160.0, 160.0])
    hip = np.array([175.0, 100.0, np.nan, 165.0, 165.0])
    ratio = np.array([1.0, 0.7, np.nan, 0.98, 0.99])
    side = "both" if view == "front" else "left"
    reps = RepResult(
        ViewResult(view, 0.1), side, "knee_deg", [_rep(1, 1), _rep(2, 3)], 0.8,
        ratio if view == "front" else knee, knee, hip,
    )  # fmt: skip
    signals = FormSignals(
        thigh_deg=np.array([85.0, 20.0, np.nan, 80.0, 80.0]),
        knee_width_ratio=np.array([1.1, 0.8, np.nan, 1.0, 1.0]),
    )
    if view == "front":
        forms = [
            RepForm(1, None, ValgusResult(0.95, False)),
            RepForm(2, None, ValgusResult(0.74, True)),
        ]
    else:
        forms = [
            RepForm(1, DepthResult(20.0, True), None),
            RepForm(2, DepthResult(38.0, False), None),
        ]
    return HudContext(reps, forms, signals, max_thigh_deg=28)


def _text(frame: int, ctx: HudContext) -> list[str]:
    return [t for t, _ in hud_lines(FramePose(frame, 0.0, None, None), ctx)]


def test_hud_without_reps_is_one_line():
    assert len(hud_lines(FramePose(0, 0.0, None, None))) == 1


@pytest.mark.parametrize("view", ["side", "front"])
@pytest.mark.parametrize(("frame", "count"), [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2)])
def test_hud_rep_count_matches_completed_reps(view, frame, count):
    # every overlay, side or front, shows the rep count
    assert _text(frame, _ctx(view))[1].startswith(f"REP {count} ")


def test_hud_side_shows_angles_and_depth_marker():
    lines = _text(1, _ctx("side"))
    assert "(side view, left leg)" in lines[1]
    assert "knee 120 deg" in lines[2]
    assert "hip 100 deg" in lines[2]
    assert "thigh 20 deg   DEPTH" in lines[2]  # 20 <= 28 threshold
    assert "DEPTH" not in _text(0, _ctx("side"))[2]  # 85 deg: standing


def test_hud_untrusted_angle_shows_dashes():
    line = _text(2, _ctx("side"))[2]
    assert "knee --" in line
    assert "thigh --" in line


def test_hud_front_shows_ratios():
    lines = _text(1, _ctx("front"))
    assert "(front view)" in lines[1]
    assert "hip height 0.70" in lines[2]
    assert "knee/hip width 0.80" in lines[2]


@pytest.mark.parametrize(
    ("view", "frame", "text", "color"),
    [
        ("side", 1, "rep 1: DEPTH OK", GREEN),
        ("side", 3, "rep 2: ABOVE PARALLEL", RED),
        ("front", 1, "rep 1: KNEES OK", GREEN),
        ("front", 4, "rep 2: KNEES CAVING", RED),
    ],
)
def test_hud_verdict_for_last_rep(view, frame, text, color):
    verdict_text, verdict_color = hud_lines(
        FramePose(frame, 0.0, None, None), _ctx(view)
    )[-1]
    assert verdict_text.startswith(text)
    assert verdict_color == color


def test_hud_no_verdict_before_first_rep():
    assert len(_text(0, _ctx("side"))) == 3


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
