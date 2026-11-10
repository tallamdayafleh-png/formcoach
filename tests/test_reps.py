import math

import numpy as np
import pytest

from cv.config import RepThresholds
from cv.reps import RepCounter, choose_side
from cv.schema import FramePose, Landmark, PoseSequence, VideoMeta

CFG = RepThresholds(down=130, up=155, min_rep_s=0.5, min_drop=25)
DT = 1000 / 30  # ms per frame


def squat_wave(n_reps: int, bottom: float = 85, top: float = 170, frames: int = 90):
    """Knee angle for n_reps squats: top -> bottom -> top, `frames` per rep."""
    phase = np.linspace(0, 2 * np.pi * n_reps, frames * n_reps, endpoint=False)
    return top - (top - bottom) * (1 - np.cos(phase)) / 2


def count(angles, cfg: RepThresholds = CFG):
    counter = RepCounter(cfg)
    spans = []
    for i, a in enumerate(angles):
        span = counter.update(i, i * DT, float(a))
        if span is not None:
            spans.append(span)
    return spans


@pytest.mark.parametrize("n", [1, 3, 5, 10])
def test_counts_each_cycle(n):
    assert len(count(squat_wave(n))) == n


def test_span_frames():
    (span,) = count(squat_wave(1))
    assert span.start_frame < span.bottom_frame < span.end_frame
    assert span.bottom_frame == pytest.approx(45, abs=1)  # middle of the cycle


def test_jitter_at_down_threshold_counts_once():
    # Hovering around down_deg for a while must not add reps.
    angles = [170] * 10 + [131, 129] * 15 + [170] * 10
    assert len(count(angles)) == 1


def test_jitter_at_up_threshold_counts_once():
    angles = [170] * 10 + [100] * 20 + [156, 154] * 10 + [170] * 5
    assert len(count(angles)) == 1


def test_shallow_dip_not_counted():
    assert count(squat_wave(3, bottom=135)) == []


def test_unfinished_last_rep_not_counted():
    wave = squat_wave(3)
    assert len(count(wave[: 2 * 90 + 45])) == 2  # ends at the bottom of rep 3


def test_must_start_from_standing():
    # Clip starts mid-squat: the half rep at the start isn't counted.
    wave = squat_wave(3)[45:]
    assert len(count(wave)) == 2


def test_lost_tracking_cancels_rep():
    angles = list(squat_wave(2))
    angles[100:120] = [math.nan] * 20  # lose tracking during rep 2's descent
    assert len(count(angles)) == 1


def test_drop_measured_from_full_standing():
    # Standing at 170 then down to 140 is a 30 deg drop (>= 25), even though
    # the last frame above up_deg (155) is only 15 deg above the bottom.
    cfg = RepThresholds(down=145, up=155, min_rep_s=0.5, min_drop=25)
    assert len(count(squat_wave(2, bottom=140), cfg)) == 2
    # Standing only reaches 160: 20 deg drop, rejected.
    assert count(squat_wave(2, bottom=140, top=160), cfg) == []


def test_too_fast_rep_rejected():
    fast = squat_wave(1, frames=12)  # 0.4 s, below min_rep_s
    assert count(np.concatenate([[170] * 5, fast, [170] * 5])) == []


def test_uses_config_values():
    wave = squat_wave(2, bottom=133)  # too shallow for down=130...
    assert count(wave) == []
    loose = RepThresholds(down=135, up=155, min_rep_s=0.5, min_drop=25)
    assert len(count(wave, loose)) == 2  # ...but counts with a looser config


def _seq(left_vis: float, right_vis: float) -> PoseSequence:
    lms = [Landmark(0.5, 0.5, 0, 0.9, 1) for _ in range(33)]
    for i in (23, 25, 27):
        lms[i] = Landmark(0.5, 0.5, 0, left_vis, 1)
    for i in (24, 26, 28):
        lms[i] = Landmark(0.5, 0.5, 0, right_vis, 1)
    meta = VideoMeta("v", 30, 100, 100, 1)
    return PoseSequence(meta, "m", [FramePose(0, 0, lms, None)])


def test_choose_side_picks_more_visible_leg():
    assert choose_side(_seq(0.9, 0.2)) == "left"
    assert choose_side(_seq(0.2, 0.9)) == "right"
