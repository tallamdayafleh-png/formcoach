"""Draw the skeleton from a keypoints file onto the video (roadmap item 2).

The overlay reads the saved keypoints JSON rather than re-running the model,
so what you see is exactly the data rep detection and form checks will use.

    python -m cv.overlay data/squat.mov out/squat_keypoints.json -o out/overlay.mp4
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv.config import default_thresholds
from cv.form import FormSignals, RepForm, form_signals
from cv.landmarks import BODY_CONNECTIONS, LANDMARK_NAMES
from cv.pose import resize_max_side
from cv.reps import RepResult, detect_reps
from cv.schema import FramePose, Landmark, PoseSequence
from cv.summary import assess_all
from cv.video import iter_frames

DEFAULT_OUT_MAX_SIDE = 1280

# BGR colours. Left/right get different colours so a left-right swap by the
# model is easy to spot.
LEFT_COLOR = (255, 160, 0)  # blue
RIGHT_COLOR = (0, 140, 255)  # orange
CENTER_COLOR = (255, 255, 255)
LOW_VIS_COLOR = (150, 150, 150)


def to_pixel(lm: Landmark, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized (0..1) landmark to integer pixel coordinates."""
    return round(lm.x * width), round(lm.y * height)


def _side_color(index: int) -> tuple[int, int, int]:
    name = LANDMARK_NAMES[index]
    if name.startswith("left_"):
        return LEFT_COLOR
    if name.startswith("right_"):
        return RIGHT_COLOR
    return CENTER_COLOR


def _bone_color(a: int, b: int) -> tuple[int, int, int]:
    ca, cb = _side_color(a), _side_color(b)
    return ca if ca == cb else CENTER_COLOR


def draw_skeleton(
    image: np.ndarray,
    landmarks: list[Landmark] | None,
    threshold: float | None = None,
) -> np.ndarray:
    """Draw bones and joints onto image in place and return it.

    Joints below the visibility threshold (default: tracking.visibility_min
    from the shared config) are drawn hollow grey and their bones skipped, so
    weak tracking is obvious instead of looking like real data. Does nothing
    if landmarks is None (no person detected on this frame).
    """
    if landmarks is None:
        return image
    if threshold is None:
        threshold = default_thresholds().tracking.visibility_min
    h, w = image.shape[:2]
    scale = max(h, w) / 1000  # keep line width similar at any resolution
    bone_px = max(1, round(4 * scale))
    joint_px = max(2, round(6 * scale))

    joints = {i for pair in BODY_CONNECTIONS for i in pair}
    visible = {i for i in joints if landmarks[i].visibility >= threshold}

    for a, b in BODY_CONNECTIONS:
        if a in visible and b in visible:
            cv2.line(
                image,
                to_pixel(landmarks[a], w, h),
                to_pixel(landmarks[b], w, h),
                _bone_color(a, b),
                bone_px,
                cv2.LINE_AA,
            )
    for i in sorted(joints):
        center = to_pixel(landmarks[i], w, h)
        if i in visible:
            cv2.circle(image, center, joint_px, _side_color(i), -1, cv2.LINE_AA)
        else:
            cv2.circle(
                image,
                center,
                joint_px,
                LOW_VIS_COLOR,
                max(1, bone_px // 2),
                cv2.LINE_AA,
            )
    return image


WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (80, 220, 80)


def _fmt_deg(x: float) -> str:
    # OpenCV's built-in fonts can't draw the degree sign, so spell it out.
    return "--" if not np.isfinite(x) else f"{x:.0f} deg"


def _fmt(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:.2f}"


@dataclass
class HudContext:
    """Everything the HUD needs beyond the frame's own pose."""

    reps: RepResult
    forms: list[RepForm]
    signals: FormSignals
    max_thigh_deg: float  # depth threshold, for the live "DEPTH" marker


def _verdict(form: RepForm) -> tuple[str, tuple[int, int, int]] | None:
    if form.depth is not None:
        d = form.depth
        if d.hit:
            return f"rep {form.rep}: DEPTH OK (thigh {_fmt_deg(d.thigh_deg)})", GREEN
        return f"rep {form.rep}: ABOVE PARALLEL (thigh {_fmt_deg(d.thigh_deg)})", RED
    if form.valgus is not None:
        v = form.valgus
        ratio = _fmt(v.knee_hip_width_ratio)
        if v.caving:
            return f"rep {form.rep}: KNEES CAVING (knee/hip width {ratio})", RED
        return f"rep {form.rep}: KNEES OK (knee/hip width {ratio})", GREEN
    return None


def hud_lines(
    pose: FramePose, ctx: HudContext | None = None
) -> list[tuple[str, tuple[int, int, int]]]:
    """HUD text and colour per line: status, rep count, live values, verdict."""
    status = "pose" if pose.detected else "NO POSE"
    lines = [
        (
            f"frame {pose.index}  {pose.timestamp_ms / 1000:.2f}s  {status}",
            WHITE if pose.detected else RED,
        )
    ]
    if ctx is None:
        return lines
    i, reps = pose.index, ctx.reps
    view = reps.view.view
    where = f"{view} view" + (f", {reps.side} leg" if view != "front" else "")
    lines.append((f"REP {reps.reps_completed_by(i)}   ({where})", WHITE))
    if view == "front":
        lines.append(
            (
                f"hip height {_fmt(reps.signal[i])}   "
                f"knee/hip width {_fmt(ctx.signals.knee_width_ratio[i])}",
                WHITE,
            )
        )
    else:
        thigh = ctx.signals.thigh_deg[i]
        depth = "   DEPTH" if np.isfinite(thigh) and thigh <= ctx.max_thigh_deg else ""
        lines.append(
            (
                f"knee {_fmt_deg(reps.knee_deg[i])}   hip {_fmt_deg(reps.hip_deg[i])}"
                f"   thigh {_fmt_deg(thigh)}{depth}",
                WHITE,
            )
        )
    last = reps.last_rep_by(i)
    if last is not None:
        verdict = _verdict(ctx.forms[last.number - 1])
        if verdict is not None:
            lines.append(verdict)
    return lines


def draw_hud(
    image: np.ndarray, pose: FramePose, ctx: HudContext | None = None
) -> np.ndarray:
    """Write the HUD lines in the top-left corner."""
    scale = max(image.shape[:2]) / 1600
    thickness = max(1, round(2 * scale))
    for n, (text, color) in enumerate(hud_lines(pose, ctx)):
        org = (round(20 * scale), round((50 + 45 * n) * scale))
        # dark outline first so the text is readable on any background
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thickness * 3, cv2.LINE_AA)  # fmt: skip
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)  # fmt: skip
    return image


def render_overlay(
    video_path: str | Path,
    keypoints: PoseSequence | str | Path,
    out_path: str | Path,
    out_max_side: int = DEFAULT_OUT_MAX_SIDE,
) -> Path:
    """Write a copy of the video with the skeleton drawn on every frame.

    Output is MP4 with the mp4v codec: fine for local review, but browsers
    can't play it (the web UI in roadmap item 10 will need H.264).
    """
    seq = (
        keypoints
        if isinstance(keypoints, PoseSequence)
        else PoseSequence.load(keypoints)
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    th = default_thresholds()
    reps = detect_reps(seq, th)
    ctx = HudContext(
        reps=reps,
        forms=assess_all(seq, reps, th),
        signals=form_signals(seq, reps.side, th.tracking),
        max_thigh_deg=th.squat.depth.max_thigh_deg,
    )

    writer: cv2.VideoWriter | None = None
    n = 0
    try:
        for frame in iter_frames(video_path):
            if frame.index >= len(seq.frames):
                raise ValueError(
                    f"Video has more frames than the keypoints file "
                    f"({len(seq.frames)}). Was it extracted from this video?"
                )
            image = resize_max_side(frame.image, out_max_side)
            if writer is None:
                h, w = image.shape[:2]
                fps = seq.meta.fps if seq.meta.fps > 0 else 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {out_path}")
            pose = seq.frames[frame.index]
            draw_skeleton(image, pose.landmarks)
            draw_hud(image, pose, ctx)
            writer.write(image)
            n += 1
    finally:
        if writer is not None:
            writer.release()

    if n != len(seq.frames):
        raise ValueError(
            f"Keypoints file has {len(seq.frames)} frames but the video has {n}. "
            "Was it extracted from this video?"
        )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw pose skeleton onto a video.")
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("keypoints", type=Path, help="keypoints JSON from cv.extract")
    parser.add_argument(
        "-o", "--output", type=Path, help="default: out/<video name>_overlay.mp4"
    )
    parser.add_argument("--max-side", type=int, default=DEFAULT_OUT_MAX_SIDE)
    args = parser.parse_args(argv)

    output = args.output or Path("out") / f"{args.video.stem}_overlay.mp4"
    render_overlay(args.video, args.keypoints, output, out_max_side=args.max_side)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
