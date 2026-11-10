"""Draw the skeleton from a keypoints file onto the video (roadmap item 2).

The overlay reads the saved keypoints JSON rather than re-running the model,
so what you see is exactly the data rep detection and form checks will use.

    python -m cv.overlay data/squat.mov out/squat_keypoints.json -o out/overlay.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from cv.config import default_thresholds
from cv.landmarks import BODY_CONNECTIONS, LANDMARK_NAMES
from cv.pose import resize_max_side
from cv.reps import RepResult, detect_reps
from cv.schema import FramePose, Landmark, PoseSequence
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


def _fmt_deg(x: float) -> str:
    # OpenCV's built-in fonts can't draw the degree sign, so spell it out.
    return "--" if not np.isfinite(x) else f"{x:.0f} deg"


def hud_lines(pose: FramePose, reps: RepResult | None = None) -> list[str]:
    """Text for the HUD: frame/time/status, then rep count and angles."""
    status = "pose" if pose.detected else "NO POSE"
    lines = [f"frame {pose.index}  {pose.timestamp_ms / 1000:.2f}s  {status}"]
    if reps is not None:
        i = pose.index
        lines.append(
            f"REP {reps.reps_completed_by(i)}   "
            f"knee {_fmt_deg(reps.knee_deg[i])}   "
            f"hip {_fmt_deg(reps.hip_deg[i])}   ({reps.side} leg)"
        )
    return lines


def draw_hud(
    image: np.ndarray, pose: FramePose, reps: RepResult | None = None
) -> np.ndarray:
    """Write the HUD lines in the top-left corner."""
    scale = max(image.shape[:2]) / 1600
    thickness = max(1, round(2 * scale))
    for n, text in enumerate(hud_lines(pose, reps)):
        org = (round(20 * scale), round((50 + 45 * n) * scale))
        color = (255, 255, 255) if pose.detected else (0, 0, 255)
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
    reps = detect_reps(seq)

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
            draw_hud(image, pose, reps)
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
