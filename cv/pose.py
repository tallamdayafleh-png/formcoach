"""Run MediaPipe Pose Landmarker over a video and collect per-frame keypoints."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from cv.models import ensure_pose_model
from cv.schema import FramePose, Landmark, PoseSequence, VideoMeta
from cv.video import iter_frames, read_info

# Frames are shrunk so their longest side is at most this before inference.
# The model itself runs at 256x256, so feeding it 4K only costs time; landmark
# x/y are fractions of the frame, so they are unaffected by the resize.
DEFAULT_MAX_SIDE = 1280


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is <= max_side, keeping aspect ratio."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _to_landmarks(raw: list) -> list[Landmark]:
    return [
        Landmark(
            x=lm.x,
            y=lm.y,
            z=lm.z,
            visibility=lm.visibility if lm.visibility is not None else 0.0,
            presence=lm.presence if lm.presence is not None else 0.0,
        )
        for lm in raw
    ]


def extract_keypoints(
    video_path: str | Path,
    model_path: Path | None = None,
    max_side: int = DEFAULT_MAX_SIDE,
    max_frames: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> PoseSequence:
    """Detect one person's pose on every frame of a video.

    Frames where no person is found are kept, with landmarks set to None, so
    later steps can tell "no data" apart from "bad form".
    """
    video_path = Path(video_path)
    model_path = ensure_pose_model() if model_path is None else model_path
    info = read_info(video_path)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    frames: list[FramePose] = []
    width = height = 0
    last_ts_int = -1
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame in iter_frames(video_path, max_frames=max_frames):
            height, width = frame.image.shape[:2]
            rgb = cv2.cvtColor(
                resize_max_side(frame.image, max_side), cv2.COLOR_BGR2RGB
            )
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # MediaPipe needs strictly increasing integer milliseconds.
            ts_int = max(round(frame.timestamp_ms), last_ts_int + 1)
            last_ts_int = ts_int
            result = landmarker.detect_for_video(mp_image, ts_int)

            if result.pose_landmarks:
                lms = _to_landmarks(result.pose_landmarks[0])
                world = _to_landmarks(result.pose_world_landmarks[0])
            else:
                lms = world = None
            frames.append(
                FramePose(
                    index=frame.index,
                    timestamp_ms=frame.timestamp_ms,
                    landmarks=lms,
                    world_landmarks=world,
                )
            )
            if on_progress is not None:
                on_progress(frame.index + 1)

    meta = VideoMeta(
        source=video_path.name,
        fps=info.fps,
        width=width or info.width,
        height=height or info.height,
        frame_count=len(frames),
    )
    return PoseSequence(meta=meta, model=model_path.stem, frames=frames)
