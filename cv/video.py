"""Video reading with correct orientation and per-frame timestamps.

Phone video is usually variable frame rate (VFR): frames are not evenly
spaced, so `frame_index / fps` drifts. We use each frame's own presentation
timestamp from the decoder instead, and repair it if it ever fails to
increase (MediaPipe's VIDEO mode rejects non-increasing timestamps).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Container-level info. frame_count is only the container's estimate."""

    fps: float
    width: int
    height: int
    frame_count_estimate: int


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: float
    image: np.ndarray  # BGR, upright


def open_video(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")
    # iPhones store portrait video sideways plus a rotation flag. Keep OpenCV's
    # auto-rotation on (the default) so frames come out upright.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def read_info(path: str | Path) -> VideoInfo:
    cap = open_video(path)
    try:
        return VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count_estimate=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


_LENS_MODEL_KEY = b"com.apple.quicktime.camera.lens_model"
_SCAN_BYTES = 4 * 1024 * 1024  # metadata lives in the moov atom: start or end


def lens_model(path: str | Path) -> str | None:
    """Camera lens description from iPhone video metadata, if present.

    e.g. "iPhone 16 Pro front camera 2.69mm f/1.9". Only the first and last
    few MB are scanned, which is where QuickTime stores its metadata.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(_SCAN_BYTES)
        f.seek(max(0, size - _SCAN_BYTES))
        tail = f.read()
    for chunk in (head, tail):
        if _LENS_MODEL_KEY not in chunk:
            continue
        match = re.search(rb"iPhone[ -~]{0,80}?camera[ -~]{0,40}", chunk)
        if match:
            return match.group().decode("ascii").strip()
    return None


def is_front_camera(path: str | Path) -> bool | None:
    """True/False from iPhone metadata; None if the file doesn't say."""
    model = lens_model(path)
    if model is None:
        return None
    return "front camera" in model.lower()


def fix_timestamp(
    raw_ms: float | None, prev_ms: float | None, frame_ms: float
) -> float:
    """Return a usable timestamp for a frame.

    Uses the decoder's timestamp when it is valid and strictly after the
    previous one; otherwise falls back to previous + one nominal frame.
    """
    if prev_ms is None:
        if raw_ms is None or not math.isfinite(raw_ms) or raw_ms < 0:
            return 0.0
        return raw_ms
    if raw_ms is None or not math.isfinite(raw_ms) or raw_ms <= prev_ms:
        return prev_ms + frame_ms
    return raw_ms


def iter_frames(path: str | Path, max_frames: int | None = None) -> Iterator[Frame]:
    """Yield upright BGR frames with monotonic timestamps in milliseconds."""
    cap = open_video(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ms = 1000.0 / fps if fps and fps > 0 else 1000.0 / 30.0
        prev_ms: float | None = None
        index = 0
        while max_frames is None or index < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            raw_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = fix_timestamp(raw_ms, prev_ms, frame_ms)
            yield Frame(index=index, timestamp_ms=ts, image=image)
            prev_ms = ts
            index += 1
    finally:
        cap.release()
