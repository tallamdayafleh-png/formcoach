"""Data model for the keypoints file written by pose extraction.

The keypoints JSON is the single source of truth for everything downstream
(overlay, rep detection, form checks). Bump SCHEMA_VERSION whenever the shape
of the file changes so old files fail loudly instead of being misread.

Version history:
  1: initial
  2: meta.mirrored (version 1 files load as mirrored = false)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cv.landmarks import MIRROR_INDEX

SCHEMA_VERSION = 2
_READABLE_VERSIONS = (1, 2)

# 5 decimals on normalized coords is ~0.02 px on a 4K frame: plenty, and it
# keeps the JSON roughly 3x smaller than full float precision.
_DECIMALS = 5


@dataclass(frozen=True)
class Landmark:
    """One body point.

    For image landmarks, x and y are fractions of frame width/height (0..1,
    origin top-left) and z is relative depth. For world landmarks, x/y/z are
    meters from the hip center. visibility and presence are 0..1 scores.
    """

    x: float
    y: float
    z: float
    visibility: float
    presence: float

    def to_list(self) -> list[float]:
        return [
            round(v, _DECIMALS)
            for v in (self.x, self.y, self.z, self.visibility, self.presence)
        ]

    @classmethod
    def from_list(cls, values: list[float]) -> Landmark:
        return cls(*values)


@dataclass(frozen=True)
class FramePose:
    """Pose for one decoded frame. Landmark lists are None if no person found."""

    index: int
    timestamp_ms: float
    landmarks: list[Landmark] | None
    world_landmarks: list[Landmark] | None

    @property
    def detected(self) -> bool:
        return self.landmarks is not None


@dataclass(frozen=True)
class VideoMeta:
    source: str
    fps: float  # nominal fps from the container; phone video is often variable
    width: int  # after rotation is applied, i.e. as the frames are displayed
    height: int
    frame_count: int  # frames actually decoded, not the container's estimate
    # True if the video is a mirror image (front camera with "Mirror Front
    # Camera" on). Landmark labels have then been swapped so "left_knee" is
    # the lifter's real left knee; x/y still match the video as recorded.
    mirrored: bool = False


@dataclass(frozen=True)
class PoseSequence:
    meta: VideoMeta
    model: str
    frames: list[FramePose]

    @property
    def detection_rate(self) -> float:
        """Fraction of frames where a person was detected."""
        if not self.frames:
            return 0.0
        return sum(f.detected for f in self.frames) / len(self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "landmark_format": ["x", "y", "z", "visibility", "presence"],
            "meta": {
                "source": self.meta.source,
                "fps": self.meta.fps,
                "width": self.meta.width,
                "height": self.meta.height,
                "frame_count": self.meta.frame_count,
                "mirrored": self.meta.mirrored,
            },
            "model": self.model,
            "frames": [
                {
                    "index": f.index,
                    "timestamp_ms": round(f.timestamp_ms, 3),
                    "landmarks": _lms_to_lists(f.landmarks),
                    "world_landmarks": _lms_to_lists(f.world_landmarks),
                }
                for f in self.frames
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseSequence:
        version = data.get("schema_version")
        if version not in _READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported keypoints schema_version {version!r}; "
                f"expected one of {_READABLE_VERSIONS}"
            )
        m = data["meta"]
        meta = VideoMeta(
            source=m["source"],
            fps=m["fps"],
            width=m["width"],
            height=m["height"],
            frame_count=m["frame_count"],
            mirrored=bool(m.get("mirrored", False)) if version >= 2 else False,
        )
        frames = [
            FramePose(
                index=f["index"],
                timestamp_ms=f["timestamp_ms"],
                landmarks=_lists_to_lms(f["landmarks"]),
                world_landmarks=_lists_to_lms(f["world_landmarks"]),
            )
            for f in data["frames"]
        ]
        return cls(meta=meta, model=data["model"], frames=frames)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")))

    @classmethod
    def load(cls, path: str | Path) -> PoseSequence:
        return cls.from_dict(json.loads(Path(path).read_text()))


MIRRORED_CHOICES = ("auto", "yes", "no")


def parse_mirrored(choice: str) -> bool | None:
    """CLI --mirrored value -> True/False, or None for auto."""
    return {"auto": None, "yes": True, "no": False}[choice]


def _swap_sides(lms: list[Landmark] | None) -> list[Landmark] | None:
    return None if lms is None else [lms[MIRROR_INDEX[i]] for i in range(len(lms))]


def with_mirrored(seq: PoseSequence, mirrored: bool) -> PoseSequence:
    """Return seq with left/right labels set for a mirrored or normal video.

    In a mirror image the lifter's real left leg appears where a right leg
    would be, so MediaPipe labels it "right". Swapping every left/right pair
    fixes the labels. Positions are untouched, so the skeleton still lines up
    with the video as recorded. Swapping twice restores the original, so this
    also undoes a wrong setting.
    """
    if seq.meta.mirrored == mirrored:
        return seq
    frames = [
        replace(
            f,
            landmarks=_swap_sides(f.landmarks),
            world_landmarks=_swap_sides(f.world_landmarks),
        )
        for f in seq.frames
    ]
    return replace(seq, meta=replace(seq.meta, mirrored=mirrored), frames=frames)


def _lms_to_lists(lms: list[Landmark] | None) -> list[list[float]] | None:
    return None if lms is None else [lm.to_list() for lm in lms]


def _lists_to_lms(values: list[list[float]] | None) -> list[Landmark] | None:
    return None if values is None else [Landmark.from_list(v) for v in values]
