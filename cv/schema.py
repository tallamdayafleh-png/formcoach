"""Data model for the keypoints file written by pose extraction.

The keypoints JSON is the single source of truth for everything downstream
(overlay, rep detection, form checks). Bump SCHEMA_VERSION whenever the shape
of the file changes so old files fail loudly instead of being misread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

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
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported keypoints schema_version {version!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        m = data["meta"]
        meta = VideoMeta(
            source=m["source"],
            fps=m["fps"],
            width=m["width"],
            height=m["height"],
            frame_count=m["frame_count"],
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


def _lms_to_lists(lms: list[Landmark] | None) -> list[list[float]] | None:
    return None if lms is None else [lm.to_list() for lm in lms]


def _lists_to_lms(values: list[list[float]] | None) -> list[Landmark] | None:
    return None if values is None else [Landmark.from_list(v) for v in values]
