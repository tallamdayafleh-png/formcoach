"""Locate (and if needed download) the MediaPipe pose model file."""

from __future__ import annotations

import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# "full" is the middle of MediaPipe's lite / full / heavy trade-off. The URL is
# pinned to version 1 (not "latest") so results don't change under us.
POSE_MODEL_NAME = "pose_landmarker_full"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
DEFAULT_POSE_MODEL = MODELS_DIR / f"{POSE_MODEL_NAME}.task"


def ensure_pose_model(path: Path = DEFAULT_POSE_MODEL) -> Path:
    """Return the model path, downloading the default model if it is missing."""
    if path.is_file():
        return path
    if path != DEFAULT_POSE_MODEL:
        raise FileNotFoundError(f"Pose model not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading pose model to {path} ...")
    urllib.request.urlretrieve(POSE_MODEL_URL, tmp)
    tmp.replace(path)
    return path


if __name__ == "__main__":
    print(ensure_pose_model())
