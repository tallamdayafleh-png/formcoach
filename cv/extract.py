"""Command line: extract pose keypoints from a video into a JSON file.

python -m cv.extract data/squat.mov -o out/squat_keypoints.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cv.pose import (
    DEFAULT_MAX_SIDE,
    MIRRORED_CHOICES,
    extract_keypoints,
    parse_mirrored,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output JSON (default: out/<video name>_keypoints.json)",
    )
    parser.add_argument("--model", type=Path, help="path to a .task model file")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    parser.add_argument(
        "--mirrored",
        choices=MIRRORED_CHOICES,
        default="auto",
        help="is the video a mirror image? auto = yes for iPhone front camera",
    )
    args = parser.parse_args(argv)

    output = args.output or Path("out") / f"{args.video.stem}_keypoints.json"

    def progress(n: int) -> None:
        if n % 50 == 0:
            print(f"  {n} frames", flush=True)

    start = time.perf_counter()
    seq = extract_keypoints(
        args.video,
        model_path=args.model,
        max_side=args.max_side,
        max_frames=args.max_frames,
        on_progress=progress,
        mirrored=parse_mirrored(args.mirrored),
    )
    elapsed = time.perf_counter() - start
    seq.save(output)

    n = seq.meta.frame_count
    print(
        f"{n} frames in {elapsed:.1f}s ({n / elapsed:.1f} fps), "
        f"person detected in {seq.detection_rate:.0%}, "
        f"mirrored: {'yes' if seq.meta.mirrored else 'no'} -> {output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
