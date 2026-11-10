"""Command line: video (or keypoints JSON) -> pose-summary JSON.

    python -m cv.analyze data/squat2.mov
    python -m cv.analyze out/squat2_keypoints.json

A video is run through pose extraction first, and its keypoints are saved to
out/<name>_keypoints.json so the overlay can reuse them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cv.schema import MIRRORED_CHOICES, PoseSequence, parse_mirrored, with_mirrored
from cv.summary import summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a squat set.")
    parser.add_argument("input", type=Path, help="video file or keypoints JSON")
    parser.add_argument(
        "-o", "--output", type=Path, help="default: out/<name>_summary.json"
    )
    parser.add_argument(
        "--mirrored",
        choices=MIRRORED_CHOICES,
        default="auto",
        help=(
            "is the video a mirror image? auto = yes for iPhone front camera "
            "(for a keypoints JSON, auto keeps the file's setting)"
        ),
    )
    args = parser.parse_args(argv)
    mirrored = parse_mirrored(args.mirrored)

    if args.input.suffix.lower() == ".json":
        seq = PoseSequence.load(args.input)
        if mirrored is not None:
            seq = with_mirrored(seq, mirrored)
        stem = args.input.stem.removesuffix("_keypoints")
    else:
        from cv.pose import extract_keypoints  # slow import; only when needed

        stem = args.input.stem
        seq = extract_keypoints(args.input, mirrored=mirrored)
        seq.save(Path("out") / f"{stem}_keypoints.json")

    summary = summarize(seq)
    output = args.output or Path("out") / f"{stem}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
