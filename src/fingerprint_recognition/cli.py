"""Command-line interface: ``fingerprint-recognition FIRST SECOND``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2

from . import __version__
from .matching import DEFAULT_MATCH_THRESHOLD, extract_features, load_image, match_features

EXIT_MATCH = 0
EXIT_NO_MATCH = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fingerprint-recognition",
        description="Compare two fingerprint images. Exits 0 on a match, 1 otherwise.",
    )
    parser.add_argument("first", type=Path, help="path to the first fingerprint image")
    parser.add_argument("second", type=Path, help="path to the second fingerprint image")
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=DEFAULT_MATCH_THRESHOLD,
        help="paired minutiae needed for a match (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--save", type=Path, metavar="PATH", help="write an image of the matches")
    parser.add_argument(
        "--show", action="store_true", help="display keypoints and matches (needs matplotlib)"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        first_image, second_image = load_image(args.first), load_image(args.second)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    first, second = extract_features(first_image), extract_features(second_image)
    result = match_features(first, second, threshold=args.threshold)

    if args.save or args.show:
        matches_image = cv2.drawMatches(  # type: ignore[call-overload]
            first_image,
            first.keypoints,
            second_image,
            second.keypoints,
            result.matches,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
        )
        if args.save:
            cv2.imwrite(str(args.save), matches_image)
        if args.show:
            from .plot import show

            show(first_image, first, second_image, second, matches_image)

    if args.json:
        print(
            json.dumps(
                {
                    "first": str(args.first),
                    "second": str(args.second),
                    "score": result.score,
                    "threshold": result.threshold,
                    "minutiae": [len(first.minutiae), len(second.minutiae)],
                    "is_match": result.is_match,
                }
            )
        )
    else:
        verdict = "Fingerprint matches." if result.is_match else "Fingerprint does not match."
        print(f"{verdict} ({result.score} paired minutiae, threshold {result.threshold})")

    return EXIT_MATCH if result.is_match else EXIT_NO_MATCH


if __name__ == "__main__":
    raise SystemExit(main())
