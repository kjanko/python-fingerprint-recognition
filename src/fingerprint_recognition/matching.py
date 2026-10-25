"""Feature extraction and fingerprint matching.

Matching works in two stages. ORB descriptors computed at each minutia
propose correspondences, and the similarity transform (rotation, translation
and scale) that the largest number of them agree on aligns the two prints.
Under that alignment, minutiae of the first print are paired with minutiae
of the second that sit at the same place with the same ridge direction. The
score is the number of such pairs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from .enhance import RidgeMap, enhance
from .minutiae import Minutia, extract_minutiae

DEFAULT_MATCH_THRESHOLD = 12
DESCRIPTOR_PATCH_SIZE = 31
# Lowe's ratio test: keep a match only if clearly better than the runner-up.
RATIO_TEST = 0.9
# Correspondences further than this from where the alignment puts them are outliers.
ALIGNMENT_TOLERANCE = 12.0
MIN_ALIGNMENT_INLIERS = 3
# Same sensor and resolution, so the scale between two prints should be close to 1.
SCALE_RANGE = (0.8, 1.25)
# Two aligned minutiae pair up if they are this close, in pixels and radians.
PAIR_DISTANCE = 10.0
PAIR_ANGLE = np.radians(20)

GrayImage = MatLike


@dataclass(frozen=True)
class Features:
    """Minutiae found on a fingerprint, plus ORB descriptors used for alignment."""

    minutiae: Sequence[Minutia]
    keypoints: Sequence[cv2.KeyPoint]
    descriptors: MatLike | None
    ridge_map: RidgeMap


@dataclass(frozen=True)
class MatchResult:
    """Outcome of comparing two fingerprints.

    ``score`` is the number of minutiae that line up once the prints are
    aligned; higher is more similar. ``matches`` are the descriptor matches
    consistent with the alignment, and ``transform`` is the 2x3 matrix that
    maps the first print onto the second (``None`` if no alignment was found).
    """

    score: int
    threshold: int
    matches: Sequence[cv2.DMatch]
    transform: npt.NDArray[np.float64] | None

    @property
    def is_match(self) -> bool:
        return self.score >= self.threshold


def load_image(path: str | os.PathLike[str]) -> GrayImage:
    """Read an image from disk as 8-bit grayscale."""
    image = cv2.imread(os.fspath(path), cv2.IMREAD_GRAYSCALE) if os.path.isfile(path) else None
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def extract_features(image: GrayImage) -> Features:
    """Enhance a grayscale fingerprint, find its minutiae and describe them."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ridge_map = enhance(clahe.apply(image))
    minutiae = extract_minutiae(ridge_map)

    # Describe the clean ridge pattern rather than the noisy scan, with each
    # patch rotated to the local ridge direction so rotation doesn't matter.
    # Ridge directions are ambiguous by 180 degrees, so every minutia is also
    # described upside down (class_id 1); matching compares those against the
    # upright descriptors (class_id 0) of the other print.
    ridge_image = cv2.GaussianBlur(np.where(ridge_map.ridges, 0, 255).astype(np.uint8), (5, 5), 1)
    candidates = [
        cv2.KeyPoint(
            float(m.x),
            float(m.y),
            DESCRIPTOR_PATCH_SIZE,
            (float(np.degrees(m.angle)) + 180 * flipped) % 360,
            class_id=flipped,
        )
        for m in minutiae
        for flipped in (0, 1)
    ]
    orb = cv2.ORB.create(edgeThreshold=DESCRIPTOR_PATCH_SIZE, patchSize=DESCRIPTOR_PATCH_SIZE)
    keypoints, descriptors = orb.compute(ridge_image, candidates)
    return Features(minutiae, keypoints, descriptors, ridge_map)


def _estimate_alignment(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> tuple[complex, complex, npt.NDArray[np.bool_]] | None:
    """Find the similarity transform ``z -> a*z + b`` most correspondences agree on.

    Points are complex numbers ``x + iy``. Every pair of correspondences
    defines a candidate transform; trying them all keeps the result
    deterministic, which random sampling (RANSAC) would not.
    """
    zs = source[:, 0] + 1j * source[:, 1]
    zt = target[:, 0] + 1j * target[:, 1]
    i, j = np.triu_indices(len(zs), 1)
    usable = np.abs(zs[j] - zs[i]) >= 1
    i, j = i[usable], j[usable]
    if i.size == 0:
        return None

    a = (zt[j] - zt[i]) / (zs[j] - zs[i])
    b = zt[i] - a * zs[i]
    errors = np.abs(a[:, None] * zs[None, :] + b[:, None] - zt[None, :])
    support = (errors < ALIGNMENT_TOLERANCE).sum(axis=1)
    support[(np.abs(a) < SCALE_RANGE[0]) | (np.abs(a) > SCALE_RANGE[1])] = 0

    best = int(np.argmax(support))
    if support[best] < MIN_ALIGNMENT_INLIERS:
        return None

    # Refine with a least-squares fit to the inliers of the best candidate.
    inliers = errors[best] < ALIGNMENT_TOLERANCE
    design = np.stack([zs[inliers], np.ones(inliers.sum())], axis=1)
    (a_fit, b_fit), *_ = np.linalg.lstsq(design, zt[inliers], rcond=None)
    if not SCALE_RANGE[0] <= abs(a_fit) <= SCALE_RANGE[1]:
        return None
    return complex(a_fit), complex(b_fit), inliers


def _pair_minutiae(
    first: Sequence[Minutia], second: Sequence[Minutia], a: complex, b: complex
) -> int:
    """Count one-to-one pairs of minutiae that coincide after alignment."""
    if not first or not second:
        return 0
    moved = a * np.array([m.x + 1j * m.y for m in first]) + b
    fixed = np.array([m.x + 1j * m.y for m in second])
    distances = np.abs(moved[:, None] - fixed[None, :])

    # Ridge directions are only defined up to 180 degrees.
    moved_angles = np.array([m.angle for m in first]) + np.angle(a)
    fixed_angles = np.array([m.angle for m in second])
    turn = np.abs((moved_angles[:, None] - fixed_angles[None, :] + np.pi / 2) % np.pi - np.pi / 2)

    rows, cols = np.nonzero((distances < PAIR_DISTANCE) & (turn < PAIR_ANGLE))
    used_first: set[int] = set()
    used_second: set[int] = set()
    for row, col in sorted(zip(rows, cols, strict=True), key=lambda rc: distances[rc]):
        if row not in used_first and col not in used_second:
            used_first.add(row)
            used_second.add(col)
    return len(used_first)


def _match_one_way(first: Features, second: Features, threshold: int) -> MatchResult:
    no_match = MatchResult(score=0, threshold=threshold, matches=[], transform=None)
    if (
        first.descriptors is None
        or second.descriptors is None
        or len(first.descriptors) < 2
        or len(second.descriptors) < 2
    ):
        return no_match

    upright = np.array([k.class_id == 0 for k in second.keypoints])
    upright_index = np.flatnonzero(upright)
    if upright_index.size < 2:
        return no_match

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates = [
        cv2.DMatch(best.queryIdx, int(upright_index[best.trainIdx]), best.distance)
        for best, runner_up in matcher.knnMatch(first.descriptors, second.descriptors[upright], k=2)
        if best.distance < RATIO_TEST * runner_up.distance
    ]
    if len(candidates) < MIN_ALIGNMENT_INLIERS:
        return no_match

    source = np.array([first.keypoints[m.queryIdx].pt for m in candidates], dtype=float)
    target = np.array([second.keypoints[m.trainIdx].pt for m in candidates], dtype=float)
    alignment = _estimate_alignment(source, target)
    if alignment is None:
        return no_match

    a, b, inliers = alignment
    return MatchResult(
        score=_pair_minutiae(first.minutiae, second.minutiae, a, b),
        threshold=threshold,
        matches=[m for m, keep in zip(candidates, inliers, strict=True) if keep],
        transform=np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]]),
    )


def match_features(
    first: Features, second: Features, threshold: int = DEFAULT_MATCH_THRESHOLD
) -> MatchResult:
    """Align two feature sets and count the minutiae they share.

    Both directions are tried and the weaker result is kept, so the score
    doesn't depend on the order of the prints.
    """
    forward = _match_one_way(first, second, threshold)
    backward = _match_one_way(second, first, threshold)
    return forward if forward.score <= backward.score else _swap(backward)


def _swap(result: MatchResult) -> MatchResult:
    """Express a second-to-first result as first-to-second."""
    if result.transform is None:
        return result
    inverse = cv2.invertAffineTransform(result.transform)
    return MatchResult(
        score=result.score,
        threshold=result.threshold,
        matches=[cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in result.matches],
        transform=np.asarray(inverse, dtype=np.float64),
    )


def compare(
    first: GrayImage | str | os.PathLike[str],
    second: GrayImage | str | os.PathLike[str],
    threshold: int = DEFAULT_MATCH_THRESHOLD,
) -> MatchResult:
    """Compare two fingerprints given as grayscale arrays or image paths."""
    first_image = first if isinstance(first, np.ndarray) else load_image(first)
    second_image = second if isinstance(second, np.ndarray) else load_image(second)
    return match_features(
        extract_features(first_image), extract_features(second_image), threshold=threshold
    )
