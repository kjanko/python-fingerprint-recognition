"""Minutiae extraction from an enhanced ridge map."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage.morphology import skeletonize

from .enhance import RidgeMap

# Minutiae closer than this to the edge of the print are usually artefacts.
BORDER_MARGIN = 12
# Minutiae closer than this to another one are usually spurs or ridge breaks.
MIN_DISTANCE = 6


class MinutiaKind(Enum):
    ENDING = 1
    BIFURCATION = 3


@dataclass(frozen=True)
class Minutia:
    """A ridge ending or bifurcation; ``angle`` is the ridge direction in radians."""

    x: int
    y: int
    angle: float
    kind: MinutiaKind


def crossing_number(skeleton: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    """Half the number of 0/1 transitions around each pixel's 8 neighbours.

    1 marks a ridge ending, 2 a ridge continuing and 3 a bifurcation.
    """
    p = np.pad(skeleton.astype(int), 1)
    # Neighbours in circular order, starting above the pixel.
    ring = [
        p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
        p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2],
    ]  # fmt: skip
    transitions = sum(np.abs(ring[i] - ring[(i + 1) % 8]) for i in range(8))
    return np.asarray(transitions, dtype=int) // 2


def extract_minutiae(ridge_map: RidgeMap) -> list[Minutia]:
    """Find ridge endings and bifurcations inside the print."""
    skeleton = skeletonize(ridge_map.ridges)
    numbers = crossing_number(skeleton)
    inside = ndimage.binary_erosion(
        ndimage.binary_fill_holes(ridge_map.mask), iterations=BORDER_MARGIN
    )

    candidates = [
        Minutia(int(x), int(y), float(ridge_map.orientation[y, x]), kind)
        for kind in MinutiaKind
        for y, x in zip(*np.nonzero(skeleton & (numbers == kind.value) & inside), strict=True)
    ]
    if len(candidates) < 2:
        return candidates

    points = np.array([(m.x, m.y) for m in candidates], dtype=float)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    isolated = distances.min(axis=1) >= MIN_DISTANCE
    return [m for m, keep in zip(candidates, isolated, strict=True) if keep]
