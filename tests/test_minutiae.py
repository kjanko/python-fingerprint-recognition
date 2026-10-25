import numpy as np

from fingerprint_recognition import MinutiaKind, RidgeMap, extract_minutiae
from fingerprint_recognition.minutiae import crossing_number


def _ridge_map(ridges):
    return RidgeMap(
        ridges=ridges,
        mask=np.ones(ridges.shape, dtype=bool),
        orientation=np.full(ridges.shape, np.pi / 2),
    )


def test_crossing_number_classifies_skeleton_pixels():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[3, 1:6] = True  # horizontal line
    skeleton[1:3, 3] = True  # branch going up from the middle

    numbers = crossing_number(skeleton)

    assert numbers[3, 1] == 1  # left end
    assert numbers[3, 2] == 2  # ridge continues
    assert numbers[3, 3] == 3  # bifurcation
    assert numbers[1, 3] == 1  # top end


def test_finds_a_ridge_ending():
    ridges = np.zeros((80, 80), dtype=bool)
    for row in (20, 30, 50, 60):
        ridges[row - 1 : row + 2, :] = True
    ridges[39:42, :40] = True  # the middle ridge stops halfway

    minutiae = extract_minutiae(_ridge_map(ridges))

    assert len(minutiae) == 1
    assert minutiae[0].kind is MinutiaKind.ENDING
    assert abs(minutiae[0].x - 39) <= 2
    assert abs(minutiae[0].y - 40) <= 1


def test_finds_a_bifurcation():
    ridges = np.zeros((80, 80), dtype=bool)
    for row in (15, 65):
        ridges[row - 1 : row + 2, :] = True
    ridges[39:42, :40] = True
    for col in range(40, 80):  # the ridge forks into two branches
        offset = (col - 40) // 3
        ridges[39 - offset : 42 - offset, col] = True
        ridges[39 + offset : 42 + offset, col] = True

    minutiae = extract_minutiae(_ridge_map(ridges))

    bifurcations = [m for m in minutiae if m.kind is MinutiaKind.BIFURCATION]
    assert len(bifurcations) == 1
    assert abs(bifurcations[0].x - 40) <= 5


def test_ignores_minutiae_at_the_edge_of_the_print():
    ridges = np.zeros((80, 80), dtype=bool)
    ridges[39:42, 5:] = True  # ends right at the border of the print

    assert extract_minutiae(_ridge_map(ridges)) == []
