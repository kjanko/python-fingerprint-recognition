import itertools

import cv2
import numpy as np
import pytest

from fingerprint_recognition import compare, extract_features, load_image, match_features

PRINTS = ["101_1", "101_2", "102_1", "102_2"]
GENUINE = [(a, b) for a, b in itertools.combinations(PRINTS, 2) if a[:3] == b[:3]]
IMPOSTOR = [(a, b) for a, b in itertools.combinations(PRINTS, 2) if a[:3] != b[:3]]


@pytest.fixture(scope="module")
def features(request):
    database = request.path.parent.parent / "database"
    return {name: extract_features(load_image(database / f"{name}.tif")) for name in PRINTS}


def _rotate(image, degrees):
    """Rotate without cropping, padding with white like the scanner background."""
    rows, cols = image.shape
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    width, height = int(rows * sin + cols * cos), int(rows * cos + cols * sin)
    matrix[:, 2] += (width - cols) / 2, (height - rows) / 2
    return cv2.warpAffine(image, matrix, (width, height), borderValue=255)


@pytest.mark.parametrize(("first", "second"), GENUINE)
def test_two_prints_of_the_same_finger_match(features, first, second):
    result = match_features(features[first], features[second])

    assert result.is_match
    assert result.score >= 2 * result.threshold


@pytest.mark.parametrize(("first", "second"), IMPOSTOR)
def test_prints_of_different_fingers_do_not_match(features, first, second):
    result = match_features(features[first], features[second])

    assert not result.is_match
    assert result.score <= result.threshold // 2


@pytest.mark.parametrize(("first", "second"), GENUINE + IMPOSTOR)
def test_score_does_not_depend_on_order(features, first, second):
    forward = match_features(features[first], features[second])
    backward = match_features(features[second], features[first])

    assert forward.score == backward.score


@pytest.mark.parametrize("degrees", [30, 90, 180])
def test_rotated_print_still_matches(database, degrees):
    reference = load_image(database / "101_1.tif")
    rotated = _rotate(load_image(database / "101_2.tif"), degrees)

    assert compare(reference, rotated).is_match
    assert not compare(load_image(database / "102_1.tif"), rotated).is_match


def test_transform_maps_the_first_print_onto_the_second(features):
    result = match_features(features["101_1"], features["101_2"])

    assert result.transform is not None
    first, second = features["101_1"], features["101_2"]
    for match in result.matches:
        x, y = first.keypoints[match.queryIdx].pt
        projected = result.transform @ np.array([x, y, 1])
        assert np.hypot(*(projected - second.keypoints[match.trainIdx].pt)) < 12


def test_threshold_decides_the_verdict(features):
    score = match_features(features["101_1"], features["101_2"]).score

    assert match_features(features["101_1"], features["101_2"], threshold=score).is_match
    assert not match_features(features["101_1"], features["101_2"], threshold=score + 1).is_match


def test_print_matches_itself(database):
    image = load_image(database / "101_1.tif")

    assert compare(image, image).is_match


def test_blank_image_never_matches():
    blank = np.full((200, 200), 128, dtype=np.uint8)
    blank[50:150, 50:150] = 0

    result = compare(blank, blank)

    assert not result.is_match
    assert result.transform is None


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.tif")
