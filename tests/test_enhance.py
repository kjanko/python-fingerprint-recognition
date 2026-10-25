import numpy as np
from scipy import ndimage

from fingerprint_recognition.enhance import (
    enhance,
    ridge_filter,
    ridge_frequency,
    ridge_orient,
    ridge_segment,
)


def _reference_ridge_filter(im, orient, freq, kx=0.65, ky=0.65, angle_inc=3):
    """The original per-pixel loop, kept to check the vectorised version."""
    rows, cols = im.shape
    newim = np.zeros((rows, cols))
    unfreq = np.unique(np.round(freq[freq > 0] * 100) / 100)
    sigmax, sigmay = kx / unfreq[0], ky / unfreq[0]
    sze = int(np.round(3 * max(sigmax, sigmay)))
    x, y = np.meshgrid(np.linspace(-sze, sze, 2 * sze + 1), np.linspace(-sze, sze, 2 * sze + 1))
    reffilter = np.exp(-(x**2 / sigmax**2 + y**2 / sigmay**2)) * np.cos(2 * np.pi * unfreq[0] * x)
    filters = [
        ndimage.rotate(reffilter, -(o * angle_inc + 90), reshape=False)
        for o in range(180 // angle_inc)
    ]
    orientindex = np.round(orient / np.pi * 180 / angle_inc)
    for i in range(rows):
        for j in range(cols):
            if orientindex[i, j] < 1:
                orientindex[i, j] += 180 // angle_inc
            if orientindex[i, j] > 180 // angle_inc:
                orientindex[i, j] -= 180 // angle_inc
    for r, c in zip(*np.nonzero(freq > 0), strict=True):
        if sze < r < rows - sze and sze < c < cols - sze:
            block = im[r - sze : r + sze + 1, c - sze : c + sze + 1]
            newim[r, c] = np.sum(block * filters[int(orientindex[r, c]) - 1])
    return newim


def test_segment_masks_out_the_flat_background(ridges):
    normalised, mask = ridge_segment(ridges, block_size=16, threshold=0.1)

    assert mask[:, :112].all()
    assert not mask[:, 128:].any()
    assert abs(normalised[mask].mean()) < 1e-9
    assert abs(normalised[mask].std() - 1) < 1e-9


def test_orientation_follows_the_ridges(ridges):
    normalised, _ = ridge_segment(ridges)
    orientation = ridge_orient(normalised)

    # Ridges run along x = -y, i.e. 135 degrees in image coordinates.
    centre = orientation[40:120, 30:90]
    assert np.allclose(np.degrees(centre), 135, atol=3)


def test_frequency_matches_the_ridge_spacing(ridges):
    normalised, mask = ridge_segment(ridges)
    orientation = ridge_orient(normalised)
    _, mean_freq = ridge_frequency(normalised, mask, orientation)

    assert abs(1 / mean_freq - 9) < 1


def test_vectorised_filter_matches_the_reference_loop(ridges):
    normalised, mask = ridge_segment(ridges)
    orientation = ridge_orient(normalised)
    _, mean_freq = ridge_frequency(normalised, mask, orientation)
    freq = mean_freq * mask

    expected = _reference_ridge_filter(normalised, orientation, freq)
    actual = ridge_filter(normalised, orientation, freq)

    assert np.allclose(actual, expected, atol=1e-9)


def test_filter_without_frequency_returns_zeros(ridges):
    result = ridge_filter(ridges.astype(float), np.zeros(ridges.shape), np.zeros(ridges.shape))

    assert not result.any()


def test_enhance_returns_a_binary_ridge_map_of_the_same_size(ridges):
    result = enhance(ridges)

    assert result.ridges.dtype == bool
    assert result.ridges.shape == result.mask.shape == result.orientation.shape == ridges.shape
    assert result.ridges[:, :112].any()
    assert not result.mask[:, 128:].any()
