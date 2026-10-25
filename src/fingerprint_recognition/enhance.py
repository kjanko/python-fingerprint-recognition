"""Gabor-filter fingerprint enhancement.

Port of Peter Kovesi's MATLAB fingerprint enhancement code, by way of
Utkarsh Deshmukh's Python translation
(https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python),
vectorised with NumPy/OpenCV. See THIRD_PARTY_NOTICES.md for their licenses.

Reference: Hong, L., Wan, Y., and Jain, A. K. Fingerprint image enhancement:
algorithm and performance evaluation. IEEE Transactions on Pattern Analysis
and Machine Intelligence 20, 8 (1998), 777-789.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from scipy import ndimage, signal

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _gaussian_kernel(sigma: float, *, force_odd: bool = True) -> FloatArray:
    size = int(6 * sigma)
    if force_odd and size % 2 == 0:
        size += 1
    return np.asarray(cv2.getGaussianKernel(size, sigma), dtype=np.float64).ravel()


def _gaussian_smooth(image: FloatArray, kernel: FloatArray) -> FloatArray:
    # The 2D Gaussian is separable, so two 1D passes equal one 2D convolution.
    smoothed = ndimage.convolve1d(ndimage.convolve1d(image, kernel, axis=0), kernel, axis=1)
    return np.asarray(smoothed, dtype=np.float64)


def ridge_segment(
    image: npt.ArrayLike, block_size: int = 16, threshold: float = 0.1
) -> tuple[FloatArray, BoolArray]:
    """Normalise the image and find the region that contains ridges.

    The image is split into ``block_size`` blocks; blocks whose standard
    deviation exceeds ``threshold`` are considered part of the fingerprint.

    Returns the image renormalised to zero mean and unit standard deviation
    over the ridge region, and the boolean ridge mask.
    """
    im = np.asarray(image, dtype=np.float64)
    rows, cols = im.shape
    im = (im - im.mean()) / im.std()

    padded_rows = block_size * math.ceil(rows / block_size)
    padded_cols = block_size * math.ceil(cols / block_size)
    padded = np.zeros((padded_rows, padded_cols))
    padded[:rows, :cols] = im

    blocks = padded.reshape(
        padded_rows // block_size, block_size, padded_cols // block_size, block_size
    )
    block_std = blocks.std(axis=(1, 3))
    std_image = np.kron(block_std, np.ones((block_size, block_size)))[:rows, :cols]

    mask = std_image > threshold
    normalised = (im - im[mask].mean()) / im[mask].std()
    return normalised, mask


def ridge_orient(
    image: FloatArray,
    gradient_sigma: float = 1,
    block_sigma: float = 7,
    orient_smooth_sigma: float = 7,
) -> FloatArray:
    """Estimate the local ridge orientation, in radians, of every pixel.

    Orientations are positive clockwise and point *along* the ridges.
    """
    gauss = _gaussian_kernel(gradient_sigma)
    fy, fx = np.gradient(np.outer(gauss, gauss))
    gx = signal.convolve2d(image, fx, mode="same")
    gy = signal.convolve2d(image, fy, mode="same")

    # Smooth the covariance data to get a weighted sum of gradient moments.
    weights = _gaussian_kernel(block_sigma, force_odd=False)
    gxx = _gaussian_smooth(gx * gx, weights)
    gyy = _gaussian_smooth(gy * gy, weights)
    gxy = 2 * _gaussian_smooth(gx * gy, weights)

    # Analytic solution of the principal direction.
    denom = np.sqrt(gxy**2 + (gxx - gyy) ** 2) + np.finfo(float).eps
    sin2theta = gxy / denom
    cos2theta = (gxx - gyy) / denom

    if orient_smooth_sigma:
        smooth = _gaussian_kernel(orient_smooth_sigma)
        cos2theta = _gaussian_smooth(cos2theta, smooth)
        sin2theta = _gaussian_smooth(sin2theta, smooth)

    return np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2


def _block_frequency(
    block: FloatArray,
    orient_block: FloatArray,
    window_size: int,
    min_wavelength: float,
    max_wavelength: float,
) -> float:
    """Estimate the ridge frequency of one image block (0 if not found)."""
    rows = block.shape[0]

    # Average the doubled angles to avoid wraparound at the origin.
    orient = math.atan2(np.mean(np.sin(2 * orient_block)), np.mean(np.cos(2 * orient_block))) / 2

    # Rotate so the ridges are vertical, then crop away the invalid corners.
    rotated = ndimage.rotate(
        block, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode="nearest"
    )
    crop = int(rows / np.sqrt(2))
    offset = int((rows - crop) / 2)
    rotated = rotated[offset : offset + crop, offset : offset + crop]

    # Project the grey values down the ridges and find the peaks.
    projection = rotated.sum(axis=0)
    dilation = ndimage.grey_dilation(projection, window_size, structure=np.ones(window_size))
    peaks = np.flatnonzero((np.abs(dilation - projection) < 2) & (projection > projection.mean()))

    if peaks.size < 2:
        return 0.0
    wavelength = (peaks[-1] - peaks[0]) / (peaks.size - 1)
    if min_wavelength <= wavelength <= max_wavelength:
        return 1 / float(wavelength)
    return 0.0


def ridge_frequency(
    image: FloatArray,
    mask: BoolArray,
    orientation: FloatArray,
    block_size: int = 38,
    window_size: int = 5,
    min_wavelength: float = 5,
    max_wavelength: float = 15,
) -> tuple[FloatArray, float]:
    """Estimate the ridge frequency per block.

    Returns the per-pixel frequency image (0 where no frequency was found)
    and the mean frequency over the valid regions.
    """
    rows, cols = image.shape
    freq = np.zeros((rows, cols))
    for r in range(0, rows - block_size, block_size):
        for c in range(0, cols - block_size, block_size):
            freq[r : r + block_size, c : c + block_size] = _block_frequency(
                image[r : r + block_size, c : c + block_size],
                orientation[r : r + block_size, c : c + block_size],
                window_size,
                min_wavelength,
                max_wavelength,
            )

    freq *= mask
    valid = freq[freq > 0]
    mean_freq = float(valid.mean()) if valid.size else 0.0
    return freq, mean_freq


def ridge_filter(
    image: FloatArray,
    orientation: FloatArray,
    frequency: FloatArray,
    kx: float = 0.65,
    ky: float = 0.65,
    angle_increment: int = 3,
) -> FloatArray:
    """Enhance the image with Gabor filters tuned to the local ridges.

    ``kx`` and ``ky`` scale the filter sigma relative to the ridge
    wavelength, along and across the ridges respectively.
    """
    im = np.asarray(image, dtype=np.float64)
    rows, cols = im.shape
    enhanced = np.zeros((rows, cols))

    nonzero = frequency[frequency > 0]
    if nonzero.size == 0:
        return enhanced

    # Round frequencies to 0.01; the filter is built from the lowest one.
    base_freq = np.unique(np.round(nonzero * 100) / 100)[0]
    sigma_x = kx / base_freq
    sigma_y = ky / base_freq
    size = int(np.round(3 * max(sigma_x, sigma_y)))

    x, y = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1))
    reference = np.exp(-(x**2 / sigma_x**2 + y**2 / sigma_y**2)) * np.cos(2 * np.pi * base_freq * x)

    # Map orientations to filter indices 1..n_orientations.
    n_orientations = 180 // angle_increment
    orient_index = np.round(orientation / np.pi * 180 / angle_increment)
    orient_index[orient_index < 1] += n_orientations
    orient_index[orient_index > n_orientations] -= n_orientations

    # Only filter pixels far enough from the border for a full window.
    valid = np.zeros((rows, cols), dtype=bool)
    valid[size + 1 : rows - size, size + 1 : cols - size] = True
    valid &= frequency > 0

    for index in range(1, n_orientations + 1):
        selected = valid & (orient_index == index)
        if not selected.any():
            continue
        # The orientation image points along the ridges, hence +90 degrees;
        # rotation is positive anticlockwise, hence the minus sign.
        kernel = ndimage.rotate(reference, -((index - 1) * angle_increment + 90), reshape=False)
        response = cv2.filter2D(im, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)
        enhanced[selected] = response[selected]

    return enhanced


@dataclass(frozen=True)
class RidgeMap:
    """Result of enhancing a fingerprint.

    ``ridges`` is the binary ridge image, ``mask`` the region that contains
    the print and ``orientation`` the ridge direction of every pixel.
    """

    ridges: BoolArray
    mask: BoolArray
    orientation: FloatArray


def enhance(image: npt.ArrayLike) -> RidgeMap:
    """Run the full enhancement pipeline on a grayscale fingerprint."""
    normalised, mask = ridge_segment(image, block_size=16, threshold=0.1)
    orientation = ridge_orient(normalised, gradient_sigma=1, block_sigma=7, orient_smooth_sigma=7)
    _, mean_freq = ridge_frequency(
        normalised,
        mask,
        orientation,
        block_size=38,
        window_size=5,
        min_wavelength=5,
        max_wavelength=15,
    )
    filtered = ridge_filter(
        normalised, orientation, np.where(mask, mean_freq, 0.0), kx=0.65, ky=0.65
    )
    return RidgeMap(ridges=filtered < -3, mask=mask, orientation=orientation)
