"""Optional matplotlib visualisation (``pip install fingerprint-recognition[plot]``)."""

from __future__ import annotations

import cv2

from .matching import Features, GrayImage


def show(
    first_image: GrayImage,
    first: Features,
    second_image: GrayImage,
    second: Features,
    matches_image: GrayImage,
) -> None:
    """Display both prints with their keypoints, then the matches between them."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "--show needs matplotlib: pip install 'fingerprint-recognition[plot]'"
        ) from exc

    _, axes = plt.subplots(1, 2)
    for axis, image, features in ((axes[0], first_image, first), (axes[1], second_image, second)):
        axis.imshow(cv2.drawKeypoints(image, features.keypoints, None))  # type: ignore[call-overload]
        axis.set_axis_off()
    plt.show()

    plt.imshow(matches_image)
    plt.axis("off")
    plt.show()
