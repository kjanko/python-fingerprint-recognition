from pathlib import Path

import numpy as np
import pytest

DATABASE = Path(__file__).resolve().parent.parent / "database"


@pytest.fixture
def database() -> Path:
    return DATABASE


@pytest.fixture
def ridges() -> np.ndarray:
    """A synthetic print: diagonal ridges on the left, flat background on the right."""
    y, x = np.mgrid[0:160, 0:200]
    image = 128 + 100 * np.sin(2 * np.pi * (x + y) / (9 * np.sqrt(2)))
    image[:, 120:] = 128
    rng = np.random.default_rng(0)
    return np.clip(image + rng.normal(0, 5, image.shape), 0, 255).astype(np.uint8)
