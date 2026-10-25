"""Minutiae-based fingerprint recognition with Gabor enhancement and ORB alignment."""

from importlib.metadata import PackageNotFoundError, version

from .enhance import RidgeMap, enhance
from .matching import (
    Features,
    MatchResult,
    compare,
    extract_features,
    load_image,
    match_features,
)
from .minutiae import Minutia, MinutiaKind, extract_minutiae

try:
    __version__ = version("fingerprint-recognition")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "Features",
    "MatchResult",
    "Minutia",
    "MinutiaKind",
    "RidgeMap",
    "compare",
    "enhance",
    "extract_features",
    "extract_minutiae",
    "load_image",
    "match_features",
]
