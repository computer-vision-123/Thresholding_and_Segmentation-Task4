"""
thresholding.py – Part A public API (thin Python wrappers).

Responsibilities of this layer
-------------------------------
1. Accept uint8 numpy arrays from the UI.
2. Normalise them to float32 in [0, 1] before calling into cv_backend.
3. Unpack the C++ py::tuple return value and wrap it in the appropriate
   result dataclass.
4. Raise a clear ImportError if cv_backend has not been compiled yet.
"""

from __future__ import annotations
from typing import Union

import numpy as np

from core.result_types import (
    OptimalThresholdResult,
    OtsuThresholdResult,
    SpectralThresholdResult,
    LocalThresholdResult,
)

try:
    import cv_backend
except ImportError as e:
    raise ImportError(
        "cv_backend C++ extension not found. "
        "Build it first with:  pip install -e ."
    ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _to_float32(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 grayscale image to float32 in [0, 1]."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got shape {image.shape}")
    return image.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def apply_optimal_thresholding(
    image: np.ndarray,
    tol: int = 1,
) -> OptimalThresholdResult:
    """
    Apply optimal (iterative) thresholding to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.
    tol : int
        Convergence tolerance in uint8 units (converted to [0,1] internally).

    Returns
    -------
    OptimalThresholdResult
    """
    f32 = _to_float32(image)
    binary, threshold = cv_backend.threshold_optimal(f32, tol / 255.0)
    return OptimalThresholdResult(image=binary, threshold=float(threshold))


def apply_otsu_thresholding(image: np.ndarray) -> OtsuThresholdResult:
    """
    Apply Otsu's global thresholding to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.

    Returns
    -------
    OtsuThresholdResult
    """
    f32 = _to_float32(image)
    binary, threshold = cv_backend.threshold_otsu(f32)
    return OtsuThresholdResult(image=binary, threshold=float(threshold))


def apply_spectral_thresholding(
    image: np.ndarray,
    n_classes: int = 3,
) -> SpectralThresholdResult:
    """
    Apply spectral (multi-Otsu) thresholding to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.
    n_classes : int
        Number of output classes (≥ 3).

    Returns
    -------
    SpectralThresholdResult
    """
    if n_classes < 3:
        raise ValueError("spectral thresholding requires n_classes >= 3")
    f32 = _to_float32(image)
    label_image, thresholds = cv_backend.threshold_spectral(f32, n_classes)
    return SpectralThresholdResult(
        image=label_image,
        thresholds=[float(t) for t in thresholds],
    )


def apply_local_thresholding(
    image: np.ndarray,
    block_size: int = 35,
    offset: int = 10,
) -> LocalThresholdResult:
    """
    Apply local (adaptive) thresholding to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.
    block_size : int
        Odd window size.
    offset : int
        Offset (in uint8 units) subtracted from the local mean.

    Returns
    -------
    LocalThresholdResult
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")
    f32 = _to_float32(image)
    binary = cv_backend.threshold_local(f32, block_size, offset / 255.0)
    return LocalThresholdResult(image=binary)
