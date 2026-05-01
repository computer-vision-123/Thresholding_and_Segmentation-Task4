"""
result_types.py – Dataclass definitions for algorithm outputs.

All result objects carry both the processed image (float32 numpy array,
values in [0, 1]) and the computed parameter(s) for UI display.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Part A – Thresholding results
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimalThresholdResult:
    """Result of optimal (iterative) thresholding."""

    image: np.ndarray
    """Binary float32 image (0.0 / 1.0), shape (H, W)."""

    threshold: float
    """Converged threshold value in [0, 1] (normalised)."""

    @property
    def threshold_uint8(self) -> int:
        """Threshold rescaled to [0, 255] for display."""
        return int(round(self.threshold * 255))


@dataclass
class OtsuThresholdResult:
    """Result of Otsu's global thresholding."""

    image: np.ndarray
    """Binary float32 image (0.0 / 1.0), shape (H, W)."""

    threshold: float
    """Otsu threshold value in [0, 1] (normalised)."""

    @property
    def threshold_uint8(self) -> int:
        return int(round(self.threshold * 255))


@dataclass
class SpectralThresholdResult:
    """Result of spectral (multi-Otsu) thresholding."""

    image: np.ndarray
    """Float32 label image, shape (H, W), values in {0 … n_classes−1}."""

    thresholds: List[float]
    """Sorted list of (n_classes−1) threshold values in [0, 1]."""

    @property
    def thresholds_uint8(self) -> List[int]:
        """Thresholds rescaled to [0, 255] for display."""
        return [int(round(t * 255)) for t in self.thresholds]


@dataclass
class LocalThresholdResult:
    """Result of local (adaptive) thresholding."""

    image: np.ndarray
    """Binary float32 image (0.0 / 1.0), shape (H, W)."""


# ══════════════════════════════════════════════════════════════════════════════
# Part B – Segmentation results
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentationResult:
    """
    Generic result for all segmentation methods.

    The label image contains integer segment IDs stored as float32
    (e.g. 0.0, 1.0, 2.0, …).  Use utils.image_utils.colorise_labels()
    to convert to an RGB image for display.
    """

    image: np.ndarray
    """Float32 label image, shape (H, W)."""

    method: str
    """Name of the segmentation method that produced this result."""

    n_segments: int = field(init=False)
    """Number of distinct segments found (computed automatically)."""

    def __post_init__(self):
        self.n_segments = int(np.unique(self.image).size)
