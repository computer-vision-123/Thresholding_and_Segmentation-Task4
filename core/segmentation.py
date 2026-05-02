"""
segmentation.py – Part B public API (thin Python wrappers).

Responsibilities of this layer
-------------------------------
1. Accept uint8 numpy arrays (gray or color) from the UI.
2. Normalise to float32 in [0, 1] before calling into cv_backend.
3. Wrap the returned float32 label image in a SegmentationResult dataclass.
4. Raise a clear ImportError if cv_backend has not been compiled yet.
"""

from __future__ import annotations
from typing import List, Tuple, Union

import numpy as np

from core.result_types import SegmentationResult

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
    """
    Convert a uint8 image (gray or color) to float32 in [0, 1].

    Accepts shapes:
      - (H, W)    – grayscale
      - (H, W, 3) – colour (BGR or RGB, order preserved)
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2-D or 3-D image, got shape {image.shape}")
    return image.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def apply_kmeans(
    image: np.ndarray,
    k: int = 4,
    max_iter: int = 100,
) -> SegmentationResult:
    """
    Segment image using K-Means clustering.

    Parameters
    ----------
    image : np.ndarray
        uint8 image, shape (H, W) or (H, W, 3).
    k : int
        Number of clusters.
    max_iter : int
        Maximum EM iterations.

    Returns
    -------
    SegmentationResult
    """
    f32 = _to_float32(image)
    labels = cv_backend.segment_kmeans(f32, k, max_iter)
    return SegmentationResult(image=labels, method="K-Means")


def apply_region_growing(
    image: np.ndarray,
    seeds: List[Tuple[int, int]],
    tolerance: int = 15,
) -> SegmentationResult:
    """
    Segment image using multi-seed region growing (parallel BFS).

    Parameters
    ----------
    image : np.ndarray
        uint8 grayscale image, shape (H, W).
    seeds : list of (row, col) tuples
        One or more seed pixel coordinates. Each seed spawns its own region;
        the region label matches the 1-based seed index (seed 0 → label 1, etc.).
    tolerance : int
        Max intensity difference from each seed's value (uint8 units, 0–255).

    Returns
    -------
    SegmentationResult
        label image: 0 = background, 1..N = region for seed N.
    """
    if image.ndim != 2:
        raise ValueError("Region growing requires a grayscale (2-D) image.")
    if not seeds:
        raise ValueError("At least one seed must be provided.")

    f32 = _to_float32(image)
    seed_rows = [int(s[0]) for s in seeds]
    seed_cols = [int(s[1]) for s in seeds]

    labels = cv_backend.segment_region_growing(
        f32,
        seed_rows,
        seed_cols,
        tolerance / 255.0,
    )
    return SegmentationResult(image=labels, method="Region Growing")


def apply_agglomerative(
    image: np.ndarray,
    k: int = 4,
) -> SegmentationResult:
    """
    Segment image using agglomerative (hierarchical) clustering.

    Parameters
    ----------
    image : np.ndarray
        uint8 image, shape (H, W) or (H, W, 3).
    k : int
        Target number of clusters.

    Returns
    -------
    SegmentationResult
    """
    f32 = _to_float32(image)
    labels = cv_backend.segment_agglomerative(f32, k)
    return SegmentationResult(image=labels, method="Agglomerative")


def apply_mean_shift(
    image: np.ndarray,
    bandwidth: float = 30.0,
) -> SegmentationResult:
    """
    Segment image using Mean Shift.

    Parameters
    ----------
    image : np.ndarray
        uint8 image, shape (H, W) or (H, W, 3).
    bandwidth : float
        Kernel bandwidth in uint8 units (converted to [0,1] internally).

    Returns
    -------
    SegmentationResult
    """
    f32 = _to_float32(image)
    labels = cv_backend.segment_mean_shift(f32, bandwidth / 255.0)
    return SegmentationResult(image=labels, method="Mean Shift")