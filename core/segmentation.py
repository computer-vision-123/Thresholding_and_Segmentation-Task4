"""
segmentation.py – Part B processing module.

Public API
----------
apply_kmeans(image, k, max_iter)      -> np.ndarray  (label image)
apply_region_growing(image, seed, tol) -> np.ndarray  (label image)
apply_agglomerative(image, k)          -> np.ndarray  (label image)
apply_mean_shift(image, bandwidth)     -> np.ndarray  (label image)

Each function accepts:
  - image : np.ndarray  – H×W (grayscale) or H×W×3 (color) uint8 array
and returns a label image (np.ndarray uint8) where each unique value
identifies a distinct segment.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# K-Means Clustering
# ──────────────────────────────────────────────────────────────────────────────

def apply_kmeans(
    image: np.ndarray,
    k: int = 4,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Segment image pixels via K-Means clustering.

    Parameters
    ----------
    image : np.ndarray
        H×W or H×W×3 uint8 image.
    k : int
        Number of clusters.
    max_iter : int
        Maximum number of EM iterations.

    Returns
    -------
    labels : np.ndarray
        H×W uint8 label image, values in {0 … k-1}.
    """
    raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Region Growing
# ──────────────────────────────────────────────────────────────────────────────

def apply_region_growing(
    image: np.ndarray,
    seed: Tuple[int, int],
    tolerance: int = 15,
) -> np.ndarray:
    """
    Grow a region from a seed pixel based on intensity similarity.


    Parameters
    ----------
    image : np.ndarray
        H×W uint8 grayscale image.
    seed : Tuple[int, int]
        (row, col) seed pixel coordinate.
    tolerance : int
        Maximum allowed intensity difference from the seed.

    Returns
    -------
    labels : np.ndarray
        H×W uint8 binary mask (255 = region, 0 = background).
    """
    raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Agglomerative (Hierarchical) Clustering
# ──────────────────────────────────────────────────────────────────────────────

def apply_agglomerative(
    image: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """
    Segment image using agglomerative (bottom-up) hierarchical clustering.


    Parameters
    ----------
    image : np.ndarray
        H×W or H×W×3 uint8 image.
    k : int
        Target number of clusters.

    Returns
    -------
    labels : np.ndarray
        H×W uint8 label image, values in {0 … k-1}.
    """
    raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Mean Shift Clustering
# ──────────────────────────────────────────────────────────────────────────────

def apply_mean_shift(
    image: np.ndarray,
    bandwidth: float = 30.0,
) -> np.ndarray:
    """
    Segment image using the Mean Shift algorithm.


    Parameters
    ----------
    image : np.ndarray
        H×W or H×W×3 uint8 image.
    bandwidth : float
        Kernel bandwidth (radius) controlling cluster granularity.

    Returns
    -------
    labels : np.ndarray
        H×W uint8 label image.
    """
    raise NotImplementedError
