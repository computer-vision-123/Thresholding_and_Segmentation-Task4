"""
thresholding.py – Part A processing module.

Public API
----------
apply_optimal_thresholding(image: np.ndarray) -> Tuple[np.ndarray, int]
apply_otsu_thresholding(image: np.ndarray) -> Tuple[np.ndarray, int]
apply_spectral_thresholding(image: np.ndarray, n_classes: int) -> Tuple[np.ndarray, List[int]]
apply_local_thresholding(image: np.ndarray, block_size: int, offset: int) -> np.ndarray

Each function accepts a 2-D uint8 numpy array (grayscale image) and returns:
  - binary / multi-level thresholded image (np.ndarray uint8)
  - the computed threshold value(s) for display
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Optimal (Iterative) Thresholding
# ──────────────────────────────────────────────────────────────────────────────

def apply_optimal_thresholding(
    image: np.ndarray,
    tol: int = 1,
) -> Tuple[np.ndarray, int]:
    """
    Iteratively estimate the optimal global threshold.


    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.
    tol : int
        Convergence tolerance.

    Returns
    -------
    binary : np.ndarray
        Binary image (0 or 255).
    threshold : int
        Final computed threshold value.
    """
    raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Otsu's Method
# ──────────────────────────────────────────────────────────────────────────────

def apply_otsu_thresholding(
    image: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Compute the globally optimal threshold by maximising between-class variance.


    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.

    Returns
    -------
    binary : np.ndarray
        Binary image (0 or 255).
    threshold : int
        Optimal Otsu threshold value.
    """
    raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Spectral (Multi-modal) Thresholding
# ──────────────────────────────────────────────────────────────────────────────

def apply_spectral_thresholding(
    image: np.ndarray,
    n_classes: int = 3,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extend Otsu's criterion to n_classes > 2 (multi-threshold / spectral).


    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.
    n_classes : int
        Number of output intensity classes (>= 3 for multi-modal).

    Returns
    -------
    labelled : np.ndarray
        Label image with values in {0 … n_classes-1}, scaled to uint8.
    thresholds : List[int]
        List of (n_classes − 1) threshold values.
    """
    raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Local (Adaptive) Thresholding
# ──────────────────────────────────────────────────────────────────────────────

def apply_local_thresholding(
    image: np.ndarray,
    block_size: int = 35,
    offset: int = 10,
) -> np.ndarray:
    """
    Compute a per-pixel threshold from the local neighbourhood mean.


    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.
    block_size : int
        Odd integer; size of the local neighbourhood window.
    offset : int
        Constant subtracted from the local mean.

    Returns
    -------
    binary : np.ndarray
        Binary image (0 or 255).
    """
    raise NotImplementedError
