"""
image_utils.py – Shared image I/O and conversion helpers.

Functions
---------
load_grayscale(path: str) -> np.ndarray
load_image(path: str) -> np.ndarray
ndarray_to_qpixmap(image: np.ndarray) -> QPixmap
colorise_labels(label_image: np.ndarray) -> np.ndarray
"""

from __future__ import annotations
import numpy as np
from PyQt5.QtGui import QPixmap, QImage


def load_grayscale(path: str) -> np.ndarray:
    """
    Load an image from disk and convert it to a single-channel grayscale array.

    Parameters
    ----------
    path : str
        Filesystem path to the image file.

    Returns
    -------
    image : np.ndarray
        2-D uint8 array of shape (H, W).
    """
    raise NotImplementedError


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk preserving its native channel count.

    Returns
    -------
    image : np.ndarray
        uint8 array of shape (H, W) for grayscale or (H, W, 3) for colour.
    """
    raise NotImplementedError


def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    """
    Convert a numpy uint8 array to a QPixmap for display in a QLabel.

    Supports:
      - 2-D (H, W)    → grayscale QImage
      - 3-D (H, W, 3) → RGB QImage (assumes RGB channel order)

    Returns
    -------
    pixmap : QPixmap
    """
    raise NotImplementedError


def colorise_labels(label_image: np.ndarray) -> np.ndarray:
    """
    Map an integer label image to a colour (H, W, 3) array for visualisation.

    Each unique label is assigned a deterministic, visually distinct colour.

    Parameters
    ----------
    label_image : np.ndarray
        2-D array of non-negative integer labels.

    Returns
    -------
    colour_image : np.ndarray
        (H, W, 3) uint8 RGB image.
    """
    raise NotImplementedError
