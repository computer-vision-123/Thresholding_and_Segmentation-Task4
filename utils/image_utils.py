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


import cv2
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
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk preserving its native channel count.

    Returns
    -------
    image : np.ndarray
        uint8 array of shape (H, W) for grayscale or (H, W, 3) for colour.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    # Convert BGR to RGB if colour
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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
    if image.dtype != np.uint8:
        # Assume float32 [0, 1] if not uint8
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if image.ndim == 2:
        h, w = image.shape
        bytes_per_line = w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    elif image.ndim == 3:
        h, w, c = image.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels for colour image, got {c}")
        bytes_per_line = 3 * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # We must copy the QImage because it references the numpy array's memory,
    # which might be garbage collected or mutated.
    return QPixmap.fromImage(q_img.copy())


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
    labels = label_image.astype(np.int32)
    unique_labels = np.unique(labels)
    h, w = labels.shape
    colour_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Simple deterministic color palette
    palette = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [128, 0, 0], [0, 128, 0], [0, 0, 128],
        [128, 128, 0], [128, 0, 128], [0, 128, 128],
    ]

    for i, label in enumerate(unique_labels):
        if label < 0: continue
        color = palette[i % len(palette)]
        colour_img[labels == label] = color

    return colour_img

