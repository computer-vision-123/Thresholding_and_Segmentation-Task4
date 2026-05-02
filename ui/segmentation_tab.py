"""
SegmentationTab – Part B UI.

Multi-seed region growing:
  - Click anywhere on the image to place a seed (any number of seeds).
  - Each seed is shown as a numbered crosshair overlay.
  - "Clear Seeds" removes all placed seeds.
  - Pixel mapping is exact regardless of image / label size.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QButtonGroup, QRadioButton, QGroupBox, QLabel,
    QStackedWidget, QSpinBox, QDoubleSpinBox, QSizePolicy,
    QFileDialog, QMessageBox, QScrollArea,
)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QBrush

from utils.image_utils import load_image, ndarray_to_qpixmap, colorise_labels
from core.segmentation import (
    apply_kmeans,
    apply_region_growing,
    apply_agglomerative,
    apply_mean_shift,
)


# ──────────────────────────────────────────────────────────────────────────────
# Seed colour palette  (one per seed, cycles if more than 12)
# ──────────────────────────────────────────────────────────────────────────────
SEED_COLORS = [
    QColor(255,  60,  60),   # red
    QColor( 60, 200,  60),   # green
    QColor( 60, 120, 255),   # blue
    QColor(255, 200,   0),   # yellow
    QColor(255,  60, 220),   # magenta
    QColor(  0, 220, 220),   # cyan
    QColor(255, 140,   0),   # orange
    QColor(160,  60, 255),   # purple
    QColor(  0, 200, 120),   # teal
    QColor(200, 200,  60),   # olive
    QColor(255, 120, 160),   # pink
    QColor( 80, 160, 255),   # sky
]


class SeedOverlayLabel(QLabel):
    """
    A QLabel that:
      - Displays an image scaled to fit (aspect-ratio preserved, centred).
      - Paints numbered crosshair markers for each seed.
      - Reports *exact original-image pixel coordinates* for every click,
        correctly accounting for letterbox / pillarbox offsets at any widget size.

    The widget never distorts the image – it always uses Qt.KeepAspectRatio
    so the mapping back to image coordinates is unambiguous.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source_pixmap: Optional[QPixmap] = None   # full-res pixmap
        self._seeds: List[Tuple[int, int]] = []          # image-space (row, col)
        self._click_callback = None                      # callable(row, col) | None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)

    # ── Public interface ──────────────────────────────────────────────────────

    def set_click_callback(self, callback) -> None:
        self._click_callback = callback

    def set_pixmap_source(self, pixmap: QPixmap) -> None:
        """Set the base image (replaces any existing one)."""
        self._source_pixmap = pixmap
        self.update()

    def set_seeds(self, seeds: List[Tuple[int, int]]) -> None:
        """Update the list of seed positions (image-space row, col) and repaint."""
        self._seeds = list(seeds)
        self.update()

    def clear(self) -> None:
        """Remove image and seeds."""
        self._source_pixmap = None
        self._seeds = []
        super().clear()
        self.setText("No image loaded")

    # ── Qt overrides ──────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        """Draw image + seed markers."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._source_pixmap is None:
            # Fall back to default QLabel text rendering
            super().paintEvent(event)
            return

        # Compute the rectangle where the scaled pixmap is drawn
        img_rect = self._image_rect()
        scaled_pm = self._source_pixmap.scaled(
            img_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        # Re-centre (scaled might differ by 1px from img_rect due to integer rounding)
        dx = (img_rect.width()  - scaled_pm.width())  // 2
        dy = (img_rect.height() - scaled_pm.height()) // 2
        draw_x = img_rect.x() + dx
        draw_y = img_rect.y() + dy
        actual_rect = QRect(draw_x, draw_y, scaled_pm.width(), scaled_pm.height())

        painter.drawPixmap(actual_rect.topLeft(), scaled_pm)

        # Draw seed markers
        for idx, (row, col) in enumerate(self._seeds):
            wx, wy = self._image_to_widget(row, col, actual_rect)
            color = SEED_COLORS[idx % len(SEED_COLORS)]
            self._draw_seed_marker(painter, wx, wy, idx + 1, color)

        painter.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._click_callback is None or self._source_pixmap is None:
            return

        actual_rect = self._actual_pixmap_rect()
        px = event.x() - actual_rect.x()
        py = event.y() - actual_rect.y()

        if 0 <= px < actual_rect.width() and 0 <= py < actual_rect.height():
            img_w = self._source_pixmap.width()
            img_h = self._source_pixmap.height()

            # Map widget pixels → original image pixels
            col = int(px * img_w / actual_rect.width())
            row = int(py * img_h / actual_rect.height())

            # Clamp to valid range
            col = max(0, min(col, img_w - 1))
            row = max(0, min(row, img_h - 1))

            self._click_callback(row, col)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _image_rect(self) -> QRect:
        """The bounding rectangle available for the image (whole widget)."""
        return QRect(0, 0, self.width(), self.height())

    def _actual_pixmap_rect(self) -> QRect:
        """
        The exact rectangle occupied by the scaled pixmap inside the widget,
        accounting for aspect-ratio letterboxing/pillarboxing.
        """
        if self._source_pixmap is None:
            return QRect()
        img_w = self._source_pixmap.width()
        img_h = self._source_pixmap.height()
        lw, lh = self.width(), self.height()

        scale = min(lw / img_w, lh / img_h)
        pw = int(img_w * scale)
        ph = int(img_h * scale)
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        return QRect(ox, oy, pw, ph)

    def _image_to_widget(self, row: int, col: int, actual_rect: QRect) -> Tuple[int, int]:
        """Convert image-space (row, col) → widget-space (x, y)."""
        if self._source_pixmap is None:
            return (0, 0)
        img_w = self._source_pixmap.width()
        img_h = self._source_pixmap.height()
        x = actual_rect.x() + int(col * actual_rect.width()  / img_w)
        y = actual_rect.y() + int(row * actual_rect.height() / img_h)
        return x, y

    @staticmethod
    def _draw_seed_marker(painter: QPainter, x: int, y: int,
                           number: int, color: QColor) -> None:
        """Draw a crosshair + circle + label at widget coords (x, y)."""
        ARM   = 10
        THICK =  2
        R     =  6

        outer_pen = QPen(Qt.black, THICK + 2)
        outer_pen.setCosmetic(True)
        inner_pen = QPen(color, THICK)
        inner_pen.setCosmetic(True)

        for pen in (outer_pen, inner_pen):
            painter.setPen(pen)
            painter.drawLine(x - ARM, y, x + ARM, y)
            painter.drawLine(x, y - ARM, x, y + ARM)

        # Filled circle
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPoint(x, y), R, R)

        # Number label
        painter.setPen(QPen(Qt.white))
        font = QFont("Arial", 6, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRect(x - R, y - R, R * 2, R * 2),
                         Qt.AlignCenter, str(number))
        painter.setBrush(Qt.NoBrush)


# ══════════════════════════════════════════════════════════════════════════════
# SegmentationTab
# ══════════════════════════════════════════════════════════════════════════════

class SegmentationTab(QWidget):
    """UI container for Part B – Unsupervised Segmentation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_image: Optional[np.ndarray] = None  # uint8 (H,W) or (H,W,3)
        self._result_image:  Optional[np.ndarray] = None  # uint8 (H,W,3) colourised
        self._seeds: List[Tuple[int, int]] = []            # image-space (row, col)
        self._build_ui()
        self._wire_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addLayout(self._create_toolbar())

        content = QHBoxLayout()
        content.addWidget(self._create_original_panel(), stretch=1)
        content.addWidget(self._create_controls_panel(), stretch=1)
        content.addWidget(self._create_result_panel(),   stretch=1)
        root.addLayout(content)

    def _create_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.btn_load_image = QPushButton("Load Image (Gray / Color)")
        layout.addWidget(self.btn_load_image)
        layout.addStretch()
        return layout

    def _create_original_panel(self) -> QGroupBox:
        box = QGroupBox("Original Image  –  click to place seeds")
        layout = QVBoxLayout(box)

        self.lbl_original = SeedOverlayLabel()
        self.lbl_original.set_click_callback(self._on_image_clicked)

        layout.addWidget(self.lbl_original)

        # Seed info bar
        seed_bar = QHBoxLayout()
        self.lbl_seed_info = QLabel("No seeds placed")
        self.lbl_seed_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.btn_clear_seeds = QPushButton("Clear Seeds")
        self.btn_clear_seeds.setFixedWidth(100)
        seed_bar.addWidget(self.lbl_seed_info)
        seed_bar.addWidget(self.btn_clear_seeds)
        layout.addLayout(seed_bar)

        return box

    def _create_controls_panel(self) -> QGroupBox:
        box = QGroupBox("Segmentation Method")
        layout = QVBoxLayout(box)

        self.method_group = QButtonGroup(self)

        self.rb_kmeans    = QRadioButton("K-Means")
        self.rb_region    = QRadioButton("Region Growing  (multi-seed)")
        self.rb_agglom    = QRadioButton("Agglomerative Clustering")
        self.rb_meanshift = QRadioButton("Mean Shift")

        self.rb_kmeans.setChecked(True)

        for idx, rb in enumerate([self.rb_kmeans, self.rb_region,
                                   self.rb_agglom, self.rb_meanshift]):
            self.method_group.addButton(rb, idx)
            layout.addWidget(rb)

        layout.addSpacing(8)

        self.param_stack = QStackedWidget()
        self.param_stack.addWidget(self._create_kmeans_params())      # 0
        self.param_stack.addWidget(self._create_region_params())      # 1
        self.param_stack.addWidget(self._create_agglom_params())      # 2
        self.param_stack.addWidget(self._create_meanshift_params())   # 3
        layout.addWidget(self.param_stack)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        self.btn_apply = QPushButton("Apply")
        layout.addWidget(self.btn_apply)

        return box

    # ── Per-method parameter widgets ──────────────────────────────────────────

    def _create_kmeans_params(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Number of clusters (k):"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 20)
        self.spin_k.setValue(4)
        layout.addWidget(self.spin_k)
        layout.addWidget(QLabel("Max iterations:"))
        self.spin_kmeans_iter = QSpinBox()
        self.spin_kmeans_iter.setRange(1, 500)
        self.spin_kmeans_iter.setValue(100)
        layout.addWidget(self.spin_kmeans_iter)
        layout.addStretch()
        return w

    def _create_region_params(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Intensity tolerance (0–255):"))
        self.spin_rg_tolerance = QSpinBox()
        self.spin_rg_tolerance.setRange(1, 255)
        self.spin_rg_tolerance.setValue(15)
        layout.addWidget(self.spin_rg_tolerance)

        hint = QLabel(
            "Click the image to add seed pixels.\n"
            "Each seed grows its own labelled region.\n"
            "Use 'Clear Seeds' to start over."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(hint)
        layout.addStretch()
        return w

    def _create_agglom_params(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Number of clusters:"))
        self.spin_agg_k = QSpinBox()
        self.spin_agg_k.setRange(2, 20)
        self.spin_agg_k.setValue(4)
        layout.addWidget(self.spin_agg_k)
        layout.addStretch()
        return w

    def _create_meanshift_params(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Bandwidth:"))
        self.spin_bandwidth = QDoubleSpinBox()
        self.spin_bandwidth.setRange(1.0, 200.0)
        self.spin_bandwidth.setValue(30.0)
        self.spin_bandwidth.setSingleStep(5.0)
        layout.addWidget(self.spin_bandwidth)
        layout.addStretch()
        return w

    def _create_result_panel(self) -> QGroupBox:
        box = QGroupBox("Segmentation Result")
        layout = QVBoxLayout(box)
        self.lbl_result = QLabel("Result will appear here")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_result)
        self.lbl_n_segments = QLabel("")
        self.lbl_n_segments.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_n_segments)
        return box

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _wire_signals(self) -> None:
        self.btn_load_image.clicked.connect(self._on_load_image)
        self.btn_apply.clicked.connect(self._on_apply_segmentation)
        self.btn_clear_seeds.clicked.connect(self._on_clear_seeds)
        self.method_group.buttonClicked.connect(self._on_method_changed)

    # ──────────────────────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────────────────────

    def _on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            self._current_image = load_image(path)
            self._seeds = []
            self._result_image = None
            self.lbl_n_segments.setText("")
            self.lbl_result.setText("Result will appear here")
            self.lbl_result.setPixmap(QPixmap())

            # Push source pixmap to the overlay label
            pm = ndarray_to_qpixmap(self._current_image)
            self.lbl_original.set_pixmap_source(pm)
            self.lbl_original.set_seeds([])
            self._update_seed_info()

            self.lbl_status.setText(
                f"Loaded: {self._current_image.shape[1]}×"
                f"{self._current_image.shape[0]} px, "
                f"{'grayscale' if self._current_image.ndim == 2 else 'colour'}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _on_method_changed(self, button) -> None:
        self.param_stack.setCurrentIndex(self.method_group.id(button))
        self.lbl_status.setText("")

    def _on_image_clicked(self, row: int, col: int) -> None:
        """
        Called by SeedOverlayLabel with exact image-space (row, col).
        Seeds are accepted regardless of which method is active, but
        only used by Region Growing.
        """
        if self._current_image is None:
            return

        # Clamp (the label already clamps, but be defensive)
        H, W = self._current_image.shape[:2]
        row = max(0, min(row, H - 1))
        col = max(0, min(col, W - 1))

        self._seeds.append((row, col))
        self.lbl_original.set_seeds(self._seeds)
        self._update_seed_info()

    def _on_clear_seeds(self) -> None:
        self._seeds = []
        self.lbl_original.set_seeds([])
        self._update_seed_info()

    def _on_apply_segmentation(self) -> None:
        if self._current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        method_id = self.method_group.checkedId()

        try:
            self.lbl_status.setText("Running…")
            self.repaint()

            if method_id == 0:
                result = apply_kmeans(
                    self._current_image,
                    k=self.spin_k.value(),
                    max_iter=self.spin_kmeans_iter.value(),
                )

            elif method_id == 1:
                # ── Multi-seed Region Growing ─────────────────────────────────
                if not self._seeds:
                    QMessageBox.warning(
                        self, "No Seeds",
                        "Click on the image to place one or more seed pixels first."
                    )
                    self.lbl_status.setText("")
                    return

                gray = (
                    self._current_image
                    if self._current_image.ndim == 2
                    else self._to_gray(self._current_image)
                )
                result = apply_region_growing(
                    gray,
                    seeds=self._seeds,
                    tolerance=self.spin_rg_tolerance.value(),
                )

            elif method_id == 2:
                result = apply_agglomerative(
                    self._current_image,
                    k=self.spin_agg_k.value(),
                )

            elif method_id == 3:
                result = apply_mean_shift(
                    self._current_image,
                    bandwidth=self.spin_bandwidth.value(),
                )
            else:
                return

            self._result_image = colorise_labels(result.image)
            self._display_image(self.lbl_result, self._result_image)
            self.lbl_n_segments.setText(f"Segments found: {result.n_segments}")
            self.lbl_status.setText(f"Done  ({result.method})")

        except RuntimeError as e:
            err = str(e)
            if "Not implemented" in err:
                QMessageBox.information(
                    self, "Not Implemented",
                    f"This method has not been implemented yet.\n\n{err}"
                )
            else:
                QMessageBox.critical(self, "Error", err)
            self.lbl_status.setText("Error – see dialog.")

        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))
            self.lbl_status.setText("Error – see dialog.")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _update_seed_info(self) -> None:
        n = len(self._seeds)
        if n == 0:
            self.lbl_seed_info.setText("No seeds placed")
        elif n == 1:
            r, c = self._seeds[0]
            self.lbl_seed_info.setText(f"1 seed: ({r}, {c})")
        else:
            self.lbl_seed_info.setText(
                f"{n} seeds: " + ", ".join(f"({r},{c})" for r, c in self._seeds)
            )

    def _display_image(self, label: QLabel, image: np.ndarray) -> None:
        if image is None:
            return
        pixmap = ndarray_to_qpixmap(image)
        label.setPixmap(
            pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Seeds repaint automatically via SeedOverlayLabel.resizeEvent
        if self._result_image is not None:
            self._display_image(self.lbl_result, self._result_image)