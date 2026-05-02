"""
SegmentationTab – Part B UI.

Implements:
  • btn_load_image.clicked     →  _on_load_image()
  • btn_apply.clicked          →  _on_apply_segmentation()
  • method_group.buttonClicked →  _on_method_changed()
  • mouse click on original image label → _on_image_clicked() (seed for region growing)
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QButtonGroup, QRadioButton, QGroupBox, QLabel,
    QStackedWidget, QSpinBox, QDoubleSpinBox, QSizePolicy,
    QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor

from utils.image_utils import load_image, ndarray_to_qpixmap, colorise_labels
from core.segmentation import (
    apply_kmeans,
    apply_region_growing,
    apply_agglomerative,
    apply_mean_shift,
)


class ClickableLabel(QLabel):
    """
    A QLabel that emits the *relative* pixel coordinate of a mouse press.

    Used to let the user pick a seed pixel for Region Growing by clicking
    directly on the displayed image.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._click_callback = None   # callable(row, col) or None

    def set_click_callback(self, callback):
        """Register a callable that receives (row, col) on click."""
        self._click_callback = callback

    def mousePressEvent(self, event):
        if self._click_callback is None or self.pixmap() is None:
            return

        # The pixmap is scaled/centred inside the label via Qt.KeepAspectRatio.
        # We need to map the click position back to original-image coordinates.
        pm   = self.pixmap()
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()

        # Top-left corner of the pixmap inside the label (centred)
        x_offset = (lw - pw) // 2
        y_offset  = (lh - ph) // 2

        px = event.x() - x_offset
        py = event.y() - y_offset

        if 0 <= px < pw and 0 <= py < ph:
            self._click_callback(py, px)   # (row, col)


class SegmentationTab(QWidget):
    """UI container for Part B – Unsupervised Segmentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None   # numpy ndarray  uint8  (H,W) or (H,W,3)
        self._result_image  = None   # numpy ndarray  uint8  (H,W,3) colourised
        self._seed          = None   # (row, col) for region growing
        self._build_ui()
        self._wire_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        toolbar = self._create_toolbar()
        root.addLayout(toolbar)

        content = QHBoxLayout()
        content.addWidget(self._create_original_panel(),  stretch=1)
        content.addWidget(self._create_controls_panel(),  stretch=1)
        content.addWidget(self._create_result_panel(),    stretch=1)
        root.addLayout(content)

    def _create_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.btn_load_image = QPushButton("Load Image (Gray / Color)")
        layout.addWidget(self.btn_load_image)
        layout.addStretch()
        return layout

    def _create_original_panel(self) -> QGroupBox:
        box = QGroupBox("Original Image")
        layout = QVBoxLayout(box)

        self.lbl_original = ClickableLabel("No image loaded")
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_original.set_click_callback(self._on_image_clicked)

        self.lbl_seed_info = QLabel("")
        self.lbl_seed_info.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.lbl_original)
        layout.addWidget(self.lbl_seed_info)
        return box

    def _create_controls_panel(self) -> QGroupBox:
        box = QGroupBox("Segmentation Method")
        layout = QVBoxLayout(box)

        # ── Method radio buttons ──────────────────────────────────────────────
        self.method_group = QButtonGroup(self)

        self.rb_kmeans    = QRadioButton("K-Means")
        self.rb_region    = QRadioButton("Region Growing")
        self.rb_agglom    = QRadioButton("Agglomerative Clustering")
        self.rb_meanshift = QRadioButton("Mean Shift")

        self.rb_kmeans.setChecked(True)

        for idx, rb in enumerate([self.rb_kmeans, self.rb_region,
                                   self.rb_agglom, self.rb_meanshift]):
            self.method_group.addButton(rb, idx)
            layout.addWidget(rb)

        layout.addSpacing(8)

        # ── Dynamic parameter stack ───────────────────────────────────────────
        self.param_stack = QStackedWidget()
        self.param_stack.addWidget(self._create_kmeans_params())      # index 0
        self.param_stack.addWidget(self._create_region_params())      # index 1
        self.param_stack.addWidget(self._create_agglom_params())      # index 2
        self.param_stack.addWidget(self._create_meanshift_params())   # index 3
        layout.addWidget(self.param_stack)

        # ── Status / info label ───────────────────────────────────────────────
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
        layout.addWidget(QLabel("Click the image to set the seed pixel."))
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

    def _wire_signals(self):
        self.btn_load_image.clicked.connect(self._on_load_image)
        self.btn_apply.clicked.connect(self._on_apply_segmentation)
        self.method_group.buttonClicked.connect(self._on_method_changed)

    # ──────────────────────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────────────────────

    def _on_load_image(self):
        """Open a file dialog and load a gray or colour image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        try:
            self._current_image = load_image(path)
            self._seed = None
            self._result_image = None
            self.lbl_seed_info.setText("")
            self.lbl_n_segments.setText("")
            self.lbl_result.setText("Result will appear here")
            self.lbl_result.setPixmap(QPixmap())  # clear previous result
            self._display_image(self.lbl_original, self._current_image)
            self.lbl_status.setText(
                f"Loaded: {self._current_image.shape[1]}×"
                f"{self._current_image.shape[0]} px, "
                f"{'grayscale' if self._current_image.ndim == 2 else 'colour'}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _on_method_changed(self, button):
        """Switch the parameter panel to match the selected method."""
        method_id = self.method_group.id(button)
        self.param_stack.setCurrentIndex(method_id)
        self.lbl_status.setText("")

    def _on_image_clicked(self, row: int, col: int):
        """
        Record a seed pixel for Region Growing.

        The callback fires only when Region Growing is the active method so
        that clicks during other methods are harmlessly ignored.
        """
        if self.method_group.checkedId() != 1:   # 1 == Region Growing
            return
        if self._current_image is None:
            return

        # Clamp to image bounds
        H = self._current_image.shape[0]
        W = self._current_image.shape[1]
        row = max(0, min(row, H - 1))
        col = max(0, min(col, W - 1))

        self._seed = (row, col)
        self.lbl_seed_info.setText(f"Seed: ({row}, {col})")

    def _on_apply_segmentation(self):
        """Dispatch to the selected segmentation algorithm and display the result."""
        if self._current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        method_id = self.method_group.checkedId()

        try:
            self.lbl_status.setText("Running…")
            self.repaint()   # flush the UI so the user sees the message

            if method_id == 0:
                # ── K-Means ───────────────────────────────────────────────────
                result = apply_kmeans(
                    self._current_image,
                    k=self.spin_k.value(),
                    max_iter=self.spin_kmeans_iter.value(),
                )

            elif method_id == 1:
                # ── Region Growing ────────────────────────────────────────────
                if self._seed is None:
                    QMessageBox.warning(
                        self, "No Seed",
                        "Click on the image to set a seed pixel first."
                    )
                    self.lbl_status.setText("")
                    return

                # Region growing requires a grayscale image
                gray = (
                    self._current_image
                    if self._current_image.ndim == 2
                    else self._to_gray(self._current_image)
                )
                result = apply_region_growing(
                    gray,
                    seed=self._seed,
                    tolerance=self.spin_rg_tolerance.value(),
                )

            elif method_id == 2:
                # ── Agglomerative ─────────────────────────────────────────────
                result = apply_agglomerative(
                    self._current_image,
                    k=self.spin_agg_k.value(),
                )

            elif method_id == 3:
                # ── Mean Shift ────────────────────────────────────────────────
                result = apply_mean_shift(
                    self._current_image,
                    bandwidth=self.spin_bandwidth.value(),
                )

            else:
                return

            # ── Colourise label image and display ─────────────────────────────
            self._result_image = colorise_labels(result.image)
            self._display_image(self.lbl_result, self._result_image)
            self.lbl_n_segments.setText(f"Segments found: {result.n_segments}")
            self.lbl_status.setText(f"Done  ({result.method})")

        except RuntimeError as e:
            # Friendly message for unimplemented stubs
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

    def _display_image(self, label: QLabel, image: np.ndarray):
        """Convert a numpy array to a QPixmap and scale it to fit the label."""
        if image is None:
            return
        pixmap = ndarray_to_qpixmap(image)
        label.setPixmap(
            pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """
        Convert an (H, W, 3) uint8 RGB image to (H, W) uint8 grayscale.

        Uses the standard luminosity weights (no external library required).
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.uint8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_image is not None:
            self._display_image(self.lbl_original, self._current_image)
        if self._result_image is not None:
            self._display_image(self.lbl_result, self._result_image)