"""
ThresholdingTab – Part A UI.


Signals / Slots (to be wired during implementation):
  • btn_load_image.clicked  →  _on_load_image()
  • btn_apply.clicked       →  _on_apply_threshold()
  • method_group.buttonClicked → _on_method_changed()
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QButtonGroup, QRadioButton, QGroupBox, QLabel,
    QSpinBox, QSizePolicy, QFileDialog, QFormLayout,
)
from PyQt5.QtCore import Qt

from utils.image_utils import load_grayscale, ndarray_to_qpixmap, colorise_labels
from core.thresholding import (
    apply_optimal_thresholding,
    apply_otsu_thresholding,
    apply_spectral_thresholding,
    apply_local_thresholding,
)


class ThresholdingTab(QWidget):
    """UI container for Part A – Thresholding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None   # numpy ndarray (grayscale)
        self._result_image  = None   # numpy ndarray (binary or labels)
        self._build_ui()
        self._wire_signals()

    def _wire_signals(self):
        self.btn_load_image.clicked.connect(self._on_load_image)
        self.btn_apply.clicked.connect(self._on_apply_threshold)
        self.method_group.buttonClicked.connect(self._on_method_changed)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Toolbar ──────────────────────────────────────────────────
        toolbar = self._create_toolbar()
        root.addLayout(toolbar)

        # ── Main content area ─────────────────────────────────────────
        self.btn_apply = QPushButton("Apply")
        
        content = QHBoxLayout()
        content.addWidget(self._create_original_panel(), stretch=1)
        
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self._create_method_panel())
        controls_layout.addWidget(self._create_params_panel())
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_apply)
        
        content.addLayout(controls_layout, stretch=1)
        content.addWidget(self._create_result_panel(),   stretch=1)
        root.addLayout(content)

    def _create_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.btn_load_image = QPushButton("Load Grayscale Image")
        layout.addWidget(self.btn_load_image)
        layout.addStretch()
        return layout

    def _create_original_panel(self) -> QGroupBox:
        box = QGroupBox("Original Image")
        layout = QVBoxLayout(box)
        self.lbl_original = QLabel("No image loaded")
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_original)
        return box

    def _create_method_panel(self) -> QGroupBox:
        box = QGroupBox("Thresholding Method")
        layout = QVBoxLayout(box)
        self.method_group = QButtonGroup(self)

        self.rb_optimal   = QRadioButton("Optimal Thresholding")
        self.rb_otsu      = QRadioButton("Otsu's Method")
        self.rb_spectral  = QRadioButton("Spectral (Multi-modal)")
        self.rb_local     = QRadioButton("Local Thresholding")

        self.rb_optimal.setChecked(True)

        for idx, rb in enumerate([self.rb_optimal, self.rb_otsu,
                                   self.rb_spectral, self.rb_local]):
            self.method_group.addButton(rb, idx)
            layout.addWidget(rb)
        return box

    def _create_params_panel(self) -> QGroupBox:
        self.params_box = QGroupBox("Parameters")
        self.params_layout = QFormLayout(self.params_box)

        # Optimal params
        self.spin_tol = QSpinBox()
        self.spin_tol.setRange(1, 50)
        self.spin_tol.setValue(1)
        
        # Spectral params
        self.spin_classes = QSpinBox()
        self.spin_classes.setRange(3, 5)
        self.spin_classes.setValue(3)

        # Local params
        self.spin_block = QSpinBox()
        self.spin_block.setRange(3, 255)
        self.spin_block.setSingleStep(2)
        self.spin_block.setValue(35)
        
        self.spin_offset = QSpinBox()
        self.spin_offset.setRange(0, 100)
        self.spin_offset.setValue(10)

        self.lbl_threshold_val = QLabel("Computed threshold: –")
        
        self._update_params_visibility()
        return self.params_box

    def _update_params_visibility(self):
        # Clear layout
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().hide()

        method_idx = self.method_group.checkedId()
        if method_idx == 0: # Optimal
            self.params_layout.addRow("Tolerance (uint8):", self.spin_tol)
            self.spin_tol.show()
        elif method_idx == 2: # Spectral
            self.params_layout.addRow("N Classes:", self.spin_classes)
            self.spin_classes.show()
        elif method_idx == 3: # Local
            self.params_layout.addRow("Block Size (odd):", self.spin_block)
            self.params_layout.addRow("Offset (uint8):", self.spin_offset)
            self.spin_block.show()
            self.spin_offset.show()
        
        self.params_layout.addRow(self.lbl_threshold_val)
        self.lbl_threshold_val.show()

    def _create_result_panel(self) -> QGroupBox:
        box = QGroupBox("Thresholded Result")
        layout = QVBoxLayout(box)
        self.lbl_result = QLabel("Result will appear here")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_result)
        return box

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.tif)")
        if path:
            try:
                self._current_image = load_grayscale(path)
                self._display_image(self.lbl_original, self._current_image)
            except Exception as e:
                print(f"Error loading image: {e}")

    def _on_method_changed(self, button):
        self._update_params_visibility()

    def _on_apply_threshold(self):
        if self._current_image is None:
            return

        method_idx = self.method_group.checkedId()
        
        try:
            if method_idx == 0: # Optimal
                res = apply_optimal_thresholding(self._current_image, self.spin_tol.value())
                self._result_image = res.image
                self.lbl_threshold_val.setText(f"Computed threshold: {res.threshold_uint8}")
            elif method_idx == 1: # Otsu
                res = apply_otsu_thresholding(self._current_image)
                self._result_image = res.image
                self.lbl_threshold_val.setText(f"Computed threshold: {res.threshold_uint8}")
            elif method_idx == 2: # Spectral
                res = apply_spectral_thresholding(self._current_image, self.spin_classes.value())
                self._result_image = colorise_labels(res.image)
                self.lbl_threshold_val.setText(f"Thresholds: {res.thresholds_uint8}")
            elif method_idx == 3: # Local
                block = self.spin_block.value()
                if block % 2 == 0: block += 1
                res = apply_local_thresholding(self._current_image, block, self.spin_offset.value())
                self._result_image = res.image
                self.lbl_threshold_val.setText("Local thresholding applied.")

            self._display_image(self.lbl_result, self._result_image)
        except Exception as e:
            print(f"Error applying threshold: {e}")

    def _display_image(self, label: QLabel, image):
        if image is None:
            return
        pixmap = ndarray_to_qpixmap(image)
        # Scale pixmap to fit label while preserving aspect ratio
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_image is not None:
            self._display_image(self.lbl_original, self._current_image)
        if self._result_image is not None:
            self._display_image(self.lbl_result, self._result_image)

