"""
ThresholdingTab – Part A UI.


Signals / Slots (to be wired during implementation):
  • btn_load_image.clicked  →  _on_load_image()
  • btn_apply.clicked       →  _on_apply_threshold()
  • method_group.buttonClicked → _on_method_changed()
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QButtonGroup, QRadioButton, QGroupBox, QLabel,
    QSpinBox, QSizePolicy,
)


class ThresholdingTab(QWidget):
    """UI container for Part A – Thresholding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None   # numpy ndarray (grayscale)
        self._result_image  = None   # numpy ndarray (binary)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Toolbar ──────────────────────────────────────────────────
        toolbar = self._create_toolbar()
        root.addLayout(toolbar)

        # ── Main content area ─────────────────────────────────────────
        content = QHBoxLayout()
        content.addWidget(self._create_original_panel(), stretch=1)
        content.addWidget(self._create_controls_panel(), stretch=1)
        content.addWidget(self._create_result_panel(),   stretch=1)
        root.addLayout(content)

    def _create_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.btn_load_image = QPushButton("Load Grayscale Image")
        layout.addWidget(self.btn_load_image)
        layout.addStretch()
        return layout

    def _create_original_panel(self) -> QGroupBox:
        """Panel that displays the loaded original image."""
        box = QGroupBox("Original Image")
        layout = QVBoxLayout(box)
        self.lbl_original = QLabel("No image loaded")
        self.lbl_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_original)
        return box

    def _create_controls_panel(self) -> QGroupBox:
        """Panel with method selection and Apply button."""
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

        layout.addSpacing(12)
        self.lbl_threshold_val = QLabel("Computed threshold: –")
        layout.addWidget(self.lbl_threshold_val)

        self.btn_apply = QPushButton("Apply")
        layout.addStretch()
        layout.addWidget(self.btn_apply)

        return box

    def _create_result_panel(self) -> QGroupBox:
        """Panel that displays the thresholded result image."""
        box = QGroupBox("Thresholded Result")
        layout = QVBoxLayout(box)
        self.lbl_result = QLabel("Result will appear here")
        self.lbl_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_result)
        return box

    # ------------------------------------------------------------------
    # Slots (stubs – to be implemented)
    # ------------------------------------------------------------------

    def _on_load_image(self):
        """Open a file dialog, load a grayscale image. (stub)"""
        pass

    def _on_apply_threshold(self):
        """Dispatch to the correct thresholding algorithm. (stub)"""
        pass

    def _on_method_changed(self, button):
        """React to method radio-button change. (stub)"""
        pass

    def _display_image(self, label: QLabel, image):
        """Convert numpy array → QPixmap and set on label. (stub)"""
        pass
