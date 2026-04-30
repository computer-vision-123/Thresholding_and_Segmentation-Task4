"""
SegmentationTab – Part B UI.

Signals / Slots (to be wired during implementation):
  • btn_load_image.clicked     →  _on_load_image()
  • btn_apply.clicked          →  _on_apply_segmentation()
  • method_group.buttonClicked →  _on_method_changed()
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QButtonGroup, QRadioButton, QGroupBox, QLabel,
    QStackedWidget, QSpinBox, QDoubleSpinBox, QSizePolicy,
)


class SegmentationTab(QWidget):
    """UI container for Part B – Unsupervised Segmentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None   # numpy ndarray (gray or color)
        self._result_image  = None   # numpy ndarray (labelled / colourised)
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
        self.lbl_original = QLabel("No image loaded")
        self.lbl_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_original)
        return box

    def _create_controls_panel(self) -> QGroupBox:
        """Panel with method radio buttons, dynamic parameters, and Apply."""
        box = QGroupBox("Segmentation Method")
        layout = QVBoxLayout(box)

        # ── Method selection ─────────────────────────────────────────
        self.method_group = QButtonGroup(self)

        self.rb_kmeans      = QRadioButton("K-Means")
        self.rb_region      = QRadioButton("Region Growing")
        self.rb_agglom      = QRadioButton("Agglomerative Clustering")
        self.rb_meanshift   = QRadioButton("Mean Shift")

        self.rb_kmeans.setChecked(True)

        for idx, rb in enumerate([self.rb_kmeans, self.rb_region,
                                   self.rb_agglom, self.rb_meanshift]):
            self.method_group.addButton(rb, idx)
            layout.addWidget(rb)

        layout.addSpacing(8)

        # ── Dynamic parameter stack ───────────────────────────────────
        self.param_stack = QStackedWidget()
        self.param_stack.addWidget(self._create_kmeans_params())     # index 0
        self.param_stack.addWidget(self._create_region_params())     # index 1
        self.param_stack.addWidget(self._create_agglom_params())     # index 2
        self.param_stack.addWidget(self._create_meanshift_params())  # index 3
        layout.addWidget(self.param_stack)

        layout.addStretch()
        self.btn_apply = QPushButton("Apply")
        layout.addWidget(self.btn_apply)

        return box

    # ── Per-method parameter widgets (stubs) ──────────────────────────

    def _create_kmeans_params(self) -> QWidget:
        """Parameters for K-Means: number of clusters k, max iterations."""
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
        return w

    def _create_region_params(self) -> QWidget:
        """Parameters for Region Growing: seed strategy, tolerance."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Intensity tolerance:"))
        self.spin_rg_tolerance = QSpinBox()
        self.spin_rg_tolerance.setRange(1, 255)
        self.spin_rg_tolerance.setValue(15)
        layout.addWidget(self.spin_rg_tolerance)
        # Seed point selection note
        layout.addWidget(QLabel("(Click on image to set seed)"))
        return w

    def _create_agglom_params(self) -> QWidget:
        """Parameters for Agglomerative: number of clusters, linkage."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Number of clusters:"))
        self.spin_agg_k = QSpinBox()
        self.spin_agg_k.setRange(2, 20)
        self.spin_agg_k.setValue(4)
        layout.addWidget(self.spin_agg_k)
        return w

    def _create_meanshift_params(self) -> QWidget:
        """Parameters for Mean Shift: bandwidth."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Bandwidth:"))
        self.spin_bandwidth = QDoubleSpinBox()
        self.spin_bandwidth.setRange(1.0, 200.0)
        self.spin_bandwidth.setValue(30.0)
        self.spin_bandwidth.setSingleStep(5.0)
        layout.addWidget(self.spin_bandwidth)
        return w

    def _create_result_panel(self) -> QGroupBox:
        box = QGroupBox("Segmentation Result")
        layout = QVBoxLayout(box)
        self.lbl_result = QLabel("Result will appear here")
        self.lbl_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_result)
        return box

    # ------------------------------------------------------------------
    # Slots (stubs – to be implemented)
    # ------------------------------------------------------------------

    def _on_load_image(self):
        """Open a file dialog, load a gray or color image. (stub)"""
        pass

    def _on_apply_segmentation(self):
        """Dispatch to the selected segmentation algorithm. (stub)"""
        pass

    def _on_method_changed(self, button):
        """Switch the parameter panel to match the selected method. (stub)"""
        method_id = self.method_group.id(button)
        self.param_stack.setCurrentIndex(method_id)

    def _display_image(self, label: QLabel, image):
        """Convert numpy array → QPixmap and set on label. (stub)"""
        pass
