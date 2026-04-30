"""
MainWindow – top-level application window.

Contains a QTabWidget with two tabs:
  • Tab 1: ThresholdingTab  (Part A)
  • Tab 2: SegmentationTab  (Part B)
"""

from PyQt5.QtWidgets import QMainWindow, QTabWidget
from ui.thresholding_tab import ThresholdingTab
from ui.segmentation_tab import SegmentationTab


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CV Task 4 – Thresholding & Segmentation")
        self.resize(1200, 800)

        self._tabs = QTabWidget()
        self._thresholding_tab = ThresholdingTab()
        self._segmentation_tab = SegmentationTab()

        self._tabs.addTab(self._thresholding_tab, "Part A – Thresholding")
        self._tabs.addTab(self._segmentation_tab, "Part B – Segmentation")

        self.setCentralWidget(self._tabs)
        self._build_menu_bar()
        self._build_status_bar()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_menu_bar(self):
        """Create File / Help menus. (stub)"""
        pass

    def _build_status_bar(self):
        """Initialise the status bar. (stub)"""
        self.statusBar().showMessage("Ready")
