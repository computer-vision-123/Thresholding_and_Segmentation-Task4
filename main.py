"""
Entry point for the CV Task 4 application.
Initializes the PyQt application and launches the main window.
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CV Task 4 – Thresholding & Segmentation")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
