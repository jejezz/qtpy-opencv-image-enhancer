#!/usr/bin/env python3
"""
Qt Image Enhancer - Main Application
A cross-platform image enhancement tool using QtPy
"""

import sys
from qtpy.QtWidgets import QApplication
from src.ui.main_window import MainWindow


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Qt Image Enhancer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Your Organization")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()