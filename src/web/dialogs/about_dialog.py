"""
About dialog for the MFE Toolbox GUI.

This module implements a modal dialog displaying application information,
version, credits, and links to website and documentation.
"""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtGui import QPixmap, QDesktopServices  # PyQt6 version 6.6.1
from PyQt6.QtCore import Qt, QUrl  # PyQt6 version 6.6.1

# Internal imports
from web.utils.qt_helpers import create_widget
from web.styles.theme import initialize_theme

# Constants
VERSION = "4.0"
LOGO_PATH = "src/web/assets/icons/logo.png"

class AboutDialog(QDialog):
    """
    Modal dialog displaying application information, version and links.
    """
    
    def __init__(self, parent=None):
        """
        Initializes the About dialog with logo, version info and buttons.
        
        Args:
            parent: Parent widget (QWidget)
        """
        super().__init__(parent)
        self.setWindowTitle("About ARMAX")
        
        # Initialize UI components
        self._logo_label = None
        self._version_label = None
        self._website_button = None
        self._docs_button = None
        self._ok_button = None
        
        # Create dialog UI
        self._create_ui()
        
        # Make dialog modal
        self.setModal(True)

    def _create_ui(self):
        """
        Creates and configures dialog UI components.
        """
        # Create main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Add logo
        self._logo_label = create_widget("QLabel", {
            "alignment": int(Qt.AlignmentFlag.AlignCenter)
        })
        
        pixmap = QPixmap(LOGO_PATH)
        if not pixmap.isNull():
            # Ensure logo is not too large
            if pixmap.width() > 200:
                pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
            self._logo_label.setPixmap(pixmap)
        else:
            # Fallback if logo image cannot be loaded
            self._logo_label.setText("ARMAX")
            self._logo_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        layout.addWidget(self._logo_label)
        
        # Add version information
        self._version_label = create_widget("QLabel", {
            "text": f"ARMAX Model Estimation\nVersion {VERSION}\n(c) 2009 Kevin Sheppard",
            "alignment": int(Qt.AlignmentFlag.AlignCenter)
        })
        layout.addWidget(self._version_label)
        
        # Add spacing
        layout.addSpacing(10)
        
        # Add buttons for website and documentation in horizontal layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._website_button = create_widget("QPushButton", {
            "text": "Website"
        })
        self._website_button.clicked.connect(self._on_website_clicked)
        button_layout.addWidget(self._website_button)
        
        self._docs_button = create_widget("QPushButton", {
            "text": "Documentation"
        })
        self._docs_button.clicked.connect(self._on_docs_clicked)
        button_layout.addWidget(self._docs_button)
        
        layout.addLayout(button_layout)
        
        # Add spacing
        layout.addSpacing(10)
        
        # Add OK button
        self._ok_button = create_widget("QPushButton", {
            "text": "OK",
            "default": True
        })
        self._ok_button.clicked.connect(self.accept)
        layout.addWidget(self._ok_button, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Set dialog layout
        self.setLayout(layout)
        
        # Set fixed size based on content
        self.setFixedSize(300, 350)
    
    def _on_website_clicked(self):
        """
        Opens project website in default browser.
        """
        QDesktopServices.openUrl(QUrl("https://github.com/bashtage/arch"))
    
    def _on_docs_clicked(self):
        """
        Opens documentation in default browser.
        """
        QDesktopServices.openUrl(QUrl("https://bashtage.github.io/arch/"))