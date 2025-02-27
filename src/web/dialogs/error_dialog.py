"""
Error dialog for displaying error messages and exceptions in the GUI.

This module provides a modal dialog with proper styling and theme integration
for displaying error messages to the user.
"""

import logging
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

# Internal imports
from web.utils.qt_helpers import create_widget
from web.styles.theme import initialize_theme

# Initialize logger
logger = logging.getLogger(__name__)

# Path to error/warning icon
ERROR_ICON_PATH = "src/web/assets/icons/warning.png"

class ErrorDialog(QDialog):
    """
    Modal dialog for displaying error messages with proper styling and theme integration.
    """
    
    def __init__(self, message: str, title: str = "Error", parent=None):
        """
        Initializes the error dialog with message and styling.
        
        Args:
            message: Error message to display
            title: Dialog title (defaults to "Error")
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set window title and modal behavior
        self.setWindowTitle(title)
        self.setModal(True)
        
        # Initialize UI components
        self._message_label = None
        self._ok_button = None
        self._layout = None
        
        # Setup UI components
        self._setup_ui()
        
        # Set message text
        self._message_label.setText(message)
        
        # Log error
        logger.error(f"Error dialog displayed: {title} - {message}")
    
    def _setup_ui(self):
        """
        Sets up the dialog's user interface components.
        """
        # Create main layout
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(20, 20, 20, 20)
        self._layout.setSpacing(15)
        
        # Try to add warning icon
        try:
            icon_label = create_widget("QLabel", {
                "alignment": Qt.AlignmentFlag.AlignCenter
            })
            
            icon = QIcon(ERROR_ICON_PATH)
            icon_pixmap = icon.pixmap(32, 32)
            if not icon_pixmap.isNull():
                icon_label.setPixmap(icon_pixmap)
                self._layout.addWidget(icon_label)
        except Exception as e:
            logger.warning(f"Failed to load error icon: {str(e)}")
        
        # Create and add message label
        self._message_label = self._create_message_label()
        self._layout.addWidget(self._message_label)
        
        # Create and add OK button
        self._ok_button = self._create_ok_button()
        self._layout.addWidget(self._ok_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Set dialog layout
        self.setLayout(self._layout)
        
        # Connect button signals
        self._ok_button.clicked.connect(self.accept)
    
    def _create_message_label(self):
        """
        Creates a styled label for the error message.
        
        Returns:
            QLabel: Configured message label
        """
        # Create label using the helper function
        label_properties = {
            "text": "",
            "wordWrap": True,
            "alignment": Qt.AlignmentFlag.AlignCenter,
            "textFormat": Qt.TextFormat.RichText
        }
        
        try:
            label = create_widget("QLabel", label_properties)
        except ValueError:
            # Fallback if helper fails
            label = QLabel()
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setTextFormat(Qt.TextFormat.RichText)
        
        return label
    
    def _create_ok_button(self):
        """
        Creates a styled OK button for the dialog.
        
        Returns:
            QPushButton: Configured OK button
        """
        # Create button using the helper function
        button_properties = {
            "text": "OK",
            "minimumWidth": 80
        }
        
        try:
            button = create_widget("QPushButton", button_properties)
        except ValueError:
            # Fallback if helper fails
            button = QPushButton("OK")
            button.setMinimumWidth(80)
        
        return button