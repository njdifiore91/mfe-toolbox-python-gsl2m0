"""
Confirmation dialog for the MFE Toolbox application.

This dialog prompts users when closing the application or model windows
to confirm their action and prevent accidental loss of unsaved changes.
"""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QHBoxLayout  # PyQt6 version 6.6.1
from PyQt6.QtGui import QIcon  # PyQt6 version 6.6.1

from web.utils.qt_helpers import create_widget
from web.styles.theme import initialize_theme  # For applying consistent theme to dialog


class CloseDialog(QDialog):
    """
    Modal dialog for confirming application or window closure with unsaved changes.
    
    This dialog displays a warning message and provides Yes/No buttons for user confirmation,
    helping prevent accidental data loss.
    """
    
    def __init__(self, parent=None, title="Confirm Close"):
        """
        Initializes the close confirmation dialog.
        
        Args:
            parent: Parent widget (window/application)
            title: Dialog title text
        """
        # Call parent constructor
        super().__init__(parent)
        
        # Set dialog properties
        self.setWindowTitle(title)
        self.setModal(True)
        
        # Initialize UI elements
        self._message_label = None
        self._button_box = None
        self._layout = None
        
        # Set up the user interface
        self.setup_ui()
        
    def setup_ui(self):
        """
        Configures the dialog's user interface components.
        
        This method creates and arranges all visual elements including
        warning icon, message text, and Yes/No buttons.
        """
        # Create main vertical layout
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(20, 20, 20, 20)
        self._layout.setSpacing(15)
        
        # Create warning message label
        self._message_label = create_widget("QLabel", {
            "text": "Are you sure you want to close?\nUnsaved changes will be lost."
        })
        
        # Create warning icon (use system icon if available)
        icon_label = create_widget("QLabel", {
            "pixmap": self.style().standardIcon(
                self.style().StandardPixmap.SP_MessageBoxWarning).pixmap(32, 32)
        })
        
        # Create horizontal layout for icon and message
        icon_layout = QHBoxLayout()
        icon_layout.setSpacing(15)
        icon_layout.addWidget(icon_label)
        icon_layout.addWidget(self._message_label)
        icon_layout.addStretch(1)
        
        # Add icon and message to main layout
        self._layout.addLayout(icon_layout)
        
        # Create button box with Yes/No buttons
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        
        # Set No button as default (safer option)
        no_button = self._button_box.button(QDialogButtonBox.StandardButton.No)
        if no_button:
            no_button.setDefault(True)
        
        # Connect buttons to accept/reject slots
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        
        # Add button box to layout
        self._layout.addWidget(self._button_box)
        
        # Set dialog layout
        self.setLayout(self._layout)
        
    def show_dialog(self):
        """
        Displays the dialog and returns user's choice.
        
        Returns:
            bool: True if user confirmed close (Yes), False otherwise (No)
        """
        # Execute dialog and return result
        result = self.exec()
        return result == QDialog.DialogCode.Accepted