"""
MFE Toolbox GUI dialog components.

This package provides modal dialog windows for the MFE Toolbox GUI,
implemented using PyQt6. It includes dialogs for About information,
Close confirmation, Error messages, and context-sensitive Help.

All dialogs follow a consistent visual style and support the application's
theming system.
"""

# Version information
__version__ = "4.0"  # Matches MFE Toolbox version

# Import dialog classes
from .about_dialog import AboutDialog
from .close_dialog import CloseDialog
from .error_dialog import ErrorDialog
from .help_dialog import HelpDialog

# Define public exports
__all__ = [
    'AboutDialog',
    'CloseDialog',
    'ErrorDialog',
    'HelpDialog'
]