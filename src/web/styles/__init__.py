"""
Style package for the PyQt6-based GUI.

This package provides a unified interface for accessing colors, fonts, layouts,
and theme management functionality to ensure consistent styling across the application.

Example:
    Initialize the theme system:
    ```
    from web.styles import initialize_theme
    initialize_theme("light")  # or "dark"
    ```

    Get a theme color:
    ```
    from web.styles import get_theme_color
    primary_color = get_theme_color("primary")
    ```
"""

from PyQt6.QtWidgets import QApplication  # PyQt6 version 6.6.1

# Import functions from submodules
from web.styles.colors import get_theme_color, create_color_palette
from web.styles.fonts import get_default_font, get_equation_font
from web.styles.layouts import adjust_layout_for_theme
from web.styles.theme import initialize_theme, apply_theme, get_current_theme

# Package version
__version__ = "1.0.0"

def is_theme_initialized() -> bool:
    """
    Check if the theme system has been initialized.
    
    Returns:
        bool: True if theme system is initialized, False otherwise
    """
    app = QApplication.instance()
    if app is None:
        return False
    
    # Check if application has a stylesheet set
    return bool(app.styleSheet())

# Re-export all imported functions
__all__ = [
    'get_theme_color',
    'create_color_palette',
    'get_default_font',
    'get_equation_font',
    'adjust_layout_for_theme',
    'initialize_theme',
    'apply_theme',
    'get_current_theme',
    'is_theme_initialized',
]