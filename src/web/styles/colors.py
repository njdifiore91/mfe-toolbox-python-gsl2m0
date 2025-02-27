"""
Color constants and utility functions for the GUI theme system.

This module provides consistent color schemes for both light and dark themes,
along with utility functions for managing colors in the PyQt6-based GUI.
"""

from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt

# Primary color scheme
PRIMARY_COLOR = QColor(51, 122, 183)      # Bootstrap primary blue
SECONDARY_COLOR = QColor(108, 117, 125)   # Bootstrap secondary gray
SUCCESS_COLOR = QColor(40, 167, 69)       # Bootstrap success green
WARNING_COLOR = QColor(255, 193, 7)       # Bootstrap warning yellow
ERROR_COLOR = QColor(220, 53, 69)         # Bootstrap danger red

# Theme-specific colors
BACKGROUND_LIGHT = QColor(255, 255, 255)  # White
BACKGROUND_DARK = QColor(33, 37, 41)      # Dark gray
TEXT_LIGHT = QColor(33, 37, 41)           # Dark gray
TEXT_DARK = QColor(255, 255, 255)         # White
BORDER_LIGHT = QColor(222, 226, 230)      # Light gray
BORDER_DARK = QColor(73, 80, 87)          # Medium gray


def get_theme_color(color_name: str, theme: str = "light") -> QColor:
    """
    Returns the appropriate color for the current theme.
    
    Args:
        color_name: The name of the color to retrieve
        theme: The theme to use ('light' or 'dark')
        
    Returns:
        QColor: Theme-appropriate color object
        
    Raises:
        ValueError: If color_name is not recognized or theme is invalid
    """
    # Validate theme
    if theme not in ["light", "dark"]:
        raise ValueError(f"Invalid theme: {theme}. Must be 'light' or 'dark'")
        
    # Universal colors (same in both themes)
    universal_colors = {
        "primary": PRIMARY_COLOR,
        "secondary": SECONDARY_COLOR,
        "success": SUCCESS_COLOR,
        "warning": WARNING_COLOR,
        "error": ERROR_COLOR,
    }
    
    # Theme-specific colors
    theme_specific_colors = {
        "light": {
            "background": BACKGROUND_LIGHT,
            "text": TEXT_LIGHT,
            "border": BORDER_LIGHT,
        },
        "dark": {
            "background": BACKGROUND_DARK,
            "text": TEXT_DARK,
            "border": BORDER_DARK,
        }
    }
    
    # First check universal colors
    if color_name in universal_colors:
        return universal_colors[color_name]
    
    # Then check theme-specific colors
    if color_name in theme_specific_colors[theme]:
        return theme_specific_colors[theme][color_name]
    
    # If not found, raise error
    raise ValueError(f"Color name not recognized: {color_name}")


def create_color_palette(theme: str = "light") -> dict:
    """
    Creates a complete color palette for the specified theme.
    
    Args:
        theme: The theme to create a palette for ('light' or 'dark')
        
    Returns:
        dict: Dictionary mapping color roles to QColor objects
        
    Raises:
        ValueError: If theme is not 'light' or 'dark'
    """
    # Validate theme
    if theme not in ["light", "dark"]:
        raise ValueError(f"Invalid theme: {theme}. Must be 'light' or 'dark'")
    
    # Create palette dictionary
    palette = {
        "primary": PRIMARY_COLOR,
        "secondary": SECONDARY_COLOR,
        "success": SUCCESS_COLOR,
        "warning": WARNING_COLOR,
        "error": ERROR_COLOR,
        "background": BACKGROUND_LIGHT if theme == "light" else BACKGROUND_DARK,
        "text": TEXT_LIGHT if theme == "light" else TEXT_DARK,
        "border": BORDER_LIGHT if theme == "light" else BORDER_DARK,
        "disabled": QColor(200, 200, 200) if theme == "light" else QColor(70, 70, 70),
        "highlight": QColor(PRIMARY_COLOR).lighter(120),
        "shadow": QColor(0, 0, 0, 30) if theme == "light" else QColor(0, 0, 0, 70),
    }
    
    return palette


def get_contrast_color(background_color: QColor) -> QColor:
    """
    Returns appropriate contrast color (light/dark) for given background.
    
    Uses the relative luminance formula to determine if a background is light or dark
    and returns the appropriate contrasting text color.
    
    Args:
        background_color: The background color to find a contrast for
        
    Returns:
        QColor: Contrasting color for text visibility
    """
    # Calculate relative luminance using the formula for perceived brightness
    # Luminance = 0.299*R + 0.587*G + 0.114*B
    r, g, b = background_color.red(), background_color.green(), background_color.blue()
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Return appropriate text color based on background luminance
    return TEXT_LIGHT if luminance > 0.5 else TEXT_DARK