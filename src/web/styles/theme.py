"""
Theme configuration and styling for the PyQt6-based GUI.

This module provides a unified theming system that coordinates colors, fonts,
and layouts across the application interface, supporting both light and dark themes.
"""

from PyQt6.QtWidgets import QApplication, QStyle
from PyQt6.QtGui import QPalette

# Internal imports
from web.styles.colors import get_theme_color, create_color_palette
from web.styles.fonts import get_default_font, get_equation_font
from web.styles.layouts import adjust_layout_for_theme

# Available themes dictionary mapping theme ID to display name
THEMES = {
    "light": "Light Theme",
    "dark": "Dark Theme"
}

# Default theme to use if not specified
DEFAULT_THEME = "light"

# Stylesheet file paths for each theme
STYLE_SHEETS = {
    "light": "src/web/assets/styles/light.qss",
    "dark": "src/web/assets/styles/dark.qss"
}


def initialize_theme(theme_name: str = DEFAULT_THEME) -> None:
    """
    Initializes the application theme system with default or specified theme.
    
    Args:
        theme_name: Name of the theme to initialize ('light' or 'dark')
    
    Raises:
        ValueError: If theme_name is not a valid theme
        RuntimeError: If QApplication instance is not found
    """
    # Validate theme name
    if theme_name not in THEMES:
        raise ValueError(f"Invalid theme: {theme_name}. Valid themes are: {', '.join(THEMES.keys())}")
    
    # Get application instance
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("No QApplication instance found. Theme can only be initialized after QApplication creation.")
    
    # Create color palette for theme
    palette = create_theme_palette(theme_name)
    
    # Set application-wide palette
    app.setPalette(palette)
    
    # Load and apply stylesheet
    try:
        stylesheet_path = STYLE_SHEETS[theme_name]
        with open(stylesheet_path, 'r') as file:
            app.setStyleSheet(file.read())
    except (KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not load stylesheet for theme '{theme_name}': {e}")
    
    # Configure default fonts
    app.setFont(get_default_font())
    
    # Update layout styling for theme happens when layouts are created
    # This function just initializes the application-wide theme settings


def apply_theme(theme_name: str) -> bool:
    """
    Applies the specified theme to the entire application.
    
    Args:
        theme_name: Name of the theme to apply ('light' or 'dark')
    
    Returns:
        bool: True if theme was successfully applied, False otherwise
        
    Raises:
        ValueError: If theme_name is not a valid theme
    """
    # Validate theme name
    if theme_name not in THEMES:
        raise ValueError(f"Invalid theme: {theme_name}. Valid themes are: {', '.join(THEMES.keys())}")
    
    # Get application instance
    app = QApplication.instance()
    if app is None:
        return False
    
    try:
        # Update application palette
        palette = create_theme_palette(theme_name)
        app.setPalette(palette)
        
        # Apply theme stylesheet
        stylesheet_path = STYLE_SHEETS[theme_name]
        with open(stylesheet_path, 'r') as file:
            app.setStyleSheet(file.read())
        
        # Update widget styles - force style refresh
        app.setStyle(app.style().objectName())
        
        return True
    
    except Exception as e:
        print(f"Error applying theme '{theme_name}': {e}")
        return False


def get_current_theme() -> str:
    """
    Returns the currently active theme name.
    
    Returns:
        str: Current theme name ('light' or 'dark')
    """
    # Get application instance
    app = QApplication.instance()
    if app is None:
        return DEFAULT_THEME
    
    # Get current palette
    palette = app.palette()
    
    # Determine active theme from palette
    # Use background color as indicator of theme type
    background = palette.color(QPalette.ColorRole.Window)
    
    # Simple heuristic: if background is dark, assume dark theme
    # Check if the color is closer to black than white
    brightness = background.red() + background.green() + background.blue()
    is_dark = brightness < (3 * 255 / 2)  # 3 channels, middle point between black and white
    
    return "dark" if is_dark else "light"


def create_theme_palette(theme_name: str) -> QPalette:
    """
    Creates a QPalette for the specified theme.
    
    Args:
        theme_name: Name of the theme to create palette for ('light' or 'dark')
    
    Returns:
        QPalette: Theme-specific color palette
        
    Raises:
        ValueError: If theme_name is not a valid theme
    """
    # Validate theme name
    if theme_name not in THEMES:
        raise ValueError(f"Invalid theme: {theme_name}. Valid themes are: {', '.join(THEMES.keys())}")
    
    # Create new palette instance
    palette = QPalette()
    
    # Get theme colors
    background = get_theme_color("background", theme_name)
    text = get_theme_color("text", theme_name)
    highlight = get_theme_color("primary", theme_name)
    border = get_theme_color("border", theme_name)
    disabled = get_theme_color("secondary", theme_name).lighter(150) if theme_name == "light" else get_theme_color("secondary", theme_name).darker(150)
    
    # Set color roles for all widget states
    # Window and base (used for background)
    palette.setColor(QPalette.ColorRole.Window, background)
    palette.setColor(QPalette.ColorRole.Base, background.lighter(105))
    palette.setColor(QPalette.ColorRole.AlternateBase, background.darker(105))
    
    # Text colors
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    
    # Highlight colors
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, background)
    
    # Button colors
    palette.setColor(QPalette.ColorRole.Button, background.lighter(105))
    
    # Link colors
    palette.setColor(QPalette.ColorRole.Link, highlight)
    palette.setColor(QPalette.ColorRole.LinkVisited, highlight.darker(120))
    
    # Configure disabled state
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, background.darker(105) if theme_name == "light" else background.lighter(105))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, background.darker(105) if theme_name == "light" else background.lighter(105))
    
    # Configure inactive state (usually same as active)
    palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight, highlight.lighter(110))
    
    # Border and shadow colors
    palette.setColor(QPalette.ColorRole.Mid, border)
    palette.setColor(QPalette.ColorRole.Dark, border.darker(120))
    palette.setColor(QPalette.ColorRole.Shadow, border.darker(150))
    
    return palette