"""
Font configurations and management for the PyQt6-based GUI.

This module provides consistent typography across the application interface by defining
standard font families, sizes, weights, and utility functions to create font instances.
"""

from PyQt6.QtGui import QFont, QFontDatabase  # PyQt6 version 6.6.1

# Standard font family definitions
FONT_FAMILIES = {
    'default': 'Segoe UI',
    'monospace': 'Consolas',
    'equation': 'Times New Roman'
}

# Standard font size definitions
FONT_SIZES = {
    'small': 9,
    'normal': 10,
    'large': 12,
    'header': 14
}

# Standard font weight definitions
FONT_WEIGHTS = {
    'normal': 400,  # QFont.Weight.Normal
    'medium': 500,  # QFont.Weight.Medium
    'bold': 700     # QFont.Weight.Bold
}


def get_font(family: str, size: int, weight: int) -> QFont:
    """
    Creates and returns a QFont with specified properties.
    
    Args:
        family: Font family name (e.g., 'Segoe UI')
        size: Font size in points
        weight: Font weight (400=normal, 700=bold)
    
    Returns:
        QFont: Configured font instance
    """
    # Validate font family exists in system
    font_db = QFontDatabase()
    families = font_db.families()
    
    # If specified family doesn't exist, fall back to a system default
    if family not in families:
        if 'default' in FONT_FAMILIES and FONT_FAMILIES['default'] in families:
            family = FONT_FAMILIES['default']
        else:
            family = font_db.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    
    # Create and configure font
    font = QFont(family)
    font.setPointSize(size)
    font.setWeight(weight)
    
    return font


def get_default_font() -> QFont:
    """
    Returns the default application font with standard size and weight.
    
    Returns:
        QFont: Default font instance
    """
    return get_font(
        FONT_FAMILIES['default'],
        FONT_SIZES['normal'],
        FONT_WEIGHTS['normal']
    )


def get_monospace_font() -> QFont:
    """
    Returns a monospace font suitable for code or equation display.
    
    Returns:
        QFont: Monospace font instance
    """
    font = get_font(
        FONT_FAMILIES['monospace'],
        FONT_SIZES['normal'],
        FONT_WEIGHTS['normal']
    )
    font.setFixedPitch(True)
    return font


def get_equation_font() -> QFont:
    """
    Returns a font suitable for mathematical equation rendering.
    
    Returns:
        QFont: Equation font instance
    """
    font = get_font(
        FONT_FAMILIES['equation'],
        FONT_SIZES['large'],
        FONT_WEIGHTS['normal']
    )
    
    # Enable kerning and standard ligatures for better equation rendering
    font.setKerning(True)
    font.setStyleStrategy(QFont.StyleStrategy.PreferDefault)
    
    return font