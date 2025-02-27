"""
Standardized layout configurations and utility functions for the GUI.

This module provides consistent layout configurations for PyQt6-based interfaces,
supporting both form-based and plot display layouts with theme-aware spacing and alignment.
"""

from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLayout
from PyQt6.QtCore import Qt

from web.styles.colors import get_theme_color

# Layout configuration constants
LAYOUT_MARGINS = {
    "default": 10,
    "compact": 5, 
    "spacious": 15
}

LAYOUT_SPACING = {
    "default": 8,
    "compact": 4,
    "spacious": 12
}

# Default alignment for form labels
FORM_LABEL_ALIGNMENT = Qt.AlignmentFlag.AlignRight


def create_form_layout(style: str = "default") -> QFormLayout:
    """
    Creates a standardized form layout with consistent spacing and alignment.
    
    Args:
        style: Layout style ('default', 'compact', or 'spacious')
        
    Returns:
        QFormLayout: Configured form layout instance
        
    Raises:
        ValueError: If style parameter is not recognized
    """
    # Validate style parameter
    if style not in LAYOUT_MARGINS:
        raise ValueError(f"Invalid style: {style}. Valid options are: {', '.join(LAYOUT_MARGINS.keys())}")
    
    # Create new form layout
    layout = QFormLayout()
    
    # Set margins based on style parameter
    margin = LAYOUT_MARGINS[style]
    layout.setContentsMargins(margin, margin, margin, margin)
    
    # Configure label alignment using FORM_LABEL_ALIGNMENT
    layout.setLabelAlignment(FORM_LABEL_ALIGNMENT)
    
    # Set field growth policy for consistent sizing
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    
    # Apply spacing based on style
    layout.setSpacing(LAYOUT_SPACING[style])
    
    return layout


def create_grid_layout(style: str = "default") -> QGridLayout:
    """
    Creates a standardized grid layout for plot displays and statistical results.
    
    Args:
        style: Layout style ('default', 'compact', or 'spacious')
        
    Returns:
        QGridLayout: Configured grid layout instance
        
    Raises:
        ValueError: If style parameter is not recognized
    """
    # Validate style parameter
    if style not in LAYOUT_MARGINS:
        raise ValueError(f"Invalid style: {style}. Valid options are: {', '.join(LAYOUT_MARGINS.keys())}")
    
    # Create new grid layout
    layout = QGridLayout()
    
    # Set margins based on style parameter
    margin = LAYOUT_MARGINS[style]
    layout.setContentsMargins(margin, margin, margin, margin)
    
    # Configure spacing between elements
    spacing = LAYOUT_SPACING[style]
    layout.setSpacing(spacing)
    
    # Set size constraints for consistent cell sizing
    layout.setHorizontalSpacing(spacing)
    layout.setVerticalSpacing(spacing)
    
    # Apply default alignment for grid cells
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    return layout


def create_plot_layout(style: str = "default") -> QVBoxLayout:
    """
    Creates a specialized layout for plot displays with appropriate spacing for matplotlib figures.
    
    Args:
        style: Layout style ('default', 'compact', or 'spacious')
        
    Returns:
        QVBoxLayout: Configured vertical layout for plots
        
    Raises:
        ValueError: If style parameter is not recognized
    """
    # Validate style parameter
    if style not in LAYOUT_MARGINS:
        raise ValueError(f"Invalid style: {style}. Valid options are: {', '.join(LAYOUT_MARGINS.keys())}")
    
    # Create new vertical layout
    layout = QVBoxLayout()
    
    # Set plot-specific margins with extra space for toolbar
    margin = LAYOUT_MARGINS[style]
    layout.setContentsMargins(margin, margin, margin, margin + 5)
    
    # Configure spacing for plot elements
    layout.setSpacing(LAYOUT_SPACING[style])
    
    # Add placeholder for matplotlib canvas
    # The actual canvas should be added by the caller
    layout.addStretch()
    
    # Add toolbar container with proper alignment
    # The actual toolbar should be added by the caller
    toolbar_layout = QHBoxLayout()
    toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addLayout(toolbar_layout)
    
    return layout


def adjust_layout_for_theme(layout: QLayout, theme: str) -> None:
    """
    Adjusts layout properties based on current theme settings.
    
    Args:
        layout: The layout to adjust
        theme: The theme to apply ('light' or 'dark')
        
    Raises:
        ValueError: If theme is not 'light' or 'dark'
        TypeError: If layout is not a QLayout instance
    """
    # Validate layout and theme parameters
    if not isinstance(layout, QLayout):
        raise TypeError(f"Expected QLayout instance, got {type(layout).__name__}")
    
    if theme not in ["light", "dark"]:
        raise ValueError(f"Invalid theme: {theme}. Must be 'light' or 'dark'")
    
    # Get theme-specific spacing values
    spacing = LAYOUT_SPACING["default"]
    if theme == "dark":
        # Dark theme uses slightly increased spacing for better readability
        spacing += 2
    
    # Update layout margins for theme
    margins = layout.contentsMargins()
    
    # Adjust content margins based on theme
    if theme == "dark":
        layout.setContentsMargins(margins.left() + 2, margins.top() + 2, 
                                margins.right() + 2, margins.bottom() + 2)
    
    # Update spacing between elements
    if isinstance(layout, QFormLayout):
        # For form layouts
        layout.setSpacing(spacing)
        
        # Apply theme-specific alignments
        if theme == "dark":
            layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        else:
            layout.setLabelAlignment(FORM_LABEL_ALIGNMENT)
        
    elif isinstance(layout, QGridLayout):
        # For grid layouts
        layout.setHorizontalSpacing(spacing)
        layout.setVerticalSpacing(spacing)
        
    elif isinstance(layout, QVBoxLayout) or isinstance(layout, QHBoxLayout):
        # For box layouts
        layout.setSpacing(spacing)
    
    # Handle any theme-specific customizations
    # Note: Background colors should be handled by the widget's stylesheet or palette