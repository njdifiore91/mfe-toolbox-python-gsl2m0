"""
Utility module providing helper functions for PyQt6 widget creation, styling and management.
Implements common patterns for GUI components while ensuring proper theme application and widget configuration.
"""

import logging
from typing import Dict, Any, Optional
import importlib
import weakref

from PyQt6.QtWidgets import QWidget, QApplication, QStyle  # PyQt6 version 6.6.1
from PyQt6.QtGui import QPalette  # PyQt6 version 6.6.1
from web.styles.theme import initialize_theme  # For initializing widget themes

# Global logger instance
logger = logging.getLogger(__name__)

def create_widget(widget_type: str, properties: Dict[str, Any]) -> QWidget:
    """
    Creates a themed PyQt6 widget with proper styling and configuration.
    
    Args:
        widget_type: String name of the widget class to create
        properties: Dictionary of properties to apply to the widget
        
    Returns:
        Configured widget instance
    
    Raises:
        ValueError: If widget_type is not a valid PyQt6 widget class
    """
    try:
        # Import the widget class dynamically from PyQt6.QtWidgets
        module = importlib.import_module('PyQt6.QtWidgets')
        widget_class = getattr(module, widget_type)
        
        # Create widget instance
        widget = widget_class()
        
        # Apply theme styling and properties
        configure_widget(widget, properties)
        
        return widget
    except (AttributeError, ImportError) as e:
        logger.error(f"Failed to create widget of type {widget_type}: {str(e)}")
        raise ValueError(f"Invalid widget type: {widget_type}")

def configure_widget(widget: QWidget, properties: Dict[str, Any]) -> None:
    """
    Configures an existing widget with theme and properties.
    
    Args:
        widget: The QWidget instance to configure
        properties: Dictionary of properties to apply to the widget
        
    Raises:
        TypeError: If widget is not a QWidget instance
    """
    if not isinstance(widget, QWidget):
        raise TypeError(f"Expected QWidget instance, got {type(widget).__name__}")
    
    # Apply current theme to widget
    app = QApplication.instance()
    if app:
        widget.setPalette(app.palette())
    
    # Update widget properties from dictionary
    for prop_name, prop_value in properties.items():
        try:
            if hasattr(widget, f"set{prop_name[0].upper()}{prop_name[1:]}"):
                # Use setter method if available (e.g., setText for 'text')
                setter = getattr(widget, f"set{prop_name[0].upper()}{prop_name[1:]}")
                setter(prop_value)
            else:
                # Otherwise set property directly
                widget.setProperty(prop_name, prop_value)
        except Exception as e:
            logger.warning(f"Failed to set property '{prop_name}' on {type(widget).__name__}: {str(e)}")
    
    # Refresh widget style
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    
    # Update widget geometry if needed
    if 'geometry' in properties or 'size' in properties:
        widget.updateGeometry()

def get_themed_palette() -> QPalette:
    """
    Creates a QPalette configured for the current theme.
    
    Returns:
        Theme-configured color palette
    """
    # Get current application theme
    app = QApplication.instance()
    if not app:
        logger.warning("No QApplication instance found. Creating default palette.")
        return QPalette()
    
    # Create new palette instance based on application palette
    palette = QPalette(app.palette())
    
    # Set up disabled and inactive states
    # Disabled state colors
    disabled_color = palette.color(QPalette.ColorRole.WindowText).lighter(150)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_color)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color)
    
    # Inactive state colors - often same as active for most controls
    palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight, 
                     palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight).lighter(110))
    
    return palette

class WidgetFactory:
    """
    Factory class for creating themed PyQt6 widgets with consistent styling.
    
    This factory maintains a cache of created widgets and provides methods to
    update their themes when the application theme changes.
    """
    
    def __init__(self):
        """
        Initializes the widget factory with theme configuration.
        """
        # Initialize widget cache using weak references to avoid memory leaks
        self._widget_cache = weakref.WeakValueDictionary()
        
        # Create theme palette
        self._theme_palette = get_themed_palette()
        
        # Set up logging
        self._logger = logging.getLogger(__name__ + ".WidgetFactory")
        
        # Configure default properties for common widgets
        self._default_properties = {
            'QLabel': {'alignment': 4},  # Qt.AlignmentFlag.AlignLeft
            'QPushButton': {'autoDefault': False},
            'QLineEdit': {'clearButtonEnabled': True}
        }
    
    def create(self, widget_type: str, properties: Dict[str, Any]) -> QWidget:
        """
        Creates a new themed widget instance.
        
        Args:
            widget_type: String name of the widget class to create
            properties: Dictionary of properties to apply to the widget
            
        Returns:
            New widget instance
            
        Raises:
            ValueError: If widget_type is not a valid PyQt6 widget class
        """
        try:
            # Merge with default properties for this widget type if available
            merged_properties = {}
            if widget_type in self._default_properties:
                merged_properties.update(self._default_properties[widget_type])
            merged_properties.update(properties)
            
            # Create widget instance
            widget = create_widget(widget_type, merged_properties)
            
            # Generate a unique identifier for caching
            cache_key = f"{widget_type}_{id(widget)}"
            
            # Cache widget
            self._widget_cache[cache_key] = widget
            
            return widget
        except ValueError as e:
            self._logger.error(f"Widget creation failed: {str(e)}")
            raise
    
    def update_theme(self) -> None:
        """
        Updates theme for all cached widgets.
        """
        # Get new theme palette
        new_palette = get_themed_palette()
        self._theme_palette = new_palette
        
        # Update cached widgets with new theme
        # Using weakref dictionary ensures we don't have references to destroyed widgets
        for widget_id, widget in list(self._widget_cache.items()):
            try:
                widget.setPalette(new_palette)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
            except Exception as e:
                self._logger.debug(f"Could not update widget {widget_id}: {str(e)}")