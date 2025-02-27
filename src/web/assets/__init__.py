"""
Web UI assets package for the MFE Toolbox.

This module provides access to PyQt6 GUI resources including stylesheets, icons, 
and theme configurations. It offers utility functions to load theme-specific 
stylesheets and access icon resources.
"""

import os
import logging
from pathlib import Path
from PyQt6.QtCore import QResource  # PyQt6 version 6.6.1

# Constants for resource paths
ICON_PATH = 'src/web/assets/icons/'
STYLE_PATH = 'src/web/assets/styles/'

# Initialize logger
logger = logging.getLogger(__name__)

# Import stylesheets
try:
    with open(os.path.join(STYLE_PATH, 'dark.qss'), 'r') as f:
        dark_theme_stylesheet = f.read()
        
    with open(os.path.join(STYLE_PATH, 'light.qss'), 'r') as f:
        light_theme_stylesheet = f.read()
        
    with open(os.path.join(STYLE_PATH, 'main.qss'), 'r') as f:
        main_stylesheet = f.read()
except FileNotFoundError as e:
    logger.error(f"Error loading stylesheets: {e}")
    # Provide empty defaults to prevent runtime errors
    dark_theme_stylesheet = ""
    light_theme_stylesheet = ""
    main_stylesheet = ""


def load_stylesheet(theme_name: str) -> str:
    """
    Loads and returns the appropriate QSS stylesheet based on theme selection.
    
    Args:
        theme_name: The theme to load ('light' or 'dark')
        
    Returns:
        The complete QSS stylesheet string for the selected theme
        
    Raises:
        ValueError: If the theme_name is not 'light' or 'dark'
    """
    # Validate theme name input
    if theme_name not in ['light', 'dark']:
        raise ValueError(f"Invalid theme: {theme_name}. Must be 'light' or 'dark'")
    
    # Load base styles from main stylesheet
    stylesheet = main_stylesheet
    
    # Load theme-specific styles based on theme_name
    if theme_name == 'light':
        theme_stylesheet = light_theme_stylesheet
    else:  # theme_name == 'dark'
        theme_stylesheet = dark_theme_stylesheet
    
    # Combine and return complete stylesheet
    return f"{stylesheet}\n{theme_stylesheet}"


def get_icon_path(icon_name: str) -> str:
    """
    Returns the full path to an icon resource.
    
    Args:
        icon_name: The filename of the icon (with extension)
        
    Returns:
        Full path to the requested icon file
        
    Raises:
        ValueError: If the icon name is empty or invalid
        FileNotFoundError: If the icon file does not exist
    """
    # Validate icon name input
    if not icon_name or not isinstance(icon_name, str):
        raise ValueError("Icon name must be a non-empty string")
    
    # Combine ICON_PATH with icon name
    icon_path = os.path.join(ICON_PATH, icon_name)
    
    # Verify icon file exists
    if not os.path.exists(icon_path):
        raise FileNotFoundError(f"Icon file not found: {icon_path}")
    
    # Return complete icon path
    return icon_path