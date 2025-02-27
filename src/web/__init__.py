"""
Package initialization file for the MFE Toolbox web UI module that provides a PyQt6-based graphical interface for time series analysis and visualization.
"""

# External imports
import logging  # Python 3.12.0 - Error logging
import asyncio  # Python 3.12.0 - Asynchronous I/O support
import PyQt6  # PyQt6 version 6.6.1 - Qt GUI framework for Python

# Internal imports
from .components.main_window import MainWindow  # Main application window implementation
from .components.navigation import NavigationWidget  # Navigation controls for plot and results navigation

# Set up global logger
logger = logging.getLogger(__name__)

# Package metadata
__version__ = '4.0.0'

# Define what is available via `from mfe.web import *`
__all__ = ['MainWindow', 'NavigationWidget']