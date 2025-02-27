"""
Package initializer for the web/utils module that exposes key utility functions 
and classes for PyQt6-based GUI components, asynchronous operations, plotting 
utilities, and input validation. Provides a clean public API for the utilities 
package while maintaining proper encapsulation.

This module serves as the main entry point for all utility functions used by 
the PyQt6-based GUI components of the MFE Toolbox. It exposes an organized 
set of functions and classes that handle:

- Asynchronous operations through run_async and AsyncRunner
- Widget creation and management via create_widget and WidgetFactory
- Plot generation with create_residual_plot and create_diagnostic_plots
- Input validation using validate_widget_input

By importing from this module, consumers get a consistent, well-typed, and
documented API without needing to know the internal package structure.
"""

import logging

# Set up globals
logger = logging.getLogger(__name__)
__version__: str = '4.0.0'
__all__: list[str] = [
    'run_async', 
    'AsyncRunner',
    'create_widget', 
    'WidgetFactory',
    'create_residual_plot', 
    'create_diagnostic_plots',
    'validate_widget_input'
]

# Import internal modules
from .async_helpers import run_async, AsyncRunner
from .qt_helpers import create_widget, WidgetFactory
from .plot_utils import create_residual_plot, create_diagnostic_plots
from .validation import validate_widget_input