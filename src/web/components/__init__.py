"""
Package initialization file for the MFE Toolbox web components module that exports
PyQt6-based GUI components for time series analysis and visualization.
"""

# All components use PyQt6 version 6.6.1 for GUI framework

# Import GUI components to make them available at package level
from .main_window import MainWindow
from .navigation import NavigationWidget
from .parameter_input import ParameterInput
from .plot_display import PlotDisplay
from .model_config import ModelConfig
from .diagnostic_plots import DiagnosticPlots
from .statistical_tests import StatisticalTests
from .results_viewer import ResultsViewer

# Define public API
__all__ = [
    'MainWindow',
    'NavigationWidget',
    'ParameterInput',
    'PlotDisplay',
    'ModelConfig',
    'DiagnosticPlots',
    'StatisticalTests',
    'ResultsViewer'
]