"""
Plot display component for statistical visualizations and diagnostic plots.

This module provides a PyQt6-based widget for displaying interactive plots
and diagnostic visualizations using Matplotlib backend. Supports theme-aware
styling and asynchronous updates for responsive user interface.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np  # numpy version 1.26.3
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy  # PyQt6 version 6.6.1
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT  # matplotlib version 3.8.2
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Internal imports
from utils.plot_utils import create_residual_plot, create_acf_plot, create_diagnostic_plots
from utils.qt_helpers import create_widget


class PlotDisplay(QWidget):
    """
    Interactive plot display widget for statistical visualizations with theme awareness.
    
    Provides a container for Matplotlib plots with theme-aware styling, navigation toolbar,
    and support for asynchronous updates. Used for displaying model diagnostics, residual
    analysis, and time series visualization.
    """
    
    # Signal emitted when a plot is updated
    plot_updated = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None, theme_settings: Optional[Dict[str, Any]] = None):
        """
        Initializes the plot display widget with container, canvas and theme support.
        
        Args:
            parent: Parent widget
            theme_settings: Theme configuration dictionary for styling plots
        """
        super().__init__(parent)
        
        # Set up logger
        self._logger = logging.getLogger(__name__)
        
        # Initialize storage for plots and cache
        self._current_plots = {}
        self._plot_cache = {}
        self._theme_settings = theme_settings or {}
        
        # Create the main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        
        # Create the plot container widget
        properties = {
            'sizePolicy': QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        }
        self._plot_container = create_widget('QWidget', properties)
        self._container_layout = QVBoxLayout(self._plot_container)
        
        # Create initial matplotlib figure and canvas
        self._figure = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set up the toolbar
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        
        # Add widgets to layouts
        self._container_layout.addWidget(self._canvas)
        self._container_layout.addWidget(self._toolbar)
        self._main_layout.addWidget(self._plot_container)
        
        # Apply theme settings
        self._apply_theme_settings()
    
    async def display_residual_plot(self, residuals: np.ndarray, 
                                   fitted_values: np.ndarray) -> None:
        """
        Displays residual analysis plot with async updates.
        
        Shows the relationship between model residuals and fitted values
        to help diagnose heteroscedasticity and other anomalies.
        
        Args:
            residuals: Array of model residuals
            fitted_values: Array of fitted values from the model
        """
        try:
            self._logger.debug("Creating residual plot")
            
            # Clear previous plots
            self._clear_container()
            
            # Create the residual plot widget asynchronously
            residual_widget = await create_residual_plot(residuals, fitted_values)
            
            # Get the figure and canvas from the widget
            figure = getattr(residual_widget, '_figure', None)
            canvas = getattr(residual_widget, '_canvas', None)
            
            if figure is None or canvas is None:
                raise ValueError("Failed to create residual plot: missing figure or canvas")
            
            # Update our container with the new plot
            self._figure = figure
            self._canvas = canvas
            
            # Update the navigation toolbar
            if hasattr(self, '_toolbar'):
                self._toolbar.canvas = self._canvas
            
            # Rebuild the layout with the new widgets
            self._rebuild_layout()
            
            # Cache the plot data
            self._current_plots["residual"] = {
                "type": "residual",
                "data": {
                    "residuals": residuals,
                    "fitted_values": fitted_values
                }
            }
            
            # Emit signal that plot has been updated
            self.plot_updated.emit("residual")
            
        except Exception as e:
            self._logger.error(f"Error displaying residual plot: {str(e)}")
            raise RuntimeError(f"Failed to display residual plot: {str(e)}") from e
    
    async def display_acf_plot(self, data: np.ndarray, lags: int = 20) -> None:
        """
        Displays autocorrelation function plot with async updates.
        
        Shows the correlation between time series observations
        separated by different time lags to help identify patterns.
        
        Args:
            data: Time series data array
            lags: Number of lags to include in ACF
        """
        try:
            self._logger.debug(f"Creating ACF plot with {lags} lags")
            
            # Clear previous plots
            self._clear_container()
            
            # Create the ACF plot widget asynchronously
            acf_widget = await create_acf_plot(data, lags)
            
            # Get the figure and canvas from the widget
            figure = getattr(acf_widget, '_figure', None)
            canvas = getattr(acf_widget, '_canvas', None)
            
            if figure is None or canvas is None:
                raise ValueError("Failed to create ACF plot: missing figure or canvas")
            
            # Update our container with the new plot
            self._figure = figure
            self._canvas = canvas
            
            # Update the navigation toolbar
            if hasattr(self, '_toolbar'):
                self._toolbar.canvas = self._canvas
            
            # Rebuild the layout with the new widgets
            self._rebuild_layout()
            
            # Cache the plot data
            self._current_plots["acf"] = {
                "type": "acf",
                "data": {
                    "data": data,
                    "lags": lags
                }
            }
            
            # Emit signal that plot has been updated
            self.plot_updated.emit("acf")
            
        except Exception as e:
            self._logger.error(f"Error displaying ACF plot: {str(e)}")
            raise RuntimeError(f"Failed to display ACF plot: {str(e)}") from e
    
    async def display_diagnostic_plots(self, residuals: np.ndarray, 
                                      fitted_values: np.ndarray) -> None:
        """
        Displays comprehensive set of diagnostic plots with async updates.
        
        Creates a grid of diagnostic visualizations including residual analysis,
        QQ plot, histogram, and time series plots to evaluate model fit.
        
        Args:
            residuals: Array of model residuals
            fitted_values: Array of fitted values from the model
        """
        try:
            self._logger.debug("Creating comprehensive diagnostic plots")
            
            # Clear previous plots
            self._clear_container()
            
            # Create diagnostic plots asynchronously
            plot_widgets = await create_diagnostic_plots(residuals, fitted_values)
            
            if not plot_widgets or not isinstance(plot_widgets, dict):
                raise ValueError("Failed to create diagnostic plots")
            
            # Create a grid layout for the plots
            grid_layout = QGridLayout()
            grid_layout.setContentsMargins(5, 5, 5, 5)
            grid_layout.setSpacing(10)
            
            # Place the plots in a 2x2 grid
            if 'residual' in plot_widgets:
                grid_layout.addWidget(plot_widgets['residual'], 0, 0)
            
            if 'acf' in plot_widgets:
                grid_layout.addWidget(plot_widgets['acf'], 0, 1)
            
            if 'qqplot' in plot_widgets:
                grid_layout.addWidget(plot_widgets['qqplot'], 1, 0)
            
            if 'histogram' in plot_widgets:
                grid_layout.addWidget(plot_widgets['histogram'], 1, 1)
            
            if 'timeseries' in plot_widgets:
                # If we have 5 plots, add the time series plot in a row by itself
                grid_layout.addWidget(plot_widgets['timeseries'], 2, 0, 1, 2)
            
            # Replace our container layout with the grid layout
            self._clear_container()
            self._container_layout = grid_layout
            self._plot_container.setLayout(self._container_layout)
            
            # Cache the plot data
            self._current_plots = {
                "diagnostic": {
                    "type": "diagnostic",
                    "data": {
                        "residuals": residuals,
                        "fitted_values": fitted_values
                    },
                    "widgets": plot_widgets
                }
            }
            
            # Emit signal that plots have been updated
            self.plot_updated.emit("diagnostic")
            
        except Exception as e:
            self._logger.error(f"Error displaying diagnostic plots: {str(e)}")
            raise RuntimeError(f"Failed to display diagnostic plots: {str(e)}") from e
    
    def clear_plots(self) -> None:
        """
        Clears all current plots and caches.
        
        Removes all plots from the display and resets internal state.
        """
        try:
            self._logger.debug("Clearing all plots")
            
            # Clear the container
            self._clear_container()
            
            # Reset plot tracking and caches
            self._current_plots = {}
            self._plot_cache = {}
            
            # Re-create empty figure
            self._figure = Figure(figsize=(8, 6), dpi=100)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            # Update the toolbar's canvas reference
            if hasattr(self, '_toolbar'):
                self._toolbar.canvas = self._canvas
            
            # Rebuild the layout with the empty figure
            self._rebuild_layout()
            
            # Emit signal that plots have been cleared
            self.plot_updated.emit("cleared")
            
        except Exception as e:
            self._logger.error(f"Error clearing plots: {str(e)}")
    
    def update_theme(self, theme_settings: Dict[str, Any]) -> None:
        """
        Updates plot theme and styling.
        
        Applies new theme settings to existing plots and updates the
        display to reflect the current theme.
        
        Args:
            theme_settings: Theme configuration dictionary
        """
        try:
            self._logger.debug("Updating plot theme")
            
            # Update theme settings
            self._theme_settings = theme_settings
            
            # Apply theme to container and widgets
            self._apply_theme_settings()
            
            # Redraw current plots with new theme
            self._redraw_current_plots()
            
        except Exception as e:
            self._logger.error(f"Error updating theme: {str(e)}")
    
    def _clear_container(self) -> None:
        """
        Clears the plot container efficiently.
        
        Internal method to remove existing plots and prepare for new content.
        """
        # Remove all widgets from the layout
        if hasattr(self, '_container_layout'):
            while self._container_layout.count():
                item = self._container_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
    
    def _rebuild_layout(self) -> None:
        """
        Rebuilds the layout with current canvas and toolbar.
        
        Internal method to update the container layout with the current plot widgets.
        """
        # Clear existing layout
        self._clear_container()
        
        # Add canvas and toolbar back to layout
        self._container_layout.addWidget(self._canvas)
        if hasattr(self, '_toolbar'):
            self._container_layout.addWidget(self._toolbar)
        
        # Force layout update
        self._plot_container.setLayout(self._container_layout)
        self.updateGeometry()
    
    def _apply_theme_settings(self) -> None:
        """
        Applies current theme settings to the plot display.
        
        Internal method to update styling based on theme configuration.
        """
        # Get theme name from settings
        theme_name = self._theme_settings.get('name', 'default')
        
        # Apply appropriate styling to the container
        if theme_name == 'dark':
            self._plot_container.setStyleSheet("""
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #404040;
            """)
        else:
            self._plot_container.setStyleSheet("""
                background-color: #FFFFFF;
                color: #333333;
                border: 1px solid #E5E5E5;
            """)
        
        # Update the figure background
        if hasattr(self, '_figure'):
            if theme_name == 'dark':
                self._figure.patch.set_facecolor('#2D2D2D')
                for ax in self._figure.get_axes():
                    ax.set_facecolor('#3D3D3D')
                    ax.tick_params(colors='#E0E0E0')
                    for spine in ax.spines.values():
                        spine.set_color('#E0E0E0')
            else:
                self._figure.patch.set_facecolor('#FFFFFF')
                for ax in self._figure.get_axes():
                    ax.set_facecolor('#F9F9F9')
                    ax.tick_params(colors='#333333')
                    for spine in ax.spines.values():
                        spine.set_color('#333333')
            
            # Update the canvas
            if hasattr(self, '_canvas'):
                self._canvas.draw_idle()
    
    def _redraw_current_plots(self) -> None:
        """
        Redraws all current plots with updated theme.
        
        Internal method to refresh plot display after theme changes.
        """
        # Check if we have any plots to redraw
        if not self._current_plots:
            return
        
        # Redraw based on current plot type
        if "residual" in self._current_plots:
            data = self._current_plots["residual"]["data"]
            asyncio.create_task(
                self.display_residual_plot(
                    data["residuals"], 
                    data["fitted_values"]
                )
            )
        
        elif "acf" in self._current_plots:
            data = self._current_plots["acf"]["data"]
            asyncio.create_task(
                self.display_acf_plot(
                    data["data"], 
                    data["lags"]
                )
            )
        
        elif "diagnostic" in self._current_plots:
            data = self._current_plots["diagnostic"]["data"]
            asyncio.create_task(
                self.display_diagnostic_plots(
                    data["residuals"], 
                    data["fitted_values"]
                )
            )