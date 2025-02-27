"""
PyQt6 component implementing interactive diagnostic plots for statistical model analysis.

This module provides specialized widgets for visualizing model performance and diagnostic
statistics, including residual plots, autocorrelation function plots, and comprehensive
model diagnostics with async support and theme-aware styling.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np  # numpy version 1.26.3
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget  # PyQt6 version 6.6.1

# Internal imports
from components.plot_display import PlotDisplay
from utils.qt_helpers import create_widget
from utils.plot_utils import create_residual_plot, create_acf_plot


class DiagnosticPlotsWidget(QWidget):
    """
    Interactive widget for displaying and managing statistical diagnostic plots with async updates and theme support.
    
    This widget organizes multiple diagnostic plot types in a tabbed interface with support
    for asynchronous updates, theme-aware styling, and smooth transitions between views.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the diagnostic plots widget with tab container, plot displays, and theme support.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Set up logger
        self._logger = logging.getLogger(__name__)
        
        # Initialize data storage and plot cache
        self._current_data = {}
        self._plot_cache = {}
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create theme-aware tab widget container
        self._tab_widget = create_widget('QTabWidget', {
            'tabPosition': QTabWidget.TabPosition.North,
            'documentMode': True,
            'movable': False,
            'tabsClosable': False
        })
        
        # Initialize async-enabled plot display widgets
        self._residual_display = PlotDisplay(self)
        self._acf_display = PlotDisplay(self)
        self._diagnostic_display = PlotDisplay(self)
        
        # Add tabs for different plot types
        self._tab_widget.addTab(self._residual_display, "Residuals")
        self._tab_widget.addTab(self._acf_display, "Autocorrelation")
        self._tab_widget.addTab(self._diagnostic_display, "Diagnostics")
        
        # Add tab widget to main layout
        main_layout.addWidget(self._tab_widget)
        
        # Configure widget styling and accessibility
        self.setLayout(main_layout)
        self.setAccessibleName("Diagnostic Plots Widget")
        self.setAccessibleDescription("Displays statistical diagnostic plots for model analysis")
        
        # Set up error handling and logging
        self._logger.debug("DiagnosticPlotsWidget initialized")
    
    async def update_plots(self, residuals: np.ndarray, fitted_values: np.ndarray) -> None:
        """
        Asynchronously updates all diagnostic plots with new data.
        
        Args:
            residuals: Array of model residuals
            fitted_values: Array of fitted values from the model
        """
        try:
            # Validate input arrays
            if not isinstance(residuals, np.ndarray) or not isinstance(fitted_values, np.ndarray):
                raise TypeError("Residuals and fitted_values must be NumPy arrays")
            
            if residuals.shape != fitted_values.shape:
                raise ValueError(f"Shape mismatch: residuals {residuals.shape} vs fitted_values {fitted_values.shape}")
            
            # Store new data in current_data
            self._current_data = {
                "residuals": residuals,
                "fitted_values": fitted_values
            }
            
            # Update residual plot tab
            await self._residual_display.display_residual_plot(residuals, fitted_values)
            
            # Update ACF plot tab
            await self._acf_display.display_acf_plot(residuals)
            
            # Update comprehensive diagnostics tab
            await self._diagnostic_display.display_diagnostic_plots(residuals, fitted_values)
            
            # Update plot cache if needed
            self._plot_cache = {
                "residual": {"type": "residual", "data": {"residuals": residuals, "fitted_values": fitted_values}},
                "acf": {"type": "acf", "data": {"data": residuals}},
                "diagnostic": {"type": "diagnostic", "data": {"residuals": residuals, "fitted_values": fitted_values}}
            }
            
            # Refresh tab widget display
            self._tab_widget.update()
            
            # Log update completion
            self._logger.debug("Diagnostic plots updated successfully")
            
        except Exception as e:
            self._logger.error(f"Error updating diagnostic plots: {str(e)}")
            raise RuntimeError(f"Failed to update diagnostic plots: {str(e)}") from e
    
    def clear_plots(self) -> None:
        """
        Clears all diagnostic plots and cache.
        """
        try:
            # Clear plot displays asynchronously
            self._residual_display.clear_plots()
            self._acf_display.clear_plots()
            self._diagnostic_display.clear_plots()
            
            # Reset plot cache
            self._plot_cache = {}
            
            # Clear current data dictionary
            self._current_data = {}
            
            # Reset tab widget state
            self._tab_widget.setCurrentIndex(0)
            
            # Log clearing operation
            self._logger.debug("Diagnostic plots cleared")
            
        except Exception as e:
            self._logger.error(f"Error clearing diagnostic plots: {str(e)}")
    
    def switch_to_tab(self, tab_name: str) -> None:
        """
        Switches to specified diagnostic plot tab with smooth transition.
        
        Args:
            tab_name: Name of the tab to switch to ('Residuals', 'Autocorrelation', 'Diagnostics')
        """
        try:
            # Validate tab name
            tab_map = {
                "residuals": 0,
                "autocorrelation": 1,
                "diagnostics": 2
            }
            
            # Allow for case-insensitive tab names
            tab_key = tab_name.lower()
            
            if tab_key not in tab_map:
                raise ValueError(f"Invalid tab name: {tab_name}. Valid options are: Residuals, Autocorrelation, Diagnostics")
            
            # Get tab index
            tab_index = tab_map[tab_key]
            
            # Prepare transition animation
            # (animation is handled implicitly by Qt's tab switching mechanism)
            
            # Set current tab index
            self._tab_widget.setCurrentIndex(tab_index)
            
            # Refresh current tab display
            current_tab = self._tab_widget.currentWidget()
            if current_tab:
                current_tab.update()
            
            # Update tab widget focus
            self._tab_widget.setFocus()
            
            # Handle accessibility updates
            current_tab = self._tab_widget.currentWidget()
            if current_tab:
                self.setAccessibleDescription(f"Viewing {tab_name} diagnostic plot")
            
            # Log tab switch
            self._logger.debug(f"Switched to tab: {tab_name}")
            
        except Exception as e:
            self._logger.error(f"Error switching to tab {tab_name}: {str(e)}")