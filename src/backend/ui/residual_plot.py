"""
PyQt6-based widget for displaying and updating residual diagnostic plots using Matplotlib backend.

This module provides interactive visualization of model residuals, ACF/PACF plots, and
other diagnostic visualizations. It integrates with the ARMAX model to display
comprehensive model diagnostics and residual analysis.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any
import asyncio
import matplotlib.pyplot as plt  # version: 3.8.2

from PyQt6.QtWidgets import QWidget, QVBoxLayout  # version: 6.6.1
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QSize, QResizeEvent, QCloseEvent  # version: 6.6.1
from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas  # version: 3.8.2
from matplotlib.backends.backend_qt6agg import NavigationToolbar2QT as NavigationToolbar  # version: 3.8.2
from matplotlib.figure import Figure  # version: 3.8.2
import statsmodels.graphics.tsaplots as tsaplots  # version: 0.14.1

from ..models.armax import ARMAX

# Configure logger
logger = logging.getLogger(__name__)

# Global constants
PLOT_DPI = 100
DEFAULT_FIGSIZE = (8, 6)
SUBPLOT_GRID = (2, 2)


class ResidualPlotWidget(QWidget):
    """
    Widget for displaying and updating residual diagnostic plots.
    
    This widget provides interactive visualization of model residuals, 
    ACF/PACF plots, and other diagnostic visualizations. It uses Matplotlib
    as a backend for high-quality plotting and integrates with ARMAX models
    for comprehensive diagnostics.
    
    Attributes
    ----------
    plot_updated : pyqtSignal
        Signal emitted when plots are updated
    """
    plot_updated = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the residual plot widget with Matplotlib canvas.
        
        Parameters
        ----------
        parent : Optional[QWidget]
            Parent widget, if any
        """
        super().__init__(parent)
        
        # Create Matplotlib figure and canvas
        self._figure = Figure(figsize=DEFAULT_FIGSIZE, dpi=PLOT_DPI)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setParent(self)
        
        # Create Matplotlib toolbar
        self._toolbar = NavigationToolbar(self._canvas, self)
        
        # Create layout and add components
        self._layout = QVBoxLayout()
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        self.setLayout(self._layout)
        
        # Initialize plot axes
        self._axes = []
        nrows, ncols = SUBPLOT_GRID
        for i in range(nrows * ncols):
            self._axes.append(self._figure.add_subplot(nrows, ncols, i + 1))
        
        # Set tight layout
        self._figure.tight_layout()
        
        logger.debug("ResidualPlotWidget initialized")
    
    def update_plots(self, residuals: np.ndarray, diagnostic_results: Dict[str, Any]) -> None:
        """
        Updates all residual diagnostic plots with new data.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals from ARMAX model
        diagnostic_results : Dict[str, Any]
            Dictionary of diagnostic test results
            
        Returns
        -------
        None
        """
        try:
            # Clear existing plots
            for ax in self._axes:
                ax.clear()
            
            # Get axes for specific plots
            residual_ax = self._axes[0]
            acf_ax = self._axes[1]
            pacf_ax = self._axes[2]
            hist_ax = self._axes[3]
            
            # Plot 1: Residual time series
            residual_ax.plot(residuals, 'b-', alpha=0.7)
            residual_ax.axhline(y=0, color='r', linestyle='-', linewidth=0.8)
            residual_ax.set_title('Residuals')
            residual_ax.set_xlabel('Observation')
            residual_ax.set_ylabel('Residual')
            
            # Plot 2: ACF of residuals
            tsaplots.plot_acf(residuals, ax=acf_ax, lags=20, alpha=0.05)
            acf_ax.set_title('Autocorrelation Function')
            
            # Plot 3: PACF of residuals
            tsaplots.plot_pacf(residuals, ax=pacf_ax, lags=20, alpha=0.05)
            pacf_ax.set_title('Partial Autocorrelation Function')
            
            # Plot 4: Residual histogram with normal distribution overlay
            hist_ax.hist(residuals, bins=20, density=True, alpha=0.7, color='b')
            
            # Add normal distribution overlay if residual stats are available
            if 'residual_stats' in diagnostic_results:
                # Extract mean and std dev
                mean = diagnostic_results['residual_stats']['mean']
                std = diagnostic_results['residual_stats']['std_dev']
                
                # Generate points for normal distribution curve
                x = np.linspace(min(residuals), max(residuals), 100)
                y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
                
                # Plot normal distribution
                hist_ax.plot(x, y, 'r-', linewidth=2)
                
                # Add Jarque-Bera test results if available
                if 'jarque_bera' in diagnostic_results:
                    jb_stat = diagnostic_results['jarque_bera']['statistic']
                    jb_pval = diagnostic_results['jarque_bera']['p_value']
                    hist_ax.text(0.05, 0.95, f"Jarque-Bera: {jb_stat:.2f}\np-value: {jb_pval:.4f}",
                                transform=hist_ax.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            hist_ax.set_title('Residual Distribution')
            hist_ax.set_xlabel('Residual')
            hist_ax.set_ylabel('Density')
            
            # Update layout
            self._figure.tight_layout()
            self._canvas.draw()
            
            # Emit signal that plots have been updated
            self.plot_updated.emit()
            
            logger.debug("Residual plots updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating residual plots: {str(e)}")
            # Re-raise exception to notify caller
            raise
    
    def clear_plots(self) -> None:
        """
        Clears all plots and resets the canvas.
        
        Returns
        -------
        None
        """
        try:
            # Clear all axes
            for ax in self._axes:
                ax.clear()
            
            # Reset titles
            titles = ['Residuals', 'Autocorrelation Function', 
                      'Partial Autocorrelation Function', 'Residual Distribution']
            for ax, title in zip(self._axes, titles):
                ax.set_title(title)
            
            # Update canvas
            self._figure.tight_layout()
            self._canvas.draw()
            
            logger.debug("Residual plots cleared")
            
        except Exception as e:
            logger.error(f"Error clearing plots: {str(e)}")
    
    @pyqtSlot()
    async def async_update_plots(self, model: ARMAX) -> None:
        """
        Asynchronously updates plots to maintain UI responsiveness.
        
        This method retrieves residuals and diagnostic information from the model
        and updates the plots without blocking the UI. It's designed to be used
        with Qt's event loop for asynchronous operations.
        
        Parameters
        ----------
        model : ARMAX
            Fitted ARMAX model containing residuals and diagnostic information
            
        Returns
        -------
        None
        """
        try:
            # Check if model has been fit
            if model.residuals is None:
                logger.warning("Cannot update plots: Model has not been fit")
                return
            
            # Get residuals
            residuals = model.residuals
            
            # Compute diagnostic tests
            loop = asyncio.get_event_loop()
            diagnostic_results = await loop.run_in_executor(None, model.diagnostic_tests)
            
            # Update plots with the new data
            self.update_plots(residuals, diagnostic_results)
            
            logger.debug("Asynchronous plot update completed")
            
        except Exception as e:
            logger.error(f"Error in async_update_plots: {str(e)}")
            # Don't re-raise here to avoid crashing the UI
    
    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Handles widget resize events.
        
        Ensures that the figure and canvas are properly resized when
        the widget is resized, maintaining the plot quality.
        
        Parameters
        ----------
        event : QResizeEvent
            Resize event information
            
        Returns
        -------
        None
        """
        super().resizeEvent(event)
        
        try:
            # Get new size
            size = self.size()
            width = size.width()
            height = size.height()
            
            # Adjust figure size (accounting for toolbar and margins)
            # The factor 0.8 is an approximation, might need adjustment
            width_inches = width / PLOT_DPI * 0.8
            height_inches = height / PLOT_DPI * 0.8
            
            # Set new figure size
            self._figure.set_size_inches(width_inches, height_inches)
            
            # Update layout
            self._figure.tight_layout()
            self._canvas.draw()
            
        except Exception as e:
            logger.error(f"Error during resize: {str(e)}")
    
    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handles widget close events and cleanup.
        
        Ensures proper cleanup of Matplotlib resources when the widget
        is closed to prevent memory leaks.
        
        Parameters
        ----------
        event : QCloseEvent
            Close event information
            
        Returns
        -------
        None
        """
        try:
            # Clear plot resources
            for ax in self._axes:
                ax.clear()
            
            # Close figure to free memory
            if self._figure is not None:
                plt.close(self._figure)
            
            # Release canvas
            self._canvas = None
            self._figure = None
            
            logger.debug("ResidualPlotWidget resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            # Accept the close event
            event.accept()