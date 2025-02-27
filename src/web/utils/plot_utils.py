"""
Utility module providing helper functions for creating and managing statistical plots and visualizations.

This module provides functions for creating interactive diagnostic plots, residual analysis,
and time series visualization using Matplotlib with PyQt6 integration, with support for
asynchronous updates.
"""

import asyncio
import logging
from typing import Dict, Optional, Tuple, Union, List

import numpy as np  # numpy version 1.26.3
import matplotlib  # matplotlib version 3.8.2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from scipy import stats  # scipy version 1.11.4
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy  # PyQt6 version 6.6.1
from PyQt6.QtCore import pyqtSignal, Qt

# Internal imports
from web.utils.qt_helpers import create_widget

# Set up logger
logger = logging.getLogger(__name__)

async def create_residual_plot(residuals: np.ndarray, fitted_values: np.ndarray, 
                              parent: Optional[QWidget] = None) -> QWidget:
    """
    Creates an interactive residual diagnostic plot showing model residuals against fitted values
    with zoom/pan capabilities.
    
    Args:
        residuals: Array of model residuals
        fitted_values: Array of fitted values from the model
        parent: Optional parent widget
        
    Returns:
        QWidget containing the interactive residual plot
    """
    # Create container widget
    properties = {
        'sizePolicy': QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    }
    
    widget = create_widget('QWidget', properties)
    layout = QVBoxLayout(widget)
    
    # Create figure and canvas
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Set up the plot
    ax.scatter(fitted_values, residuals, alpha=0.7, marker='o', color='navy')
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted Values')
    
    # Configure grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    configure_plot_style(fig, 'default')
    
    # Add toolbar for interactivity
    toolbar = NavigationToolbar(canvas, widget)
    
    # Add to layout
    layout.addWidget(canvas)
    layout.addWidget(toolbar)
    
    # Tight layout for better appearance
    fig.tight_layout()
    canvas.draw()
    
    # Tag this plot for update_plot_async
    ax._plot_type = 'residual'
    widget._plot_type = 'residual'
    widget._figure = fig
    widget._canvas = canvas
    
    return widget

async def create_acf_plot(data: np.ndarray, lags: int = 20, 
                         parent: Optional[QWidget] = None) -> QWidget:
    """
    Creates interactive autocorrelation function plot for residual analysis with confidence intervals.
    
    Args:
        data: Time series data array
        lags: Number of lags to include in ACF
        parent: Optional parent widget
        
    Returns:
        QWidget containing the interactive ACF plot
    """
    # Create container widget
    properties = {
        'sizePolicy': QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    }
    
    widget = create_widget('QWidget', properties)
    layout = QVBoxLayout(widget)
    
    # Create figure and canvas
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Calculate ACF asynchronously
    acf_values = np.zeros(lags + 1)
    for i in range(lags + 1):
        # Simple ACF calculation for demonstration
        # For a full implementation, use statsmodels.tsa.stattools.acf
        if i == 0:
            acf_values[i] = 1.0  # Lag 0 is always 1.0
        else:
            # Calculate autocorrelation at lag i
            n = len(data)
            slice1 = data[:n-i]
            slice2 = data[i:n]
            acf_values[i] = np.corrcoef(slice1, slice2)[0, 1]
    
    # Calculate confidence intervals (approximate method)
    confidence = 1.96 / np.sqrt(len(data))  # 95% confidence interval
    
    # Plot ACF values
    ax.bar(range(lags + 1), acf_values, width=0.3, color='navy', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-')
    ax.axhline(y=confidence, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-confidence, color='red', linestyle='--', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.set_title('Autocorrelation Function')
    
    # Configure grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    configure_plot_style(fig, 'default')
    
    # Add toolbar for interactivity
    toolbar = NavigationToolbar(canvas, widget)
    
    # Add to layout
    layout.addWidget(canvas)
    layout.addWidget(toolbar)
    
    # Tight layout for better appearance
    fig.tight_layout()
    canvas.draw()
    
    # Tag this plot for update_plot_async
    ax._plot_type = 'acf'
    widget._plot_type = 'acf'
    widget._figure = fig
    widget._canvas = canvas
    widget._data = data
    widget._lags = lags
    
    return widget

async def create_diagnostic_plots(residuals: np.ndarray, fitted_values: np.ndarray, 
                                 parent: Optional[QWidget] = None) -> Dict[str, QWidget]:
    """
    Creates a comprehensive set of interactive diagnostic plots for model evaluation.
    
    Args:
        residuals: Array of model residuals
        fitted_values: Array of fitted values from the model
        parent: Optional parent widget
        
    Returns:
        Dictionary of named diagnostic plot widgets
    """
    # Initialize dictionary to store plot widgets
    plots = {}
    
    # Create residual plot
    plots['residual'] = await create_residual_plot(residuals, fitted_values, parent)
    
    # Create ACF plot
    plots['acf'] = await create_acf_plot(residuals, 20, parent)
    
    # Create QQ plot for normality check
    properties = {
        'sizePolicy': QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    }
    
    qq_widget = create_widget('QWidget', properties)
    layout = QVBoxLayout(qq_widget)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Create QQ plot
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Residuals')
    
    # Configure grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    configure_plot_style(fig, 'default')
    
    # Add toolbar for interactivity
    toolbar = NavigationToolbar(canvas, qq_widget)
    
    # Add to layout
    layout.addWidget(canvas)
    layout.addWidget(toolbar)
    
    # Tight layout for better appearance
    fig.tight_layout()
    canvas.draw()
    
    # Tag this plot for update_plot_async
    ax._plot_type = 'qqplot'
    qq_widget._plot_type = 'qqplot'
    qq_widget._figure = fig
    qq_widget._canvas = canvas
    
    plots['qqplot'] = qq_widget
    
    # Create histogram with density plot
    hist_widget = create_widget('QWidget', properties)
    layout = QVBoxLayout(hist_widget)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Create histogram
    _, bins, _ = ax.hist(residuals, bins=30, density=True, alpha=0.7, color='navy')
    
    # Add a density curve
    kde_xs = np.linspace(min(residuals), max(residuals), 200)
    kde = stats.gaussian_kde(residuals)
    ax.plot(kde_xs, kde(kde_xs), 'r-', label='Density')
    
    # Add normal distribution curve
    mu, sigma = stats.norm.fit(residuals)
    norm_pdf = stats.norm.pdf(kde_xs, mu, sigma)
    ax.plot(kde_xs, norm_pdf, 'g--', label='Normal')
    
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Histogram of Residuals')
    ax.legend()
    
    # Configure grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    configure_plot_style(fig, 'default')
    
    # Add toolbar for interactivity
    toolbar = NavigationToolbar(canvas, hist_widget)
    
    # Add to layout
    layout.addWidget(canvas)
    layout.addWidget(toolbar)
    
    # Tight layout for better appearance
    fig.tight_layout()
    canvas.draw()
    
    # Tag this plot for update_plot_async
    ax._plot_type = 'histogram'
    hist_widget._plot_type = 'histogram'
    hist_widget._figure = fig
    hist_widget._canvas = canvas
    
    plots['histogram'] = hist_widget
    
    # Time series plot of residuals
    ts_widget = create_widget('QWidget', properties)
    layout = QVBoxLayout(ts_widget)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Create time series plot
    ax.plot(residuals, marker='o', markersize=3, linestyle='-', color='navy', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    ax.set_xlabel('Observation')
    ax.set_ylabel('Residuals')
    ax.set_title('Time Series of Residuals')
    
    # Configure grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    configure_plot_style(fig, 'default')
    
    # Add toolbar for interactivity
    toolbar = NavigationToolbar(canvas, ts_widget)
    
    # Add to layout
    layout.addWidget(canvas)
    layout.addWidget(toolbar)
    
    # Tight layout for better appearance
    fig.tight_layout()
    canvas.draw()
    
    # Tag this plot for update_plot_async
    ax._plot_type = 'timeseries'
    ts_widget._plot_type = 'timeseries'
    ts_widget._figure = fig
    ts_widget._canvas = canvas
    
    plots['timeseries'] = ts_widget
    
    return plots

def configure_plot_style(figure: matplotlib.figure.Figure, theme_name: str = 'default') -> None:
    """
    Configures consistent plot styling across all diagnostic plots with PyQt6 theme integration.
    
    Args:
        figure: Matplotlib figure to style
        theme_name: Name of the theme to apply ('default', 'light', 'dark')
        
    Returns:
        None: Updates figure in place
    """
    # Define theme parameters
    if theme_name == 'dark':
        # Dark theme parameters
        bg_color = '#2D2D2D'
        text_color = '#E0E0E0'
        grid_color = '#404040'
        face_color = '#3D3D3D'
    else:
        # Light/default theme parameters
        bg_color = '#FFFFFF'
        text_color = '#333333'
        grid_color = '#E5E5E5'
        face_color = '#F9F9F9'
    
    # Apply theme to figure and axes
    figure.patch.set_facecolor(bg_color)
    
    for ax in figure.get_axes():
        ax.set_facecolor(face_color)
        
        # Set text colors
        ax.title.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        
        # Set tick colors
        ax.tick_params(axis='x', colors=text_color)
        ax.tick_params(axis='y', colors=text_color)
        
        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(text_color)
        
        # Configure grid (if enabled)
        ax.grid(True, linestyle='--', alpha=0.3, color=grid_color)
    
    # Update font properties
    font_properties = {
        'family': 'sans-serif',
        'weight': 'normal',
        'size': 10
    }
    
    # Apply font properties (to local figure only, not globally)
    for ax in figure.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            if hasattr(item, 'set_fontfamily'):
                item.set_fontfamily(font_properties['family'])
            if hasattr(item, 'set_fontweight'):
                item.set_fontweight(font_properties['weight'])
            if hasattr(item, 'set_fontsize'):
                item.set_fontsize(font_properties['size'])
    
    # Update figure layout
    figure.tight_layout()

async def update_plot_async(plot_widget: QWidget, new_data: np.ndarray) -> None:
    """
    Asynchronously updates plot data and refreshes display.
    
    Args:
        plot_widget: The plot widget to update
        new_data: New data array to display
        
    Returns:
        None: Updates plot in place
    """
    try:
        # Validate input
        if not isinstance(plot_widget, QWidget):
            raise TypeError("plot_widget must be a QWidget instance")
            
        if not isinstance(new_data, np.ndarray):
            raise TypeError("new_data must be a NumPy array")
        
        # Get plot type and canvas from widget
        if not hasattr(plot_widget, '_plot_type') or not hasattr(plot_widget, '_canvas'):
            raise ValueError("Widget is not a recognized plot widget. Use functions from plot_utils to create plots.")
        
        plot_type = plot_widget._plot_type
        canvas = plot_widget._canvas
        fig = plot_widget._figure
        ax = fig.axes[0]  # Assuming single axes
        
        # Different update logic based on plot type
        if plot_type == 'residual':
            # Update residual plot: need fitted values too
            if len(ax.collections) > 0:
                # If we have both residuals and fitted values in new_data
                if isinstance(new_data, tuple) and len(new_data) == 2:
                    residuals, fitted_values = new_data
                    ax.collections[0].set_offsets(np.column_stack([fitted_values, residuals]))
                else:
                    # If we only have residuals, keep original x coordinates
                    x_data = ax.collections[0].get_offsets()[:, 0]
                    ax.collections[0].set_offsets(np.column_stack([x_data, new_data]))
                    
                ax.relim()
                ax.autoscale_view()
                
        elif plot_type == 'acf':
            # Recalculate ACF
            lags = getattr(plot_widget, '_lags', 20)
            
            # Recalculate ACF values
            acf_values = np.zeros(lags + 1)
            for i in range(lags + 1):
                if i == 0:
                    acf_values[i] = 1.0
                else:
                    # Calculate autocorrelation at lag i
                    n = len(new_data)
                    slice1 = new_data[:n-i]
                    slice2 = new_data[i:n]
                    acf_values[i] = np.corrcoef(slice1, slice2)[0, 1]
            
            # Clear previous plot and redraw
            ax.clear()
            
            # Calculate confidence intervals
            confidence = 1.96 / np.sqrt(len(new_data))
            
            # Plot ACF values
            ax.bar(range(lags + 1), acf_values, width=0.3, color='navy', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-')
            ax.axhline(y=confidence, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=-confidence, color='red', linestyle='--', alpha=0.7)
            
            # Re-add labels and title
            ax.set_xlabel('Lag')
            ax.set_ylabel('Correlation')
            ax.set_title('Autocorrelation Function')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Update saved data
            plot_widget._data = new_data
            
        elif plot_type == 'qqplot':
            # Redraw QQ plot
            ax.clear()
            stats.probplot(new_data, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot of Residuals')
            ax.grid(True, linestyle='--', alpha=0.7)
            
        elif plot_type == 'histogram':
            # Recreate histogram
            ax.clear()
            ax.hist(new_data, bins=30, density=True, alpha=0.7, color='navy')
            
            # Recreate density curve
            kde_xs = np.linspace(min(new_data), max(new_data), 200)
            kde = stats.gaussian_kde(new_data)
            ax.plot(kde_xs, kde(kde_xs), 'r-', label='Density')
            
            # Recreate normal curve
            mu, sigma = stats.norm.fit(new_data)
            norm_pdf = stats.norm.pdf(kde_xs, mu, sigma)
            ax.plot(kde_xs, norm_pdf, 'g--', label='Normal')
            
            # Re-add labels and formatting
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Density')
            ax.set_title('Histogram of Residuals')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
        elif plot_type == 'timeseries':
            # Update time series plot
            if len(ax.lines) > 0:
                ax.lines[0].set_ydata(new_data)
                # Keep x-limit as is, update y-limit
                ax.relim()
                ax.autoscale_view(scaley=True, scalex=False)
            else:
                # If line doesn't exist, recreate it
                ax.clear()
                ax.plot(new_data, marker='o', markersize=3, linestyle='-', color='navy', alpha=0.7)
                ax.axhline(y=0, color='red', linestyle='-', alpha=0.7)
                
                # Re-add labels and formatting
                ax.set_xlabel('Observation')
                ax.set_ylabel('Residuals')
                ax.set_title('Time Series of Residuals')
                ax.grid(True, linestyle='--', alpha=0.7)
        else:
            logger.warning(f"Unknown plot type for update: {plot_type}")
            
            # Try to clear and redraw completely as a simple line plot
            ax.clear()
            ax.plot(new_data)
            ax.set_title('Plot')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Ensure styling is consistent
        configure_plot_style(fig, 'default')
        
        # Update the canvas
        canvas.draw_idle()
        
        # Emit signal if widget has update_complete signal
        if hasattr(plot_widget, 'update_complete'):
            plot_widget.update_complete.emit()
            
    except Exception as e:
        logger.error(f"Error updating plot: {str(e)}")
        raise RuntimeError(f"Plot update failed: {str(e)}") from e