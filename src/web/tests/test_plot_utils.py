"""
Test suite for plot utility functions that verify the creation and configuration 
of statistical plots and visualizations using Matplotlib with PyQt6 integration.
"""
import asyncio
import pytest
import numpy as np
from matplotlib import pyplot as plt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt

# Import functions to test
from web.utils.plot_utils import (
    create_residual_plot,
    create_acf_plot,
    create_diagnostic_plots,
    configure_plot_style
)

# Fixtures
@pytest.fixture(scope="function")
def sample_data():
    """Provides sample numpy arrays for plot testing"""
    np.random.seed(42)  # Ensure reproducibility
    size = 100
    residuals = np.random.normal(0, 1, size)
    fitted_values = np.random.normal(5, 2, size)
    time_series = np.random.normal(0, 1, size)
    return {
        'residuals': residuals,
        'fitted_values': fitted_values,
        'time_series': time_series
    }

@pytest.fixture(scope="function")
def mock_figure():
    """Provides mock matplotlib figure for style testing"""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    return fig

@pytest.fixture(scope="function")
async def async_test_data():
    """Async data generator for plot updates"""
    np.random.seed(43)  # Different seed for update data
    size = 100
    residuals = np.random.normal(0, 1.5, size)  # Different distribution
    fitted_values = np.random.normal(6, 2.5, size)  # Different distribution
    return residuals, fitted_values

@pytest.fixture(scope="function")
def theme_config():
    """Theme configuration for style testing"""
    return {
        'default': {
            'fontsize': 10,
            'grid_alpha': 0.7
        },
        'dark': {
            'fontsize': 12,
            'grid_alpha': 0.5
        }
    }

@pytest.mark.asyncio
async def test_create_residual_plot(sample_data, qtbot, async_test_data):
    """Tests the creation and properties of residual diagnostic plots with PyQt6 widget integration"""
    # Configure matplotlib backend for PyQt6
    plt.switch_backend('QtAgg')
    
    # Create a container widget
    container = QWidget()
    layout = QVBoxLayout(container)
    
    # Call create_residual_plot with test data
    plot_widget = await create_residual_plot(
        sample_data['residuals'],
        sample_data['fitted_values'],
        parent=container
    )
    
    # Add widget to layout
    layout.addWidget(plot_widget)
    container.show()
    qtbot.addWidget(container)
    
    # Verify the widget was created and is a QWidget
    assert plot_widget is not None
    assert isinstance(plot_widget, QWidget)
    
    # Check that the figure was set on the widget
    assert hasattr(plot_widget, '_figure')
    assert hasattr(plot_widget, '_canvas')
    assert hasattr(plot_widget, '_plot_type')
    assert plot_widget._plot_type == 'residual'
    
    # Get the figure and axes
    fig = plot_widget._figure
    ax = fig.axes[0]
    
    # Verify plot properties
    assert ax.get_xlabel() == 'Fitted Values'
    assert ax.get_ylabel() == 'Residuals'
    assert ax.get_title() == 'Residuals vs Fitted Values'
    
    # Verify scatter plot exists and has correct data
    assert len(ax.collections) > 0
    scatter = ax.collections[0]
    assert scatter.get_offsets().shape[0] == len(sample_data['residuals']), "Scatter plot data count mismatch"
    
    # Verify reference line exists
    has_reference_line = False
    for line in ax.lines:
        if len(line.get_ydata()) > 0 and line.get_ydata()[0] == 0:
            has_reference_line = True
            break
    assert has_reference_line, "Reference line at y=0 not found"
    
    # Verify grid is enabled
    assert ax.get_grid()
    
    # Test widget integration
    assert plot_widget.layout() is not None
    assert plot_widget.children() is not None
    
    # Clean up resources
    plt.close(fig)
    container.close()

@pytest.mark.asyncio
async def test_create_acf_plot(sample_data, qtbot):
    """Tests the creation and properties of autocorrelation function plots with PyQt6 integration"""
    # Configure matplotlib backend for PyQt6
    plt.switch_backend('QtAgg')
    
    # Create a container widget
    container = QWidget()
    layout = QVBoxLayout(container)
    
    # Call create_acf_plot with test data
    plot_widget = await create_acf_plot(
        sample_data['time_series'],
        lags=20,
        parent=container
    )
    
    # Add widget to layout
    layout.addWidget(plot_widget)
    container.show()
    qtbot.addWidget(container)
    
    # Verify the widget was created and is a QWidget
    assert plot_widget is not None
    assert isinstance(plot_widget, QWidget)
    
    # Check that the figure was set on the widget
    assert hasattr(plot_widget, '_figure')
    assert hasattr(plot_widget, '_canvas')
    assert hasattr(plot_widget, '_plot_type')
    assert plot_widget._plot_type == 'acf'
    
    # Get the figure and axes
    fig = plot_widget._figure
    ax = fig.axes[0]
    
    # Verify plot properties
    assert ax.get_xlabel() == 'Lag'
    assert ax.get_ylabel() == 'Correlation'
    assert ax.get_title() == 'Autocorrelation Function'
    
    # Verify ACF bars exist
    assert len(ax.containers) > 0
    
    # Verify confidence intervals and zero line
    confidence_lines = 0
    zero_line_found = False
    for line in ax.lines:
        # Horizontal lines have same y value
        if len(line.get_ydata()) > 1 and all(y == line.get_ydata()[0] for y in line.get_ydata()):
            if line.get_ydata()[0] == 0:
                zero_line_found = True
            else:
                confidence_lines += 1
    
    assert confidence_lines >= 2, f"Expected at least 2 confidence interval lines, found {confidence_lines}"
    assert zero_line_found, "Zero reference line not found"
    
    # Verify grid is enabled
    assert ax.get_grid()
    
    # Test widget interaction
    # Check toolbar integration
    toolbar_found = False
    for child in plot_widget.children():
        if 'NavigationToolbar' in child.__class__.__name__:
            toolbar_found = True
            break
    assert toolbar_found, "Navigation toolbar not found"
    
    # Clean up resources
    plt.close(fig)
    container.close()

@pytest.mark.asyncio
async def test_create_diagnostic_plots(sample_data, qtbot):
    """Tests the creation of comprehensive diagnostic plot set with PyQt6 integration"""
    # Configure matplotlib backend for PyQt6
    plt.switch_backend('QtAgg')
    
    # Create a container widget
    container = QWidget()
    layout = QVBoxLayout(container)
    
    # Call create_diagnostic_plots with test data
    plot_widgets = await create_diagnostic_plots(
        sample_data['residuals'],
        sample_data['fitted_values'],
        parent=container
    )
    
    # Verify the dictionary of widgets was returned
    assert plot_widgets is not None
    assert isinstance(plot_widgets, dict)
    
    # Verify all expected plot types exist
    expected_plots = ['residual', 'acf', 'qqplot', 'histogram', 'timeseries']
    for plot_type in expected_plots:
        assert plot_type in plot_widgets, f"Missing plot type: {plot_type}"
        
        # Add each widget to layout and verify properties
        plot_widget = plot_widgets[plot_type]
        layout.addWidget(plot_widget)
        
        # Verify each widget has correct attributes
        assert hasattr(plot_widget, '_figure')
        assert hasattr(plot_widget, '_canvas')
        assert hasattr(plot_widget, '_plot_type')
        assert plot_widget._plot_type == plot_type
        
        # Verify figure has at least one axes
        fig = plot_widget._figure
        assert len(fig.axes) > 0
        
        # Check for consistent styling across plots
        ax = fig.axes[0]
        assert ax.get_grid(), f"Grid should be enabled for {plot_type} plot"
    
    # Show container and add to qtbot
    container.show()
    qtbot.addWidget(container)
    
    # Test widget layout
    assert container.layout() is not None
    assert container.layout().count() == len(expected_plots)
    
    # Clean up resources
    for plot_type, widget in plot_widgets.items():
        plt.close(widget._figure)
    container.close()

def test_configure_plot_style(mock_figure):
    """Tests the plot style configuration function with theme integration"""
    # Create test figure
    fig = mock_figure
    ax = fig.axes[0]
    
    # Configure default style
    configure_plot_style(fig, 'default')
    
    # Verify font sizes are set correctly
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
                 ax.get_xticklabels() + ax.get_yticklabels()):
        if hasattr(item, 'get_fontsize'):
            assert item.get_fontsize() == 10, f"Font size should be 10, got {item.get_fontsize()}"
    
    # Verify grid style configuration
    assert ax.get_grid(), "Grid should be enabled"
    
    # Get original background color
    original_bg_color = fig.get_facecolor()
    
    # Check with dark theme
    configure_plot_style(fig, 'dark')
    
    # Verify color scheme application
    assert fig.get_facecolor() != original_bg_color, "Background color should change with theme"
    
    # Verify tick parameter settings
    tick_color = ax.xaxis.get_ticklabels()[0].get_color() if ax.xaxis.get_ticklabels() else None
    assert tick_color is not None, "Tick color not set"
    
    # Clean up resources
    plt.close(fig)