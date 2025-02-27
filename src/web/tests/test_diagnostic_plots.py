"""
Test suite for the diagnostic plots component, validating visualization functionality,
plot updates, and interactive features of the DiagnosticPlotsWidget.
"""

import pytest
import pytest_asyncio
import numpy as np
from PyQt6.QtTest import QTest  # PyQt6 version 6.6.1

# Import component under test
from components.diagnostic_plots import DiagnosticPlotsWidget
from utils.plot_utils import create_residual_plot, create_acf_plot

# Global constants
SAMPLE_SIZE = 100  # Size of sample data arrays for testing
TEST_TABS = ['Residuals', 'ACF', 'PACF', 'Diagnostics']
PLOT_TIMEOUT = 5000  # Timeout for plot operations in milliseconds

@pytest.mark.qt
def test_diagnostic_plots_initialization(qtbot):
    """
    Tests proper initialization of DiagnosticPlotsWidget including theme-aware styling.
    """
    # Create widget instance
    widget = DiagnosticPlotsWidget()
    qtbot.addWidget(widget)
    
    # Verify tab widget initialization
    assert hasattr(widget, '_tab_widget')
    
    # Check plot display widgets creation
    assert hasattr(widget, '_residual_display')
    assert hasattr(widget, '_acf_display')
    assert hasattr(widget, '_diagnostic_display')
    
    # Validate initial empty state
    assert widget._current_data == {}
    assert widget._plot_cache == {}
    
    # Verify theme-aware styling has been applied
    assert widget.layout() is not None
    assert widget._tab_widget.tabPosition() == 0  # North position

@pytest.mark.qt
@pytest.mark.asyncio
async def test_update_plots_async(qtbot):
    """
    Tests asynchronous plot update functionality with sample data.
    """
    # Create test data arrays
    residuals = np.random.normal(0, 1, SAMPLE_SIZE)
    fitted_values = np.random.normal(2, 0.5, SAMPLE_SIZE)
    
    # Initialize DiagnosticPlotsWidget
    widget = DiagnosticPlotsWidget()
    qtbot.addWidget(widget)
    
    # Await update_plots with test data
    await widget.update_plots(residuals, fitted_values)
    
    # Verify plot updates in all tabs
    assert '_residual_display' in vars(widget)
    assert '_acf_display' in vars(widget)
    assert '_diagnostic_display' in vars(widget)
    
    # Check data storage
    assert "residuals" in widget._current_data
    assert "fitted_values" in widget._current_data
    np.testing.assert_array_equal(widget._current_data["residuals"], residuals)
    np.testing.assert_array_equal(widget._current_data["fitted_values"], fitted_values)
    
    # Check plot cache
    assert "residual" in widget._plot_cache
    assert "acf" in widget._plot_cache
    assert "diagnostic" in widget._plot_cache
    
    # Validate memory cleanup
    # Memory usage should be reasonable after plot creation
    # This is a simple check that cached data isn't duplicated unnecessarily
    assert len(widget._plot_cache) == 3  # Expected number of plot types

@pytest.mark.qt
@pytest.mark.asyncio
async def test_clear_plots(qtbot):
    """
    Tests plot clearing functionality and resource cleanup.
    """
    # Initialize widget with test data
    widget = DiagnosticPlotsWidget()
    qtbot.addWidget(widget)
    
    # Create test data and populate plots
    residuals = np.random.normal(0, 1, SAMPLE_SIZE)
    fitted_values = np.random.normal(2, 0.5, SAMPLE_SIZE)
    await widget.update_plots(residuals, fitted_values)
    
    # Verify plots were created
    assert len(widget._plot_cache) > 0
    
    # Call clear_plots
    widget.clear_plots()
    
    # Verify all plots are cleared
    assert widget._plot_cache == {}
    assert widget._current_data == {}
    
    # Check data storage reset
    assert len(widget._current_data) == 0
    
    # Verify resource cleanup
    assert widget._tab_widget.currentIndex() == 0  # First tab should be selected

@pytest.mark.qt
def test_tab_switching(qtbot):
    """
    Tests tab switching functionality and plot state preservation.
    """
    # Initialize widget
    widget = DiagnosticPlotsWidget()
    qtbot.addWidget(widget)
    
    # Switch to each available tab
    tabs = ["residuals", "autocorrelation", "diagnostics"]
    
    for i, tab_name in enumerate(tabs):
        # Switch to tab
        widget.switch_to_tab(tab_name)
        
        # Verify correct tab activation
        assert widget._tab_widget.currentIndex() == i
        
        # Check plot display updates
        current_tab = widget._tab_widget.currentWidget()
        assert current_tab is not None
        
        # Validate state preservation
        assert widget._tab_widget.hasFocus()

@pytest.mark.qt
@pytest.mark.asyncio
async def test_interactive_features(qtbot):
    """
    Tests interactive plot features and user input handling.
    """
    # Initialize widget with test data
    widget = DiagnosticPlotsWidget()
    qtbot.addWidget(widget)
    
    # Create test data and populate plots
    residuals = np.random.normal(0, 1, SAMPLE_SIZE)
    fitted_values = np.random.normal(2, 0.5, SAMPLE_SIZE)
    await widget.update_plots(residuals, fitted_values)
    
    # Simulate zoom interactions
    for tab_name in ["residuals", "autocorrelation", "diagnostics"]:
        # Switch to tab
        widget.switch_to_tab(tab_name)
        
        # Get current tab widget and verify interactive components
        current_tab = widget._tab_widget.currentWidget()
        assert current_tab is not None
        
        # Test tab interaction by clicking on the tab
        tab_index = widget._tab_widget.currentIndex()
        tab_rect = widget._tab_widget.tabBar().tabRect(tab_index)
        tab_center = tab_rect.center()
        QTest.mouseClick(widget._tab_widget.tabBar(), 1, pos=tab_center)
        
        # Verify we're still on the right tab after clicking
        assert widget._tab_widget.currentIndex() == tab_index
        
        # Verify tooltip displays
        assert widget.accessibleDescription() is not None
        assert "diagnostic plot" in widget.accessibleDescription().lower()
        
        # Check click handlers
        # Verify tab has focus after clicking
        assert widget._tab_widget.hasFocus()