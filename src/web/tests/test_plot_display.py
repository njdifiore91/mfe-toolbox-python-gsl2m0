"""
Test suite for the PlotDisplay widget component.

This module validates the statistical plot visualization functionality,
interactive features, and proper PyQt6/Matplotlib integration of the
PlotDisplay widget.
"""

import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication  # PyQt6 version 6.6.1

# Internal imports
from components.plot_display import PlotDisplay
from utils.qt_helpers import create_widget


class TestPlotDisplay:
    """Test class for PlotDisplay widget functionality."""
    
    def setup_method(self, method):
        """Setup method run before each test."""
        # Initialize QApplication for tests if it doesn't exist
        self._qapp = QApplication.instance() or QApplication([])
        
        # Create fresh widget instance
        self._widget = PlotDisplay()
        
        # Reset test data
        self._test_residuals = np.random.normal(0, 1, 100)
        self._test_fitted_values = np.random.normal(0, 1, 100)
        
    def teardown_method(self, method):
        """Cleanup method run after each test."""
        # Clear all plots
        if hasattr(self, '_widget') and self._widget:
            self._widget.clear_plots()
        
        # Clean up widget
        if hasattr(self, '_widget') and self._widget:
            self._widget.setParent(None)
            self._widget.deleteLater()
            self._widget = None


@pytest.mark.qt
def test_plot_display_initialization():
    """Tests proper initialization of PlotDisplay widget."""
    # Create QApplication instance for test
    app = QApplication.instance() or QApplication([])
    
    # Initialize PlotDisplay widget
    widget = PlotDisplay()
    
    try:
        # Verify widget properties and layout
        assert widget is not None
        assert hasattr(widget, '_main_layout')
        assert hasattr(widget, '_plot_container')
        assert hasattr(widget, '_container_layout')
        
        # Validate canvas and toolbar initialization
        assert hasattr(widget, '_figure')
        assert hasattr(widget, '_canvas')
        assert hasattr(widget, '_toolbar')
        
        # Check initial plot dictionary state
        assert hasattr(widget, '_current_plots')
        assert widget._current_plots == {}
    finally:
        # Clean up
        widget.deleteLater()


@pytest.mark.qt
@pytest.mark.asyncio
async def test_display_residual_plot():
    """Tests residual plot display functionality."""
    # Create test data arrays for residuals and fitted values
    residuals = np.random.normal(0, 1, 100)
    fitted_values = np.random.normal(0, 1, 100)
    
    # Initialize PlotDisplay widget
    app = QApplication.instance() or QApplication([])
    widget = PlotDisplay()
    
    try:
        # Call display_residual_plot method
        await widget.display_residual_plot(residuals, fitted_values)
        
        # Verify plot creation and display
        assert "residual" in widget._current_plots
        assert widget._current_plots["residual"]["type"] == "residual"
        assert "residuals" in widget._current_plots["residual"]["data"]
        assert "fitted_values" in widget._current_plots["residual"]["data"]
        
        # Validate plot properties and styling
        assert np.array_equal(
            widget._current_plots["residual"]["data"]["residuals"], 
            residuals
        )
        assert np.array_equal(
            widget._current_plots["residual"]["data"]["fitted_values"], 
            fitted_values
        )
    finally:
        # Clean up
        widget.clear_plots()
        widget.deleteLater()


@pytest.mark.qt
@pytest.mark.asyncio
async def test_display_acf_plot():
    """Tests autocorrelation function plot display."""
    # Create test time series data array
    data = np.random.normal(0, 1, 100)
    
    # Initialize PlotDisplay widget
    app = QApplication.instance() or QApplication([])
    widget = PlotDisplay()
    
    try:
        # Call display_acf_plot method
        await widget.display_acf_plot(data)
        
        # Verify ACF plot creation
        assert "acf" in widget._current_plots
        assert widget._current_plots["acf"]["type"] == "acf"
        assert "data" in widget._current_plots["acf"]["data"]
        
        # Validate plot properties and styling
        assert np.array_equal(
            widget._current_plots["acf"]["data"]["data"], 
            data
        )
        assert "lags" in widget._current_plots["acf"]["data"]
    finally:
        # Clean up
        widget.clear_plots()
        widget.deleteLater()


@pytest.mark.qt
@pytest.mark.asyncio
async def test_display_diagnostic_plots():
    """Tests comprehensive diagnostic plots display."""
    # Create test data for residuals and fitted values
    residuals = np.random.normal(0, 1, 100)
    fitted_values = np.random.normal(0, 1, 100)
    
    # Initialize PlotDisplay widget
    app = QApplication.instance() or QApplication([])
    widget = PlotDisplay()
    
    try:
        # Call display_diagnostic_plots method
        await widget.display_diagnostic_plots(residuals, fitted_values)
        
        # Verify all diagnostic plots are created
        assert "diagnostic" in widget._current_plots
        assert widget._current_plots["diagnostic"]["type"] == "diagnostic"
        assert "residuals" in widget._current_plots["diagnostic"]["data"]
        assert "fitted_values" in widget._current_plots["diagnostic"]["data"]
        
        # Validate plot layout and styling
        assert np.array_equal(
            widget._current_plots["diagnostic"]["data"]["residuals"], 
            residuals
        )
        assert np.array_equal(
            widget._current_plots["diagnostic"]["data"]["fitted_values"], 
            fitted_values
        )
    finally:
        # Clean up
        widget.clear_plots()
        widget.deleteLater()


@pytest.mark.qt
@pytest.mark.asyncio
async def test_clear_plots():
    """Tests plot clearing functionality."""
    # Initialize PlotDisplay widget
    app = QApplication.instance() or QApplication([])
    widget = PlotDisplay()
    
    try:
        # Create test data
        residuals = np.random.normal(0, 1, 100)
        fitted_values = np.random.normal(0, 1, 100)
        
        # Create a plot first
        await widget.display_residual_plot(residuals, fitted_values)
        
        # Verify plot was created
        assert "residual" in widget._current_plots
        assert len(widget._current_plots) > 0
        
        # Call clear_plots method
        widget.clear_plots()
        
        # Verify all plots are removed
        assert widget._current_plots == {}
        
        # Validate canvas and toolbar state
        assert hasattr(widget, '_figure')
        assert hasattr(widget, '_canvas')
    finally:
        # Clean up
        widget.deleteLater()