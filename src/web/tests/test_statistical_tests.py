"""
Test suite for the statistical tests GUI component in the MFE Toolbox.

This module validates the functionality of the StatisticalTests widget,
including test execution, result display, and plot updates in the PyQt6-based
interface.
"""

import asyncio
import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

from web.components.statistical_tests import StatisticalTests
from web.utils.qt_helpers import create_widget


@pytest.mark.qt
def test_statistical_tests_initialization(qtbot):
    """
    Tests proper initialization of the StatisticalTests widget including theme application and layout setup.
    
    Args:
        qtbot: pytest_qt.plugin.QtBot - Qt testing bot
    """
    # Create StatisticalTests widget
    widget = StatisticalTests()
    qtbot.addWidget(widget)
    
    # Verify widget is properly initialized
    assert widget is not None
    assert hasattr(widget, '_results_grid')
    assert hasattr(widget, '_plot_display')
    
    # Verify results grid headers
    header_row = 0
    expected_headers = ["Test", "Statistic", "p-value", "Result"]
    for col, header_text in enumerate(expected_headers):
        header_item = widget._results_grid.itemAtPosition(header_row, col)
        assert header_item is not None
        assert header_item.widget() is not None
        assert header_item.widget().text() == header_text
    
    # Verify plot display is initialized
    assert widget._plot_display is not None
    
    # Verify clear button exists and is connected
    clear_button = None
    for child in widget.findChildren(QPushButton):
        if child.text() == 'Clear Results':
            clear_button = child
            break
    
    assert clear_button is not None


@pytest.mark.qt
@pytest.mark.asyncio
async def test_ljung_box_execution(qtbot):
    """
    Tests asynchronous Ljung-Box test execution with real-time plot updates.
    
    Args:
        qtbot: pytest_qt.plugin.QtBot - Qt testing bot
    """
    # Create widget
    widget = StatisticalTests()
    qtbot.addWidget(widget)
    
    # Generate test data with known autocorrelation
    np.random.seed(42)  # For reproducibility
    n = 500
    ar_coef = 0.7
    residuals = np.zeros(n)
    residuals[0] = np.random.randn()
    for i in range(1, n):
        residuals[i] = ar_coef * residuals[i-1] + np.random.randn()
    
    # Record initial state
    initial_row_count = widget._results_grid.rowCount()
    
    # Connect to plot update signal for verification
    plot_updates = []
    widget._plot_display.plot_updated.connect(lambda x: plot_updates.append(x))
    
    # Execute Ljung-Box test
    await widget.run_ljung_box(residuals, lags=20)
    
    # Verify results were added to grid
    assert widget._results_grid.rowCount() > initial_row_count
    
    # Verify "Ljung-Box" appears in first column of results
    found_result = False
    for row in range(1, widget._results_grid.rowCount()):
        item = widget._results_grid.itemAtPosition(row, 0)
        if item and item.widget() and item.widget().text() == "Ljung-Box":
            found_result = True
            
            # Verify statistic, p-value and result columns have content
            for col in range(1, 4):
                col_item = widget._results_grid.itemAtPosition(row, col)
                assert col_item is not None
                assert col_item.widget() is not None
                assert col_item.widget().text()  # Not empty
            break
    
    assert found_result, "Ljung-Box test results not found in grid"
    
    # Verify plot was updated
    assert len(plot_updates) > 0
    assert "acf" in plot_updates
    
    # Verify results are stored in the _test_results dictionary
    assert "ljung_box" in widget._test_results
    assert "statistic" in widget._test_results["ljung_box"]
    assert "p_value" in widget._test_results["ljung_box"]
    assert "result" in widget._test_results["ljung_box"]


@pytest.mark.qt
@pytest.mark.asyncio
async def test_arch_lm_execution(qtbot):
    """
    Tests asynchronous ARCH-LM test execution with progress tracking.
    
    Args:
        qtbot: pytest_qt.plugin.QtBot - Qt testing bot
    """
    # Create widget
    widget = StatisticalTests()
    qtbot.addWidget(widget)
    
    # Generate test data with ARCH effects
    np.random.seed(42)  # For reproducibility
    n = 500
    residuals = np.zeros(n)
    volatility = np.ones(n)
    
    # Generate ARCH process
    alpha0 = 0.1
    alpha1 = 0.7
    
    residuals[0] = np.random.randn() * np.sqrt(volatility[0])
    for i in range(1, n):
        volatility[i] = alpha0 + alpha1 * residuals[i-1]**2
        residuals[i] = np.random.randn() * np.sqrt(volatility[i])
    
    # Record initial state
    initial_row_count = widget._results_grid.rowCount()
    
    # Connect to plot update signal for verification
    plot_updates = []
    widget._plot_display.plot_updated.connect(lambda x: plot_updates.append(x))
    
    # Execute ARCH-LM test
    await widget.run_arch_lm(residuals, lags=12)
    
    # Verify results were added to grid
    assert widget._results_grid.rowCount() > initial_row_count
    
    # Verify "ARCH-LM" appears in first column of results
    found_result = False
    for row in range(1, widget._results_grid.rowCount()):
        item = widget._results_grid.itemAtPosition(row, 0)
        if item and item.widget() and item.widget().text() == "ARCH-LM":
            found_result = True
            
            # Verify statistic, p-value and result columns have content
            for col in range(1, 4):
                col_item = widget._results_grid.itemAtPosition(row, col)
                assert col_item is not None
                assert col_item.widget() is not None
                assert col_item.widget().text()  # Not empty
            break
    
    assert found_result, "ARCH-LM test results not found in grid"
    
    # Verify plot was updated
    assert len(plot_updates) > 0
    assert "diagnostic" in plot_updates
    
    # Verify results are stored in the _test_results dictionary
    assert "arch_lm" in widget._test_results
    assert "statistic" in widget._test_results["arch_lm"]
    assert "p_value" in widget._test_results["arch_lm"]
    assert "result" in widget._test_results["arch_lm"]


@pytest.mark.qt
def test_durbin_watson_execution(qtbot):
    """
    Tests Durbin-Watson test execution with error handling.
    
    Args:
        qtbot: pytest_qt.plugin.QtBot - Qt testing bot
    """
    # Create widget
    widget = StatisticalTests()
    qtbot.addWidget(widget)
    
    # Generate test data with positive autocorrelation
    np.random.seed(42)  # For reproducibility
    n = 200
    ar_coef = 0.7  # Strong positive autocorrelation
    residuals = np.zeros(n)
    residuals[0] = np.random.randn()
    for i in range(1, n):
        residuals[i] = ar_coef * residuals[i-1] + np.random.randn()
    
    # Record initial state
    initial_row_count = widget._results_grid.rowCount()
    
    # First test with invalid input to check error handling
    # Create a completely invalid input - not numpy array
    with pytest.raises(TypeError):
        # This synchronous call will wrap the async function
        loop = asyncio.get_event_loop()
        loop.run_until_complete(widget.run_durbin_watson("not an array"))
    
    # Now test with proper input
    # Execute Durbin-Watson test
    loop = asyncio.get_event_loop()
    loop.run_until_complete(widget.run_durbin_watson(residuals))
    
    # Verify results were added to grid
    assert widget._results_grid.rowCount() > initial_row_count
    
    # Verify "Durbin-Watson" appears in first column of results
    found_result = False
    for row in range(1, widget._results_grid.rowCount()):
        item = widget._results_grid.itemAtPosition(row, 0)
        if item and item.widget() and item.widget().text() == "Durbin-Watson":
            found_result = True
            
            # Verify statistic and result columns have content
            stat_item = widget._results_grid.itemAtPosition(row, 1)
            assert stat_item is not None
            assert stat_item.widget() is not None
            assert stat_item.widget().text()  # Not empty
            
            # Check p-value is "N/A" for Durbin-Watson
            p_value_item = widget._results_grid.itemAtPosition(row, 2)
            assert p_value_item.widget().text() == "N/A"
            
            # With ar_coef = 0.7, we expect positive autocorrelation
            # DW statistic should be less than 2
            result_item = widget._results_grid.itemAtPosition(row, 3)
            assert "Positive autocorrelation" in result_item.widget().text()
            break
    
    assert found_result, "Durbin-Watson test results not found in grid"
    
    # Verify results are stored in the _test_results dictionary
    assert "durbin_watson" in widget._test_results
    assert "statistic" in widget._test_results["durbin_watson"]
    assert widget._test_results["durbin_watson"]["p_value"] is None  # DW doesn't have p-value
    assert "result" in widget._test_results["durbin_watson"]


@pytest.mark.qt
@pytest.mark.asyncio
async def test_white_test_execution(qtbot):
    """
    Tests White's test execution with large datasets.
    
    Args:
        qtbot: pytest_qt.plugin.QtBot - Qt testing bot
    """
    # Create widget
    widget = StatisticalTests()
    qtbot.addWidget(widget)
    
    # Generate larger test dataset with heteroskedasticity
    np.random.seed(42)  # For reproducibility
    n = 1000  # Large sample
    
    # Create predictors (including a constant term)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])
    
    # Create heteroskedastic errors (variance increasing with x1)
    beta = np.array([1.0, 2.0, -1.0])
    y = X @ beta + np.random.randn(n) * (1 + 0.5 * np.abs(x1))
    
    # Calculate residuals
    from numpy.linalg import lstsq
    residuals = y - X @ lstsq(X, y, rcond=None)[0]
    
    # Record initial state
    initial_row_count = widget._results_grid.rowCount()
    
    # Connect to plot update signal for verification
    plot_updates = []
    widget._plot_display.plot_updated.connect(lambda x: plot_updates.append(x))
    
    # Execute White's test
    await widget.run_white_test(residuals, X)
    
    # Verify results were added to grid
    assert widget._results_grid.rowCount() > initial_row_count
    
    # Verify "White's Test" appears in first column of results
    found_result = False
    for row in range(1, widget._results_grid.rowCount()):
        item = widget._results_grid.itemAtPosition(row, 0)
        if item and item.widget() and item.widget().text() == "White's Test":
            found_result = True
            
            # Verify statistic, p-value and result columns have content
            for col in range(1, 4):
                col_item = widget._results_grid.itemAtPosition(row, col)
                assert col_item is not None
                assert col_item.widget() is not None
                assert col_item.widget().text()  # Not empty
            
            # Since we generated data with heteroskedasticity, 
            # the result should indicate this
            result_item = widget._results_grid.itemAtPosition(row, 3)
            result_text = result_item.widget().text()
            assert "Heteroskedasticity" in result_text
            break
    
    assert found_result, "White's test results not found in grid"
    
    # Verify plot was updated
    assert len(plot_updates) > 0
    
    # Verify results are stored in the _test_results dictionary
    assert "white_test" in widget._test_results
    assert "statistic" in widget._test_results["white_test"]
    assert "p_value" in widget._test_results["white_test"]
    assert "result" in widget._test_results["white_test"]


@pytest.mark.qt
def test_clear_results(qtbot):
    """
    Tests clearing of test results and plot reset.
    
    Args:
        qtbot: pytest_qt.plugin.QtBot - Qt testing bot
    """
    # Create widget
    widget = StatisticalTests()
    qtbot.addWidget(widget)
    
    # Generate test data
    np.random.seed(42)  # For reproducibility
    n = 200
    residuals = np.random.randn(n)
    X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
    
    # Execute multiple statistical tests
    loop = asyncio.get_event_loop()
    
    # Run Durbin-Watson test
    loop.run_until_complete(widget.run_durbin_watson(residuals))
    
    # Run Ljung-Box test
    loop.run_until_complete(widget.run_ljung_box(residuals))
    
    # Verify tests were added to grid
    assert widget._results_grid.rowCount() > 1  # Header row plus at least two test rows
    
    # Verify tests are in _test_results dictionary
    assert len(widget._test_results) > 0
    
    # Connect to plot update signal for verification
    plot_updates = []
    widget._plot_display.plot_updated.connect(lambda x: plot_updates.append(x))
    
    # Clear results
    widget.clear_results()
    
    # Verify results grid has been cleared (only header row remains)
    assert widget._results_grid.rowCount() == 1  # Only header row
    
    # Verify _test_results dictionary is empty
    assert len(widget._test_results) == 0
    
    # Verify plot has been cleared
    assert "cleared" in plot_updates