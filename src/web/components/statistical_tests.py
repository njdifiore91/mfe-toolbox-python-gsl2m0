"""
PyQt6 component for displaying and managing statistical test results in the MFE Toolbox GUI.
Provides interactive widgets for test selection, execution, and result visualization 
with async support and comprehensive error handling.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt

# Internal imports
from backend.core.tests import ljung_box, arch_lm, durbin_watson, white_test
from utils.qt_helpers import create_widget
from components.plot_display import PlotDisplay

# Set up logger
logger = logging.getLogger(__name__)

class StatisticalTests(QWidget):
    """
    PyQt6 widget for displaying and managing statistical test results with 
    async execution support and comprehensive error handling.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the statistical tests widget with PyQt6 components and async support.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing StatisticalTests widget")
        
        # Initialize main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create container widget with proper styling
        self._container = create_widget('QWidget', {
            'objectName': 'statisticalTestsContainer',
            'styleSheet': """
                QWidget#statisticalTestsContainer {
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
            """
        })
        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(10)
        
        # Create results grid with proper styling
        self._results_grid = QGridLayout()
        self._results_grid.setColumnStretch(0, 1)  # Test name column
        self._results_grid.setColumnStretch(1, 1)  # Statistic column
        self._results_grid.setColumnStretch(2, 1)  # p-value column
        self._results_grid.setColumnStretch(3, 2)  # Result column
        self._results_grid.setSpacing(8)
        
        # Set up the results grid header
        self._setup_results_grid()
        
        # Initialize plot display for visualization
        self._plot_display = PlotDisplay(self)
        
        # Create clear button
        clear_button = create_widget('QPushButton', {
            'text': 'Clear Results',
            'objectName': 'clearResultsButton',
            'cursor': Qt.CursorShape.PointingHandCursor
        })
        clear_button.clicked.connect(self.clear_results)
        
        # Add components to layouts
        container_layout.addLayout(self._results_grid)
        container_layout.addWidget(self._plot_display)
        container_layout.addWidget(clear_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        main_layout.addWidget(self._container)
        
        # Initialize test results dictionary
        self._test_results = {}
        
        logger.debug("StatisticalTests widget initialization complete")
    
    def _setup_results_grid(self) -> None:
        """
        Sets up the grid layout for displaying test results.
        """
        # Create header labels
        headers = ["Test", "Statistic", "p-value", "Result"]
        
        for col, header in enumerate(headers):
            label = create_widget('QLabel', {
                'text': header, 
                'alignment': Qt.AlignmentFlag.AlignCenter,
                'styleSheet': 'font-weight: bold; padding: 4px; background-color: #e0e0e0; border-radius: 2px;'
            })
            self._results_grid.addWidget(label, 0, col)
    
    async def run_ljung_box(self, residuals: np.ndarray, lags: int = 20) -> None:
        """
        Executes Ljung-Box test asynchronously and updates display.
        
        Args:
            residuals: Array of model residuals
            lags: Number of lags to include in test
            
        Returns:
            None: Updates display in place with test results
        """
        try:
            logger.info(f"Running Ljung-Box test with {lags} lags")
            
            # Validate inputs
            if not isinstance(residuals, np.ndarray):
                raise TypeError("Residuals must be a NumPy array")
            
            if not isinstance(lags, int) or lags <= 0:
                raise ValueError(f"Lags must be a positive integer, got {lags}")
            
            # Execute test
            q_stat, p_value = ljung_box(residuals, lags)
            
            # Format results
            stat_formatted = f"{q_stat:.4f}"
            p_formatted = f"{p_value:.4f}"
            result = "No serial correlation" if p_value > 0.05 else "Serial correlation detected"
            
            # Update results dictionary
            self._test_results["ljung_box"] = {
                "statistic": q_stat,
                "p_value": p_value,
                "result": result,
                "params": {"lags": lags}
            }
            
            # Update display
            self._update_results_display("Ljung-Box", stat_formatted, p_formatted, result)
            
            # Update plot display
            await self._plot_display.display_acf_plot(residuals, lags)
            
            logger.info(f"Ljung-Box test completed: {result}")
            
        except Exception as e:
            logger.error(f"Error executing Ljung-Box test: {str(e)}")
            self._handle_test_error("Ljung-Box", str(e))
    
    async def run_arch_lm(self, residuals: np.ndarray, lags: int = 12) -> None:
        """
        Executes ARCH-LM test asynchronously and updates display.
        
        Args:
            residuals: Array of model residuals
            lags: Number of lags to include in test
            
        Returns:
            None: Updates display in place with test results
        """
        try:
            logger.info(f"Running ARCH-LM test with {lags} lags")
            
            # Validate inputs
            if not isinstance(residuals, np.ndarray):
                raise TypeError("Residuals must be a NumPy array")
            
            if not isinstance(lags, int) or lags <= 0:
                raise ValueError(f"Lags must be a positive integer, got {lags}")
            
            # Execute test
            lm_stat, p_value = arch_lm(residuals, lags)
            
            # Format results
            stat_formatted = f"{lm_stat:.4f}"
            p_formatted = f"{p_value:.4f}"
            result = "No ARCH effects" if p_value > 0.05 else "ARCH effects detected"
            
            # Update results dictionary
            self._test_results["arch_lm"] = {
                "statistic": lm_stat,
                "p_value": p_value,
                "result": result,
                "params": {"lags": lags}
            }
            
            # Update display
            self._update_results_display("ARCH-LM", stat_formatted, p_formatted, result)
            
            # Update plot display - show diagnostic plots with squared residuals
            squared_residuals = residuals**2
            await self._plot_display.display_diagnostic_plots(residuals, squared_residuals)
            
            logger.info(f"ARCH-LM test completed: {result}")
            
        except Exception as e:
            logger.error(f"Error executing ARCH-LM test: {str(e)}")
            self._handle_test_error("ARCH-LM", str(e))
    
    async def run_durbin_watson(self, residuals: np.ndarray) -> None:
        """
        Executes Durbin-Watson test asynchronously and updates display.
        
        Args:
            residuals: Array of model residuals
            
        Returns:
            None: Updates display in place with test results
        """
        try:
            logger.info("Running Durbin-Watson test")
            
            # Validate inputs
            if not isinstance(residuals, np.ndarray):
                raise TypeError("Residuals must be a NumPy array")
            
            # Execute test
            dw_stat = durbin_watson(residuals)
            
            # Determine result (DW test doesn't have a p-value)
            # DW statistic interpretation: 
            # - 0 to 2: positive autocorrelation (0=perfect positive)
            # - 2: no autocorrelation
            # - 2 to 4: negative autocorrelation (4=perfect negative)
            if dw_stat < 1.5:
                result = "Positive autocorrelation detected"
            elif dw_stat > 2.5:
                result = "Negative autocorrelation detected"
            else:
                result = "No autocorrelation detected"
            
            # Format result
            stat_formatted = f"{dw_stat:.4f}"
            p_formatted = "N/A"  # DW test doesn't provide a p-value
            
            # Update results dictionary
            self._test_results["durbin_watson"] = {
                "statistic": dw_stat,
                "p_value": None,
                "result": result
            }
            
            # Update display
            self._update_results_display("Durbin-Watson", stat_formatted, p_formatted, result)
            
            # Show time series plot of residuals
            index_array = np.arange(len(residuals))
            await self._plot_display.display_residual_plot(residuals, index_array)
            
            logger.info(f"Durbin-Watson test completed: {result}")
            
        except Exception as e:
            logger.error(f"Error executing Durbin-Watson test: {str(e)}")
            self._handle_test_error("Durbin-Watson", str(e))
    
    async def run_white_test(self, residuals: np.ndarray, regressors: np.ndarray) -> None:
        """
        Executes White's test asynchronously and updates display.
        
        Args:
            residuals: Array of model residuals
            regressors: Matrix of regressors
            
        Returns:
            None: Updates display in place with test results
        """
        try:
            logger.info("Running White's test for heteroskedasticity")
            
            # Validate inputs
            if not isinstance(residuals, np.ndarray):
                raise TypeError("Residuals must be a NumPy array")
            
            if not isinstance(regressors, np.ndarray) or regressors.ndim != 2:
                raise TypeError("Regressors must be a 2D NumPy array")
            
            if len(residuals) != regressors.shape[0]:
                raise ValueError("Residuals and regressors must have the same number of observations")
            
            # Execute test
            white_stat, p_value = white_test(residuals, regressors)
            
            # Format results
            stat_formatted = f"{white_stat:.4f}"
            p_formatted = f"{p_value:.4f}"
            result = "Homoskedasticity" if p_value > 0.05 else "Heteroskedasticity detected"
            
            # Update results dictionary
            self._test_results["white_test"] = {
                "statistic": white_stat,
                "p_value": p_value,
                "result": result
            }
            
            # Update display
            self._update_results_display("White's Test", stat_formatted, p_formatted, result)
            
            # Update plot display
            squared_residuals = residuals**2
            fitted_values = regressors @ np.linalg.lstsq(regressors, residuals, rcond=None)[0]
            await self._plot_display.display_residual_plot(squared_residuals, fitted_values)
            
            logger.info(f"White's test completed: {result}")
            
        except Exception as e:
            logger.error(f"Error executing White's test: {str(e)}")
            self._handle_test_error("White's Test", str(e))
    
    def _update_results_display(self, test_name: str, statistic: str, p_value: str, result: str) -> None:
        """
        Updates the results grid with test results.
        
        Args:
            test_name: Name of the test
            statistic: Formatted test statistic
            p_value: Formatted p-value
            result: Test result interpretation
        """
        # Find existing row for this test or add a new one
        row = -1
        for i in range(1, self._results_grid.rowCount()):
            item = self._results_grid.itemAtPosition(i, 0)
            if item and item.widget() and item.widget().text() == test_name:
                row = i
                break
        
        # If row not found, add a new one
        if row == -1:
            row = self._results_grid.rowCount()
        
        # Create or update widgets in the row
        test_label = create_widget('QLabel', {'text': test_name})
        stat_label = create_widget('QLabel', {'text': statistic, 'alignment': Qt.AlignmentFlag.AlignCenter})
        p_label = create_widget('QLabel', {'text': p_value, 'alignment': Qt.AlignmentFlag.AlignCenter})
        result_label = create_widget('QLabel', {'text': result})
        
        # Style the labels
        test_label.setStyleSheet("padding: 4px;")
        stat_label.setStyleSheet("padding: 4px;")
        p_label.setStyleSheet("padding: 4px;")
        
        # Style result based on significance
        if "detected" in result.lower():
            result_label.setStyleSheet("color: red; padding: 4px;")
        else:
            result_label.setStyleSheet("color: green; padding: 4px;")
        
        # Add widgets to grid
        self._results_grid.addWidget(test_label, row, 0)
        self._results_grid.addWidget(stat_label, row, 1)
        self._results_grid.addWidget(p_label, row, 2)
        self._results_grid.addWidget(result_label, row, 3)
    
    def _handle_test_error(self, test_name: str, error_message: str) -> None:
        """
        Handles errors during test execution and updates display.
        
        Args:
            test_name: Name of the test
            error_message: Error message to display
        """
        row = self._results_grid.rowCount()
        
        # Create error message widgets
        test_label = create_widget('QLabel', {'text': test_name})
        error_label = create_widget('QLabel', {'text': "Error", 'alignment': Qt.AlignmentFlag.AlignCenter})
        error_label.setStyleSheet("color: red; font-weight: bold; padding: 4px;")
        
        empty_label = create_widget('QLabel', {'text': ""})
        message_label = create_widget('QLabel', {'text': error_message})
        message_label.setStyleSheet("color: red; padding: 4px;")
        
        # Add widgets to grid
        self._results_grid.addWidget(test_label, row, 0)
        self._results_grid.addWidget(error_label, row, 1)
        self._results_grid.addWidget(empty_label, row, 2)
        self._results_grid.addWidget(message_label, row, 3)
    
    def clear_results(self) -> None:
        """
        Clears all test results and resets display state.
        
        Returns:
            None: Clears display in place
        """
        try:
            logger.info("Clearing statistical test results")
            
            # Clear results dictionary
            self._test_results = {}
            
            # Clear grid except for header row
            for row in range(self._results_grid.rowCount() - 1, 0, -1):
                for col in range(self._results_grid.columnCount()):
                    item = self._results_grid.itemAtPosition(row, col)
                    if item and item.widget():
                        widget = item.widget()
                        self._results_grid.removeWidget(widget)
                        widget.deleteLater()
            
            # Clear plot display
            self._plot_display.clear_plots()
            
            logger.debug("Test results cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing test results: {str(e)}")