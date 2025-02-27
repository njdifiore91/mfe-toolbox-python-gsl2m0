"""
Main ARMAX application window implementation using PyQt6.

This module implements the primary user interface for the MFE Toolbox's ARMAX model
component, providing an interactive environment for model configuration, estimation,
and results visualization with comprehensive error handling and async support.
"""

import logging
import asyncio
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (  # PyQt6 version 6.6.1
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QMessageBox, QSplitter, QApplication, QLabel,
    QProgressDialog, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer  # PyQt6 version 6.6.1
import numpy as np  # numpy version 1.26.3

# Internal imports
from web.components.diagnostic_plots import DiagnosticPlotsWidget
from web.components.statistical_tests import StatisticalTests as StatisticalTestsWidget
from web.components.results_viewer import ResultsViewer
from web.components.model_config import ModelConfig

# Configure logger
logger = logging.getLogger(__name__)


class ARMAXMainWindow(QMainWindow):
    """
    Main application window for ARMAX model estimation and analysis with async support.
    
    This window integrates model configuration, diagnostic plots, statistical tests,
    and results viewing into a comprehensive interface for time series analysis using
    ARMA/ARMAX models with real-time visualization and async estimation.
    """
    
    def __init__(self, parent: Optional[QMainWindow] = None):
        """
        Initializes the main ARMAX application window with all required components.
        
        Args:
            parent: Optional parent window
        """
        super().__init__(parent)
        
        # Initialize properties
        self._model_config = None
        self._diagnostic_plots = None
        self._statistical_tests = None
        self._results_viewer = None
        self._estimate_button = None
        self._reset_button = None
        self._view_results_button = None
        self._close_button = None
        
        # Model-related properties
        self._armax_model = None
        self._estimation_task = None
        self._progress_dialog = None
        
        # Set up UI components
        self.setup_ui()
        
        # Set window properties
        self.setWindowTitle("ARMAX Model Estimation")
        self.resize(1000, 700)
        
        # Create status bar
        self.setStatusBar(QStatusBar())
        
        # Set up logging
        logger.info("ARMAX application window initialized")
        
    def setup_ui(self):
        """
        Creates and arranges all UI components with proper layout and styling.
        """
        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create model configuration section
        config_group = QGroupBox("Model Configuration")
        config_layout = QVBoxLayout(config_group)
        self._model_config = ModelConfig(self)
        config_layout.addWidget(self._model_config)
        
        # Create diagnostic plots section
        plots_group = QGroupBox("Diagnostic Plots")
        plots_layout = QVBoxLayout(plots_group)
        self._diagnostic_plots = DiagnosticPlotsWidget(self)
        plots_layout.addWidget(self._diagnostic_plots)
        
        # Create statistical tests section
        tests_group = QGroupBox("Statistical Tests")
        tests_layout = QVBoxLayout(tests_group)
        self._statistical_tests = StatisticalTestsWidget(self)
        tests_layout.addWidget(self._statistical_tests)
        
        # Create control buttons
        buttons_layout = QHBoxLayout()
        
        self._estimate_button = QPushButton("Estimate Model")
        self._estimate_button.clicked.connect(self.estimate_model)
        
        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self.reset_model)
        
        self._view_results_button = QPushButton("View Results")
        self._view_results_button.clicked.connect(self.view_results)
        self._view_results_button.setEnabled(False)  # Disabled until estimation is done
        
        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.close_window)
        
        buttons_layout.addWidget(self._estimate_button)
        buttons_layout.addWidget(self._reset_button)
        buttons_layout.addWidget(self._view_results_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self._close_button)
        
        # Create splitter for flexible layout
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Add configuration to top of splitter
        config_container = QWidget()
        config_container_layout = QVBoxLayout(config_container)
        config_container_layout.addWidget(config_group)
        config_container_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(config_container)
        
        # Add diagnostics to bottom of splitter
        diagnostics_container = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_container)
        diagnostics_layout.addWidget(plots_group)
        diagnostics_layout.addWidget(tests_group)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(diagnostics_container)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 400])
        
        # Add components to main layout
        main_layout.addWidget(splitter)
        main_layout.addLayout(buttons_layout)
        
        # Connect model config signals
        self._model_config.model_changed.connect(self._on_model_config_changed)
        self._model_config.config_error.connect(self._on_config_error)
        
        # Initialize results viewer (will be shown when needed)
        self._results_viewer = None
        
        # Set accessibility names and descriptions
        self.setAccessibleName("ARMAX Main Window")
        self.setAccessibleDescription("Main application window for ARMAX model estimation")
        
    @pyqtSlot()
    def estimate_model(self):
        """
        Handles model estimation asynchronously with progress updates.
        """
        try:
            # Validate model configuration
            config = self._model_config.get_config()
            if not config.get('is_valid', False):
                QMessageBox.warning(
                    self,
                    "Invalid Configuration",
                    "Please correct the model configuration errors before estimating."
                )
                return
            
            # Get model parameters
            parameters = config.get('parameters', {})
            
            # Create ARMAX model
            from backend.models.armax import ARMAX
            p = parameters.get('AR_ORDER', 1)
            q = parameters.get('MA_ORDER', 1)
            include_constant = parameters.get('CONSTANT', True)
            
            # Initialize model
            self._armax_model = ARMAX(p, q, include_constant)
            
            # In a real implementation, we would get actual data
            # For now, create simulated data
            data = np.random.randn(200)
            
            # Create and show progress dialog
            self._progress_dialog = QProgressDialog("Estimating model...", "Cancel", 0, 100, self)
            self._progress_dialog.setWindowTitle("Estimation Progress")
            self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self._progress_dialog.setValue(0)
            self._progress_dialog.show()
            
            # Disable buttons during estimation
            self._estimate_button.setEnabled(False)
            self._reset_button.setEnabled(False)
            
            # Start async estimation
            self._start_async_estimation(data)
            
        except Exception as e:
            logger.error(f"Error during model estimation: {str(e)}")
            QMessageBox.critical(
                self,
                "Estimation Error",
                f"An error occurred during model estimation:\n{str(e)}"
            )
            
            # Re-enable buttons
            self._estimate_button.setEnabled(True)
            self._reset_button.setEnabled(True)
    
    def _start_async_estimation(self, data: np.ndarray):
        """
        Starts the asynchronous model estimation process.
        
        Args:
            data: Time series data for model estimation
        """
        # Create the async task
        async def run_estimation():
            try:
                # Run the async fit method
                if self._armax_model:
                    converged = await self._armax_model.async_fit(data)
                    
                    # Update progress to 100%
                    if self._progress_dialog:
                        self._progress_dialog.setValue(100)
                    
                    # Process results
                    if converged:
                        # Get model residuals and fitted values
                        residuals = self._armax_model.residuals
                        fitted_values = self._armax_model._fitted
                        
                        # Update plots asynchronously
                        await self._diagnostic_plots.update_plots(residuals, fitted_values)
                        
                        # Run statistical tests asynchronously
                        await self._statistical_tests.run_ljung_box(residuals)
                        await self._statistical_tests.run_durbin_watson(residuals)
                        
                        # Show success message
                        QTimer.singleShot(0, lambda: self._show_estimation_complete(True))
                    else:
                        # Show warning about convergence issues
                        QTimer.singleShot(0, lambda: self._show_estimation_complete(False))
                    
                return converged
            except Exception as e:
                logger.error(f"Async estimation error: {str(e)}")
                # Show error dialog on the main thread
                QTimer.singleShot(0, lambda: self._show_estimation_error(str(e)))
                return False
            finally:
                # Update UI on the main thread
                QTimer.singleShot(0, self._update_ui_after_estimation)
        
        # Create and start the task
        self._estimation_task = asyncio.create_task(run_estimation())
        
        # Set up a timer to update progress periodically
        self._setup_progress_updates()
    
    def _setup_progress_updates(self):
        """
        Sets up periodic progress updates during estimation.
        """
        # Create a timer to update progress
        timer = QTimer(self)
        progress_value = 0
        
        def update_progress():
            nonlocal progress_value
            if self._progress_dialog and not self._progress_dialog.wasCanceled():
                # Increment progress value (simulated progress)
                progress_value = min(progress_value + 5, 99)  # Max 99% until complete
                self._progress_dialog.setValue(progress_value)
            else:
                # If canceled, stop the timer and cancel the task
                timer.stop()
                if self._estimation_task and not self._estimation_task.done():
                    self._estimation_task.cancel()
                self._update_ui_after_estimation()
        
        # Connect timer to update function and start it
        timer.timeout.connect(update_progress)
        timer.start(200)  # Update every 200ms
    
    def _show_estimation_complete(self, success: bool):
        """
        Shows the estimation completion message.
        
        Args:
            success: Whether estimation was successful
        """
        if success:
            QMessageBox.information(
                self,
                "Estimation Complete",
                "Model estimation completed successfully."
            )
        else:
            QMessageBox.warning(
                self,
                "Estimation Warning",
                "Model estimation completed with convergence issues."
            )
    
    def _show_estimation_error(self, error_message: str):
        """
        Shows an error message for estimation failures.
        
        Args:
            error_message: The error message to display
        """
        QMessageBox.critical(
            self,
            "Estimation Error",
            f"An error occurred during model estimation:\n{error_message}"
        )
    
    def _update_ui_after_estimation(self):
        """
        Updates the UI state after estimation is complete.
        """
        # Close progress dialog if it exists
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        
        # Re-enable buttons
        self._estimate_button.setEnabled(True)
        self._reset_button.setEnabled(True)
        
        # Enable results viewing if model exists
        if self._armax_model and hasattr(self._armax_model, 'params') and self._armax_model.params is not None:
            self._view_results_button.setEnabled(True)
        
        logger.info("Model estimation UI updated")
    
    @pyqtSlot()
    def reset_model(self):
        """
        Handles reset button click with proper cleanup.
        """
        try:
            # Reset model configuration
            self._model_config.reset_config()
            
            # Clear diagnostic plots
            self._diagnostic_plots.clear_plots()
            
            # Clear statistical tests
            self._statistical_tests.clear_results()
            
            # Reset model instance
            self._armax_model = None
            
            # Disable results viewing
            self._view_results_button.setEnabled(False)
            
            logger.info("Model reset completed")
            
        except Exception as e:
            logger.error(f"Error during model reset: {str(e)}")
            QMessageBox.critical(
                self,
                "Reset Error",
                f"An error occurred during reset:\n{str(e)}"
            )
    
    @pyqtSlot()
    def view_results(self):
        """
        Handles view results button click with proper state management.
        """
        try:
            # Check if we have a model
            if not self._armax_model or not hasattr(self._armax_model, 'params') or self._armax_model.params is None:
                QMessageBox.warning(
                    self,
                    "No Results",
                    "No model results available. Please estimate a model first."
                )
                return
            
            # Create results viewer if needed
            if self._results_viewer is None:
                self._results_viewer = ResultsViewer(self._armax_model, self)
            
            # Show results viewer
            self._results_viewer.show()
            self._results_viewer.raise_()
            
            logger.info("Results viewer displayed")
            
        except Exception as e:
            logger.error(f"Error displaying results: {str(e)}")
            QMessageBox.critical(
                self,
                "Results Error",
                f"An error occurred while displaying results:\n{str(e)}"
            )
    
    @pyqtSlot()
    def close_window(self):
        """
        Handles close button click with proper cleanup and state saving.
        """
        try:
            # Show confirmation dialog
            reply = QMessageBox.question(
                self,
                "Confirm Close",
                "Are you sure you want to close? Unsaved changes will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Cancel any ongoing estimation
                if self._estimation_task and not self._estimation_task.done():
                    self._estimation_task.cancel()
                
                # Clean up resources
                if self._results_viewer is not None:
                    self._results_viewer.close()
                
                # Close the main window
                self.close()
                
                logger.info("Application closed")
            
        except Exception as e:
            logger.error(f"Error closing application: {str(e)}")
            # Force close even if error occurs
            self.close()
    
    def _on_model_config_changed(self, config: Dict[str, Any]):
        """
        Handles model configuration changes.
        
        Args:
            config: Updated model configuration dictionary
        """
        # Enable or disable the estimate button based on configuration validity
        self._estimate_button.setEnabled(config.get('is_valid', False))
        
        logger.debug(f"Model configuration changed: {config}")
    
    def _on_config_error(self, error_message: str):
        """
        Handles configuration error messages.
        
        Args:
            error_message: Error message from configuration component
        """
        # Log the error
        logger.warning(f"Configuration error: {error_message}")
        
        # Update status bar
        self.statusBar().showMessage(f"Error: {error_message}", 5000)