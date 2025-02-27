"""
Main application window for the MFE Toolbox GUI, implementing a PyQt6-based interface 
for time series model estimation, diagnostics and results visualization. Provides a 
comprehensive interface for model configuration, estimation, and analysis.
"""

import logging
import asyncio
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, 
    QMenuBar, QStatusBar, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QCloseEvent

# Internal imports
from components.model_config import ModelConfig
from components.diagnostic_plots import DiagnosticPlotsWidget
from components.results_viewer import ResultsViewer
from dialogs.about_dialog import AboutDialog
from dialogs.close_dialog import CloseDialog
from utils.qt_helpers import create_widget

# Configure logger
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window for the MFE Toolbox GUI"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the main window with all required components
        
        Args:
            parent: Optional parent widget
        """
        # Initialize QMainWindow
        super().__init__(parent)
        
        # Initialize properties
        self._model_config: ModelConfig = None
        self._diagnostic_plots: DiagnosticPlotsWidget = None
        self._results_viewer: ResultsViewer = None
        self._menu_bar: QMenuBar = None
        self._status_bar: QStatusBar = None
        self._has_unsaved_changes: bool = False
        
        # Setup UI components
        self.setup_ui()
        
        # Setup menu bar
        self.setup_menu_bar()
        
        # Initialize window state
        self.setWindowTitle("MFE Toolbox - ARMAX Model Estimation")
        self.resize(1000, 800)
        self.setMinimumSize(800, 600)
        
        # Show ready status
        self._status_bar.showMessage("Ready")
        
        logger.debug("MainWindow initialized")
        
    def setup_ui(self):
        """
        Creates and arranges the main window UI components
        """
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create model configuration section
        config_section = create_widget("QWidget", {})
        config_layout = QVBoxLayout(config_section)
        config_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add heading for model configuration
        config_heading = create_widget("QLabel", {
            "text": "Model Configuration",
            "alignment": int(Qt.AlignmentFlag.AlignCenter),
            "font": "bold"
        })
        config_layout.addWidget(config_heading)
        
        # Create model configuration widget
        self._model_config = ModelConfig(self)
        config_layout.addWidget(self._model_config)
        
        main_layout.addWidget(config_section)
        
        # Create diagnostic plots section
        plots_section = create_widget("QWidget", {})
        plots_layout = QVBoxLayout(plots_section)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add heading for diagnostic plots
        plots_heading = create_widget("QLabel", {
            "text": "Diagnostic Plots",
            "alignment": int(Qt.AlignmentFlag.AlignCenter),
            "font": "bold"
        })
        plots_layout.addWidget(plots_heading)
        
        # Create diagnostic plots widget
        self._diagnostic_plots = DiagnosticPlotsWidget(self)
        plots_layout.addWidget(self._diagnostic_plots)
        
        main_layout.addWidget(plots_section)
        main_layout.setStretchFactor(plots_section, 2)  # Give plots more space
        
        # Create button section
        button_section = create_widget("QWidget", {})
        button_layout = QHBoxLayout(button_section)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        
        # Create buttons
        estimate_button = create_widget("QPushButton", {
            "text": "Estimate Model",
            "toolTip": "Estimate model with current configuration"
        })
        
        reset_button = create_widget("QPushButton", {
            "text": "Reset",
            "toolTip": "Reset all settings to defaults"
        })
        
        view_results_button = create_widget("QPushButton", {
            "text": "View Results",
            "toolTip": "View detailed model results"
        })
        
        close_button = create_widget("QPushButton", {
            "text": "Close",
            "toolTip": "Close application"
        })
        
        # Add buttons to layout
        button_layout.addWidget(estimate_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(view_results_button)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        
        main_layout.addWidget(button_section)
        
        # Connect button signals
        estimate_button.clicked.connect(self._on_estimate_button_clicked)
        reset_button.clicked.connect(self.on_reset_clicked)
        view_results_button.clicked.connect(self._show_results)
        close_button.clicked.connect(self.close)
        
        # Connect model configuration signals
        self._model_config.model_changed.connect(self._on_model_changed)
        
        # Create status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        
    def setup_menu_bar(self):
        """
        Creates and configures the main window menu bar
        """
        self._menu_bar = QMenuBar()
        self.setMenuBar(self._menu_bar)
        
        # Create File menu
        file_menu = self._menu_bar.addMenu("&File")
        
        # Add actions to File menu
        new_action = file_menu.addAction("&New")
        new_action.setShortcut("Ctrl+N")
        new_action.setStatusTip("Create a new model")
        
        open_action = file_menu.addAction("&Open...")
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open an existing model")
        
        save_action = file_menu.addAction("&Save...")
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save the current model")
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Alt+F4")
        exit_action.setStatusTip("Exit the application")
        
        # Connect actions
        new_action.triggered.connect(self.on_reset_clicked)
        exit_action.triggered.connect(self.close)
        
        # Create Help menu
        help_menu = self._menu_bar.addMenu("&Help")
        
        # Add actions to Help menu
        about_action = help_menu.addAction("&About")
        about_action.setStatusTip("Show information about the application")
        
        # Connect actions
        about_action.triggered.connect(self.show_about_dialog)
    
    def _on_model_changed(self, config):
        """
        Handles model configuration changes
        
        Args:
            config: Updated model configuration
        """
        self._has_unsaved_changes = True
        
        # Update status message based on config validity
        if config.get('is_valid', False):
            self._status_bar.showMessage("Model configuration updated")
        else:
            self._status_bar.showMessage("Warning: Current model configuration is invalid")
    
    def _on_estimate_button_clicked(self):
        """
        Handles the estimate button click by starting an async task
        """
        logger.debug("Estimate button clicked")
        # Create an async task to handle the estimation
        asyncio.create_task(self.on_estimate_clicked())
    
    async def on_estimate_clicked(self):
        """
        Handles model estimation button click asynchronously
        """
        try:
            logger.debug("Starting model estimation")
            
            # Get model configuration
            config = self._model_config.get_config()
            
            if not config.get('is_valid', False):
                self._status_bar.showMessage("Invalid model configuration. Please check parameters.")
                return
            
            # Update status bar
            self._status_bar.showMessage("Estimating model...")
            
            # In a real implementation, this would call the actual model estimation
            # For now, simulate with a delay and generate synthetic data
            for progress in range(1, 11):
                # Simulate estimation progress
                await asyncio.sleep(0.2)  # Simulate computation time
                self._status_bar.showMessage(f"Estimating model... {progress * 10}%")
            
            # Generate synthetic data for testing
            import numpy as np
            residuals = np.random.normal(0, 1, 500)
            fitted_values = np.linspace(-2, 2, 500) + np.random.normal(0, 0.5, 500)
            
            # Update plots with the results
            await self._diagnostic_plots.update_plots(residuals, fitted_values)
            
            # Create or update results viewer
            if self._results_viewer is None:
                # In a real implementation, this would use the actual estimated model
                from backend.models.armax import ARMAX
                
                # Create a simple model for demonstration
                model = ARMAX(p=1, q=1)
                # Setup synthetic parameters 
                model.params = np.array([1, 1, 1, 0.7, -0.3, 0.01])  # p, q, const_flag, AR, MA, const
                model.residuals = residuals
                model._fitted = fitted_values
                model.loglikelihood = -250.0
                model.standard_errors = np.array([0, 0, 0, 0.04, 0.05, 0.01])
                
                # Create results viewer
                self._results_viewer = ResultsViewer(model, self)
            
            # Update status bar
            self._status_bar.showMessage("Model estimation completed successfully")
            
            # Reset unsaved changes flag
            self._has_unsaved_changes = False
            
            # Show results automatically after estimation
            self._show_results()
            
        except Exception as e:
            logger.error(f"Error during model estimation: {str(e)}")
            self._status_bar.showMessage(f"Error during estimation: {str(e)}")
    
    def on_reset_clicked(self):
        """
        Handles reset button click
        """
        try:
            logger.debug("Reset button clicked")
            
            # Reset model configuration
            self._model_config.reset_config()
            
            # Clear diagnostic plots
            self._diagnostic_plots.clear_plots()
            
            # Reset results viewer
            self._results_viewer = None
            
            # Reset window state
            self._has_unsaved_changes = False
            
            # Update status bar
            self._status_bar.showMessage("Reset completed")
            
        except Exception as e:
            logger.error(f"Error during reset: {str(e)}")
            self._status_bar.showMessage(f"Error during reset: {str(e)}")
    
    def _show_results(self):
        """
        Shows the results viewer dialog
        """
        if self._results_viewer is not None:
            logger.debug("Showing results viewer")
            self._results_viewer.show()
        else:
            logger.debug("No results available")
            self._status_bar.showMessage("No results available. Please estimate a model first.")
    
    def closeEvent(self, event: QCloseEvent):
        """
        Handles window close event with confirmation if needed
        
        Args:
            event: QCloseEvent object
        """
        logger.debug("Close event triggered")
        
        # Check for unsaved changes
        if self._has_unsaved_changes:
            logger.debug("Unsaved changes detected, showing confirmation dialog")
            
            # Show confirmation dialog
            close_dialog = CloseDialog(self)
            if close_dialog.show_dialog():
                # User confirmed close
                logger.debug("User confirmed close")
                event.accept()
            else:
                # User canceled close
                logger.debug("User canceled close")
                event.ignore()
        else:
            # No unsaved changes, close directly
            logger.debug("No unsaved changes, closing directly")
            event.accept()
    
    def show_about_dialog(self):
        """
        Shows the About dialog
        """
        logger.debug("Showing About dialog")
        about_dialog = AboutDialog(self)
        about_dialog.exec()