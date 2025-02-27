"""
PyQt6-based results viewer dialog for displaying ARMAX model estimation results,
parameter estimates, diagnostic statistics and interactive plots. Provides 
comprehensive visualization of model outputs with navigation capabilities.
"""

import logging
import numpy as np  # version: 1.26.3
from PyQt6.QtWidgets import (  # version: 6.6.1
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, 
    QTableWidgetItem, QFrame, QPushButton, QWidget, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot  # version: 6.6.1

from ..models.armax import ARMAX
from .residual_plot import ResidualPlotWidget

# Configure logger
logger = logging.getLogger(__name__)


class ARMAXResultsViewer(QDialog):
    """
    Dialog for displaying ARMAX model estimation results with interactive navigation.
    
    This dialog shows model equation, parameter estimates, statistical metrics,
    and diagnostic plots with navigation capabilities for comprehensive model
    evaluation.
    """
    
    def __init__(self, parent=None):
        """
        Initializes the results viewer dialog.
        
        Parameters
        ----------
        parent : Optional[QDialog]
            The parent widget, if any
        """
        super().__init__(parent)
        
        # Initialize instance variables
        self._equation_label = None
        self._parameter_table = None
        self._likelihood_label = None
        self._criteria_label = None
        self._residual_plots = None
        self._prev_button = None
        self._next_button = None
        self._page_label = None
        self._close_button = None
        self._current_page = 0
        
        # Set up the UI components
        self.setup_ui()
        
        # Set window properties
        self.setWindowTitle("ARMAX Model Results")
        self.resize(800, 600)
        
        logger.debug("ARMAXResultsViewer initialized")
    
    def setup_ui(self):
        """
        Creates and arranges all UI components.
        """
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create section for model equation
        equation_frame = QFrame()
        equation_layout = QVBoxLayout(equation_frame)
        equation_title = QLabel("<b>Model Equation</b>")
        self._equation_label = QLabel("Equation will appear here")
        self._equation_label.setFrameShape(QFrame.Shape.Panel)
        self._equation_label.setFrameShadow(QFrame.Shadow.Sunken)
        self._equation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._equation_label.setMinimumHeight(60)
        equation_layout.addWidget(equation_title)
        equation_layout.addWidget(self._equation_label)
        
        # Create parameter estimates table
        params_frame = QFrame()
        params_layout = QVBoxLayout(params_frame)
        params_title = QLabel("<b>Parameter Estimates</b>")
        self._parameter_table = QTableWidget()
        self._parameter_table.setColumnCount(5)
        self._parameter_table.setHorizontalHeaderLabels(
            ["Parameter", "Estimate", "Std.Error", "t-stat", "p-value"]
        )
        self._parameter_table.horizontalHeader().setStretchLastSection(True)
        params_layout.addWidget(params_title)
        params_layout.addWidget(self._parameter_table)
        
        # Create statistical metrics section
        stats_frame = QFrame()
        stats_layout = QGridLayout(stats_frame)
        stats_title = QLabel("<b>Statistical Metrics</b>")
        
        # Log-likelihood
        ll_title = QLabel("Log-Likelihood:")
        self._likelihood_label = QLabel("-")
        stats_layout.addWidget(ll_title, 0, 0)
        stats_layout.addWidget(self._likelihood_label, 0, 1)
        
        # Information criteria
        ic_title = QLabel("Information Criteria:")
        self._criteria_label = QLabel("AIC: - | BIC: -")
        stats_layout.addWidget(ic_title, 1, 0)
        stats_layout.addWidget(self._criteria_label, 1, 1)
        
        # Assemble metrics section
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(stats_title)
        metrics_layout.addLayout(stats_layout)
        stats_frame.setLayout(metrics_layout)
        
        # Create residual plots widget
        plots_frame = QFrame()
        plots_layout = QVBoxLayout(plots_frame)
        plots_title = QLabel("<b>Diagnostic Plots</b>")
        self._residual_plots = ResidualPlotWidget(self)
        plots_layout.addWidget(plots_title)
        plots_layout.addWidget(self._residual_plots)
        
        # Create navigation buttons
        nav_layout = QHBoxLayout()
        
        # Previous button
        self._prev_button = QPushButton("< Previous")
        self._prev_button.setEnabled(False)  # Disabled by default
        self._prev_button.clicked.connect(lambda: self.navigate_plots("prev"))
        
        # Page indicator
        self._page_label = QLabel("1/1")
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Next button
        self._next_button = QPushButton("Next >")
        self._next_button.setEnabled(False)  # Disabled by default
        self._next_button.clicked.connect(lambda: self.navigate_plots("next"))
        
        # Close button
        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.close_dialog)
        
        # Add buttons to layout
        nav_layout.addWidget(self._prev_button)
        nav_layout.addWidget(self._page_label)
        nav_layout.addWidget(self._next_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self._close_button)
        
        # Add all components to main layout
        main_layout.addWidget(equation_frame)
        main_layout.addWidget(params_frame)
        main_layout.addWidget(stats_frame)
        main_layout.addWidget(plots_frame)
        main_layout.addLayout(nav_layout)
    
    @pyqtSlot()
    def display_results(self, model):
        """
        Updates the dialog with current model results.
        
        Parameters
        ----------
        model : ARMAX
            The fitted ARMAX model containing results to display
        """
        try:
            if model.params is None:
                logger.warning("Cannot display results: Model has not been fit")
                return
            
            # Update model equation
            self._update_equation(model)
            
            # Update parameter table
            self._update_parameter_table(model)
            
            # Update statistical metrics
            self._update_metrics(model)
            
            # Update residual plots
            self._update_plots(model)
            
            logger.debug("Results display updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating results display: {str(e)}")
            # Show error message in the UI
            self._equation_label.setText(f"Error displaying results: {str(e)}")
    
    def _update_equation(self, model):
        """
        Updates the model equation display.
        
        Parameters
        ----------
        model : ARMAX
            The fitted ARMAX model
        """
        try:
            # Extract parameters
            ar_params, ma_params, constant, exog_params = model._extract_params(model.params)
            
            # Construct equation text
            equation = "y(t) = "
            
            # Add constant term if present
            if constant is not None:
                equation += f"{constant:.4f}"
            
            # Add AR terms
            for i, param in enumerate(ar_params):
                sign = "+" if param >= 0 else ""
                equation += f" {sign} {param:.4f}·y(t-{i+1})"
            
            # Add MA terms
            for i, param in enumerate(ma_params):
                sign = "+" if param >= 0 else ""
                equation += f" {sign} {param:.4f}·ε(t-{i+1})"
            
            # Add exogenous terms if present
            if len(exog_params) > 0:
                for i, param in enumerate(exog_params):
                    sign = "+" if param >= 0 else ""
                    equation += f" {sign} {param:.4f}·x{i+1}(t)"
            
            # Add error term
            equation += " + ε(t)"
            
            self._equation_label.setText(equation)
            
        except Exception as e:
            logger.error(f"Error updating equation: {str(e)}")
            self._equation_label.setText("Error: Could not display equation")
    
    def _update_parameter_table(self, model):
        """
        Updates the parameter estimates table.
        
        Parameters
        ----------
        model : ARMAX
            The fitted ARMAX model
        """
        try:
            # Get diagnostic results for parameter summary
            diagnostics = model.diagnostic_tests()
            
            if 'parameter_summary' not in diagnostics:
                logger.warning("No parameter summary available in diagnostics")
                return
            
            # Get parameter summary
            param_summary = diagnostics['parameter_summary']
            
            # Set up table
            self._parameter_table.setRowCount(len(param_summary))
            
            # Fill parameter table
            for i, param in enumerate(param_summary):
                # Parameter name
                name_item = QTableWidgetItem(param['name'])
                self._parameter_table.setItem(i, 0, name_item)
                
                # Estimate value
                value_item = QTableWidgetItem(f"{param['value']:.4f}")
                self._parameter_table.setItem(i, 1, value_item)
                
                # Standard error
                if param['std_error'] is not None:
                    std_err_item = QTableWidgetItem(f"{param['std_error']:.4f}")
                else:
                    std_err_item = QTableWidgetItem("N/A")
                self._parameter_table.setItem(i, 2, std_err_item)
                
                # t-statistic
                if param['t_statistic'] is not None:
                    t_stat_item = QTableWidgetItem(f"{param['t_statistic']:.4f}")
                else:
                    t_stat_item = QTableWidgetItem("N/A")
                self._parameter_table.setItem(i, 3, t_stat_item)
                
                # p-value
                if param['p_value'] is not None:
                    p_value_item = QTableWidgetItem(f"{param['p_value']:.4f}")
                else:
                    p_value_item = QTableWidgetItem("N/A")
                self._parameter_table.setItem(i, 4, p_value_item)
            
            # Adjust column widths
            self._parameter_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error updating parameter table: {str(e)}")
    
    def _update_metrics(self, model):
        """
        Updates the statistical metrics display.
        
        Parameters
        ----------
        model : ARMAX
            The fitted ARMAX model
        """
        try:
            # Get diagnostic results
            diagnostics = model.diagnostic_tests()
            
            # Update log-likelihood
            if model.loglikelihood is not None:
                self._likelihood_label.setText(f"{model.loglikelihood:.4f}")
            else:
                self._likelihood_label.setText("N/A")
            
            # Update information criteria
            if 'AIC' in diagnostics and 'BIC' in diagnostics:
                aic = diagnostics['AIC']
                bic = diagnostics['BIC']
                self._criteria_label.setText(f"AIC: {aic:.4f} | BIC: {bic:.4f}")
            else:
                self._criteria_label.setText("Information criteria not available")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    def _update_plots(self, model):
        """
        Updates the residual plots display.
        
        Parameters
        ----------
        model : ARMAX
            The fitted ARMAX model
        """
        try:
            # Get diagnostic results
            diagnostics = model.diagnostic_tests()
            
            # Update plots
            if model.residuals is not None:
                self._residual_plots.update_plots(model.residuals, diagnostics)
                
                # For this implementation, we're setting up two pages
                # Page 1: Main diagnostic plots (already shown by default)
                # Page 2: Additional diagnostic information could be added
                total_pages = 2
                
                # Enable/disable navigation buttons based on available pages
                self._current_page = 0
                self._prev_button.setEnabled(False)
                self._next_button.setEnabled(total_pages > 1)
                self._page_label.setText(f"1/{total_pages}")
            else:
                self._residual_plots.clear_plots()
                logger.warning("No residuals available for plotting")
            
        except Exception as e:
            logger.error(f"Error updating plots: {str(e)}")
            # Clear plots in case of error
            self._residual_plots.clear_plots()
    
    @pyqtSlot()
    def navigate_plots(self, direction):
        """
        Handles plot navigation button clicks.
        
        Parameters
        ----------
        direction : str
            Navigation direction, either "prev" or "next"
        """
        # Get current number of pages
        total_pages = 2  # For this implementation, we have 2 pages
        
        if direction == "prev" and self._current_page > 0:
            self._current_page -= 1
        elif direction == "next" and self._current_page < total_pages - 1:
            self._current_page += 1
        
        # Update button states
        self._prev_button.setEnabled(self._current_page > 0)
        self._next_button.setEnabled(self._current_page < total_pages - 1)
        
        # Update page indicator
        self._page_label.setText(f"{self._current_page + 1}/{total_pages}")
        
        # Update displayed content based on current page
        if self._current_page == 0:
            # First page: show standard residual plots (already shown by default)
            # Make sure plots are visible
            self._residual_plots.setVisible(True)
        elif self._current_page == 1:
            # Second page: could show additional diagnostic information
            # This is a placeholder - in a real implementation, you would
            # display additional content here
            pass
        
        logger.debug(f"Navigated to page {self._current_page + 1}")
    
    @pyqtSlot()
    def close_dialog(self):
        """
        Handles dialog close button click.
        """
        try:
            # Clean up resources
            if self._residual_plots:
                self._residual_plots.clear_plots()
            
            # Close the dialog
            self.accept()
            
            logger.debug("Dialog closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing dialog: {str(e)}")
            # Force close in case of error
            self.reject()