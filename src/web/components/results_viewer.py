"""
PyQt6-based results viewer component for displaying ARMAX model estimation results.

This module provides a comprehensive viewer for model estimation results including
parameter estimates, diagnostic statistics, and interactive plots for assessing
model adequacy and fit.
"""

import logging
from typing import Optional, Tuple

import numpy as np  # version: 1.26.3
from PyQt6.QtWidgets import (  # version: 6.6.1
    QWidget, QVBoxLayout, QLabel, QTableWidget, QPushButton,
    QTableWidgetItem, QHeaderView, QGridLayout, QSizePolicy, QHBoxLayout, QFrame
)
from PyQt6.QtCore import Qt

# Internal imports
from components.plot_display import PlotDisplay
from utils.qt_helpers import create_widget
from backend.models.armax import ARMAX


class ResultsViewer(QWidget):
    """
    Interactive viewer for displaying ARMAX model estimation results.
    
    This widget provides a comprehensive view of model fit results including
    the model equation, parameter estimates with standard errors, diagnostic
    statistics, and interactive diagnostic plots with navigation capabilities.
    """
    
    def __init__(self, model: ARMAX, parent: Optional[QWidget] = None):
        """
        Initializes the results viewer with model results.
        
        Args:
            model: ARMAX model with estimation results
            parent: Parent widget for Qt hierarchy
        """
        super().__init__(parent)
        
        # Set up logger
        self._logger = logging.getLogger(__name__)
        
        # Store model reference
        self._model = model
        
        # Initialize properties
        self._equation_widget = None
        self._parameter_table = None
        self._metrics_widget = None
        self._plot_display = None
        self._current_page = 0
        
        # Create UI components and set up layout
        self._create_ui()
        
        # Display initial results
        self.display_equation()
        self.display_parameters()
        self.display_metrics()
        self.display_plots()
    
    def _create_ui(self):
        """Creates and arranges all UI components for the results viewer."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create equation display widget
        self._equation_widget = create_widget('QLabel', {
            'wordWrap': True,
            'alignment': Qt.AlignmentFlag.AlignCenter,
            'sizePolicy': QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        })
        equation_frame = create_widget('QFrame', {
            'frameShape': QFrame.Shape.StyledPanel
        })
        equation_layout = QVBoxLayout(equation_frame)
        equation_layout.addWidget(create_widget('QLabel', {
            'text': 'Model Equation',
            'alignment': Qt.AlignmentFlag.AlignCenter,
            'font': 'bold'
        }))
        equation_layout.addWidget(self._equation_widget)
        main_layout.addWidget(equation_frame)
        
        # Create parameter table
        self._parameter_table = create_widget('QTableWidget', {
            'columnCount': 5,
            'rowCount': 0,
            'horizontalHeaderLabels': ['Parameter', 'Estimate', 'Std. Error', 't-statistic', 'p-value'],
            'sizePolicy': QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        })
        self._parameter_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Create metrics widget
        self._metrics_widget = create_widget('QWidget', {
            'sizePolicy': QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        })
        metrics_layout = QGridLayout(self._metrics_widget)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        metrics_layout.setSpacing(10)
        
        # Create plot display
        self._plot_display = PlotDisplay(self)
        
        # Create page container
        self._page_container = QWidget()
        page_layout = QVBoxLayout(self._page_container)
        page_layout.setContentsMargins(0, 0, 0, 0)
        
        # Page 1: Parameters and metrics
        self._page1 = QWidget()
        page1_layout = QVBoxLayout(self._page1)
        page1_layout.addWidget(create_widget('QLabel', {
            'text': 'Parameter Estimates',
            'alignment': Qt.AlignmentFlag.AlignCenter,
            'font': 'bold'
        }))
        page1_layout.addWidget(self._parameter_table)
        page1_layout.addWidget(create_widget('QLabel', {
            'text': 'Statistical Metrics',
            'alignment': Qt.AlignmentFlag.AlignCenter,
            'font': 'bold'
        }))
        page1_layout.addWidget(self._metrics_widget)
        
        # Page 2: Plots
        self._page2 = QWidget()
        page2_layout = QVBoxLayout(self._page2)
        page2_layout.addWidget(create_widget('QLabel', {
            'text': 'Diagnostic Plots',
            'alignment': Qt.AlignmentFlag.AlignCenter,
            'font': 'bold'
        }))
        page2_layout.addWidget(self._plot_display)
        
        # Start with page 1
        page_layout.addWidget(self._page1)
        
        # Add page container to main layout
        main_layout.addWidget(self._page_container)
        
        # Create navigation buttons
        nav_layout = QHBoxLayout()
        self._prev_button = create_widget('QPushButton', {'text': '< Previous'})
        self._next_button = create_widget('QPushButton', {'text': 'Next >'})
        self._page_label = create_widget('QLabel', {
            'text': '1/2',
            'alignment': Qt.AlignmentFlag.AlignCenter
        })
        nav_layout.addWidget(self._prev_button)
        nav_layout.addWidget(self._page_label)
        nav_layout.addWidget(self._next_button)
        main_layout.addLayout(nav_layout)
        
        # Connect navigation signals
        self._prev_button.clicked.connect(lambda: self.navigate('prev'))
        self._next_button.clicked.connect(lambda: self.navigate('next'))
        
        # Initial button states
        self._prev_button.setEnabled(False)
    
    def display_equation(self):
        """
        Displays the model equation with estimated parameters.
        
        Updates equation display to show the ARMA/ARMAX structure with
        estimated parameter values formatted for easy interpretation.
        """
        try:
            # Get AR and MA parameters
            ar_params, ma_params, constant, exog_params = self._extract_params()
            
            # Create equation string
            equation_parts = []
            
            # Left side of equation
            equation_parts.append("y<sub>t</sub> = ")
            
            # Constant term
            if constant is not None:
                equation_parts.append(f"{constant:.4f}")
            
            # AR terms
            if len(ar_params) > 0:
                for i, param in enumerate(ar_params):
                    sign = "+" if param > 0 else ""
                    equation_parts.append(f" {sign} {param:.4f} y<sub>t-{i+1}</sub>")
            
            # MA terms
            if len(ma_params) > 0:
                for i, param in enumerate(ma_params):
                    sign = "+" if param > 0 else ""
                    equation_parts.append(f" {sign} {param:.4f} ε<sub>t-{i+1}</sub>")
            
            # Exogenous variables
            if len(exog_params) > 0:
                for i, param in enumerate(exog_params):
                    sign = "+" if param > 0 else ""
                    equation_parts.append(f" {sign} {param:.4f} x<sub>{i+1,t}</sub>")
            
            # Error term
            equation_parts.append(" + ε<sub>t</sub>")
            
            # Combine and set text
            equation_text = "".join(equation_parts)
            self._equation_widget.setText(equation_text)
            
            # Set font for better display
            font = self._equation_widget.font()
            font.setPointSize(12)
            self._equation_widget.setFont(font)
            
            self._logger.debug("Equation display updated")
            
        except Exception as e:
            self._logger.error(f"Error displaying equation: {str(e)}")
            self._equation_widget.setText("Error displaying equation")
    
    def display_parameters(self):
        """
        Displays parameter estimates and standard errors in table.
        
        Updates the parameter table with estimates, standard errors,
        t-statistics, and p-values for all model parameters.
        """
        try:
            # Get parameter summary from diagnostic tests
            diagnostics = self._model.diagnostic_tests()
            param_summary = diagnostics.get('parameter_summary', [])
            
            # Clear existing table
            self._parameter_table.setRowCount(0)
            
            # Add parameters to table
            self._parameter_table.setRowCount(len(param_summary))
            
            for i, param in enumerate(param_summary):
                # Parameter name
                name_item = QTableWidgetItem(param['name'])
                self._parameter_table.setItem(i, 0, name_item)
                
                # Parameter estimate
                value_item = QTableWidgetItem(f"{param['value']:.4f}")
                value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._parameter_table.setItem(i, 1, value_item)
                
                # Standard error, t-statistic, p-value
                if param['std_error'] is not None:
                    se_item = QTableWidgetItem(f"{param['std_error']:.4f}")
                    se_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self._parameter_table.setItem(i, 2, se_item)
                    
                    t_item = QTableWidgetItem(f"{param['t_statistic']:.4f}")
                    t_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self._parameter_table.setItem(i, 3, t_item)
                    
                    p_item = QTableWidgetItem(f"{param['p_value']:.4f}")
                    p_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    # Highlight significant parameters
                    if param['p_value'] < 0.05:
                        p_item.setForeground(Qt.GlobalColor.blue)
                    self._parameter_table.setItem(i, 4, p_item)
                else:
                    # Add placeholders for missing statistics
                    for col in range(2, 5):
                        self._parameter_table.setItem(i, col, QTableWidgetItem("--"))
            
            self._logger.debug("Parameter table updated")
            
        except Exception as e:
            self._logger.error(f"Error displaying parameters: {str(e)}")
            self._parameter_table.setRowCount(1)
            self._parameter_table.setItem(0, 0, QTableWidgetItem("Error displaying parameters"))
    
    def display_metrics(self):
        """
        Displays model fit metrics and diagnostic statistics.
        
        Updates the metrics display with log-likelihood, information criteria,
        and diagnostic test results to assess model adequacy.
        """
        try:
            # Get diagnostic test results
            diagnostics = self._model.diagnostic_tests()
            
            # Clear existing layout
            layout = self._metrics_widget.layout()
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Add information criteria
            layout.addWidget(create_widget('QLabel', {'text': 'Information Criteria:'}), 0, 0)
            layout.addWidget(create_widget('QLabel', {'text': f"AIC: {diagnostics['AIC']:.4f}"}), 0, 1)
            layout.addWidget(create_widget('QLabel', {'text': f"BIC: {diagnostics['BIC']:.4f}"}), 0, 2)
            
            # Add log-likelihood
            layout.addWidget(create_widget('QLabel', {'text': 'Log-Likelihood:'}), 1, 0)
            layout.addWidget(create_widget('QLabel', {'text': f"{self._model.loglikelihood:.4f}"}), 1, 1)
            
            # Add diagnostic tests
            layout.addWidget(create_widget('QLabel', {'text': 'Diagnostic Tests:'}), 2, 0)
            
            # Ljung-Box test
            lb_test = diagnostics['ljung_box']
            lb_text = f"Ljung-Box Q({lb_test['lags']}): {lb_test['statistic']:.4f} [p: {lb_test['p_value']:.4f}]"
            layout.addWidget(create_widget('QLabel', {'text': lb_text}), 2, 1, 1, 2)
            
            # Jarque-Bera test
            jb_test = diagnostics['jarque_bera']
            jb_text = f"Jarque-Bera: {jb_test['statistic']:.4f} [p: {jb_test['p_value']:.4f}]"
            layout.addWidget(create_widget('QLabel', {'text': jb_text}), 3, 1, 1, 2)
            
            self._logger.debug("Metrics display updated")
            
        except Exception as e:
            self._logger.error(f"Error displaying metrics: {str(e)}")
            layout = self._metrics_widget.layout()
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            layout.addWidget(create_widget('QLabel', {'text': f"Error displaying metrics: {str(e)}"}), 0, 0)
    
    def display_plots(self):
        """
        Displays diagnostic plots for model fit.
        
        Updates the plot display with residual diagnostic plots and
        other visualizations to assess model adequacy.
        """
        try:
            # Get model residuals and fitted values
            residuals = self._model.residuals
            fitted_values = self._model._fitted
            
            if residuals is None or fitted_values is None:
                raise ValueError("Model residuals or fitted values not available")
            
            # Use display_diagnostic_plots to show comprehensive diagnostics
            import asyncio
            asyncio.create_task(self._plot_display.display_diagnostic_plots(residuals, fitted_values))
            
            self._logger.debug("Plot display update initiated")
            
        except Exception as e:
            self._logger.error(f"Error displaying plots: {str(e)}")
            self._plot_display.clear_plots()
    
    def navigate(self, direction: str):
        """
        Handles navigation between result pages.
        
        Updates display to show different result pages based on
        navigation direction.
        
        Args:
            direction: Navigation direction ('prev' or 'next')
        """
        # Total number of pages
        total_pages = 2  # Parameters and Plots
        
        # Update current page based on direction
        if direction == 'prev' and self._current_page > 0:
            self._current_page -= 1
        elif direction == 'next' and self._current_page < total_pages - 1:
            self._current_page += 1
        
        # Clear current page container
        layout = self._page_container.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        # Add new page based on current index
        if self._current_page == 0:
            layout.addWidget(self._page1)
        else:
            layout.addWidget(self._page2)
        
        # Update page indicator
        self._page_label.setText(f"{self._current_page + 1}/{total_pages}")
        
        # Update button states
        self._prev_button.setEnabled(self._current_page > 0)
        self._next_button.setEnabled(self._current_page < total_pages - 1)
        
        self._logger.debug(f"Navigated to page {self._current_page + 1}")
    
    def _extract_params(self) -> Tuple[np.ndarray, np.ndarray, Optional[float], np.ndarray]:
        """
        Extracts model parameters from the param vector.
        
        Retrieves AR parameters, MA parameters, constant term, and exogenous
        variable parameters from the model parameter vector.
        
        Returns:
            Tuple containing (ar_params, ma_params, constant, exog_params)
        """
        # Call the model's internal _extract_params method if available
        if hasattr(self._model, '_extract_params'):
            return self._model._extract_params(self._model.params)
        
        # Fallback extraction logic if not available
        try:
            params = self._model.params
            
            # Extract parameter counts
            p = int(params[0])
            q = int(params[1])
            has_constant = params[2] > 0.5
            
            # Extract parameters
            idx = 3
            ar_params = params[idx:idx+p] if p > 0 else np.empty(0)
            idx += p
            
            ma_params = params[idx:idx+q] if q > 0 else np.empty(0)
            idx += q
            
            constant = params[idx] if has_constant else None
            idx += 1 if has_constant else 0
            
            exog_params = params[idx:] if idx < len(params) else np.empty(0)
            
            return ar_params, ma_params, constant, exog_params
            
        except Exception as e:
            self._logger.error(f"Error extracting parameters: {str(e)}")
            return np.empty(0), np.empty(0), None, np.empty(0)