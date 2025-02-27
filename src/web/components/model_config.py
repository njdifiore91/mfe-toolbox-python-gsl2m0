"""
Model configuration component for the MFE Toolbox GUI.

This module provides a PyQt6-based widget for configuring time series and
volatility model parameters, providing a unified interface for model specification
with real-time validation and error feedback.
"""

import logging
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QComboBox  # PyQt6 version 6.6.1
from PyQt6.QtCore import pyqtSignal  # PyQt6 version 6.6.1

# Internal imports
from web.components.parameter_input import ParameterInput
from web.utils.validation import validate_widget_input
from web.utils.qt_helpers import create_widget

# Configure logger
logger = logging.getLogger(__name__)

# Model type definitions
MODEL_TYPES = {
    'ARMA': 'ARMA Model',
    'GARCH': 'GARCH Model',
    'EGARCH': 'EGARCH Model',
    'APARCH': 'APARCH Model',
    'FIGARCH': 'FIGARCH Model'
}


class ModelConfig(QWidget):
    """
    Widget for configuring time series and volatility model parameters.
    
    This widget provides a unified interface for specifying model parameters
    with real-time validation and error feedback. It allows selection of model
    type and configuration of model-specific parameters.
    
    Signals:
        model_changed (dict): Emitted when model configuration changes
        config_error (str): Emitted when configuration has errors
    """
    
    # Signals
    model_changed = pyqtSignal(dict)  # Emitted when model configuration changes
    config_error = pyqtSignal(str)    # Emitted when configuration has errors
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes model configuration widget with parameter inputs.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Initialize properties
        self._model_type_combo: QComboBox = None
        self._parameter_input: ParameterInput = None
        
        # Setup UI components
        self.setup_ui()
        
        # Connect signals
        self._model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        self._parameter_input.parameters_changed.connect(
            lambda params: self.model_changed.emit(self.get_config())
        )
        self._parameter_input.validation_error.connect(
            lambda msg: self.config_error.emit(f"Parameter error: {msg}")
        )
        
        # Initial validation
        self.validate_config()
    
    def setup_ui(self) -> None:
        """
        Creates and arranges UI components.
        
        This method sets up the widget's layout and creates the necessary UI
        components for model configuration.
        """
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create model type group
        model_type_group = QGroupBox("Model Type")
        model_type_layout = QVBoxLayout(model_type_group)
        model_type_layout.setContentsMargins(8, 12, 8, 8)
        model_type_layout.setSpacing(6)
        
        # Create model type combo box
        self._model_type_combo = create_widget("QComboBox", {
            "toolTip": "Select model type for estimation"
        })
        
        # Add model types to combo box
        for model_id, model_name in MODEL_TYPES.items():
            self._model_type_combo.addItem(model_name, model_id)
        
        # Add to layout
        model_type_layout.addWidget(self._model_type_combo)
        
        # Create parameter input group
        param_group = QGroupBox("Model Parameters")
        param_layout = QVBoxLayout(param_group)
        param_layout.setContentsMargins(8, 12, 8, 8)
        param_layout.setSpacing(6)
        
        # Create parameter input widget
        self._parameter_input = ParameterInput(self)
        param_layout.addWidget(self._parameter_input)
        
        # Add groups to main layout
        main_layout.addWidget(model_type_group)
        main_layout.addWidget(param_group)
        
        # Set minimum width for better display
        self.setMinimumWidth(300)
    
    def on_model_type_changed(self, model_type_text: str) -> None:
        """
        Handles model type selection changes.
        
        This method is called when the user selects a different model type.
        It updates the parameter input widget for the selected model type and
        performs validation.
        
        Args:
            model_type_text: Display text of the selected model type
        """
        try:
            # Get the model ID from the current selection
            model_id = self._model_type_combo.currentData()
            
            logger.debug(f"Model type changed to: {model_id}")
            
            # Update parameter input for selected model type
            # Reset parameters for the new model type
            self._parameter_input.reset_inputs()
            
            # Validate the new configuration
            self.validate_config()
            
            # Emit model changed signal
            self.model_changed.emit(self.get_config())
            
        except Exception as e:
            logger.error(f"Error handling model type change: {str(e)}")
            self.config_error.emit(f"Error changing model type: {str(e)}")
    
    def validate_config(self) -> bool:
        """
        Validates complete model configuration.
        
        This method performs comprehensive validation of the model configuration,
        including model type selection, parameter values, and model-specific
        constraints.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        is_valid = True
        error_messages = []
        
        try:
            # Validate model type selection
            if self._model_type_combo.currentIndex() == -1:
                is_valid = False
                error_messages.append("Please select a model type")
                
                # Highlight the combo box with error style
                validate_widget_input(self._model_type_combo, {'required': True})
            
            # Validate parameter inputs
            param_valid = self._parameter_input.validate_inputs()
            if not param_valid:
                is_valid = False
                error_messages.append("Invalid parameter values")
            
            # Check compatibility of parameters with selected model
            model_id = self._model_type_combo.currentData()
            parameters = self._parameter_input.get_parameters()
            
            # Model-specific validation
            if model_id == 'ARMA':
                # ARMA-specific validation
                if 'AR_ORDER' in parameters and 'MA_ORDER' in parameters:
                    p = parameters.get('AR_ORDER', 0)
                    q = parameters.get('MA_ORDER', 0)
                    if p == 0 and q == 0:
                        is_valid = False
                        error_messages.append("ARMA model must have at least one non-zero order")
            
            elif model_id in ['GARCH', 'EGARCH']:
                # GARCH model validation
                if 'ALPHA' in parameters and 'BETA' in parameters:
                    alpha = parameters.get('ALPHA', 0)
                    beta = parameters.get('BETA', 0)
                    # Check GARCH stability condition
                    if isinstance(alpha, (int, float)) and isinstance(beta, (int, float)):
                        if alpha + beta >= 1:
                            is_valid = False
                            error_messages.append("GARCH stability condition: α + β must be < 1")
            
            # Emit config error if validation failed
            if not is_valid and error_messages:
                error_message = "; ".join(error_messages)
                self.config_error.emit(error_message)
                logger.warning(f"Configuration validation errors: {error_message}")
        
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            is_valid = False
            self.config_error.emit(f"Validation error: {str(e)}")
        
        return is_valid
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retrieves current model configuration.
        
        This method returns a dictionary containing the complete model configuration,
        including model type, parameters, and validation status.
        
        Returns:
            Dict[str, Any]: Model configuration dictionary
        """
        # Validate configuration before returning
        is_valid = self.validate_config()
        
        # Get model type
        model_id = self._model_type_combo.currentData()
        model_text = self._model_type_combo.currentText()
        
        # Get parameters
        parameters = self._parameter_input.get_parameters()
        
        # Build configuration dictionary
        config = {
            'model_type': model_id,
            'model_name': model_text,
            'parameters': parameters,
            'is_valid': is_valid
        }
        
        return config
    
    def reset_config(self) -> None:
        """
        Resets configuration to defaults.
        
        This method resets the model type selection and parameter inputs to
        their default values and performs validation.
        """
        try:
            # Reset model type to first option
            self._model_type_combo.setCurrentIndex(0)
            
            # Reset parameter inputs
            self._parameter_input.reset_inputs()
            
            # Validate new configuration
            self.validate_config()
            
            # Emit model changed signal
            self.model_changed.emit(self.get_config())
            
            logger.debug("Model configuration reset to defaults")
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {str(e)}")
            self.config_error.emit(f"Reset error: {str(e)}")