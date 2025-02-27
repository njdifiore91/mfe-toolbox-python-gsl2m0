"""
Parameter input component for the MFE Toolbox GUI.

This module provides a PyQt6-based widget for handling model parameter inputs
with real-time validation, providing an interface for configuring time series 
and volatility model parameters with type checking and error feedback.
"""

import logging
from typing import Dict, Any, Optional, Union

from PyQt6.QtWidgets import QWidget, QLineEdit, QSpinBox, QVBoxLayout, QLabel  # PyQt6 version 6.6.1
from PyQt6.QtCore import pyqtSignal  # PyQt6 version 6.6.1

# Internal imports
from web.utils.validation import validate_widget_input
from web.utils.qt_helpers import create_widget

# Configure logger
logger = logging.getLogger(__name__)

# Parameter type definitions
PARAMETER_TYPES = {
    'AR_ORDER': 'int', 
    'MA_ORDER': 'int',
    'CONSTANT': 'bool',
    'ALPHA': 'float',
    'BETA': 'float'
}


class ParameterInput(QWidget):
    """Widget for handling model parameter inputs with real-time validation."""
    
    # Signals
    parameters_changed = pyqtSignal(dict)  # Emitted when any parameter value changes
    validation_error = pyqtSignal(str)     # Emitted when validation fails
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes parameter input widget with validation.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Initialize widget storage
        self._input_widgets: Dict[str, QWidget] = {}
        self._current_values: Dict[str, Any] = {}
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Set up input widgets
        self.create_input_widgets()
    
    def create_input_widgets(self) -> None:
        """Creates input widgets for different parameter types."""
        # Main layout
        layout = self.layout()
        
        # Create widgets for each parameter type
        for param_name, param_type in PARAMETER_TYPES.items():
            # Create label with descriptive text
            label_text = param_name.replace('_', ' ').title() + ":"
            label = create_widget("QLabel", {
                "text": label_text,
                "toolTip": f"Parameter: {param_name}"
            })
            
            # Create appropriate input widget based on parameter type
            widget_properties = {
                "toolTip": f"Enter {param_name.lower().replace('_', ' ')} value"
            }
            
            if param_type == 'int':
                widget_type = "QSpinBox"
                widget_properties.update({
                    "minimum": 0,
                    "maximum": 30,  # Maximum order based on system constants
                    "value": 1,
                    "toolTip": f"Enter {param_name.lower().replace('_', ' ')} (0-30)"
                })
            elif param_type == 'float':
                widget_type = "QLineEdit"
                widget_properties.update({
                    "text": "0.5",
                    "toolTip": f"Enter {param_name.lower().replace('_', ' ')} value (0-1)"
                })
            elif param_type == 'bool':
                widget_type = "QCheckBox"
                widget_properties.update({
                    "checked": True,
                    "text": "Include",
                    "toolTip": f"Toggle {param_name.lower().replace('_', ' ')}"
                })
            else:
                widget_type = "QLineEdit"
            
            # Create widget
            widget = create_widget(widget_type, widget_properties)
            
            # Store widget
            self._input_widgets[param_name] = widget
            
            # Set up connections using helper function to avoid closure issues
            self._connect_widget_signals(widget, param_name)
            
            # Set initial value
            self._update_current_value(param_name, widget)
            
            # Add widgets to layout
            param_layout = QVBoxLayout()
            param_layout.addWidget(label)
            param_layout.addWidget(widget)
            param_layout.setContentsMargins(0, 0, 0, 8)  # Add some spacing between parameters
            layout.addLayout(param_layout)
        
        # Apply initial validation
        self.validate_inputs()
    
    def _connect_widget_signals(self, widget: QWidget, param_name: str) -> None:
        """
        Connects appropriate signals for a widget to handle value changes.
        
        Args:
            widget: The widget to connect signals for
            param_name: The parameter name associated with the widget
        """
        if isinstance(widget, QSpinBox):
            # Using a named parameter to connect to prevent closure issues
            widget.valueChanged.connect(
                lambda val, name=param_name: self.on_input_changed(name)
            )
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(
                lambda text, name=param_name: self.on_input_changed(name)
            )
        else:  # QCheckBox or others with toggled signal
            if hasattr(widget, 'toggled'):
                widget.toggled.connect(
                    lambda checked, name=param_name: self.on_input_changed(name)
                )
            elif hasattr(widget, 'stateChanged'):
                widget.stateChanged.connect(
                    lambda state, name=param_name: self.on_input_changed(name)
                )
    
    def _update_current_value(self, param_name: str, widget: QWidget) -> None:
        """
        Updates the current value for a parameter based on its widget.
        
        Args:
            param_name: The parameter name
            widget: The associated widget
        """
        param_type = PARAMETER_TYPES.get(param_name, 'str')
        
        try:
            if isinstance(widget, QSpinBox):
                self._current_values[param_name] = widget.value()
            elif isinstance(widget, QLineEdit):
                if param_type == 'float':
                    try:
                        self._current_values[param_name] = float(widget.text())
                    except ValueError:
                        self._current_values[param_name] = 0.5  # Default
                else:
                    self._current_values[param_name] = widget.text()
            else:  # QCheckBox or other
                if hasattr(widget, 'isChecked'):
                    self._current_values[param_name] = widget.isChecked()
                else:
                    # Fallback for other widget types
                    self._current_values[param_name] = None
        except Exception as e:
            logger.error(f"Error updating value for {param_name}: {str(e)}")
            # Set a default value based on parameter type
            if param_type == 'int':
                self._current_values[param_name] = 1
            elif param_type == 'float':
                self._current_values[param_name] = 0.5
            elif param_type == 'bool':
                self._current_values[param_name] = True
            else:
                self._current_values[param_name] = ""
    
    def validate_inputs(self) -> bool:
        """
        Validates all input widgets.
        
        Returns:
            bool: True if all inputs are valid
        """
        all_valid = True
        error_messages = []
        
        # Validate each input widget
        for param_name, widget in self._input_widgets.items():
            param_type = PARAMETER_TYPES.get(param_name, 'str')
            
            # Set up validation rules based on parameter type
            validation_rules = {}
            
            if param_type == 'float':
                validation_rules = {
                    'numeric': True,
                    'bounds': (0.0, 1.0),
                    'required': True
                }
            elif param_type == 'int':
                # For AR/MA orders, we need model-specific validation
                if param_name in ['AR_ORDER', 'MA_ORDER']:
                    validation_rules = {
                        'model_type': 'ARMA',
                        'p_widget': self._input_widgets.get('AR_ORDER'),
                        'q_widget': self._input_widgets.get('MA_ORDER')
                    }
            
            # Validate the widget
            try:
                is_valid = validate_widget_input(widget, validation_rules)
            except Exception as e:
                logger.error(f"Validation error for {param_name}: {str(e)}")
                is_valid = False
            
            if not is_valid:
                all_valid = False
                error_messages.append(f"Invalid {param_name.replace('_', ' ').lower()} value")
        
        # Special validation: ARMA model must have at least one non-zero order
        if 'AR_ORDER' in self._input_widgets and 'MA_ORDER' in self._input_widgets:
            ar_value = 0
            ma_value = 0
            
            if isinstance(self._input_widgets['AR_ORDER'], QSpinBox):
                ar_value = self._input_widgets['AR_ORDER'].value()
            
            if isinstance(self._input_widgets['MA_ORDER'], QSpinBox):
                ma_value = self._input_widgets['MA_ORDER'].value()
            
            if ar_value == 0 and ma_value == 0:
                all_valid = False
                error_messages.append("ARMA model must have at least one non-zero order")
        
        # Emit validation error if any
        if not all_valid and error_messages:
            error_message = "; ".join(error_messages)
            self.validation_error.emit(error_message)
            logger.warning(f"Validation errors: {error_message}")
        
        return all_valid
    
    def on_input_changed(self, parameter_name: str) -> None:
        """
        Handles input value changes with validation.
        
        Args:
            parameter_name: Name of the parameter that changed
        """
        # Get widget
        widget = self._input_widgets.get(parameter_name)
        
        if not widget:
            logger.error(f"Widget for parameter {parameter_name} not found")
            return
        
        # Update current value
        self._update_current_value(parameter_name, widget)
        
        # Validate the input
        is_valid = self.validate_inputs()
        
        # Emit parameters changed signal
        self.parameters_changed.emit(self._current_values)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieves current parameter values.
        
        Returns:
            Dict[str, Any]: Dictionary of parameter values
        """
        # Validate all inputs before returning
        is_valid = self.validate_inputs()
        
        if not is_valid:
            logger.warning("Returning parameters despite validation errors")
        
        return self._current_values.copy()
    
    def reset_inputs(self) -> None:
        """Resets all input widgets to default values."""
        # Reset each input widget to default value
        for param_name, widget in self._input_widgets.items():
            param_type = PARAMETER_TYPES.get(param_name, 'str')
            
            try:
                if isinstance(widget, QSpinBox):
                    widget.setValue(1)  # Default integer value
                elif isinstance(widget, QLineEdit):
                    if param_type == 'float':
                        widget.setText("0.5")  # Default float value
                    else:
                        widget.setText("")  # Default text value
                else:  # QCheckBox or other
                    if hasattr(widget, 'setChecked'):
                        widget.setChecked(True)  # Default boolean value
            except Exception as e:
                logger.error(f"Error resetting widget for {param_name}: {str(e)}")
        
        # Update current values after reset
        for param_name, widget in self._input_widgets.items():
            self._update_current_value(param_name, widget)
        
        # Validate inputs after reset
        self.validate_inputs()
        
        # Emit parameters changed signal
        self.parameters_changed.emit(self._current_values)