"""
Validation utilities for PyQt6 GUI components.

This module provides validation functionality for input widgets in the PyQt6-based
GUI. It ensures data integrity through widget-specific validation and leverages 
backend validation capabilities to maintain consistent validation rules across
the application.
"""

import logging
import numpy as np
from typing import Optional, Union, Dict, Any

from PyQt6.QtWidgets import QWidget, QLineEdit, QSpinBox, QComboBox  # PyQt6 version 6.6.1

# Internal imports
from backend.utils.validation import (
    validate_array_input,
    validate_parameters,
    validate_model_order,
)
from web.utils.qt_helpers import create_widget

# Configure logger
logger = logging.getLogger(__name__)

# Mapping of widget types to validation functions
WIDGET_VALIDATION_RULES = {
    'QLineEdit': 'validate_line_edit',
    'QSpinBox': 'validate_spin_box',
    'QComboBox': 'validate_combo_box'
}


def validate_widget_input(widget: QWidget, validation_rules: Optional[dict] = None) -> bool:
    """
    Validates input from a PyQt6 widget based on its type and validation rules.
    
    This function acts as the main entry point for widget validation, delegating
    to specific validators based on the widget type.
    
    Parameters
    ----------
    widget : QWidget
        The widget to validate
    validation_rules : Optional[dict]
        Dictionary containing validation rules specific to the widget type
        
    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    try:
        # Get widget class name for dispatching to appropriate validator
        widget_type = widget.__class__.__name__
        
        # Check if we have a validator for this widget type
        if widget_type not in WIDGET_VALIDATION_RULES:
            logger.warning(f"No validator defined for widget type: {widget_type}")
            return True  # Pass validation if no validator exists
        
        # Get the validation function name
        validator_name = WIDGET_VALIDATION_RULES[widget_type]
        
        # Get the actual function
        if hasattr(globals(), validator_name):
            validator_func = globals()[validator_name]
            return validator_func(widget, validation_rules)
        else:
            logger.error(f"Validator function {validator_name} not found")
            return False
    
    except Exception as e:
        logger.error(f"Error validating widget input: {str(e)}")
        return False


def validate_line_edit(widget: QLineEdit, rules: Optional[dict] = None) -> bool:
    """
    Validates QLineEdit input for numerical values and ranges.
    
    Performs validation on text input fields, typically used for numerical
    parameters like alpha values, confidence levels, etc.
    
    Parameters
    ----------
    widget : QLineEdit
        The QLineEdit widget to validate
    rules : Optional[dict]
        Dictionary containing validation rules such as:
        - numeric: bool - Whether input should be numeric
        - bounds: tuple - (min, max) allowed values
        - required: bool - Whether the field is required
        
    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    if not isinstance(widget, QLineEdit):
        logger.error(f"Expected QLineEdit, got {type(widget).__name__}")
        return False
    
    # Default rules
    default_rules = {
        'numeric': True,
        'bounds': None,
        'required': True
    }
    
    # Merge with provided rules
    if rules is not None:
        default_rules.update(rules)
    
    rules = default_rules
    
    # Get the input text
    text = widget.text().strip()
    
    # Check if required
    if rules.get('required') and not text:
        widget.setStyleSheet("background-color: #FFDDDD;")
        return False
    
    # If not required and empty, it's valid
    if not rules.get('required') and not text:
        widget.setStyleSheet("")
        return True
    
    # Validate numeric if required
    if rules.get('numeric'):
        try:
            value = float(text)
            
            # Validate bounds if provided
            if rules.get('bounds') is not None:
                min_val, max_val = rules['bounds']
                if value < min_val or value > max_val:
                    widget.setStyleSheet("background-color: #FFDDDD;")
                    return False
            
            # Validate using backend parameter validation
            try:
                if validate_parameters(value, bounds=rules.get('bounds')):
                    widget.setStyleSheet("")
                    return True
            except Exception as e:
                logger.warning(f"Backend validation failed: {str(e)}")
                widget.setStyleSheet("background-color: #FFDDDD;")
                return False
                
        except ValueError:
            widget.setStyleSheet("background-color: #FFDDDD;")
            return False
    
    # Input passed validation
    widget.setStyleSheet("")
    return True


def validate_spin_box(widget: QSpinBox, rules: Optional[dict] = None) -> bool:
    """
    Validates QSpinBox input for model orders and integer parameters.
    
    Validates spin box inputs, typically used for model orders (AR, MA),
    lag lengths, and other integer parameters.
    
    Parameters
    ----------
    widget : QSpinBox
        The QSpinBox widget to validate
    rules : Optional[dict]
        Dictionary containing validation rules such as:
        - model_type: str - Type of model for order validation
        - p_widget: QSpinBox - Paired AR order widget for combined validation
        - q_widget: QSpinBox - Paired MA order widget for combined validation
        
    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    if not isinstance(widget, QSpinBox):
        logger.error(f"Expected QSpinBox, got {type(widget).__name__}")
        return False
    
    # Get the value
    value = widget.value()
    
    # Default rules
    default_rules = {
        'model_type': None,
        'p_widget': None,
        'q_widget': None
    }
    
    # Merge with provided rules
    if rules is not None:
        default_rules.update(rules)
    
    rules = default_rules
    
    # If this is part of a p/q pair, validate together
    if rules.get('model_type') and (rules.get('p_widget') or rules.get('q_widget')):
        p = value if widget == rules.get('p_widget') else (rules.get('p_widget').value() if rules.get('p_widget') else 0)
        q = value if widget == rules.get('q_widget') else (rules.get('q_widget').value() if rules.get('q_widget') else 0)
        
        try:
            # Validate using backend model order validation
            if validate_model_order(p, q, model_type=rules.get('model_type')):
                widget.setStyleSheet("")
                return True
        except Exception as e:
            logger.warning(f"Backend validation failed: {str(e)}")
            widget.setStyleSheet("background-color: #FFDDDD;")
            return False
    
    # If no specific rules or not part of a pair, basic validation
    if value < 0:
        widget.setStyleSheet("background-color: #FFDDDD;")
        return False
    
    # Input passed validation
    widget.setStyleSheet("")
    return True


def validate_combo_box(widget: QComboBox, rules: Optional[dict] = None) -> bool:
    """
    Validates QComboBox selections for model types and distributions.
    
    Validates dropdown selections, typically used for model types,
    distribution selections, and other categorical choices.
    
    Parameters
    ----------
    widget : QComboBox
        The QComboBox widget to validate
    rules : Optional[dict]
        Dictionary containing validation rules such as:
        - allowed_values: list - List of allowed values
        - required: bool - Whether a selection is required
        
    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    if not isinstance(widget, QComboBox):
        logger.error(f"Expected QComboBox, got {type(widget).__name__}")
        return False
    
    # Default rules
    default_rules = {
        'allowed_values': None,
        'required': True
    }
    
    # Merge with provided rules
    if rules is not None:
        default_rules.update(rules)
    
    rules = default_rules
    
    # Get the selected value
    current_text = widget.currentText()
    current_data = widget.currentData()
    
    # Check if required and nothing is selected
    if rules.get('required') and widget.currentIndex() == -1:
        widget.setStyleSheet("background-color: #FFDDDD;")
        return False
    
    # Check against allowed values if specified
    if rules.get('allowed_values') is not None:
        allowed = rules['allowed_values']
        if current_text not in allowed and current_data not in allowed:
            widget.setStyleSheet("background-color: #FFDDDD;")
            return False
    
    # Input passed validation
    widget.setStyleSheet("")
    return True


def validate_numeric_input(value: Union[str, int, float], bounds: Optional[dict] = None) -> bool:
    """
    Validates numeric input from text or spin box widgets.
    
    Generic validation function for numeric inputs that can be used
    across different widget types.
    
    Parameters
    ----------
    value : Union[str, int, float]
        The value to validate
    bounds : Optional[dict]
        Dictionary containing validation bounds:
        - min: float - Minimum allowed value
        - max: float - Maximum allowed value
        
    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    try:
        # Convert to float if it's a string
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return False
        
        # Check bounds if provided
        if bounds is not None:
            min_val = bounds.get('min', float('-inf'))
            max_val = bounds.get('max', float('inf'))
            
            if value < min_val or value > max_val:
                return False
        
        # Validate using backend parameter validation
        try:
            param_bounds = None
            if bounds is not None:
                min_val = bounds.get('min', float('-inf'))
                max_val = bounds.get('max', float('inf'))
                param_bounds = (min_val, max_val)
                
            return validate_parameters(value, bounds=param_bounds)
        except Exception as e:
            logger.warning(f"Backend validation failed: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating numeric input: {str(e)}")
        return False