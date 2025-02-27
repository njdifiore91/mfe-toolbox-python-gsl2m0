"""
Test suite for validating PyQt6 GUI component input validation.

This module tests the widget validation utilities in web.utils.validation,
ensuring robust error handling and data integrity for the GUI layer.
"""

import pytest
import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QSpinBox, QComboBox  # PyQt6 version 6.6.1

from web.utils.validation import (
    validate_widget_input,
    validate_line_edit,
    validate_spin_box,
    validate_combo_box
)


# Fixture for QLineEdit
@pytest.fixture
def line_edit():
    return QLineEdit()


# Fixture for QSpinBox
@pytest.fixture
def spin_box():
    widget = QSpinBox()
    widget.setRange(-100, 100)  # Ensure we can set any test value
    return widget


# Fixture for QComboBox
@pytest.fixture
def combo_box():
    return QComboBox()


@pytest.mark.parametrize('widget_type', ['QLineEdit', 'QSpinBox', 'QComboBox'])
def test_validate_widget_input(widget_type, line_edit, spin_box, combo_box):
    """Tests the generic widget input validation function."""
    # Create test widget of specified type
    if widget_type == 'QLineEdit':
        widget = line_edit
        widget.setText('123')
        rules = {'numeric': True}
    elif widget_type == 'QSpinBox':
        widget = spin_box
        widget.setValue(5)
        rules = {}
    else:  # QComboBox
        widget = combo_box
        widget.addItems(['GARCH', 'EGARCH'])
        widget.setCurrentIndex(0)
        rules = {'allowed_values': ['GARCH', 'EGARCH']}
    
    # Test validation with valid input
    result = validate_widget_input(widget, rules)
    assert result is True, f"Validation failed for valid {widget_type}"
    
    # Test with invalid inputs
    if widget_type == 'QLineEdit':
        widget.setText('abc')  # Invalid numeric input
        rules = {'numeric': True}
    elif widget_type == 'QSpinBox':
        widget.setValue(-1)  # Invalid negative order
        rules = {'model_type': 'GARCH'}
    else:  # QComboBox
        widget.clear()
        widget.addItems(['invalid'])
        widget.setCurrentIndex(0)
        rules = {'allowed_values': ['GARCH', 'EGARCH']}
    
    # Validation should fail for invalid inputs
    result = validate_widget_input(widget, rules)
    assert result is False, f"Validation incorrectly passed for invalid {widget_type}"


@pytest.mark.parametrize('input_value,expected_valid', [
    ('1.23', True),    # Valid float
    ('abc', False),    # Invalid non-numeric
    ('-0.5', True)     # Valid negative float
])
def test_validate_line_edit(line_edit, input_value, expected_valid):
    """Tests QLineEdit validation for numerical inputs."""
    line_edit.setText(input_value)
    
    # Test with rules: numeric required
    rules = {
        'numeric': True,
        'required': True
    }
    
    result = validate_line_edit(line_edit, rules)
    assert result == expected_valid, f"Unexpected validation result for '{input_value}'"
    
    # Verify styling changes
    if expected_valid:
        assert line_edit.styleSheet() == "", "Valid input should reset style"
    else:
        assert "background-color: #FFDDDD" in line_edit.styleSheet(), "Invalid input should have error style"
    
    # Test with bounds if input is numeric and valid
    if expected_valid:
        # Test within bounds
        rules['bounds'] = (-1, 10)
        result = validate_line_edit(line_edit, rules)
        assert result is True, f"Value {input_value} should be within bounds (-1, 10)"
        
        # Test outside bounds
        if float(input_value) < 2:  # Ensure our test value is out of bounds
            rules['bounds'] = (2, 10)
            result = validate_line_edit(line_edit, rules)
            assert result is False, f"Value {input_value} should be outside bounds (2, 10)"
    
    # Test empty input for required field
    line_edit.setText('')
    result = validate_line_edit(line_edit, rules)
    assert result is False, "Empty value should be invalid for required field"
    
    # Test empty input for optional field
    rules['required'] = False
    result = validate_line_edit(line_edit, rules)
    assert result is True, "Empty value should be valid for optional field"


@pytest.mark.parametrize('value,expected_valid', [
    (1, True),     # Valid order
    (0, True),     # Valid zero order
    (31, False)    # Invalid (exceeds MAX_ORDER)
])
def test_validate_spin_box(spin_box, value, expected_valid):
    """Tests QSpinBox validation for model orders."""
    spin_box.setValue(value)
    
    # Basic validation without model_type
    result = validate_spin_box(spin_box)
    assert result == expected_valid, f"Unexpected validation result for value {value}"
    
    # Check styling
    if expected_valid:
        assert spin_box.styleSheet() == "", "Valid input should reset style"
    else:
        assert "background-color: #FFDDDD" in spin_box.styleSheet(), "Invalid input should have error style"
    
    # Only test valid model orders with ARMA model_type
    if expected_valid:
        # Create paired p/q widgets for ARMA model
        p_widget = QSpinBox()
        p_widget.setValue(value)
        
        q_widget = QSpinBox()
        q_widget.setValue(0)
        
        rules = {
            'model_type': 'ARMA',
            'p_widget': p_widget,
            'q_widget': q_widget
        }
        
        # Special case: for ARMA, p=0 and q=0 is invalid
        if value == 0:
            result = validate_spin_box(p_widget, rules)
            assert result is False, "ARMA with p=0, q=0 should be invalid"
            
            # But if q > 0, it should be valid
            q_widget.setValue(1)
            result = validate_spin_box(p_widget, rules)
            assert result is True, "ARMA with p=0, q=1 should be valid"


@pytest.mark.parametrize('items,selection,expected_valid', [
    (['GARCH', 'EGARCH'], 'GARCH', True),        # Valid selection
    (['normal', 'student-t'], 'invalid', False)  # Invalid selection
])
def test_validate_combo_box(combo_box, items, selection, expected_valid):
    """Tests QComboBox validation for model types and distributions."""
    combo_box.clear()
    combo_box.addItems(items)
    
    # Set selection if it's in the items list, otherwise add and select invalid
    if selection in items:
        combo_box.setCurrentText(selection)
    else:
        # Add the invalid item and select it
        combo_box.addItem(selection)
        combo_box.setCurrentText(selection)
    
    rules = {
        'allowed_values': items,
        'required': True
    }
    
    result = validate_combo_box(combo_box, rules)
    assert result == expected_valid, f"Unexpected validation result for '{selection}'"
    
    # Verify styling changes
    if expected_valid:
        assert combo_box.styleSheet() == "", "Valid input should reset style"
    else:
        assert "background-color: #FFDDDD" in combo_box.styleSheet(), "Invalid input should have error style"
    
    # Test required validation
    combo_box.clear()  # Empty combo box
    rules = {'required': True}
    result = validate_combo_box(combo_box, rules)
    assert result is False, "Empty combobox should fail when required"
    
    rules = {'required': False}
    result = validate_combo_box(combo_box, rules)
    assert result is True, "Empty combobox should pass when not required"


def test_validation_error_handling():
    """Tests error handling in validation functions."""
    # Test with wrong widget types
    wrong_widget = QWidget()  # Not a supported widget type
    result = validate_widget_input(wrong_widget)
    assert result is True, "Unsupported widget types should pass validation by default"
    
    # Test with None for widget
    with pytest.raises(Exception):
        validate_widget_input(None)
    
    # Test validate_line_edit with wrong widget type
    with pytest.raises(Exception):
        validate_line_edit(QSpinBox())
    
    # Test validate_spin_box with wrong widget type
    with pytest.raises(Exception):
        validate_spin_box(QLineEdit())
    
    # Test validate_combo_box with wrong widget type
    with pytest.raises(Exception):
        validate_combo_box(QLineEdit())
    
    # Test exception handling in validate_widget_input
    line_edit = QLineEdit()
    line_edit.setText("1.23")
    
    # Create rules that would cause backend validation to fail
    # Invalid bounds type would trigger an exception
    rules = {'numeric': True, 'bounds': ('not_a_number', 10)}
    
    # This should catch the exception and return False
    result = validate_widget_input(line_edit, rules)
    assert result is False, "Validation should handle backend validation errors gracefully"