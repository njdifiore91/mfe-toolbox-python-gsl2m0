"""
Test suite for the ParameterInput widget component.

This module tests the functionality of the ParameterInput widget, including
input handling, parameter validation, and signal emission for the PyQt6-based
parameter configuration interface.
"""

import pytest
from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication

# Internal imports
from web.components.parameter_input import ParameterInput
from web.utils.validation import validate_widget_input
from web.utils.qt_helpers import create_widget


class TestParameterInput:
    """Test class for ParameterInput widget"""
    
    def setup_method(self, method):
        """Test setup method"""
        # Create QApplication if it doesn't exist
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        # Create fresh ParameterInput instance
        self.widget = ParameterInput()
        
        # Initialize signal spies
        self.parameters_changed_spy = QSignalSpy(self.widget.parameters_changed)
        self.validation_error_spy = QSignalSpy(self.widget.validation_error)
    
    def teardown_method(self, method):
        """Test cleanup method"""
        # Clean up widget
        if hasattr(self, 'widget') and self.widget:
            self.widget.deleteLater()
        
        # Reset application state
        if hasattr(self, 'app') and self.app:
            self.app.processEvents()
        
        # Clear signal spies
        self.parameters_changed_spy = None
        self.validation_error_spy = None


@pytest.mark.qt
def test_parameter_input_initialization():
    """Tests proper initialization of ParameterInput widget"""
    # Create QApplication if it doesn't exist
    if not QApplication.instance():
        app = QApplication([])
    
    # Create ParameterInput widget instance
    widget = ParameterInput()
    
    # Verify widget creation
    assert widget is not None
    
    # Check default parameter values
    params = widget.get_parameters()
    assert "AR_ORDER" in params
    assert "MA_ORDER" in params
    assert "CONSTANT" in params
    assert "ALPHA" in params
    assert "BETA" in params
    
    # Validate widget hierarchy
    assert widget.layout() is not None
    
    # Clean up
    widget.deleteLater()


@pytest.mark.qt
def test_parameter_validation():
    """Tests parameter validation functionality"""
    # Create QApplication if it doesn't exist
    if not QApplication.instance():
        app = QApplication([])
    
    # Create ParameterInput widget
    widget = ParameterInput()
    validation_error_spy = QSignalSpy(widget.validation_error)
    
    # Set invalid parameter values
    ar_order_widget = widget._input_widgets["AR_ORDER"]
    ma_order_widget = widget._input_widgets["MA_ORDER"]
    
    ar_order_widget.setValue(0)
    ma_order_widget.setValue(0)
    
    # Verify validation error signals
    widget.validate_inputs()
    assert validation_error_spy.count() > 0
    assert "ARMA model must have at least one non-zero order" in validation_error_spy[-1][0]
    
    # Set valid parameter values
    ar_order_widget.setValue(1)
    
    # Verify validation success
    validation_error_spy.clear()
    widget.validate_inputs()
    assert validation_error_spy.count() == 0
    
    # Clean up
    widget.deleteLater()


@pytest.mark.qt
def test_parameter_signals():
    """Tests signal emission on parameter changes"""
    # Create QApplication if it doesn't exist
    if not QApplication.instance():
        app = QApplication([])
    
    # Create ParameterInput widget
    widget = ParameterInput()
    
    # Set up QSignalSpy for parameters_changed signal
    parameters_changed_spy = QSignalSpy(widget.parameters_changed)
    parameters_changed_spy.clear()  # Clear initial signals
    
    # Modify parameter values
    ar_order_widget = widget._input_widgets["AR_ORDER"]
    ar_order_widget.setValue(2)
    
    # Verify signal emission
    assert parameters_changed_spy.count() > 0
    
    # Check signal payload
    last_signal = parameters_changed_spy[-1][0]
    assert isinstance(last_signal, dict)
    assert "AR_ORDER" in last_signal
    assert last_signal["AR_ORDER"] == 2
    
    # Clean up
    widget.deleteLater()


@pytest.mark.qt
def test_parameter_reset():
    """Tests parameter reset functionality"""
    # Create QApplication if it doesn't exist
    if not QApplication.instance():
        app = QApplication([])
    
    # Create ParameterInput widget
    widget = ParameterInput()
    parameters_changed_spy = QSignalSpy(widget.parameters_changed)
    
    # Set parameter values
    ar_order_widget = widget._input_widgets["AR_ORDER"]
    ma_order_widget = widget._input_widgets["MA_ORDER"]
    
    ar_order_widget.setValue(3)
    ma_order_widget.setValue(2)
    
    # Clear spy to only catch reset signal
    parameters_changed_spy.clear()
    
    # Call reset function
    widget.reset_inputs()
    
    # Verify default values restored
    params = widget.get_parameters()
    assert params["AR_ORDER"] == 1
    assert params["MA_ORDER"] == 1
    
    # Check reset signal emission
    assert parameters_changed_spy.count() > 0
    
    # Clean up
    widget.deleteLater()