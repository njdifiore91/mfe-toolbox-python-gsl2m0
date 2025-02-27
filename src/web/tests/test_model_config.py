"""
Test suite for the ModelConfig widget component.

Tests model configuration functionality, parameter validation,
and UI interaction using pytest and PyQt6 testing utilities.
"""

import pytest
from PyQt6.QtTest import QTest  # PyQt6 version 6.6.1
from PyQt6.QtWidgets import QApplication  # PyQt6 version 6.6.1

# Internal imports
from web.components.model_config import ModelConfig
from web.utils.validation import validate_widget_input
from web.utils.qt_helpers import create_widget


class ModelConfigFixture:
    """Pytest fixture class for ModelConfig testing."""
    
    def __init__(self):
        """Sets up test environment for ModelConfig tests."""
        # Initialize QApplication
        self.app = QApplication.instance() or QApplication([])
        
        # Create ModelConfig widget
        self.widget = ModelConfig()
        
        # Signal tracking containers
        self.model_changed_signals = []
        self.error_signals = []
        
        # Connect signal handlers
        self.widget.model_changed.connect(self._on_model_changed)
        self.widget.config_error.connect(self._on_error)
    
    def _on_model_changed(self, config):
        """Tracks model_changed signal emissions."""
        self.model_changed_signals.append(config)
    
    def _on_error(self, message):
        """Tracks error signal emissions."""
        self.error_signals.append(message)
    
    def setup(self):
        """Test setup method."""
        # Reset widget to default state
        if hasattr(self.widget, 'reset_config'):
            self.widget.reset_config()
        
        # Clear tracked signals
        self.model_changed_signals = []
        self.error_signals = []
    
    def cleanup(self):
        """Test cleanup method."""
        # Clean up widget
        if self.widget:
            self.widget.deleteLater()
            self.widget = None


@pytest.fixture
def model_config_fixture():
    """Creates ModelConfigFixture for testing."""
    fixture = ModelConfigFixture()
    yield fixture
    fixture.cleanup()


@pytest.mark.qt
def test_model_config_initialization(model_config_fixture):
    """Tests proper initialization of ModelConfig widget."""
    fixture = model_config_fixture
    fixture.setup()
    
    # Verify default model type selection
    assert fixture.widget._model_type_combo.currentIndex() == 0
    
    # Check parameter input widget initialization
    assert fixture.widget._parameter_input is not None
    
    # Validate initial configuration state
    config = fixture.widget.get_config()
    assert 'model_type' in config
    assert 'parameters' in config
    assert 'is_valid' in config
    
    # Check that the initial configuration is valid
    assert config['is_valid'] is True


@pytest.mark.qt
def test_model_type_change(model_config_fixture):
    """Tests model type selection changes and signal emission."""
    fixture = model_config_fixture
    fixture.setup()
    
    # Get initial state
    initial_model_type = fixture.widget._model_type_combo.currentData()
    
    # Change model type selection
    current_index = fixture.widget._model_type_combo.currentIndex()
    new_index = (current_index + 1) % fixture.widget._model_type_combo.count()
    fixture.widget._model_type_combo.setCurrentIndex(new_index)
    
    # Process events to ensure signal emission
    QTest.qWait(10)
    
    # Verify signal emission
    assert len(fixture.model_changed_signals) > 0
    
    # Validate updated configuration
    config = fixture.widget.get_config()
    assert config['model_type'] == fixture.widget._model_type_combo.currentData()
    assert config['model_type'] != initial_model_type  # Model type has changed


@pytest.mark.qt
def test_parameter_validation(model_config_fixture):
    """Tests parameter validation functionality."""
    fixture = model_config_fixture
    fixture.setup()
    
    # Set invalid parameter values
    param_inputs = fixture.widget._parameter_input._input_widgets
    
    # Find float parameters
    if 'ALPHA' in param_inputs:
        # Set value outside valid range (0-1)
        param_inputs['ALPHA'].setText("2.0")
        
        # Process events
        QTest.qWait(10)
        
        # Verify validation failure
        assert not fixture.widget.validate_config()
        
        # Reset to valid value
        param_inputs['ALPHA'].setText("0.5")
        
        # Process events
        QTest.qWait(10)
        
        # Verify validation success
        assert fixture.widget.validate_config()
    
    # Test ARMA-specific validation
    if 'AR_ORDER' in param_inputs and 'MA_ORDER' in param_inputs:
        # Set both orders to 0 (invalid for ARMA)
        param_inputs['AR_ORDER'].setValue(0)
        param_inputs['MA_ORDER'].setValue(0)
        
        # Process events
        QTest.qWait(10)
        
        # Verify validation failure
        assert not fixture.widget.validate_config()
        
        # Reset to valid values
        param_inputs['AR_ORDER'].setValue(1)
        
        # Process events
        QTest.qWait(10)
        
        # Verify validation success
        assert fixture.widget.validate_config()


@pytest.mark.qt
def test_config_retrieval(model_config_fixture):
    """Tests configuration dictionary retrieval."""
    fixture = model_config_fixture
    fixture.setup()
    
    # Set known configuration values
    param_inputs = fixture.widget._parameter_input._input_widgets
    
    # Test configurations for different parameters
    test_values = {
        'AR_ORDER': 2,
        'MA_ORDER': 1,
        'ALPHA': 0.3,
        'BETA': 0.6
    }
    
    # Set parameter values if available
    for param_name, value in test_values.items():
        if param_name in param_inputs:
            # Set value based on widget type
            if hasattr(param_inputs[param_name], 'setValue'):
                param_inputs[param_name].setValue(value)
            elif hasattr(param_inputs[param_name], 'setText'):
                param_inputs[param_name].setText(str(value))
    
    # Process events
    QTest.qWait(10)
    
    # Retrieve configuration dictionary
    config = fixture.widget.get_config()
    
    # Verify dictionary structure
    assert 'model_type' in config
    assert 'model_name' in config
    assert 'parameters' in config
    assert 'is_valid' in config
    
    # Verify parameter values
    parameters = config['parameters']
    for param_name, value in test_values.items():
        if param_name in parameters:
            assert parameters[param_name] == value or \
                   abs(float(parameters[param_name]) - float(value)) < 0.0001  # Float comparison


@pytest.mark.qt
def test_error_handling(model_config_fixture):
    """Tests error signal emission on invalid configurations."""
    fixture = model_config_fixture
    fixture.setup()
    
    # Clear any initialization signals
    fixture.error_signals = []
    
    # Find parameters to test
    param_inputs = fixture.widget._parameter_input._input_widgets
    
    # Test case 1: Invalid float input
    if 'ALPHA' in param_inputs:
        # Set invalid text value
        param_inputs['ALPHA'].setText("not_a_number")
        
        # Process events
        QTest.qWait(10)
        
        # Force validation
        fixture.widget.validate_config()
        
        # Process events to ensure signal emission
        QTest.qWait(10)
        
        # Verify error signal emission
        assert len(fixture.error_signals) > 0
        
        # Clear error signals for next test
        fixture.error_signals = []
    
    # Test case 2: GARCH stability condition violation
    if 'ALPHA' in param_inputs and 'BETA' in param_inputs:
        # Set values that violate α + β < 1 condition
        param_inputs['ALPHA'].setText("0.5")
        param_inputs['BETA'].setText("0.6")
        
        # Process events
        QTest.qWait(10)
        
        # Force validation when model type is GARCH
        model_type_combo = fixture.widget._model_type_combo
        for i in range(model_type_combo.count()):
            if model_type_combo.itemData(i) == 'GARCH':
                model_type_combo.setCurrentIndex(i)
                break
        
        # Process events
        QTest.qWait(10)
        
        # Force validation
        fixture.widget.validate_config()
        
        # Process events
        QTest.qWait(10)
        
        # Verify error signal emission for stability condition
        assert any("stability" in msg.lower() for msg in fixture.error_signals)