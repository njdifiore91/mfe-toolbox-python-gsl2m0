"""
Test suite for PyQt6 helper utilities, validating widget creation, 
theme application, and configuration functionality.
"""

import pytest
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel  # PyQt6 version 6.6.1
from PyQt6.QtCore import Qt  # PyQt6 version 6.6.1
from utils.qt_helpers import create_widget, configure_widget, WidgetFactory


class TestQtHelpers:
    """Test class for Qt helper utilities"""
    
    def setup_method(self, method):
        """Setup method run before each test"""
        # Reset widget factory
        self.factory = WidgetFactory()
        
        # Clear widget cache
        self.created_widgets = []
        
        # Initialize test widgets
        self.test_widget = QWidget()
        self.test_button = QPushButton("Test Button")
        self.test_label = QLabel("Test Label")
    
    def teardown_method(self, method):
        """Cleanup method run after each test"""
        # Delete test widgets
        if hasattr(self, 'test_widget'):
            self.test_widget.deleteLater()
        if hasattr(self, 'test_button'):
            self.test_button.deleteLater()
        if hasattr(self, 'test_label'):
            self.test_label.deleteLater()
            
        # Delete any additional widgets created during tests
        if hasattr(self, 'created_widgets'):
            for widget in self.created_widgets:
                if widget is not None:
                    widget.deleteLater()
        
        # Reset theme
        self.factory = None


@pytest.mark.qt
def test_create_widget_basic(qtbot):
    """Tests basic widget creation functionality"""
    # Create a simple QWidget using the helper
    widget = create_widget("QWidget", {})
    qtbot.addWidget(widget)
    
    # Verify widget was created correctly
    assert isinstance(widget, QWidget)
    assert widget.isEnabled()
    assert not widget.isVisible()  # Widgets are not visible by default
    
    # Create a QPushButton with basic properties
    button = create_widget("QPushButton", {"text": "Test Button"})
    qtbot.addWidget(button)
    
    # Verify widget type and properties
    assert isinstance(button, QPushButton)
    assert button.text() == "Test Button"
    
    # Create a QLabel
    label = create_widget("QLabel", {"text": "Test Label"})
    qtbot.addWidget(label)
    
    # Verify widget type and properties
    assert isinstance(label, QLabel)
    assert label.text() == "Test Label"
    
    # Check theme application
    # We can verify the widget has a palette applied
    assert widget.palette() is not None
    
    # Validate widget is visible and enabled
    widget.show()
    assert widget.isVisible()
    
    # Clean up
    widget.deleteLater()
    button.deleteLater()
    label.deleteLater()


@pytest.mark.qt
def test_create_widget_properties(qtbot):
    """Tests widget creation with custom properties"""
    # Create a button with multiple properties
    properties = {
        "text": "Custom Button",
        "enabled": False,
        "visible": True,
        "toolTip": "This is a tooltip",
        "styleSheet": "background-color: #f0f0f0;"
    }
    
    button = create_widget("QPushButton", properties)
    qtbot.addWidget(button)
    
    # Verify all properties were applied correctly
    assert button.text() == "Custom Button"
    assert not button.isEnabled()
    assert button.isVisible()
    assert button.toolTip() == "This is a tooltip"
    assert button.styleSheet() == "background-color: #f0f0f0;"
    
    # Test widget with size and position properties
    size_properties = {
        "minimumWidth": 100,
        "minimumHeight": 50,
        "maximumWidth": 300,
        "maximumHeight": 200
    }
    
    widget = create_widget("QWidget", size_properties)
    qtbot.addWidget(widget)
    
    # Verify size properties
    assert widget.minimumWidth() == 100
    assert widget.minimumHeight() == 50
    assert widget.maximumWidth() == 300
    assert widget.maximumHeight() == 200
    
    # Verify property application
    # Create a widget with various properties
    diverse_properties = {
        "windowTitle": "Test Window",
        "toolTipDuration": 2000,
        "styleSheet": "color: blue; background-color: #eaeaea;"
    }
    
    complex_widget = create_widget("QWidget", diverse_properties)
    qtbot.addWidget(complex_widget)
    
    # Validate style properties
    assert complex_widget.windowTitle() == "Test Window"
    assert complex_widget.toolTipDuration() == 2000
    assert complex_widget.styleSheet() == "color: blue; background-color: #eaeaea;"
    
    # Clean up
    button.deleteLater()
    widget.deleteLater()
    complex_widget.deleteLater()


@pytest.mark.qt
def test_configure_widget(qtbot):
    """Tests widget configuration functionality"""
    # Create a basic widget
    widget = QWidget()
    qtbot.addWidget(widget)
    
    # Configure with properties
    properties = {
        "enabled": False,
        "visible": True,
        "toolTip": "Widget Tooltip",
        "styleSheet": "border: 1px solid red;"
    }
    
    configure_widget(widget, properties)
    
    # Verify properties were applied
    assert not widget.isEnabled()
    assert widget.isVisible()
    assert widget.toolTip() == "Widget Tooltip"
    assert widget.styleSheet() == "border: 1px solid red;"
    
    # Update configuration with new properties
    new_properties = {
        "enabled": True,
        "minimumWidth": 200,
        "toolTip": "Updated Tooltip"
    }
    
    configure_widget(widget, new_properties)
    
    # Verify updated properties
    assert widget.isEnabled()
    assert widget.minimumWidth() == 200
    assert widget.toolTip() == "Updated Tooltip"
    assert widget.styleSheet() == "border: 1px solid red;"  # Should maintain previous value
    
    # Check theme consistency
    # We can verify the widget palette is still applied after reconfiguration
    assert widget.palette() is not None
    
    # Clean up
    widget.deleteLater()


@pytest.mark.qt
def test_widget_factory(qtbot):
    """Tests WidgetFactory class functionality"""
    # Create a factory instance
    factory = WidgetFactory()
    
    # Create multiple widgets
    button1 = factory.create("QPushButton", {"text": "Button 1"})
    button2 = factory.create("QPushButton", {"text": "Button 2"})
    label = factory.create("QLabel", {"text": "Test Label"})
    
    qtbot.addWidget(button1)
    qtbot.addWidget(button2)
    qtbot.addWidget(label)
    
    # Verify widget creation
    assert isinstance(button1, QPushButton)
    assert isinstance(button2, QPushButton)
    assert isinstance(label, QLabel)
    
    assert button1.text() == "Button 1"
    assert button2.text() == "Button 2"
    assert label.text() == "Test Label"
    
    # Verify widget caching
    # Create another button and verify it's a new instance
    button3 = factory.create("QPushButton", {"text": "Button 3"})
    qtbot.addWidget(button3)
    
    assert button3 is not button1
    assert button3 is not button2
    
    # Test theme updates
    # This should apply new palette to all cached widgets
    factory.update_theme()
    
    # All widgets should still have valid palettes
    assert button1.palette() is not None
    assert button2.palette() is not None
    assert label.palette() is not None
    
    # Test factory with default properties
    # Button should have autoDefault=False applied automatically
    default_button = factory.create("QPushButton", {"text": "Default Button"})
    qtbot.addWidget(default_button)
    
    # Check factory configuration
    assert default_button.autoDefault() is False
    
    # Test factory with merged properties (default + custom)
    custom_button = factory.create("QPushButton", {
        "text": "Custom Button",
        "autoDefault": True  # Override default
    })
    qtbot.addWidget(custom_button)
    
    assert custom_button.text() == "Custom Button"
    assert custom_button.autoDefault() is True  # Custom setting overrides default
    
    # Clean up
    button1.deleteLater()
    button2.deleteLater()
    button3.deleteLater()
    label.deleteLater()
    default_button.deleteLater()
    custom_button.deleteLater()