"""
Test suite for the MFE Toolbox GUI dialog components implemented with PyQt6.

This module provides comprehensive unit tests for the About, Close, Error,
and Help dialogs, verifying their initialization, display properties, and functionality.
"""

import pytest
import logging
from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QDialog
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QUrl

# Internal imports
from web.dialogs.about_dialog import AboutDialog
from web.dialogs.close_dialog import CloseDialog
from web.dialogs.error_dialog import ErrorDialog
from web.dialogs.help_dialog import HelpDialog


@pytest.fixture(scope="module")
def qapp():
    """Create a QApplication instance for the tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.mark.qt
def test_about_dialog(qapp):
    """Tests the About dialog initialization and display."""
    # Create About dialog instance
    dialog = AboutDialog()
    
    try:
        # Verify dialog title is set correctly
        assert dialog.windowTitle() == "About ARMAX"
        assert dialog.isModal() == True
        
        # Check version label contains expected information
        version_label_text = dialog._version_label.text()
        assert "ARMAX Model Estimation" in version_label_text
        assert "Version 4.0" in version_label_text
        assert "(c) 2009 Kevin Sheppard" in version_label_text
        
        # Verify website and docs buttons exist
        assert dialog._website_button is not None
        assert dialog._website_button.text() == "Website"
        assert dialog._docs_button is not None
        assert dialog._docs_button.text() == "Documentation"
        
        # Test OK button is properly configured
        assert dialog._ok_button is not None
        assert dialog._ok_button.text() == "OK"
        
        # Verify button click signals are connected
        assert dialog._ok_button.clicked.isConnected()
        assert dialog._website_button.clicked.isConnected()
        assert dialog._docs_button.clicked.isConnected()
    finally:
        # Ensure dialog is closed to prevent memory leaks
        dialog.close()


@pytest.mark.qt
def test_close_dialog(qapp, monkeypatch):
    """Tests the Close confirmation dialog functionality."""
    # Create Close dialog instance
    dialog = CloseDialog()
    
    try:
        # Verify dialog title and message
        assert dialog.windowTitle() == "Confirm Close"
        assert dialog.isModal() == True
        
        # Check message content
        message_text = dialog._message_label.text()
        assert "Are you sure you want to close?" in message_text
        assert "Unsaved changes will be lost" in message_text
        
        # Get Yes and No buttons from the dialog
        button_box = dialog._button_box
        yes_button = button_box.button(QDialogButtonBox.StandardButton.Yes)
        no_button = button_box.button(QDialogButtonBox.StandardButton.No)
        
        # Verify buttons exist and are configured properly
        assert yes_button is not None
        assert no_button is not None
        assert no_button.isDefault() == True  # No should be default for safety
        
        # Test custom title works
        custom_title = "Custom Close Dialog"
        custom_dialog = CloseDialog(title=custom_title)
        assert custom_dialog.windowTitle() == custom_title
        custom_dialog.close()
        
        # Test show_dialog return values by mocking QDialog.exec
        # Save original method
        original_exec = QDialog.exec
        
        # Test accepted case
        def mock_exec_accept(self):
            return QDialog.DialogCode.Accepted
        
        monkeypatch.setattr(QDialog, "exec", mock_exec_accept)
        assert dialog.show_dialog() == True
        
        # Test rejected case
        def mock_exec_reject(self):
            return QDialog.DialogCode.Rejected
        
        monkeypatch.setattr(QDialog, "exec", mock_exec_reject)
        assert dialog.show_dialog() == False
        
        # Restore original method
        monkeypatch.setattr(QDialog, "exec", original_exec)
    finally:
        # Ensure dialog is closed
        dialog.close()


@pytest.mark.qt
def test_error_dialog(qapp, caplog):
    """Tests the Error dialog display and message handling."""
    # Set up logging capture
    with caplog.at_level(logging.ERROR):
        # Create Error dialog with test message
        test_message = "Test error message"
        dialog = ErrorDialog(test_message)
        
        try:
            # Verify dialog properties
            assert dialog.windowTitle() == "Error"
            assert dialog.isModal() == True
            
            # Check error message is displayed correctly
            assert test_message in dialog._message_label.text()
            
            # Test OK button exists and is connected
            assert dialog._ok_button is not None
            assert dialog._ok_button.text() == "OK"
            assert dialog._ok_button.clicked.isConnected()
            
            # Verify error was logged
            assert "Error dialog displayed" in caplog.text
            assert test_message in caplog.text
            
            # Test custom title works
            custom_title = "Custom Error Title"
            custom_dialog = ErrorDialog(test_message, custom_title)
            assert custom_dialog.windowTitle() == custom_title
            custom_dialog.close()
        finally:
            # Ensure dialog is closed
            dialog.close()


@pytest.mark.qt
def test_help_dialog(qapp, monkeypatch):
    """Tests the Help dialog content and navigation."""
    # Create Help dialog with test topic
    test_topic = "general"
    dialog = HelpDialog(help_topic=test_topic)
    
    try:
        # Verify help content loads correctly
        assert dialog.windowTitle() == "Help"
        assert dialog.isModal() == True
        
        # Check title shows correct topic
        assert dialog._title_label.text() == "Help Topic: General"
        
        # Verify content browser has content
        assert dialog._content_browser is not None
        assert dialog._content_browser.toPlainText() != ""
        
        # Test close button exists and is connected
        assert dialog._close_button is not None
        assert dialog._close_button.text() == "Close"
        assert dialog._close_button.clicked.isConnected()
        
        # Test different topic
        another_topic = "arma_models"
        another_dialog = HelpDialog(help_topic=another_topic)
        try:
            assert another_dialog._title_label.text() == "Help Topic: Arma Models"
        finally:
            another_dialog.close()
        
        # Test link navigation by simulating a click
        # Save the original method for restoration
        original_load_content = dialog._load_help_content
        
        # Track method calls with a simple list
        called_topics = []
        
        def mock_load_content():
            called_topics.append(dialog._current_topic)
        
        # Apply the mock
        monkeypatch.setattr(dialog, "_load_help_content", mock_load_content)
        
        # Simulate topic link click
        dialog._on_link_clicked(QUrl("topic:garch_models"))
        
        # Verify topic was changed and content reload attempted
        assert dialog._current_topic == "garch_models"
        assert "garch_models" in called_topics
        assert dialog._title_label.text() == "Help Topic: Garch Models"
        
        # Restore original method
        monkeypatch.setattr(dialog, "_load_help_content", original_load_content)
    finally:
        # Ensure dialog is closed
        dialog.close()