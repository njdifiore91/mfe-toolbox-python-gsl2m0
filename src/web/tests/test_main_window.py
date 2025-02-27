"""
Test suite for the MFE Toolbox main window implementation, validating GUI functionality,
asynchronous operations, and user interactions using pytest and PyQt6 testing utilities.
"""

import logging
import pytest
import pytest_asyncio
from unittest import mock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

# Internal imports
from components.main_window import MainWindow
from utils.qt_helpers import create_widget
from utils.async_helpers import run_async

# Configure logger
logger = logging.getLogger(__name__)


class MainWindowTestFixture:
    """
    Test fixture class providing common setup for main window tests.
    """
    
    def __init__(self):
        """
        Initializes test fixture with QApplication instance.
        """
        # Create QApplication instance if it doesn't exist
        self.app = QApplication.instance() or QApplication([])
        
        # Initialize MainWindow
        self.window = MainWindow()
        
        # Configure test environment
        logger.debug("MainWindowTestFixture initialized")
    
    def setup(self):
        """
        Sets up test environment before each test.
        """
        # Reset window state
        if hasattr(self.window, 'on_reset_clicked'):
            self.window.on_reset_clicked()
        
        # Clear any previous test data
        self.window._has_unsaved_changes = False
        self.window._results_viewer = None
        
        # Initialize test configuration
        logger.debug("Test environment setup complete")
        
    def teardown(self):
        """
        Cleans up test environment after each test.
        """
        # Close main window
        self.window.close()
        
        # Clean up QApplication if needed
        # (Not closing app, as it might be used by other tests)
        
        # Reset test state
        logger.debug("Test environment cleanup complete")


@pytest.mark.asyncio
async def test_main_window_creation():
    """
    Tests successful creation and initialization of the main window.
    """
    # Create test fixture
    fixture = MainWindowTestFixture()
    
    try:
        # Set up test environment
        fixture.setup()
        
        # Verify window title and size
        assert "MFE Toolbox" in fixture.window.windowTitle()
        assert fixture.window.width() >= 800
        assert fixture.window.height() >= 600
        
        # Check menu bar existence
        assert hasattr(fixture.window, '_menu_bar')
        assert fixture.window._menu_bar is not None
        
        # Validate status bar initialization
        assert hasattr(fixture.window, '_status_bar')
        assert fixture.window._status_bar is not None
        
        # Ensure model configuration widget exists
        assert hasattr(fixture.window, '_model_config')
        assert fixture.window._model_config is not None
        
        # Verify diagnostic plots widget exists
        assert hasattr(fixture.window, '_diagnostic_plots')
        assert fixture.window._diagnostic_plots is not None
        
        # Check results viewer initialization
        assert hasattr(fixture.window, '_results_viewer')
        assert fixture.window._results_viewer is None  # Should be None initially
    finally:
        # Clean up
        fixture.teardown()


@pytest.mark.asyncio
async def test_model_estimation():
    """
    Tests asynchronous model estimation workflow.
    """
    # Create test fixture
    fixture = MainWindowTestFixture()
    
    try:
        # Set up test environment
        fixture.setup()
        
        # Configure test model parameters
        if hasattr(fixture.window, '_model_config') and hasattr(fixture.window._model_config, 'get_config'):
            # In a real test, we would set specific configuration options
            pass
        
        # Trigger estimation asynchronously
        await fixture.window.on_estimate_clicked()
        
        # Verify estimation results
        assert fixture.window._has_unsaved_changes is False
        
        # Check results display
        assert fixture.window._results_viewer is not None
        
        # Validate diagnostic plot updates
        assert hasattr(fixture.window, '_diagnostic_plots')
        
        # Check status bar message
        assert "Model estimation completed successfully" in fixture.window._status_bar.currentMessage()
    finally:
        # Clean up
        fixture.teardown()


def test_reset_functionality():
    """
    Tests reset button functionality.
    """
    # Create test fixture
    fixture = MainWindowTestFixture()
    
    try:
        # Set up test environment
        fixture.setup()
        
        # Set initial model configuration
        fixture.window._has_unsaved_changes = True
        
        # Create a mock results viewer if needed
        if fixture.window._results_viewer is None:
            fixture.window._results_viewer = mock.MagicMock()
        
        # Trigger reset action
        fixture.window.on_reset_clicked()
        
        # Verify configuration reset
        assert fixture.window._has_unsaved_changes is False
        
        # Validate results viewer reset
        assert fixture.window._results_viewer is None
        
        # Ensure window state reset
        assert "Reset completed" in fixture.window._status_bar.currentMessage()
    finally:
        # Clean up
        fixture.teardown()


def test_close_confirmation():
    """
    Tests close confirmation dialog functionality.
    """
    # Create test fixture
    fixture = MainWindowTestFixture()
    
    try:
        # Set up test environment
        fixture.setup()
        
        # Make changes to trigger unsaved state
        fixture.window._has_unsaved_changes = True
        
        # Mock CloseDialog to control behavior
        with mock.patch('components.main_window.CloseDialog') as MockDialog:
            # Configure mock to return configurable results
            mock_dialog_instance = mock.MagicMock()
            MockDialog.return_value = mock_dialog_instance
            
            # Create a mock event
            close_event = mock.MagicMock()
            
            # Test scenario 1: Dialog returns False (cancel close)
            mock_dialog_instance.show_dialog.return_value = False
            fixture.window.closeEvent(close_event)
            
            # Check that event.ignore() was called
            close_event.ignore.assert_called_once()
            close_event.accept.assert_not_called()
            
            # Reset the mock event
            close_event.reset_mock()
            
            # Test scenario 2: Dialog returns True (confirm close)
            mock_dialog_instance.show_dialog.return_value = True
            fixture.window.closeEvent(close_event)
            
            # Check that event.accept() was called
            close_event.accept.assert_called_once()
            close_event.ignore.assert_not_called()
    finally:
        # Clean up
        fixture.teardown()