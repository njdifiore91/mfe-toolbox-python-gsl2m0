"""
Pytest configuration and shared fixtures for testing the MFE Toolbox GUI components implemented with PyQt6.
Provides common test setup, teardown and utilities for GUI testing.
"""

import logging
import time
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
import pytest_asyncio

from utils.qt_helpers import create_widget

# Global logger instance
logger = logging.getLogger(__name__)

def pytest_configure(config):
    """
    Configures pytest with custom markers and settings for GUI testing
    
    Args:
        config: Pytest configuration object
    """
    # Register qt marker for GUI tests
    config.addinivalue_line("markers", 
                           "qt: mark test as requiring Qt GUI components")
    
    # Register asyncio marker for async tests
    config.addinivalue_line("markers",
                           "asyncio: mark test as requiring asyncio support")
    
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up Qt test environment
    logger.info("Configuring Qt test environment")

class QtTestHelper:
    """
    Helper class providing common Qt testing utilities and fixtures
    """
    
    def __init__(self):
        """
        Initializes Qt test environment
        """
        # Use existing application instance
        self.app = QApplication.instance()
        
        # Initialize QTest bot
        self.qtbot = QTest
        
        # Configure test environment
        logger.debug("QtTestHelper initialized")
    
    def wait_for(self, condition, timeout=1000):
        """
        Waits for a condition with timeout
        
        Args:
            condition: Callable that returns True when condition is met
            timeout: Timeout in milliseconds
            
        Returns:
            bool: Whether condition was met
        """
        # Start timeout timer
        start_time = time.time()
        check_interval = 50  # ms
        
        # Check condition periodically
        while True:
            # Process events
            self.app.processEvents()
            
            # Check condition
            if condition():
                return True
                
            # Check if timeout occurred
            if (time.time() - start_time) * 1000 > timeout:
                return False
                
            # Wait before checking again
            QTest.qWait(check_interval)

@pytest.fixture(scope="session")
def qt_app():
    """
    Provides QApplication fixture for tests
    
    Returns:
        QApplication: Application instance
    """
    # Create QApplication instance if not exists
    app = QApplication.instance() or QApplication([])
    
    # Return fixture
    yield app
    
    # No cleanup needed - Qt will handle this at the end of the session

@pytest.fixture
def qtbot():
    """
    Provides QTest bot fixture for widget testing
    
    Returns:
        QTest: Qt test utility
    """
    return QTest

@pytest.fixture
def qt_helper(qt_app):
    """
    Provides QtTestHelper fixture for common test utilities
    
    Returns:
        QtTestHelper: Helper instance with test utilities
    """
    helper = QtTestHelper()
    return helper