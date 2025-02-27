"""
Test suite for the async_helpers module, validating asynchronous operation support
in the PyQt6-based GUI environment.
"""

import pytest
import pytest_asyncio
import asyncio
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from ...utils.async_helpers import run_async, AsyncRunner


class AsyncHelpersTestCase:
    """
    Test fixture class providing common setup for async helper tests.
    """
    
    def __init__(self):
        """
        Initializes test case with Qt application instance.
        """
        self.app = QApplication.instance() or QApplication([])
        self.runner = AsyncRunner()
    
    def setUp(self):
        """
        Test case setup method.
        
        Creates a fresh Qt application and AsyncRunner instance for each test.
        """
        self.app = QApplication.instance() or QApplication([])
        self.runner = AsyncRunner()
    
    def tearDown(self):
        """
        Test case cleanup method.
        
        Stops the AsyncRunner and cleans up the Qt application.
        """
        if hasattr(self, 'runner') and self.runner:
            self.runner.stop()


@pytest.mark.asyncio
async def test_run_async_basic():
    """
    Tests basic async function execution through run_async helper.
    """
    # Ensure QApplication exists
    app = QApplication.instance() or QApplication([])
    
    # Flag to verify Qt event loop remained responsive
    qt_event_processed = False
    
    # Create a timer to verify Qt event processing
    def mark_qt_responsive():
        nonlocal qt_event_processed
        qt_event_processed = True
    
    # Schedule a Qt timer to fire during async operation
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(mark_qt_responsive)
    timer.start(50)  # 50ms should be during the 100ms sleep
    
    # Create a simple async coroutine
    async def simple_coro():
        await asyncio.sleep(0.1)  # 100ms sleep
        return 42
    
    # Execute the coroutine through run_async
    result = run_async(simple_coro)
    
    # Verify the correct result is returned
    assert result == 42
    
    # Check Qt event loop remained responsive
    assert qt_event_processed, "Qt event loop did not process events during async operation"


@pytest.mark.asyncio
async def test_run_async_exception():
    """
    Tests error handling in run_async helper.
    """
    # Ensure QApplication exists
    app = QApplication.instance() or QApplication([])
    
    # Create an async coroutine that raises an exception
    async def error_coro():
        await asyncio.sleep(0.1)
        raise ValueError("Test exception")
    
    # Verify the exception is properly propagated
    with pytest.raises(ValueError, match="Test exception"):
        run_async(error_coro)


@pytest.mark.asyncio
async def test_async_runner_lifecycle():
    """
    Tests AsyncRunner class lifecycle and resource management.
    """
    # Set up test case
    test_case = AsyncHelpersTestCase()
    test_case.setUp()
    
    try:
        # Create a test coroutine
        async def test_coro():
            await asyncio.sleep(0.1)
            return "completed"
        
        # Execute the coroutine
        result = test_case.runner.run_async(test_coro())
        
        # Verify proper execution
        assert result == "completed"
        
        # Test cleanup
        test_case.runner.stop()
        
        # Verify we can recreate and use runner after stopping
        test_case.runner = AsyncRunner()
        result = test_case.runner.run_async(test_coro())
        assert result == "completed"
    finally:
        # Clean up
        test_case.tearDown()


@pytest.mark.asyncio
async def test_async_runner_concurrent():
    """
    Tests concurrent operation handling in AsyncRunner.
    """
    # Set up test case
    test_case = AsyncHelpersTestCase()
    test_case.setUp()
    
    # Flag to verify Qt event loop remained responsive
    qt_event_processed = False
    
    # Create a timer to verify Qt event processing
    def mark_qt_responsive():
        nonlocal qt_event_processed
        qt_event_processed = True
    
    try:
        # Schedule a Qt timer to fire during async operation
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(mark_qt_responsive)
        timer.start(50)  # 50ms should be during the tasks
        
        # Create multiple async coroutines
        async def task1():
            await asyncio.sleep(0.1)
            return "task1"
            
        async def task2():
            await asyncio.sleep(0.05)
            return "task2"
        
        # Execute tasks
        result1 = test_case.runner.run_async(task1())
        result2 = test_case.runner.run_async(task2())
        
        # Verify results
        assert result1 == "task1"
        assert result2 == "task2"
        
        # Verify Qt event loop remained responsive
        assert qt_event_processed, "Qt event loop did not process events during async operations"
    finally:
        # Clean up
        test_case.tearDown()