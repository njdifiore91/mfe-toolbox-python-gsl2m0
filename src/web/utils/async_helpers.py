"""
Utility module providing asynchronous operation helpers for the MFE Toolbox GUI,
enabling non-blocking execution of long-running computations while maintaining
UI responsiveness.
"""

import asyncio
from typing import Any, Callable, Coroutine, Optional, TypeVar, cast
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop, QTimer

# Set up module logger
logger = logging.getLogger(__name__)

# Type variable for return type flexibility
T = TypeVar('T')

def run_async(coroutine_func: Callable[..., Coroutine[Any, Any, T]], *args: Any, **kwargs: Any) -> T:
    """
    Executes an asynchronous coroutine while maintaining Qt event loop responsiveness.
    
    This function creates a new event loop if necessary and integrates it with the Qt event loop
    to ensure UI responsiveness during long-running operations.
    
    Args:
        coroutine_func: The coroutine function to execute
        *args: Positional arguments to pass to the coroutine function
        **kwargs: Keyword arguments to pass to the coroutine function
        
    Returns:
        Result of the coroutine execution
        
    Raises:
        Any exception that occurs during coroutine execution
    """
    try:
        # Create the coroutine with the provided arguments
        coro = coroutine_func(*args, **kwargs)
        
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            logger.debug("Creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Create a Qt event loop for integration
        qt_loop = QEventLoop()
        
        # Create a future to hold the result
        future = asyncio.run_coroutine_threadsafe(coro, loop) if loop.is_running() else None
        
        if future:
            # Handle the case where the event loop is already running
            while not future.done():
                # Process Qt events
                QApplication.instance().processEvents()
                qt_loop.processEvents()
                # Avoid high CPU usage
                QTimer.singleShot(10, lambda: None)
                
            # Get the result or propagate exception
            return future.result()
        else:
            # Handle the case where we need to run the event loop
            
            # Set up a task to run the coroutine
            task = loop.create_task(coro)
            
            # Set up a periodic callback to process Qt events
            def process_qt_events():
                QApplication.instance().processEvents()
                qt_loop.processEvents()
                if not task.done():
                    # Schedule next callback
                    loop.call_later(0.01, process_qt_events)
            
            # Schedule the first callback
            loop.call_soon(process_qt_events)
            
            # Run the loop until the task is done
            return loop.run_until_complete(task)
    
    except Exception as e:
        logger.error(f"Failed to execute async operation: {str(e)}")
        raise


class AsyncRunner:
    """
    Helper class for managing asynchronous operations in the PyQt6 GUI environment.
    
    This class provides a reusable interface for executing asynchronous tasks
    while maintaining GUI responsiveness through integration of asyncio and Qt event loops.
    """
    
    def __init__(self) -> None:
        """
        Initializes the AsyncRunner with event loop setup.
        
        Sets up the asyncio and Qt event loops for integration and configures
        exception handling.
        """
        # Initialize event loops
        try:
            self._event_loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            logger.debug("Creating new event loop for AsyncRunner")
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            self._created_loop = True
        else:
            self._created_loop = False
            
        # Create Qt event loop
        self._qt_loop = QEventLoop()
        
        # Set up exception handlers
        self._event_loop.set_exception_handler(self._handle_exception)
        
        logger.debug("AsyncRunner initialized with event loops")
        
    def _handle_exception(self, loop: asyncio.AbstractEventLoop, context: dict) -> None:
        """
        Handles exceptions raised during asynchronous operations.
        
        Args:
            loop: The event loop where the exception occurred
            context: The context information for the exception
        """
        exception = context.get('exception')
        message = context.get('message', 'No error message available')
        
        if exception:
            logger.error(f"Async exception: {message}, Exception: {str(exception)}")
        else:
            logger.error(f"Async error: {message}")
    
    def run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        Executes an asynchronous task while maintaining GUI responsiveness.
        
        Args:
            coro: The coroutine to execute
            
        Returns:
            Result of the asynchronous operation
            
        Raises:
            Any exception that occurs during task execution
        """
        try:
            logger.debug("Running async task")
            
            # Create a future to hold the result if event loop is running
            future = asyncio.run_coroutine_threadsafe(coro, self._event_loop) if self._event_loop.is_running() else None
            
            if future:
                # Handle the case where the event loop is already running
                while not future.done():
                    # Process Qt events
                    QApplication.instance().processEvents()
                    self._qt_loop.processEvents()
                    # Avoid high CPU usage
                    QTimer.singleShot(10, lambda: None)
                    
                # Get the result or propagate exception
                return future.result()
            else:
                # Handle the case where we need to run the event loop
                
                # Set up a task to run the coroutine
                task = self._event_loop.create_task(coro)
                
                # Set up a periodic callback to process Qt events
                def process_qt_events():
                    QApplication.instance().processEvents()
                    self._qt_loop.processEvents()
                    if not task.done():
                        # Schedule next callback
                        self._event_loop.call_later(0.01, process_qt_events)
                
                # Schedule the first callback
                self._event_loop.call_soon(process_qt_events)
                
                # Run the loop until the task is done
                return self._event_loop.run_until_complete(task)
                
        except Exception as e:
            logger.error(f"Error in run_async: {str(e)}")
            raise
            
    def stop(self) -> None:
        """
        Stops the async runner and cleans up resources.
        
        This method should be called when the runner is no longer needed
        to ensure proper cleanup of event loops and resources.
        """
        try:
            # Stop Qt event loop
            if self._qt_loop.isRunning():
                self._qt_loop.quit()
                
            # Close asyncio event loop if we created it
            if self._created_loop and not self._event_loop.is_closed() and not self._event_loop.is_running():
                # Get pending tasks
                pending = asyncio.all_tasks(self._event_loop)
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                    
                # Run the event loop until tasks are cancelled
                if pending:
                    self._event_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                    
                # Close the event loop
                self._event_loop.close()
                
            logger.debug("AsyncRunner stopped and cleaned up")
            
        except Exception as e:
            logger.error(f"Error stopping AsyncRunner: {str(e)}")