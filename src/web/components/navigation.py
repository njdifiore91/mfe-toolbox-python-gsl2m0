"""
Navigation component that provides page navigation controls and state management
for the MFE Toolbox GUI using PyQt6. Handles navigation between diagnostic plots,
results pages and maintains navigation state.
"""

from typing import Optional

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel  # PyQt6 version 6.6.1
from PyQt6.QtCore import pyqtSignal, pyqtSlot  # PyQt6 version 6.6.1

from web.utils.qt_helpers import create_widget


class SignalBlocker:
    """
    Context manager for temporarily blocking Qt signals during navigation state updates.
    
    This prevents recursive signal emission when updating UI components.
    """
    
    def __init__(self, widget):
        """
        Initialize the signal blocker with the widget to block.
        
        Args:
            widget: The QWidget whose signals should be temporarily blocked
        """
        self.widget = widget
        self.was_blocked = False
    
    def __enter__(self):
        """Block signals when entering context."""
        self.was_blocked = self.widget.signalsBlocked()
        self.widget.blockSignals(True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous signal blocking state when exiting context."""
        self.widget.blockSignals(self.was_blocked)


class NavigationWidget(QWidget):
    """
    Widget providing navigation controls and state management for plot and
    results navigation with async support.
    """
    
    # Signal emitted when page changes with the new page number
    page_changed = pyqtSignal(int)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the navigation widget with navigation controls and state management.
        
        Args:
            parent: Optional parent widget
        """
        # Call parent QWidget constructor with optional parent
        super().__init__(parent)
        
        # Initialize navigation state
        self._current_page = 1
        self._total_pages = 1
        
        # UI components
        self._prev_button = None
        self._next_button = None
        self._page_label = None
        
        # Create and arrange navigation controls
        self.setup_ui()
        
        # Initialize button states based on current page
        self.update_button_states()
    
    def setup_ui(self) -> None:
        """
        Creates and arranges the navigation UI components with consistent styling.
        """
        # Create layout with proper margins and spacing
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(10)
        
        # Create previous button
        self._prev_button = create_widget('QPushButton', {
            'text': '◀ Previous',
            'toolTip': 'Go to previous page',
            'cursor': 'PointingHandCursor'
        })
        self._prev_button.clicked.connect(lambda: self.navigate('prev'))
        
        # Create page label
        self._page_label = create_widget('QLabel', {
            'text': f'{self._current_page}/{self._total_pages}',
            'alignment': 132  # Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        })
        
        # Create next button
        self._next_button = create_widget('QPushButton', {
            'text': 'Next ▶',
            'toolTip': 'Go to next page',
            'cursor': 'PointingHandCursor'
        })
        self._next_button.clicked.connect(lambda: self.navigate('next'))
        
        # Add widgets to layout with proper alignment
        layout.addStretch()
        layout.addWidget(self._prev_button)
        layout.addWidget(self._page_label)
        layout.addWidget(self._next_button)
        layout.addStretch()
        
        # Set widget layout
        self.setLayout(layout)
    
    @pyqtSlot(str)
    def navigate(self, direction: str) -> None:
        """
        Handles navigation button clicks and updates current page with validation.
        
        Args:
            direction: Navigation direction ('prev' or 'next')
        
        Raises:
            ValueError: If direction is not 'prev' or 'next'
        """
        # Validate direction parameter
        if direction not in ('prev', 'next'):
            raise ValueError(f"Invalid navigation direction: {direction}. Must be 'prev' or 'next'")
        
        # Calculate new page number
        new_page = self._current_page
        if direction == 'prev':
            new_page = max(1, self._current_page - 1)
        elif direction == 'next':
            new_page = min(self._total_pages, self._current_page + 1)
        
        # If page changed, update state and emit signal
        if new_page != self._current_page:
            # Use SignalBlocker to prevent recursive signals
            with SignalBlocker(self):
                self._current_page = new_page
                
                # Update navigation button states
                self.update_button_states()
                
                # Update page label text
                self._page_label.setText(f"{self._current_page}/{self._total_pages}")
            
            # Emit page_changed signal with new page number
            self.page_changed.emit(self._current_page)
    
    def set_total_pages(self, total: int) -> None:
        """
        Sets the total number of pages and updates navigation state with validation.
        
        Args:
            total: Total number of pages
            
        Raises:
            ValueError: If total is less than 1
        """
        # Validate total pages parameter
        if total < 1:
            raise ValueError(f"Total pages must be at least 1, got {total}")
        
        # Store previous state to detect changes
        prev_page = self._current_page
        prev_total = self._total_pages
        
        # Use SignalBlocker during state update
        with SignalBlocker(self):
            # Update total pages count
            self._total_pages = total
            
            # Reset current page to 1 if needed
            if self._current_page > self._total_pages:
                self._current_page = 1
            
            # Update navigation button states
            self.update_button_states()
            
            # Update page label text
            self._page_label.setText(f"{self._current_page}/{self._total_pages}")
        
        # Emit page_changed signal if page changed
        if self._current_page != prev_page or self._total_pages != prev_total:
            self.page_changed.emit(self._current_page)
    
    def update_button_states(self) -> None:
        """
        Updates the enabled state of navigation buttons based on current position.
        """
        # Use SignalBlocker during button state updates
        with SignalBlocker(self):
            # Set prev button state - disable if on first page
            self._prev_button.setEnabled(self._current_page > 1)
            
            # Set next button state - disable if on last page
            self._next_button.setEnabled(self._current_page < self._total_pages)