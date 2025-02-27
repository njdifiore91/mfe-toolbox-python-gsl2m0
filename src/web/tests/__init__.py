"""
Test package initialization module for the MFE Toolbox web UI component.
Configures pytest test discovery and provides common test utilities for PyQt6-based GUI testing.
"""

import pytest
try:
    from PyQt6.QtWidgets import QApplication
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

# Register pytest-asyncio plugin for async testing support
pytest_plugins = ['pytest_asyncio']

# Create a single QApplication instance to be shared across all GUI tests
qt_app = QApplication([]) if HAS_PYQT else None


def pytest_configure(config):
    """
    Pytest hook to configure test environment before test collection.
    
    Parameters
    ----------
    config : pytest.Config
        The pytest configuration object
    
    Returns
    -------
    None
        Configuration applied
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", 
        "gui: mark test as requiring PyQt6 GUI components"
    )
    config.addinivalue_line(
        "markers", 
        "async_gui: mark test as requiring both PyQt6 GUI components and async support"
    )
    
    # Configure test discovery paths
    config.addinivalue_line(
        "testpaths", "web/tests"
    )
    
    # Configure asyncio for Qt event loop if PyQt6 is available
    if HAS_PYQT:
        try:
            import pytest_asyncio
            pytest_asyncio.plugin.default_event_loop = 'qt'
        except (ImportError, AttributeError):
            pass
    
    # Set up hypothesis strategies for GUI property testing if available
    try:
        from hypothesis import settings, HealthCheck
        settings.register_profile(
            "gui_tests", 
            suppress_health_check=[HealthCheck.function_scoped_fixture],
            deadline=None
        )
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to modify test collection behavior.
    
    Parameters
    ----------
    config : pytest.Config
        The pytest configuration object
    items : List[pytest.Item]
        Collected test items
    
    Returns
    -------
    None
        Test collection modified
    """
    # Mark tests according to their requirements
    for item in items:
        # Add 'gui' marker to tests that use PyQt6 widgets
        if 'qt' in item.keywords or 'pyqt' in item.keywords:
            item.add_marker(pytest.mark.gui)
        
        # Add 'async_gui' marker to tests that use async with PyQt6
        if ('qt' in item.keywords or 'pyqt' in item.keywords) and 'async' in item.keywords:
            item.add_marker(pytest.mark.async_gui)
    
    # Skip GUI tests if PyQt6 is not available
    if not HAS_PYQT:
        skip_gui = pytest.mark.skip(reason="PyQt6 is not installed")
        for item in items:
            if 'gui' in item.keywords:
                item.add_marker(skip_gui)
    
    # Configure async test requirements
    try:
        import pytest_asyncio
        for item in items:
            if 'async_gui' in item.keywords:
                item.add_marker(pytest.mark.asyncio)
    except ImportError:
        skip_async = pytest.mark.skip(reason="pytest-asyncio is not installed")
        for item in items:
            if 'async_gui' in item.keywords:
                item.add_marker(skip_async)
    
    # Re-order tests to run GUI tests after unit tests
    gui_tests = [item for item in items if 'gui' in item.keywords]
    non_gui_tests = [item for item in items if 'gui' not in item.keywords]
    items[:] = non_gui_tests + gui_tests
    
    # Apply hypothesis settings for GUI property tests if available
    try:
        from hypothesis import settings
        for item in items:
            if 'gui' in item.keywords and 'hypothesis' in item.keywords:
                item.add_marker(pytest.mark.hypothesis.settings(settings.get_profile('gui_tests')))
    except ImportError:
        pass