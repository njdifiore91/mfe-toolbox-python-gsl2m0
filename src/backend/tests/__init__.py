"""
Test package initialization module that configures the test environment, exposes test fixtures and utilities,
and sets up pytest configuration for the MFE Toolbox test suite.
"""

import pytest
import logging
from pathlib import Path
import numba.testing
from hypothesis import settings, Verbosity

# Internal imports 
from .test_timeseries import TestARMAX as TestARMAModel
from .test_distributions import test_distribution_properties

# Define constants
TEST_DATA_DIR = Path('tests/data')  # Path to test data directory
RANDOM_SEED = 42  # Fixed random seed for reproducible test data generation
HYPOTHESIS_PROFILE = {
    'max_examples': 100,
    'deadline': None  # Disable deadline checks for performance tests
}

# Configure logging
logger = logging.getLogger(__name__)

def pytest_configure(config):
    """
    Pytest hook to configure test environment before test execution.
    
    Parameters
    ----------
    config : pytest.Config
        Pytest configuration object
        
    Returns
    -------
    None
    """
    # Register custom markers
    config.addinivalue_line("markers", "statistical: tests for statistical properties")
    config.addinivalue_line("markers", "async_test: tests for asynchronous functionality")
    config.addinivalue_line("markers", "performance: tests for performance and optimization")
    
    # Configure hypothesis settings for statistical tests
    settings.register_profile('statistical_tests', 
                             max_examples=HYPOTHESIS_PROFILE['max_examples'],
                             deadline=HYPOTHESIS_PROFILE['deadline'],
                             verbosity=Verbosity.normal)
    settings.load_profile('statistical_tests')
    
    # Initialize test data directory
    if not TEST_DATA_DIR.exists():
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created test data directory: {TEST_DATA_DIR}")
    
    # Configure logging for test execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up Numba test configuration
    numba.testing.TEST_PARALLEL = True
    numba.testing.TEST_EXTENDED_MATH = True
    
    logger.info("Test environment configured successfully")

def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to modify test collection and ordering.
    
    Parameters
    ----------
    config : pytest.Config
        Pytest configuration object
    items : List[pytest.Item]
        List of collected test items
        
    Returns
    -------
    None
    """
    # Add markers based on test names and modules
    for item in items:
        # Mark statistical tests
        if 'distribution' in item.nodeid or 'statistical' in item.nodeid:
            item.add_marker(pytest.mark.statistical)
        
        # Mark async tests
        if 'async' in item.nodeid or item.get_closest_marker('asyncio'):
            item.add_marker(pytest.mark.async_test)
            
        # Mark performance tests
        if 'performance' in item.nodeid or 'numba' in item.nodeid:
            item.add_marker(pytest.mark.performance)
    
    # Configure test ordering if needed
    # items.sort(key=lambda x: x.get_closest_marker("priority") is not None, reverse=True)
    
    # Skip tests based on markers if needed
    # if config.getoption("--skip-performance"):
    #    skip_performance = pytest.mark.skip(reason="Performance tests disabled")
    #    for item in items:
    #        if item.get_closest_marker("performance"):
    #            item.add_marker(skip_performance)

# Create wrapper classes for distribution tests
class TestGEDProperties:
    """Test class for GED distribution properties."""
    
    @staticmethod
    @pytest.mark.statistical
    def test_properties(data):
        """Test GED distribution properties using property-based testing."""
        test_distribution_properties(data)

class TestSkewedTProperties:
    """Test class for Skewed T distribution properties."""
    
    @staticmethod
    @pytest.mark.statistical
    def test_properties(data):
        """Test Skewed T distribution properties using property-based testing."""
        test_distribution_properties(data)

# Export the test classes and pytest hooks
__all__ = [
    'TestARMAModel',
    'TestGEDProperties', 
    'TestSkewedTProperties',
    'pytest_configure',
    'pytest_collection_modifyitems'
]