"""
Shared pytest fixtures and configuration for testing the MFE Toolbox backend components.

This module provides:
1. Common test data generators (returns, time series, etc.)
2. Model parameter configurations for GARCH, ARMAX, and other models
3. Async test support
4. Custom pytest markers and settings
"""

import pytest
import numpy as np
from scipy import stats
from hypothesis import settings, strategies as st, HealthCheck

# Global constants for testing
RANDOM_SEED = 42
VALID_GARCH_TYPES = ['GARCH', 'EGARCH', 'GJR-GARCH', 'TGARCH', 'APARCH']
VALID_DISTRIBUTIONS = ['normal', 'student-t', 'ged', 'skewed-t']

def pytest_configure(config):
    """
    Configures pytest with custom markers and settings.
    
    Args:
        config: pytest.Config object for the pytest session
    """
    # Register custom markers
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "slow: mark a test as slow running")
    config.addinivalue_line("markers", "numba: mark a test that validates Numba optimization")
    config.addinivalue_line("markers", "statistical: mark a test that validates statistical properties")
    config.addinivalue_line("markers", "asyncio: mark as an asynchronous test")
    
    # Configure pytest-asyncio for async test support
    try:
        import pytest_asyncio
        pytest_asyncio.plugin.pytest_configure(config)
    except ImportError:
        pass
    
    # Configure hypothesis for statistical testing
    settings.register_profile(
        "stats", 
        max_examples=100, 
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow, 
            HealthCheck.function_scoped_fixture
        ]
    )
    settings.load_profile("stats")
    
    # Configure test coverage settings
    config.option.cov_report = {"term-missing": True}
    config.option.cov_config = ".coveragerc"

@pytest.fixture
def sample_returns():
    """
    Generate synthetic return data for testing.
    
    Returns:
        np.ndarray: A numpy array of synthetic financial returns with 1000 observations
                   following a standard normal distribution
    """
    np.random.seed(RANDOM_SEED)
    return np.random.normal(0, 1, 1000)

@pytest.fixture
def garch_model_params():
    """
    Provide standard GARCH model parameters for testing.
    
    Returns:
        dict: A dictionary of GARCH model parameters with the following keys:
             - p: GARCH order
             - q: ARCH order
             - omega: Constant
             - alpha: ARCH parameter
             - beta: GARCH parameter
             - dist: Error distribution
    """
    return {
        'p': 1,          # GARCH order
        'q': 1,          # ARCH order
        'omega': 0.05,   # Constant
        'alpha': 0.1,    # ARCH parameter
        'beta': 0.85,    # GARCH parameter
        'dist': 'normal' # Error distribution
    }

@pytest.fixture
def armax_model_params():
    """
    Provide standard ARMAX model parameters for testing.
    
    Returns:
        dict: A dictionary of ARMAX model parameters with the following keys:
             - ar_order: AR order
             - ma_order: MA order
             - exog: Exogenous variables (None by default)
             - constant: Boolean indicating whether to include a constant
             - ar_params: Array of AR parameters
             - ma_params: Array of MA parameters
             - constant_value: Value of the constant term
    """
    return {
        'ar_order': 2,                   # AR order
        'ma_order': 1,                   # MA order
        'exog': None,                    # Exogenous variables
        'constant': True,                # Include constant
        'ar_params': np.array([0.5, 0.2]), # AR parameters
        'ma_params': np.array([0.3]),      # MA parameters
        'constant_value': 0.01           # Constant value
    }

@pytest.fixture
def async_test_timeout():
    """
    Provide timeout settings for async tests.
    
    Returns:
        float: Default timeout in seconds for async tests
              to prevent tests from hanging indefinitely
    """
    return 5.0  # 5 seconds default timeout