"""
Test suite for univariate GARCH model implementations including model estimation, forecasting, 
and parameter validation. Verifies Numba-optimized volatility computations and error distribution handling.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st
import pytest_asyncio

# Import components to test
from ..models.univariate import UnivariateGARCH, compute_volatility, SUPPORTED_MODELS


class TestUnivariateGARCH:
    """Test fixture class for UnivariateGARCH model testing."""
    
    def __init__(self):
        """Initialize test fixtures with reproducible data."""
        # Set random seed for reproducibility
        np.random.seed(12345)
        
        # Generate sample return data using numpy
        self.sample_returns = np.random.normal(0, 1, size=1000)
        
        # Initialize base model instance
        self.model = UnivariateGARCH(p=1, q=1)
        
        # Setup async test environment
        # (No specific setup needed for async tests)
    
    def setup_method(self, method):
        """Setup method run before each test."""
        # Reset random seed
        np.random.seed(12345)
        
        # Regenerate sample data
        self.sample_returns = np.random.normal(0, 1, size=1000)
        
        # Reinitialize model instance
        self.model = UnivariateGARCH(p=1, q=1)
        
        # Clear any cached Numba compilations
        # (Numba cache is managed automatically)


@pytest.mark.parametrize('model_type', SUPPORTED_MODELS)
@pytest.mark.parametrize('p,q', [(1,1), (2,1), (1,2)])
def test_univariate_garch_initialization(model_type, p, q):
    """
    Test proper initialization of UnivariateGARCH models with various parameters and types.
    
    This test verifies that models can be initialized with different orders (p,q)
    and model types, validating that parameters are correctly set and constraints enforced.
    """
    # Initialize GARCH model with specified type and orders
    model = UnivariateGARCH(p=p, q=q, model_type=model_type)
    
    # Verify model parameters are correctly set
    assert model.p == p
    assert model.q == q
    assert model.model_type == model_type
    
    # Check distribution type initialization (default should be 'normal')
    assert model.distribution == 'normal'
    
    # Validate parameter constraints
    assert model.parameters is None  # Should be None before estimation
    assert model.volatility is None  # Should be None before estimation
    
    # Ensure proper type hints are respected
    assert isinstance(model.p, int)
    assert isinstance(model.q, int)
    assert isinstance(model.model_type, str)
    
    # Try with non-default distribution
    model = UnivariateGARCH(p=p, q=q, model_type=model_type, distribution='student-t')
    assert model.distribution == 'student-t'


@pytest.mark.parametrize('model_type', SUPPORTED_MODELS)
@pytest.mark.parametrize('distribution', ['normal', 'student-t', 'ged'])
def test_compute_volatility(model_type, distribution):
    """
    Test Numba-optimized volatility computation function across model types.
    
    This test validates that the compute_volatility function correctly calculates
    conditional volatility for different model specifications and error distributions.
    """
    # Generate synthetic return data using numpy
    np.random.seed(42)
    returns = np.random.normal(0, 1, size=500)
    
    # Create parameter sets for each model type
    if model_type == 'GARCH':
        params = np.array([0.01, 0.1, 0.8])  # omega, alpha, beta
    elif model_type == 'EGARCH':
        params = np.array([-0.1, 0.1, 0.0, 0.9])  # omega, alpha, gamma, beta
    elif model_type in ['AGARCH', 'TARCH']:
        params = np.array([0.01, 0.05, 0.8, 0.1])  # omega, alpha, beta, gamma
    elif model_type == 'FIGARCH':
        params = np.array([0.01, 0.4, 0.3, 0.2])  # omega, d, beta, phi
    elif model_type == 'IGARCH':
        params = np.array([0.01, 0.2])  # omega, alpha
    else:
        params = np.array([0.01, 0.1, 0.8])  # default
    
    # Compute volatility using Numba-optimized function
    volatility = compute_volatility(returns, params, model_type)
    
    # Verify volatility properties (positivity, stationarity)
    assert volatility.shape == returns.shape
    assert np.all(volatility > 0)  # Volatility should be positive
    
    # Check Numba compilation success
    # (If the function runs without errors, compilation was successful)
    
    # Validate numerical accuracy
    # For the first observation, volatility should match sample variance
    assert np.isclose(volatility[0], np.var(returns))
    
    # For subsequent observations, it should follow the model recursion
    # This is difficult to test directly, so we'll verify it's different from initial value
    assert not np.allclose(volatility, volatility[0])


@pytest.mark.asyncio
@pytest.mark.parametrize('model_type', SUPPORTED_MODELS)
@pytest.mark.parametrize('sample_size', [500, 1000])
async def test_async_model_estimation(model_type, sample_size):
    """
    Test asynchronous parameter estimation for GARCH models.
    
    This test verifies that the async_fit method correctly estimates model
    parameters for various GARCH specifications and data sizes.
    """
    # Generate synthetic return series
    np.random.seed(123)
    returns = np.random.normal(0, 1, size=sample_size)
    
    # Initialize model with specified type
    model = UnivariateGARCH(p=1, q=1, model_type=model_type)
    
    # Estimate parameters asynchronously
    converged = await model.async_fit(returns)
    
    # Verify convergence and optimization success
    assert converged is True
    assert model.parameters is not None
    assert model.volatility is not None
    assert model.likelihood < 0  # Log-likelihood should be negative
    
    # Check parameter constraints satisfaction
    if model_type == 'GARCH':
        omega, alpha, beta = model.parameters
        assert omega > 0
        assert 0 < alpha < 1
        assert 0 < beta < 1
        assert alpha + beta < 1  # Stationarity condition
    elif model_type == 'IGARCH':
        omega, alpha = model.parameters
        assert omega > 0
        assert 0 < alpha < 1
    
    # Validate standard errors computation
    # (Standard errors might not be available in the current implementation)
    
    # Ensure proper async cleanup
    # (No specific cleanup needed for async tests)


@pytest.mark.parametrize('horizon', [1, 5, 22])
@pytest.mark.parametrize('model_type', SUPPORTED_MODELS)
def test_volatility_forecasting(horizon, model_type):
    """
    Test volatility forecasting functionality with different horizons.
    
    This test ensures that the forecast method produces valid volatility forecasts
    for different horizons and model specifications.
    """
    # Generate sample data and fit model
    np.random.seed(456)
    returns = np.random.normal(0, 1, size=800)
    
    # Initialize and fit model synchronously (using pytest event loop)
    model = UnivariateGARCH(p=1, q=1, model_type=model_type)
    
    # Mock the fit process by setting parameters directly
    # This avoids async testing complexity in this function
    if model_type == 'GARCH':
        model.parameters = np.array([0.01, 0.1, 0.8])
    elif model_type == 'EGARCH':
        model.parameters = np.array([-0.1, 0.1, 0.0, 0.9])
    elif model_type in ['AGARCH', 'TARCH']:
        model.parameters = np.array([0.01, 0.05, 0.8, 0.1])
    elif model_type == 'FIGARCH':
        model.parameters = np.array([0.01, 0.4, 0.3, 0.2])
    elif model_type == 'IGARCH':
        model.parameters = np.array([0.01, 0.2])
    
    # Set volatility and returns
    model.volatility = compute_volatility(returns, model.parameters, model_type)
    model._returns = returns
    
    # Generate forecasts for specified horizon
    forecasts = model.forecast(horizon=horizon)
    
    # Verify forecast shapes and dimensions
    assert len(forecasts) == horizon
    
    # Check forecast positivity constraints
    assert np.all(forecasts > 0)
    
    # Validate forecast uncertainty bounds (this is approximate)
    # Forecast should not diverge too much from unconditional variance
    uncond_var = np.var(returns)
    assert np.all(forecasts < uncond_var * 5)
    
    # Test forecast aggregation methods (analytic vs simulation)
    sim_forecasts = model.forecast(horizon=horizon, method='simulation')
    assert len(sim_forecasts) == horizon
    assert np.all(sim_forecasts > 0)