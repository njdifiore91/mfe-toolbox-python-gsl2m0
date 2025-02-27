"""
Tests for multivariate volatility models including BEKK and DCC implementations.

This module provides comprehensive tests for the volatility modeling capabilities
of the MFE Toolbox, focusing on multivariate GARCH models for cross-asset volatility
analysis. It validates parameter estimation, forecasting capabilities, and numerical
optimization using pytest and hypothesis frameworks.

The tests cover:
- BEKK model initialization and parameter validation
- DCC model setup and correlation matrix properties
- Asynchronous parameter estimation with simulated data
- Multi-step ahead volatility and correlation forecasting
- Numerical optimization and convergence properties
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st
import pytest_asyncio

# Internal imports
from ...backend.models.volatility import BEKK, DCC
from ...backend.core.optimization import optimize_garch


@pytest.mark.parametrize('p,q', [(1,1), (2,1), (1,2), (2,2)])
def test_bekk_initialization(p, q):
    """
    Tests BEKK model initialization with various parameter configurations.
    
    This test verifies that BEKK models can be properly initialized with
    different p and q orders, and that the model parameters are correctly
    set up with appropriate dimensions.
    
    Parameters
    ----------
    p, q : int
        GARCH and ARCH orders for the BEKK model
    """
    # Initialize BEKK model
    model = BEKK(p=p, q=q)
    
    # Verify model attributes are correctly set
    assert model.p == p
    assert model.parameters is None  # Parameters should be None before estimation
    assert model.covariance is None  # Covariance should be None before estimation
    assert model.likelihood is None  # Likelihood should be None before estimation
    assert model.converged is False  # Converged should be False before estimation
    
    # Verify distribution setting
    assert model.distribution == 'normal'  # Default distribution should be normal
    
    # Test with different distribution
    model_t = BEKK(p=p, q=q, distribution='student-t')
    assert model_t.distribution == 'student-t'


@pytest.mark.asyncio
async def test_bekk_parameter_estimation():
    """
    Tests BEKK model parameter estimation using simulated data.
    
    This test validates the asynchronous parameter estimation process
    for BEKK models using simulated return data, checking parameter
    convergence and covariance matrix properties.
    """
    # Generate simulated multivariate return data
    np.random.seed(42)  # For reproducibility
    T, N = 500, 2  # Time periods, number of assets
    returns = np.random.randn(T, N) * 0.01  # Low variance for stability
    
    # Add some volatility clustering
    for t in range(1, T):
        if t % 50 < 25:  # Create volatility regime shifts
            returns[t] = returns[t] * 2.0
    
    # Initialize BEKK model
    model = BEKK(p=1, q=1)
    
    # Perform async parameter estimation
    await model.async_fit(returns)
    
    # Check if estimation converged
    assert model.converged, "BEKK model estimation should converge with simulated data"
    
    # Verify parameter dimensions
    n_params = len(model.parameters)
    n_c_params = N * (N + 1) // 2  # Lower triangular elements of C
    n_a_params = N * N  # Elements of A matrix
    n_b_params = N * N  # Elements of B matrix
    expected_n_params = n_c_params + n_a_params + n_b_params
    
    assert n_params == expected_n_params, f"Expected {expected_n_params} parameters, got {n_params}"
    
    # Verify likelihood is reasonable
    assert model.likelihood is not None
    assert np.isfinite(model.likelihood)
    
    # Test forecasting
    horizon = 10
    forecasts = model.forecast(horizon)
    
    # Verify forecast dimensions
    assert forecasts.shape == (N, N, horizon)
    
    # Verify all forecasts are positive definite
    for h in range(horizon):
        # Check eigenvalues are positive
        eigvals = np.linalg.eigvals(forecasts[:, :, h])
        assert np.all(eigvals > 0), f"Forecast at horizon {h} is not positive definite"


@pytest.mark.parametrize('p,q', [(1,1), (2,1), (1,2)])
def test_dcc_initialization(p, q):
    """
    Tests DCC model initialization and parameter setup.
    
    This test ensures that DCC models can be properly initialized with
    different p and q orders, and that the model properties are
    correctly established.
    
    Parameters
    ----------
    p, q : int
        GARCH and ARCH orders for the DCC model
    """
    # Initialize DCC model
    model = DCC(p=p, q=q)
    
    # Verify model attributes are correctly set
    assert model.p == p
    assert model.parameters is None  # Parameters should be None before estimation
    assert model.correlation is None  # Correlation should be None before estimation
    assert model.likelihood is None  # Likelihood should be None before estimation
    assert model.converged is False  # Converged should be False before estimation
    
    # Verify distribution setting
    assert model.distribution == 'normal'  # Default distribution should be normal
    
    # Test with different distribution
    model_t = DCC(p=p, q=q, distribution='student-t')
    assert model_t.distribution == 'student-t'
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        DCC(p=-1, q=q)  # p should be non-negative
    
    with pytest.raises(ValueError):
        DCC(p=p, q=-1)  # q should be non-negative
    
    with pytest.raises(ValueError):
        DCC(p=0, q=0)  # At least one of p, q should be positive


@pytest.mark.asyncio
async def test_dcc_estimation():
    """
    Tests DCC model estimation and correlation forecasting.
    
    This test validates the asynchronous parameter estimation process
    for DCC models using simulated return data, and verifies the
    correlation matrix properties and forecasting capabilities.
    """
    # Generate simulated multivariate return data
    np.random.seed(42)  # For reproducibility
    T, N = 500, 2  # Time periods, number of assets
    
    # Create base returns with some correlation
    corr = 0.5
    cov = np.array([[1.0, corr], [corr, 1.0]]) * 0.01
    returns = np.random.multivariate_normal(mean=np.zeros(N), cov=cov, size=T)
    
    # Add some volatility clustering
    for t in range(1, T):
        if t % 50 < 25:  # Create volatility regime shifts
            returns[t] = returns[t] * 2.0
    
    # Initialize DCC model
    model = DCC(p=1, q=1)
    
    # Perform async parameter estimation
    await model.async_fit(returns)
    
    # Check if estimation converged
    assert model.converged, "DCC model estimation should converge with simulated data"
    
    # Verify parameters include both univariate GARCH and DCC parameters
    assert model.parameters is not None
    assert len(model.parameters) >= 3 * N + 2  # 3 params per asset + 2 DCC params
    
    # Verify likelihood is reasonable
    assert model.likelihood is not None
    assert np.isfinite(model.likelihood)
    
    # Test correlation forecasting
    horizon = 10
    forecast_corr = model.forecast_correlation(horizon)
    
    # Verify forecast dimensions
    assert forecast_corr.shape == (N, N, horizon)
    
    # Verify all correlation matrices are valid (diagonal = 1, -1 ≤ off-diagonal ≤ 1)
    for h in range(horizon):
        # Check diagonal elements
        np.testing.assert_almost_equal(np.diag(forecast_corr[:, :, h]), np.ones(N))
        
        # Check off-diagonal elements
        for i in range(N):
            for j in range(i+1, N):
                corr_value = forecast_corr[i, j, h]
                assert -1.0 <= corr_value <= 1.0, f"Invalid correlation value: {corr_value}"
                assert forecast_corr[i, j, h] == forecast_corr[j, i, h], "Correlation matrix not symmetric"


class TestVolatilityModels:
    """
    Test class for multivariate volatility model implementations.
    
    This class provides common test fixtures and helper methods for
    testing multivariate volatility models. It generates test data
    and validates model properties across multiple test scenarios.
    """
    
    def setup_method(self, method):
        """
        Setup method run before each test.
        
        This method initializes test data and model instances,
        ensuring a clean test environment for each test case.
        
        Parameters
        ----------
        method : function
            The test method being set up
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate test data
        self.T = 500  # Number of observations
        self.N = 2    # Number of assets
        self.returns = self.generate_test_data(self.T, self.N)
        
        # Initialize model instances
        self.bekk_model = BEKK(p=1, q=1)
        self.dcc_model = DCC(p=1, q=1)
    
    def generate_test_data(self, n_obs, n_assets):
        """
        Helper method to generate test data.
        
        This method creates synthetic multivariate return data with
        properties suitable for testing volatility models, including
        volatility clustering and cross-asset correlation.
        
        Parameters
        ----------
        n_obs : int
            Number of observations (time periods)
        n_assets : int
            Number of assets
            
        Returns
        -------
        ndarray
            Simulated return data with shape (n_obs, n_assets)
        """
        # Create base returns with correlation
        corr = 0.5
        cov = np.zeros((n_assets, n_assets))
        
        # Fill covariance matrix
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    cov[i, j] = 0.01  # Variance
                else:
                    cov[i, j] = corr * np.sqrt(0.01 * 0.01)  # Covariance
        
        # Generate multivariate normal returns
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=cov,
            size=n_obs
        )
        
        # Add volatility clustering
        for t in range(1, n_obs):
            if t % 50 < 25:  # Create volatility regime shifts
                returns[t] = returns[t] * 1.5 + returns[t-1] * 0.2
        
        return returns