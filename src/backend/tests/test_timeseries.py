"""
Test suite for ARMA/ARMAX time series modeling functionality.

This module contains comprehensive tests for the ARMAX model class,
validating model estimation, forecasting, and diagnostic capabilities.
It uses pytest for standard unit testing and hypothesis for property-based
testing of statistical properties.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

# Internal imports
from ..models.timeseries import ARMAX, compute_acf, compute_pacf


class TestARMAX:
    """Test class for ARMAX model functionality."""
    
    def setup_method(self, method):
        """Setup method run before each test."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create synthetic test data
        self.n_obs = 1000
        self.ar_params = np.array([0.6, -0.2])
        self.ma_params = np.array([0.3, -0.1])
        
        # Generate synthetic ARMA process
        self.data = np.random.randn(self.n_obs)
        for t in range(2, self.n_obs):
            self.data[t] += (self.ar_params[0] * self.data[t-1] + 
                            self.ar_params[1] * self.data[t-2] +
                            self.ma_params[0] * np.random.randn() +
                            self.ma_params[1] * np.random.randn())
        
        # Initialize model instances for different tests
        self.arma_model = ARMAX(p=2, q=2)
        self.armax_model = ARMAX(p=2, q=2, trend='c')


@pytest.mark.parametrize('p,q,trend', [(1,1,'n'), (2,0,'c'), (0,1,'ct')])
def test_armax_initialization(p, q, trend):
    """Tests proper initialization of ARMAX model with various parameters."""
    # Create model
    model = ARMAX(p=p, q=q, trend=trend)
    
    # Verify model attributes
    assert model.p == p
    assert model.q == q
    assert model.trend == trend
    
    # Check parameter validation
    n_trend_params = 0
    if 'c' in trend:
        n_trend_params += 1
    if 't' in trend:
        n_trend_params += 1
    
    # Verify internal attributes are set correctly
    assert model._n_trend_params == n_trend_params
    assert len(model._param_names) == p + q + n_trend_params
    
    # Test invalid trend specification
    with pytest.raises(ValueError):
        invalid_model = ARMAX(p=p, q=q, trend='invalid')


@pytest.mark.asyncio
async def test_armax_estimation():
    """Tests asynchronous model estimation with simulated data."""
    # Generate synthetic time series
    np.random.seed(42)
    n = 500
    ar_params = np.array([0.6])
    ma_params = np.array([0.4])
    
    # Generate ARMA process
    data = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    for t in range(1, n):
        if t >= 1:
            data[t] += ar_params[0] * data[t-1]
        if t >= 1:
            data[t] += ma_params[0] * errors[t-1]
        data[t] += errors[t]
    
    # Create and fit ARMAX model
    model = ARMAX(p=1, q=1, trend='c')
    converged = await model.async_fit(data)
    
    # Verify estimation converged
    assert converged
    
    # Check parameter estimates (should be close to true values)
    ar_index = model._n_trend_params  # First param after trend terms
    ma_index = model._n_trend_params + model.p
    
    # Allow some tolerance in parameter estimation
    assert abs(model.params[ar_index] - ar_params[0]) < 0.2
    assert abs(model.params[ma_index] - ma_params[0]) < 0.2
    
    # Verify residuals properties
    assert model.residuals is not None
    assert len(model.residuals) == len(data)
    assert abs(np.mean(model.residuals[5:])) < 0.1  # Mean close to zero


@pytest.mark.asyncio
async def test_armax_forecasting():
    """Tests multi-step ahead forecasting functionality."""
    # Generate synthetic time series
    np.random.seed(42)
    n = 500
    ar_params = np.array([0.6])
    ma_params = np.array([0.4])
    
    # Generate ARMA process
    data = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    for t in range(1, n):
        if t >= 1:
            data[t] += ar_params[0] * data[t-1]
        if t >= 1:
            data[t] += ma_params[0] * errors[t-1]
        data[t] += errors[t]
    
    # Split into training and test sets
    train_data = data[:450]
    test_data = data[450:]
    
    # Create and fit ARMAX model
    model = ARMAX(p=1, q=1, trend='c')
    await model.async_fit(train_data)
    
    # Generate forecasts
    forecast_steps = 50
    forecasts = model.forecast(forecast_steps)
    
    # Verify forecast dimensions
    assert len(forecasts) == forecast_steps
    
    # Check forecast accuracy (basic check)
    mse = np.mean((forecasts - test_data)**2)
    
    # Compute a naive forecast (using last observation)
    naive_forecast = np.ones(forecast_steps) * train_data[-1]
    naive_mse = np.mean((naive_forecast - test_data)**2)
    
    # The model should outperform naive forecast
    assert mse < naive_mse


@pytest.mark.parametrize('nlags', [10, 20, 30])
def test_acf_computation(nlags):
    """Tests autocorrelation function computation."""
    # Generate test data
    np.random.seed(42)
    n = 1000
    ar_params = np.array([0.7])
    
    # Generate AR(1) process
    data = np.zeros(n)
    for t in range(1, n):
        data[t] = ar_params[0] * data[t-1] + np.random.normal(0, 1)
    
    # Compute ACF
    acf = compute_acf(data, nlags)
    
    # Verify dimensions
    assert len(acf) == nlags + 1
    
    # ACF at lag 0 should be 1
    assert abs(acf[0] - 1.0) < 1e-10
    
    # ACF should decay for AR(1) process
    assert acf[1] > acf[2]
    
    # ACF values should be between -1 and 1
    assert all(-1 <= x <= 1 for x in acf)
    
    # Compare with numpy.correlate for validation
    np_acf = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    np_acf = np_acf[len(np_acf)//2:len(np_acf)//2 + nlags + 1]
    np_acf = np_acf / np_acf[0]
    
    # Should be close to numpy implementation
    assert np.allclose(acf, np_acf, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('nlags', [10, 20, 30])
def test_pacf_computation(nlags):
    """Tests partial autocorrelation function computation."""
    # Generate test data
    np.random.seed(42)
    n = 1000
    ar_params = np.array([0.7, 0.2])
    
    # Generate AR(2) process
    data = np.zeros(n)
    for t in range(2, n):
        data[t] = ar_params[0] * data[t-1] + ar_params[1] * data[t-2] + np.random.normal(0, 1)
    
    # Compute PACF
    pacf = compute_pacf(data, nlags)
    
    # Verify dimensions
    assert len(pacf) == nlags + 1
    
    # PACF at lag 0 should be 1
    assert abs(pacf[0] - 1.0) < 1e-10
    
    # PACF for AR(2) should have significant values at lags 1 and 2
    assert abs(pacf[1]) > 0.4  # Close to true AR(1) parameter
    assert abs(pacf[2]) > 0.1  # Close to true AR(2) parameter
    
    # PACF values beyond p=2 should be small for AR(2) process
    assert all(abs(pacf[i]) < 0.1 for i in range(3, nlags + 1))
    
    # PACF values should be between -1 and 1
    assert all(-1 <= x <= 1 for x in pacf)


@pytest.mark.asyncio
async def test_diagnostic_tests():
    """Tests model diagnostic statistics and residual analysis."""
    # Generate synthetic time series
    np.random.seed(42)
    n = 500
    ar_params = np.array([0.6])
    ma_params = np.array([0.4])
    
    # Generate ARMA process
    data = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    for t in range(1, n):
        if t >= 1:
            data[t] += ar_params[0] * data[t-1]
        if t >= 1:
            data[t] += ma_params[0] * errors[t-1]
        data[t] += errors[t]
    
    # Create and fit ARMAX model
    model = ARMAX(p=1, q=1, trend='c')
    await model.async_fit(data)
    
    # Compute diagnostic statistics
    diagnostics = model.diagnostic_tests()
    
    # Verify diagnostic statistics exist
    assert 'log_likelihood' in diagnostics
    assert 'aic' in diagnostics
    assert 'bic' in diagnostics
    assert 'ljung_box_q' in diagnostics
    assert 'ljung_box_p' in diagnostics
    assert 'jarque_bera' in diagnostics
    assert 'jarque_bera_p' in diagnostics
    
    # Check residual statistics
    assert 'mean_residual' in diagnostics
    assert abs(diagnostics['mean_residual']) < 0.1  # Mean close to zero
    
    # Check parameter statistics
    assert 'param_const' in diagnostics  # Constant parameter
    assert 'param_ar.1' in diagnostics   # AR parameter
    assert 'param_ma.1' in diagnostics   # MA parameter
    
    # Verify p-values are properly formatted
    assert 0 <= diagnostics['ljung_box_p'] <= 1
    assert 0 <= diagnostics['jarque_bera_p'] <= 1


@given(st.lists(st.floats(min_value=-0.9, max_value=0.9), min_size=1, max_size=3))
@settings(max_examples=50)
def test_armax_stationarity_property(ar_params):
    """Tests ARMAX model stationarity with property-based testing."""
    # Create model with AR parameters from hypothesis
    p = len(ar_params)
    model = ARMAX(p=p, q=1)
    
    # Set parameters directly (this is testing internal constraints)
    n_trend_params = 0
    if 'c' in model.trend:
        n_trend_params += 1
    
    # Create full parameter array
    params = np.zeros(n_trend_params + p + 1)
    # Fill AR parameters
    for i in range(p):
        params[n_trend_params + i] = ar_params[i]
    # Set MA parameter to a reasonable value
    params[n_trend_params + p] = 0.3
    
    # Manually set parameters and data for testing
    model.params = params
    model._endog = np.random.randn(200)
    model.residuals = np.random.randn(200)
    
    # Generate forecasts (should not explode for stationary models)
    forecasts = model.forecast(50)
    
    # For stationary models, forecasts should not explode
    assert np.all(np.isfinite(forecasts))
    assert np.max(np.abs(forecasts)) < 100  # Reasonable bound