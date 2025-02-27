"""
Test suite for GARCH model implementations including unit tests for model estimation,
simulation, forecasting and parameter validation. Validates Numba-optimized routines
and async estimation capabilities.
"""

import pytest
import numpy as np
import asyncio
from hypothesis import given, strategies as st
from typing import List, Tuple, Dict

# Internal imports
from ..models.garch import (
    GARCHModel, 
    compute_garch_likelihood, 
    simulate_garch,
    VALID_GARCH_TYPES,
    VALID_DISTRIBUTIONS
)

# Configure constants for testing
np.random.seed(42)  # Ensure reproducibility


@pytest.mark.parametrize('model_type', VALID_GARCH_TYPES)
@pytest.mark.parametrize('distribution', ['normal', 'student-t'])
def test_garch_initialization(model_type, distribution):
    """Tests GARCH model initialization with various parameters."""
    # Basic initialization
    model = GARCHModel(p=1, q=1, model_type=model_type, distribution=distribution)
    
    # Check attributes
    assert model.p == 1
    assert model.q == 1
    assert model.model_type == model_type
    assert model.distribution == distribution
    assert model.parameters is not None
    assert model.converged is False
    
    # Check model_type_id and distribution_id mapping
    model_type_id = model._get_model_type_id()
    dist_id = model._get_distribution_id()
    assert isinstance(model_type_id, int)
    assert isinstance(dist_id, int)
    
    # Check parameter count based on model type and distribution
    expected_params = 3  # GARCH base case
    if model_type in ['EGARCH', 'GJR-GARCH', 'TARCH', 'AGARCH', 'FIGARCH']:
        expected_params = 4
    if distribution == 'student-t':
        expected_params += 1
    elif distribution == 'ged':
        expected_params += 1
    elif distribution == 'skewed-t':
        expected_params += 2
        
    assert len(model.parameters) == expected_params


def test_garch_invalid_initialization():
    """Tests GARCH model initialization with invalid parameters."""
    # Invalid model type
    with pytest.raises(ValueError):
        GARCHModel(p=1, q=1, model_type="INVALID_MODEL")
    
    # Invalid distribution type
    with pytest.raises(ValueError):
        GARCHModel(p=1, q=1, distribution="INVALID_DISTRIBUTION")
    
    # Invalid orders
    with pytest.raises(ValueError):
        GARCHModel(p=-1, q=1)
    
    with pytest.raises(ValueError):
        GARCHModel(p=0, q=0)


@pytest.mark.parametrize('model_type', VALID_GARCH_TYPES)
def test_garch_parameter_validation(model_type):
    """Tests parameter validation in GARCH models."""
    model = GARCHModel(p=1, q=1, model_type=model_type)
    
    # Test with invalid dimension data
    with pytest.raises(ValueError):
        invalid_data = np.random.normal(0, 1, (10, 2))  # 2D array not allowed
        asyncio.run(model.async_fit(invalid_data))
    
    # Test with too short data
    with pytest.raises(ValueError):
        short_data = np.random.normal(0, 1, 5)  # Too few observations
        asyncio.run(model.async_fit(short_data))
    
    # Test forecasting without parameters
    with pytest.raises(RuntimeError):
        model.forecast(5)
    
    # Test simulation without parameters
    with pytest.raises(RuntimeError):
        model.simulate(100)
    
    # Mock successful estimation
    model.parameters = np.ones(model.n_params)
    model.converged = True
    model.last_returns = np.array([0.1])
    model.last_variance = 1.0
    
    # Test forecast with invalid horizon
    with pytest.raises(ValueError):
        model.forecast(0)
    
    # Test simulation with invalid sample size
    with pytest.raises(ValueError):
        model.simulate(0)


@pytest.mark.parametrize('distribution', VALID_DISTRIBUTIONS)
def test_garch_likelihood(distribution):
    """Tests GARCH likelihood computation."""
    # Generate test data
    returns = np.random.normal(0, 1, 1000)
    
    # Initialize model with different distributions
    model = GARCHModel(p=1, q=1, model_type='GARCH', distribution=distribution)
    
    # Create parameters based on distribution type
    if distribution == 'normal':
        params = np.array([0.05, 0.1, 0.8])  # omega, alpha, beta
    elif distribution == 'student-t':
        params = np.array([0.05, 0.1, 0.8, 8.0])  # omega, alpha, beta, df
    elif distribution == 'ged':
        params = np.array([0.05, 0.1, 0.8, 1.5])  # omega, alpha, beta, shape
    elif distribution == 'skewed-t':
        params = np.array([0.05, 0.1, 0.8, 8.0, 0.0])  # omega, alpha, beta, df, skew
    
    # Compute likelihood
    model_type_id = model._get_model_type_id()
    distribution_id = model._get_distribution_id()
    likelihood = compute_garch_likelihood(returns, params, model_type_id, distribution_id)
    
    # Likelihood should be finite
    assert np.isfinite(likelihood)
    assert isinstance(likelihood, float)
    
    # Test invalid parameters
    invalid_params = params.copy()
    invalid_params[0] = -0.1  # Negative omega
    
    # Should return inf for invalid params
    inf_likelihood = compute_garch_likelihood(returns, invalid_params, model_type_id, distribution_id)
    assert inf_likelihood == np.inf


@pytest.mark.parametrize('model_type', ['GARCH', 'EGARCH', 'GJR-GARCH'])
def test_garch_likelihood_stability(model_type):
    """Tests stability of likelihood computation across different parameter values."""
    # Generate test data
    returns = np.random.normal(0, 1, 500)
    
    # Initialize model
    model = GARCHModel(p=1, q=1, model_type=model_type, distribution='normal')
    model_type_id = model._get_model_type_id()
    distribution_id = model._get_distribution_id()
    
    # Define parameter sets to test
    if model_type == 'GARCH':
        param_sets = [
            np.array([0.01, 0.05, 0.90]),  # Lower omega
            np.array([0.10, 0.05, 0.90]),  # Higher omega
            np.array([0.05, 0.01, 0.94]),  # Lower alpha
            np.array([0.05, 0.20, 0.75]),  # Higher alpha
        ]
    elif model_type == 'EGARCH':
        param_sets = [
            np.array([-5.0, 0.1, -0.05, 0.9]),  # Different parameters
            np.array([-3.0, 0.2, -0.1, 0.8]),
            np.array([-4.0, 0.15, 0.0, 0.85]),
            np.array([-3.5, 0.05, -0.15, 0.95]),
        ]
    else:  # GJR-GARCH, TARCH
        param_sets = [
            np.array([0.05, 0.03, 0.9, 0.04]),  # Different parameters
            np.array([0.03, 0.05, 0.9, 0.02]),
            np.array([0.04, 0.04, 0.88, 0.06]),
            np.array([0.06, 0.02, 0.92, 0.03]),
        ]
    
    # Check likelihood for each parameter set
    for params in param_sets:
        likelihood = compute_garch_likelihood(returns, params, model_type_id, distribution_id)
        assert np.isfinite(likelihood)


@pytest.mark.asyncio
async def test_garch_estimation():
    """Tests async GARCH model estimation."""
    # Generate data with GARCH(1,1) properties
    T = 1000
    
    # True parameters
    omega, alpha, beta = 0.05, 0.1, 0.8
    
    # Generate conditional variances
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance
    
    # Generate returns
    returns = np.zeros(T)
    for t in range(T):
        # Generate return with conditional variance
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
        
        # Update volatility for next period
        if t < T - 1:
            sigma2[t+1] = omega + alpha * returns[t]**2 + beta * sigma2[t]
    
    # Create and estimate model
    model = GARCHModel(p=1, q=1, model_type='GARCH', distribution='normal')
    converged = await model.async_fit(returns)
    
    # Check that estimation converged
    assert converged
    assert model.converged
    assert model.parameters is not None
    assert model.likelihood is not None
    
    # Check parameter estimates - they may not be exactly the true values
    # but should be in a reasonable range
    estimated_omega, estimated_alpha, estimated_beta = model.parameters
    
    # Verify parameters are in reasonable bounds
    assert estimated_omega > 0, "Omega should be positive"
    assert 0 < estimated_alpha < 1, "Alpha should be between 0 and 1"
    assert 0 < estimated_beta < 1, "Beta should be between 0 and 1"
    assert estimated_alpha + estimated_beta < 1, "Alpha + Beta should be less than 1"


@pytest.mark.parametrize('model_type', ['GARCH', 'EGARCH', 'GJR-GARCH'])
@pytest.mark.parametrize('distribution', ['normal', 'student-t'])
def test_garch_simulation(model_type, distribution):
    """Tests GARCH simulation functionality."""
    # Skip some combinations to reduce test time
    if model_type in ['AGARCH', 'FIGARCH'] and distribution in ['ged', 'skewed-t']:
        pytest.skip("Skipping complex model/distribution combination")
    
    # Set parameters based on model type
    if model_type == 'GARCH':
        params = np.array([0.05, 0.1, 0.8])  # omega, alpha, beta
    elif model_type == 'EGARCH':
        params = np.array([0.05, 0.1, -0.05, 0.8])  # omega, alpha, gamma, beta
    elif model_type in ['GJR-GARCH', 'TARCH']:
        params = np.array([0.05, 0.05, 0.8, 0.05])  # omega, alpha, beta, gamma
    else:
        params = np.array([0.05, 0.1, 0.8, 0.0])  # Default case
    
    # Add distribution parameters
    if distribution == 'student-t':
        params = np.append(params, 8.0)  # degrees of freedom
    elif distribution == 'ged':
        params = np.append(params, 1.5)  # shape parameter
    elif distribution == 'skewed-t':
        params = np.append(params, [8.0, 0.0])  # df, skew
    
    # Create model and map types to IDs
    model = GARCHModel(p=1, q=1, model_type=model_type, distribution=distribution)
    model_type_id = model._get_model_type_id()
    distribution_id = model._get_distribution_id()
    
    # Simulate data
    n_samples = 500
    returns, volatility = simulate_garch(n_samples, params, model_type_id, distribution_id)
    
    # Check output shapes
    assert len(returns) == n_samples
    assert len(volatility) == n_samples
    
    # Check basic statistical properties
    assert np.all(np.isfinite(returns))
    assert np.all(np.isfinite(volatility))
    assert np.all(volatility > 0)
    
    # Mean should be close to 0 for standardized innovations
    assert abs(np.mean(returns)) < 0.5
    
    # Volatility should exhibit clustering (serial correlation in squared returns)
    squared_returns = returns**2
    acf_squared = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
    assert acf_squared > 0  # Positive autocorrelation in squared returns


@pytest.mark.parametrize('horizon', [1, 5, 22])
def test_garch_forecasting(horizon):
    """Tests GARCH forecasting capabilities."""
    # Create model and set parameters manually
    model = GARCHModel(p=1, q=1, model_type='GARCH', distribution='normal')
    
    # Set parameters directly to bypass estimation
    omega, alpha, beta = 0.05, 0.1, 0.8
    model.parameters = np.array([omega, alpha, beta])
    model.converged = True
    model.last_returns = np.array([0.1])
    model.last_variance = 1.0
    
    # Generate forecasts
    forecasts = model.forecast(horizon)
    
    # Check forecast shape
    assert len(forecasts) == horizon
    
    # Check forecast properties
    assert np.all(forecasts > 0)
    assert np.all(np.isfinite(forecasts))
    
    # For GARCH(1,1), first forecast should match formula
    expected_first_forecast = omega + alpha * model.last_returns[-1]**2 + beta * model.last_variance
    assert np.isclose(forecasts[0], expected_first_forecast)
    
    # For long horizons, check convergence toward unconditional variance
    if horizon > 20:
        unconditional_var = omega / (1 - alpha - beta)
        assert abs(forecasts[-1] - unconditional_var) < 0.1 * unconditional_var


@given(st.lists(st.floats(min_value=-10, max_value=10), min_size=20, max_size=50))
def test_garch_with_property_based_testing(data):
    """Tests GARCH model robustness using property-based testing."""
    # Convert to numpy array and filter out any NaN/Inf values
    returns = np.array(data)
    valid_indices = np.isfinite(returns)
    if np.sum(valid_indices) < 20:
        # Skip if too few valid points
        return
    
    returns = returns[valid_indices]
    
    # Simple GARCH(1,1) model
    model = GARCHModel(p=1, q=1, model_type='GARCH', distribution='normal')
    
    # Skip full estimation, just test likelihood function
    model_type_id = model._get_model_type_id()
    distribution_id = model._get_distribution_id()
    
    # Valid parameters
    params = np.array([0.05, 0.1, 0.8])
    
    # Likelihood should be finite for valid parameters
    likelihood = compute_garch_likelihood(returns, params, model_type_id, distribution_id)
    assert np.isfinite(likelihood)
    
    # Invalid parameters (alpha + beta >= 1)
    invalid_params = np.array([0.05, 0.5, 0.5])
    
    # Likelihood should be infinite for invalid parameters
    inv_likelihood = compute_garch_likelihood(returns, invalid_params, model_type_id, distribution_id)
    assert inv_likelihood == np.inf


@pytest.mark.asyncio
async def test_garch_convergence_stability():
    """Tests GARCH estimation convergence across datasets with different properties."""
    # Generate 3 datasets with different characteristics
    np.random.seed(42)
    n_samples = 1000
    
    # Dataset 1: Low volatility, normal returns
    returns1 = np.random.normal(0, 0.01, n_samples)
    
    # Dataset 2: Higher volatility, normal returns
    returns2 = np.random.normal(0, 0.05, n_samples)
    
    # Dataset 3: Returns with volatility clustering
    volatility = np.zeros(n_samples)
    volatility[0] = 0.01
    returns3 = np.zeros(n_samples)
    
    for t in range(1, n_samples):
        # Simple AR(1) volatility model
        volatility[t] = 0.001 + 0.1 * returns3[t-1]**2 + 0.8 * volatility[t-1]
        returns3[t] = np.random.normal(0, np.sqrt(volatility[t]))
    
    # Create models
    model = GARCHModel(p=1, q=1, model_type='GARCH', distribution='normal')
    
    # Test each dataset
    for returns in [returns1, returns2, returns3]:
        # Reset model state
        model.parameters = np.zeros(model.n_params)
        model.converged = False
        
        # Estimate
        converged = await model.async_fit(returns)
        
        # All datasets should lead to convergence
        assert converged, f"Model failed to converge"
        assert model.parameters is not None
        assert np.all(np.isfinite(model.parameters))
        
        # Parameters should respect constraints
        omega, alpha, beta = model.parameters
        assert omega > 0, "Omega should be positive"
        assert alpha >= 0, "Alpha should be non-negative"
        assert beta >= 0, "Beta should be non-negative"
        assert alpha + beta < 1, "Alpha + Beta should be less than 1"


def test_garch_distribution_effects():
    """Tests the effects of different distribution assumptions on GARCH likelihood."""
    # Generate data with fat tails
    np.random.seed(123)
    n = 1000
    
    # Create fat-tailed data using t-distribution
    t_data = np.random.standard_t(df=5, size=n)
    
    # Standardize to unit variance
    t_data = t_data / np.std(t_data)
    
    # Set common parameters 
    params_normal = np.array([0.05, 0.1, 0.8])
    params_t = np.array([0.05, 0.1, 0.8, 5.0])  # Add df parameter
    
    # Create models with different distributions
    model_normal = GARCHModel(p=1, q=1, model_type='GARCH', distribution='normal')
    model_t = GARCHModel(p=1, q=1, model_type='GARCH', distribution='student-t')
    
    # Compute likelihoods
    ll_normal = compute_garch_likelihood(
        t_data, 
        params_normal, 
        model_normal._get_model_type_id(),
        model_normal._get_distribution_id()
    )
    
    ll_t = compute_garch_likelihood(
        t_data, 
        params_t, 
        model_t._get_model_type_id(),
        model_t._get_distribution_id()
    )
    
    # For t-distributed data, t-distribution likelihood should be higher (less negative)
    # than normal likelihood
    assert ll_t < ll_normal, "Student-t likelihood should be better for fat-tailed data"