"""
Test suite for Numba-optimized numerical optimization routines.

This module contains comprehensive test cases for validating the performance
and correctness of optimization routines used in the MFE Toolbox, with a focus
on GARCH parameter estimation, likelihood optimization, and standard error computation.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from ..core.optimization import Optimizer, optimize_garch, compute_standard_errors
from ..utils.validation import VALID_GARCH_TYPES, VALID_DISTRIBUTIONS


class TestOptimizer:
    """Test fixture class for optimization tests."""
    
    def __init__(self):
        """Sets up test fixtures."""
        # Generate synthetic test data
        np.random.seed(42)
        self.test_data = np.random.normal(0, 1, size=1000)
        
        # Initialize test parameters
        self.initial_params = np.array([0.01, 0.1, 0.8])
        
        # Create optimizer instance
        self.optimizer = Optimizer()
    
    def setup_method(self, method):
        """Setup method run before each test."""
        # Reset test data
        np.random.seed(42)
        self.test_data = np.random.normal(0, 1, size=1000)
        
        # Reset optimizer instance
        self.optimizer = Optimizer()
        
        # Clear any cached results


def test_optimizer_initialization():
    """Tests initialization of Optimizer class with various configuration options."""
    # Create Optimizer instance with default options
    optimizer = Optimizer()
    
    # Verify default optimization options are set correctly
    assert optimizer.optimization_options['method'] == 'SLSQP'
    assert optimizer.optimization_options['tol'] == 1e-8
    assert optimizer.optimization_options['max_iter'] == 1000
    assert optimizer.optimization_options['disp'] is False
    
    # Create Optimizer with custom options
    custom_options = {
        'method': 'COBYLA',
        'tol': 1e-6,
        'max_iter': 500,
        'disp': True
    }
    optimizer = Optimizer(options=custom_options)
    
    # Verify custom options override defaults correctly
    assert optimizer.optimization_options['method'] == 'COBYLA'
    assert optimizer.optimization_options['tol'] == 1e-6
    assert optimizer.optimization_options['max_iter'] == 500
    assert optimizer.optimization_options['disp'] is True


@pytest.mark.parametrize('model_type', VALID_GARCH_TYPES)
@pytest.mark.parametrize('distribution', VALID_DISTRIBUTIONS)
def test_optimize_garch_valid_inputs(model_type, distribution):
    """Tests GARCH optimization with valid input parameters."""
    # Generate test data using numpy random
    np.random.seed(42)
    data = np.random.normal(0, 1, size=1000)
    
    # Create initial parameter vector
    if model_type in ['GARCH', 'FIGARCH']:
        initial_params = np.array([0.01, 0.1, 0.8])
    else:
        # For EGARCH, GJR-GARCH, TARCH, AGARCH
        initial_params = np.array([0.01, 0.1, 0.1, 0.8])
    
    # Call optimize_garch with test inputs
    params, likelihood, converged = optimize_garch(
        data, initial_params, model_type, distribution
    )
    
    # Verify optimization converges
    assert converged is True
    
    # Check parameter constraints are satisfied
    assert params[0] > 0  # omega > 0
    
    if model_type == 'GARCH':
        assert 0 <= params[1] < 1  # 0 <= alpha < 1
        assert 0 <= params[2] < 1  # 0 <= beta < 1
        assert params[1] + params[2] < 1  # alpha + beta < 1
    elif model_type == 'EGARCH':
        assert 0 <= params[3] < 1  # 0 <= beta < 1
    elif model_type in ['GJR-GARCH', 'TARCH', 'AGARCH']:
        assert 0 <= params[1] < 1  # 0 <= alpha < 1
        assert 0 <= params[2] < 1  # 0 <= beta < 1
        assert 0 <= params[3] < 1  # 0 <= gamma < 1
        assert params[1] + params[2] + 0.5 * params[3] < 1  # alpha + beta + 0.5*gamma < 1
    
    # Validate likelihood value is reasonable
    assert likelihood is not None
    assert np.isfinite(likelihood)


def test_optimize_garch_invalid_inputs():
    """Tests GARCH optimization error handling with invalid inputs."""
    # Generate test data
    np.random.seed(42)
    data = np.random.normal(0, 1, size=1000)
    initial_params = np.array([0.01, 0.1, 0.8])
    
    # Test with invalid data dimensions
    with pytest.raises((ValueError, TypeError)):
        optimize_garch(np.array([]), initial_params, 'GARCH', 'normal')
    
    # Test with invalid initial parameters
    with pytest.raises((ValueError, TypeError)):
        optimize_garch(data, np.array([]), 'GARCH', 'normal')
    
    # Test with invalid model type
    with pytest.raises(ValueError):
        optimize_garch(data, initial_params, 'INVALID_MODEL', 'normal')
    
    # Test with invalid distribution
    with pytest.raises(ValueError):
        optimize_garch(data, initial_params, 'GARCH', 'INVALID_DIST')


@pytest.mark.parametrize('model_type', VALID_GARCH_TYPES)
def test_compute_standard_errors(model_type):
    """Tests standard error computation for optimized parameters."""
    # Generate test data and parameters
    np.random.seed(42)
    data = np.random.normal(0, 1, size=1000)
    
    if model_type in ['GARCH', 'FIGARCH']:
        params = np.array([0.01, 0.1, 0.8])
        model_type_id = 0
    elif model_type == 'EGARCH':
        params = np.array([0.01, 0.1, 0.1, 0.8])
        model_type_id = 1
    else:
        # For GJR-GARCH, TARCH, AGARCH
        params = np.array([0.01, 0.1, 0.1, 0.8])
        model_type_id = 2
    
    # Compute standard errors
    std_errors = compute_standard_errors(params, data, model_type_id)
    
    # Verify standard error dimensions
    assert std_errors.shape == params.shape
    
    # Check standard errors are positive
    assert np.all(std_errors > 0)
    
    # Validate magnitude of standard errors
    assert np.all(std_errors < 1.0)  # Standard errors should be reasonably small


@pytest.mark.asyncio
async def test_async_optimize():
    """Tests asynchronous optimization workflow."""
    # Initialize async optimizer
    optimizer = Optimizer()
    
    # Prepare test data and parameters
    np.random.seed(42)
    data = np.random.normal(0, 1, size=1000)
    initial_params = np.array([0.01, 0.1, 0.8])
    model_type = 'GARCH'
    distribution = 'normal'
    
    # Execute async optimization
    params, likelihood = await optimizer.async_optimize(
        data, initial_params, model_type, distribution
    )
    
    # Verify convergence status
    assert optimizer.converged is True
    
    # Validate optimization results
    assert params.shape == initial_params.shape
    assert params[0] > 0  # omega > 0
    assert 0 <= params[1] < 1  # 0 <= alpha < 1
    assert 0 <= params[2] < 1  # 0 <= beta < 1
    assert params[1] + params[2] < 1  # alpha + beta < 1
    assert np.isfinite(likelihood)