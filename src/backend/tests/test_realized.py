"""
Test suite for high-frequency financial data analysis tools in the MFE Toolbox.

This module tests realized volatility estimation, noise filtering, kernel-based
covariance estimation, and related utilities. It validates Numba-optimized routines
and ensures proper handling of various sampling schemes for high-frequency financial data.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

# Internal imports
from ..models.realized import realized_variance, kernel_realized_covariance, RealizedMeasure
from ..utils.validation import validate_array_input


class TestRealizedMeasures:
    """Test fixture class for realized measure computations."""
    
    def __init__(self):
        """Initialize test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate sample test data
        self.sample_prices = np.exp(np.cumsum(np.random.randn(100) * 0.01))
        self.sample_times = np.arange(0, 100, 1.0)
        self.sample_returns = np.diff(np.log(self.sample_prices))
        
    def setup_method(self, method):
        """Setup method run before each test."""
        # Reset test data to ensure clean state
        np.random.seed(42)
        self.sample_prices = np.exp(np.cumsum(np.random.randn(100) * 0.01))
        self.sample_times = np.arange(0, 100, 1.0)
        self.sample_returns = np.diff(np.log(self.sample_prices))


def test_realized_variance_basic():
    """Tests basic functionality of realized variance computation."""
    # Generate sample price and time data
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.randn(100) * 0.01))
    times = np.arange(0, 100, 1.0)
    
    # Compute realized variance with default parameters
    rv, rv_ss = realized_variance(
        prices, 
        times, 
        'timestamp', 
        'Fixed', 
        5
    )
    
    # Verify output type and positivity
    assert isinstance(rv, float)
    assert isinstance(rv_ss, float)
    assert rv >= 0
    assert rv_ss >= 0
    
    # Check subsampled variance computation
    assert rv_ss is not None  # Ensure subsampling is computed


@pytest.mark.parametrize('sampling_type, sampling_interval', [
    ('CalendarTime', (60, 300)),
    ('CalendarUniform', (78, 390)),
    ('BusinessTime', (1, 50, 300)),
    ('BusinessUniform', (68, 390)),
    ('Fixed', 30)
])
def test_realized_variance_sampling(sampling_type, sampling_interval):
    """Tests realized variance with different sampling schemes."""
    # Generate test data for specified sampling scheme
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.randn(100) * 0.01))
    times = np.arange(0, 100, 1.0)
    
    # Compute realized variance with given parameters
    rv, rv_ss = realized_variance(
        prices,
        times,
        'timestamp',
        sampling_type,
        sampling_interval
    )
    
    # Validate output consistency
    assert isinstance(rv, float)
    assert isinstance(rv_ss, float)
    assert rv >= 0
    assert rv_ss >= 0


@pytest.mark.parametrize('kernel_type, bandwidth', [
    ('Bartlett', 10),
    ('Parzen', 20),
    ('Quadratic', 15),
    ('Truncated', 5)
])
def test_kernel_covariance(kernel_type, bandwidth):
    """Tests kernel-based realized covariance estimation."""
    # Generate multivariate return data
    np.random.seed(42)
    n_obs = 100
    n_vars = 3
    returns = np.random.randn(n_obs, n_vars) * 0.01
    
    # Compute kernel realized covariance
    cov_matrix = kernel_realized_covariance(
        returns,
        kernel_type,
        bandwidth
    )
    
    # Verify matrix properties (symmetry, positive definiteness)
    assert cov_matrix.shape == (n_vars, n_vars)
    assert np.allclose(cov_matrix, cov_matrix.T)  # Check symmetry
    
    # Check eigenvalues for positive semi-definiteness
    eigvals = np.linalg.eigvals(cov_matrix)
    assert np.all(eigvals > -1e-10)  # Allow for small numerical errors


def test_realized_measure_class():
    """Tests RealizedMeasure class functionality."""
    # Initialize RealizedMeasure with test parameters
    measure = RealizedMeasure(
        sampling_type='Fixed',
        sampling_interval=10
    )
    
    # Generate test data
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.randn(100) * 0.01))
    times = np.arange(0, 100, 1.0)
    
    # Compute measures on test data
    results = measure.compute(
        prices,
        times
    )
    
    # Validate confidence interval computation
    assert isinstance(results, np.ndarray)
    assert results.size > 0
    assert np.all(results >= 0)
    
    # Calculate confidence intervals
    lower, upper = measure.get_confidence_intervals(alpha=0.05)
    
    # Check error handling
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)
    assert np.all(lower <= upper)
    assert np.all(lower >= 0)


def test_invalid_inputs():
    """Tests error handling for invalid inputs."""
    # Test with empty arrays
    with pytest.raises(ValueError):
        # Create empty arrays
        empty_prices = np.array([])
        empty_times = np.array([])
        
        # Validate input (should raise ValueError)
        validate_array_input(empty_prices)
    
    # Test with non-finite values
    with pytest.raises(ValueError):
        # Create array with NaN
        invalid_prices = np.array([1.0, 2.0, np.nan, 4.0])
        
        # Validate input (should raise ValueError)
        validate_array_input(invalid_prices)
    
    # Test with mismatched dimensions
    with pytest.raises(ValueError):
        # Create arrays with different lengths
        prices = np.exp(np.random.randn(100).cumsum() * 0.01)
        times = np.arange(50)  # Only 50 time points
        
        # Create RealizedMeasure
        measure = RealizedMeasure(
            sampling_type='Fixed',
            sampling_interval=10
        )
        
        # Compute (should raise ValueError for mismatched dimensions)
        measure.compute(prices, times)
    
    # Verify appropriate error messages
    with pytest.raises(ValueError) as excinfo:
        # Try to create RealizedMeasure with invalid sampling type
        RealizedMeasure(
            sampling_type='InvalidType',
            sampling_interval=10
        )
    
    # Check error message content
    assert "Sampling type" in str(excinfo.value)