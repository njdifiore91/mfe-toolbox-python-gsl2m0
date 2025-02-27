"""
Test suite for bootstrap resampling functionality.

This module provides comprehensive tests for the bootstrap module, validating
block bootstrap, stationary bootstrap, and asynchronous bootstrap implementations.
Tests cover input validation, statistical properties of bootstrap samples,
and proper functioning of the asynchronous interface with Numba optimization.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from ..core.bootstrap import Bootstrap, block_bootstrap, stationary_bootstrap
from ..utils.validation import validate_array_input


class TestBootstrap:
    """Test fixture class for bootstrap tests."""
    
    def __init__(self):
        """Sets up test fixtures."""
        # Initialize test parameters
        self.sample_size = 1000
        self.num_bootstraps = 100
        np.random.seed(42)  # For reproducibility
        self.test_data = np.random.randn(self.sample_size)  # Generate test data


def test_block_bootstrap_validation():
    """Tests input validation for block bootstrap function."""
    # Create invalid input data
    with pytest.raises(TypeError):
        block_bootstrap("not an array", 100, 10)
    
    # Test invalid array with NaN
    invalid_data = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError):
        block_bootstrap(invalid_data, 100, 10)
    
    # Test invalid block sizes
    valid_data = np.random.randn(100)
    with pytest.raises(ValueError):
        block_bootstrap(valid_data, 100, -5)
    
    # Test invalid number of bootstraps
    with pytest.raises(ValueError):
        block_bootstrap(valid_data, -10, 5)


def test_stationary_bootstrap_validation():
    """Tests input validation for stationary bootstrap function."""
    # Create invalid input data
    with pytest.raises(TypeError):
        stationary_bootstrap("not an array", 100, 0.1)
    
    # Test invalid array with NaN
    invalid_data = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError):
        stationary_bootstrap(invalid_data, 100, 0.1)
    
    # Test invalid block probabilities
    valid_data = np.random.randn(100)
    with pytest.raises(ValueError):
        stationary_bootstrap(valid_data, 100, 1.5)  # Probability > 1
    
    with pytest.raises(ValueError):
        stationary_bootstrap(valid_data, 100, -0.1)  # Probability < 0
    
    # Test invalid number of bootstraps
    with pytest.raises(ValueError):
        stationary_bootstrap(valid_data, -10, 0.1)


@pytest.mark.parametrize('block_size', [10, 20, 50])
def test_block_bootstrap_properties(block_size):
    """Tests statistical properties of block bootstrap samples."""
    # Generate test time series data
    np.random.seed(42)  # For reproducibility
    sample_size = 1000
    data = np.random.randn(sample_size)
    
    # Perform block bootstrap
    num_bootstraps = 100
    bootstrap_samples = block_bootstrap(data, num_bootstraps, block_size)
    
    # Verify sample dimensions
    assert bootstrap_samples.shape == (num_bootstraps, sample_size)
    
    # Check statistical properties
    original_mean = np.mean(data)
    original_std = np.std(data)
    
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    bootstrap_stds = np.std(bootstrap_samples, axis=1)
    
    # The mean of bootstrap means should be close to the original mean
    assert np.abs(np.mean(bootstrap_means) - original_mean) < 0.1
    
    # The mean of bootstrap stds should be close to the original std
    assert np.abs(np.mean(bootstrap_stds) - original_std) < 0.1
    
    # Validate block structure
    # Test that adjacent values in samples are more likely to be from the original data
    for i in range(3):  # Check first 3 bootstrap samples
        consecutive_matches = 0
        for j in range(sample_size - 1):
            if any(np.array_equal(data[k:k+2], bootstrap_samples[i, j:j+2]) 
                   for k in range(sample_size - 1)):
                consecutive_matches += 1
        
        # We should find many consecutive matches if blocks are preserved
        assert consecutive_matches > sample_size / (block_size * 4)


@pytest.mark.parametrize('block_prob', [0.1, 0.2, 0.5])
def test_stationary_bootstrap_properties(block_prob):
    """Tests statistical properties of stationary bootstrap samples."""
    # Generate test time series data
    np.random.seed(42)  # For reproducibility
    sample_size = 1000
    data = np.random.randn(sample_size)
    
    # Perform stationary bootstrap
    num_bootstraps = 100
    bootstrap_samples = stationary_bootstrap(data, num_bootstraps, block_prob)
    
    # Verify sample dimensions
    assert bootstrap_samples.shape == (num_bootstraps, sample_size)
    
    # Check statistical properties
    original_mean = np.mean(data)
    original_std = np.std(data)
    
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    bootstrap_stds = np.std(bootstrap_samples, axis=1)
    
    # The mean of bootstrap means should be close to the original mean
    assert np.abs(np.mean(bootstrap_means) - original_mean) < 0.1
    
    # The mean of bootstrap stds should be close to the original std
    assert np.abs(np.mean(bootstrap_stds) - original_std) < 0.1
    
    # Validate random block lengths
    # Ensure samples are different from original and from each other
    assert not np.array_equal(bootstrap_samples[0], data)
    assert not np.array_equal(bootstrap_samples[0], bootstrap_samples[1])
    
    # Expected block length is 1/block_prob - verify we have some variety in patterns
    expected_block_len = 1.0 / block_prob
    run_lengths = []
    
    for i in range(3):
        j = 0
        while j < sample_size - 1:
            run_length = 1
            while (j + run_length < sample_size - 1 and 
                   np.array_equal(bootstrap_samples[i, j:j+2], 
                                  bootstrap_samples[i, j+run_length:j+run_length+2])):
                run_length += 1
            run_lengths.append(run_length)
            j += run_length
            
    # Verify the average run length is roughly what we expect
    avg_run_length = np.mean(run_lengths)
    assert 0.5 * expected_block_len < avg_run_length < 2.0 * expected_block_len


@pytest.mark.asyncio
async def test_async_bootstrap():
    """Tests asynchronous bootstrap operations."""
    # Initialize Bootstrap class
    bootstrap = Bootstrap(method="block", options={"block_size": 20})
    
    # Generate test data
    np.random.seed(42)
    sample_size = 500
    data = np.random.randn(sample_size)
    
    # Perform async bootstrap
    num_bootstraps = 200
    results = await bootstrap.async_bootstrap(data, num_bootstraps)
    
    # Verify results
    assert results.shape == (num_bootstraps, sample_size)
    
    # Check progress updates by testing with stationary bootstrap as well
    bootstrap_stationary = Bootstrap(method="stationary", options={"block_probability": 0.1})
    results_stationary = await bootstrap_stationary.async_bootstrap(data, num_bootstraps)
    
    # Verify results for stationary bootstrap
    assert results_stationary.shape == (num_bootstraps, sample_size)
    
    # Verify statistical properties maintained
    original_mean = np.mean(data)
    bootstrap_means = np.mean(results, axis=1)
    assert np.abs(np.mean(bootstrap_means) - original_mean) < 0.1