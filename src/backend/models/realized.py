"""
High-frequency financial data analysis module for MFE Toolbox.

This module provides tools for analyzing high-frequency financial data, including
realized volatility estimation, noise filtering, and kernel-based covariance
estimation. It implements efficient computation through Numba-optimized routines
and supports various sampling schemes for intraday data.

Key features:
- Realized variance estimation with noise filtering
- Kernel-based covariance estimation for high-frequency returns
- Microstructure noise variance estimation
- Support for various sampling methods (calendar time, business time, etc.)
- Numba JIT compilation for high-performance calculations
"""

import logging
import numpy as np
from scipy import stats
import numba
from numba import jit
import pandas as pd
from typing import Union, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field

from ..core.optimization import optimize_garch
from ..utils.validation import validate_array_input

# Configure logger
logger = logging.getLogger(__name__)

# Global constants
SAMPLING_METHODS = ['CalendarTime', 'CalendarUniform', 'BusinessTime', 'BusinessUniform', 'Fixed']
KERNEL_TYPES = ['Bartlett', 'Parzen', 'Quadratic', 'Truncated']


@jit(nopython=True)
def _sample_data(prices: np.ndarray, 
                times: np.ndarray, 
                sampling_type: str, 
                sampling_interval: Union[int, tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized function to sample price data according to specified method.
    
    Parameters
    ----------
    prices : np.ndarray
        Array of price data
    times : np.ndarray
        Array of time stamps corresponding to prices
    sampling_type : str
        Method for sampling ('CalendarTime', 'BusinessTime', etc.)
    sampling_interval : Union[int, tuple]
        Sampling interval or parameters for the sampling method
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing sampled prices and times
    """
    n = len(prices)
    
    # Fixed sampling is the simplest - just take every nth point
    if sampling_type == 'Fixed':
        interval = int(sampling_interval)
        indices = np.arange(0, n, interval)
        return prices[indices], times[indices]
    
    # For other sampling types, we need to implement more complex logic
    # Since Numba has limitations with string operations and conditionals,
    # we'll need to use numeric indicators for sampling types
    
    # Initialize empty arrays to store sampled data
    # Since Numba doesn't support dynamic arrays, we'll allocate max size and trim
    max_samples = n
    sampled_prices = np.zeros(max_samples)
    sampled_times = np.zeros(max_samples)
    
    # Counter for number of samples
    count = 0
    
    # Calendar Time sampling
    if sampling_type == 'CalendarTime':
        min_interval, max_interval = sampling_interval[0], sampling_interval[1]
        last_time = times[0]
        sampled_prices[0] = prices[0]
        sampled_times[0] = times[0]
        count = 1
        
        for i in range(1, n):
            time_diff = times[i] - last_time
            if time_diff >= min_interval and time_diff <= max_interval:
                sampled_prices[count] = prices[i]
                sampled_times[count] = times[i]
                last_time = times[i]
                count += 1
    
    # Calendar Uniform sampling
    elif sampling_type == 'CalendarUniform':
        min_points, max_points = sampling_interval[0], sampling_interval[1]
        time_range = times[-1] - times[0]
        grid_size = max(int(time_range / max_points), 1)
        
        for i in range(0, int(time_range), grid_size):
            # Find closest point to grid time
            grid_time = times[0] + i
            closest_idx = np.argmin(np.abs(times - grid_time))
            
            sampled_prices[count] = prices[closest_idx]
            sampled_times[count] = times[closest_idx]
            count += 1
    
    # Business Time sampling
    elif sampling_type == 'BusinessTime':
        min_ticks, max_ticks, max_time = sampling_interval[0], sampling_interval[1], sampling_interval[2]
        tick_count = 0
        last_idx = 0
        sampled_prices[0] = prices[0]
        sampled_times[0] = times[0]
        count = 1
        
        for i in range(1, n):
            tick_count += 1
            time_diff = times[i] - times[last_idx]
            
            if (tick_count >= min_ticks and tick_count <= max_ticks) or time_diff >= max_time:
                sampled_prices[count] = prices[i]
                sampled_times[count] = times[i]
                tick_count = 0
                last_idx = i
                count += 1
    
    # Business Uniform sampling
    elif sampling_type == 'BusinessUniform':
        min_points, max_points = sampling_interval[0], sampling_interval[1]
        grid_size = max(int(n / max_points), 1)
        
        for i in range(0, n, grid_size):
            sampled_prices[count] = prices[i]
            sampled_times[count] = times[i]
            count += 1
    
    # Truncate arrays to actual count
    return sampled_prices[:count], sampled_times[:count]


@jit(nopython=True)
def realized_variance(prices: np.ndarray, 
                     times: np.ndarray, 
                     time_type: str, 
                     sampling_type: str, 
                     sampling_interval: Union[int, tuple]) -> Tuple[float, float]:
    """
    Computes realized variance from high-frequency price data with noise filtering using Numba-optimized routines.
    
    This function calculates the realized variance of a high-frequency price
    series using various sampling methods to mitigate the impact of market
    microstructure noise. It also computes a subsampled estimate for improved
    robustness.
    
    Parameters
    ----------
    prices : np.ndarray
        Array of high-frequency price data
    times : np.ndarray
        Array of time stamps corresponding to prices
    time_type : str
        Type of time representation ('timestamp', 'seconds', etc.)
    sampling_type : str
        Method for sampling ('CalendarTime', 'BusinessTime', etc.)
    sampling_interval : Union[int, tuple]
        Sampling interval or parameters for the sampling method
        
    Returns
    -------
    Tuple[float, float]
        Tuple containing (realized_variance, subsampled_variance)
    """
    # Basic input checks (limited due to Numba constraints)
    n = len(prices)
    if n < 2 or len(times) != n:
        return 0.0, 0.0
    
    # Sample data according to specified method
    sampled_prices, sampled_times = _sample_data(prices, times, sampling_type, sampling_interval)
    
    # Compute returns (using log prices)
    log_prices = np.log(sampled_prices)
    returns = np.diff(log_prices)
    
    # Calculate realized variance (sum of squared returns)
    rv = np.sum(returns**2)
    
    # Compute subsampled estimate for improved accuracy
    # For simplicity in Numba, we'll use a basic 2-point subsampling
    sub_returns1 = returns[::2]  # Every 2nd point starting from 0
    sub_returns2 = returns[1::2]  # Every 2nd point starting from 1
    
    # Compute subsample realized variances and average
    sub_rv1 = np.sum(sub_returns1**2) * 2  # Scale by 2 since we're using half the data
    sub_rv2 = np.sum(sub_returns2**2) * 2
    
    # Average the subsampled estimates
    rv_ss = (sub_rv1 + sub_rv2) / 2
    
    return rv, rv_ss


@jit(nopython=True)
def _compute_kernel_weights(kernel_type: str, 
                           bandwidth: int) -> np.ndarray:
    """
    Compute kernel weights for various kernel types.
    
    Parameters
    ----------
    kernel_type : str
        Type of kernel ('Bartlett', 'Parzen', 'Quadratic', 'Truncated')
    bandwidth : int
        Kernel bandwidth parameter
        
    Returns
    -------
    np.ndarray
        Array of kernel weights
    """
    # Create array for weights (symmetric around 0)
    weights = np.zeros(2 * bandwidth + 1)
    
    # Bartlett kernel (triangular)
    if kernel_type == 'Bartlett':
        for i in range(-bandwidth, bandwidth + 1):
            if abs(i) <= bandwidth:
                weights[i + bandwidth] = 1.0 - abs(i) / (bandwidth + 1)
    
    # Parzen kernel
    elif kernel_type == 'Parzen':
        for i in range(-bandwidth, bandwidth + 1):
            q = abs(i) / (bandwidth + 1)
            if q <= 0.5:
                weights[i + bandwidth] = 1.0 - 6.0 * q**2 + 6.0 * q**3
            elif q <= 1.0:
                weights[i + bandwidth] = 2.0 * (1.0 - q)**3
    
    # Quadratic kernel
    elif kernel_type == 'Quadratic':
        for i in range(-bandwidth, bandwidth + 1):
            q = abs(i) / (bandwidth + 1)
            if q <= 1.0:
                weights[i + bandwidth] = (1.0 - q**2)
    
    # Truncated kernel (flat window)
    elif kernel_type == 'Truncated':
        for i in range(-bandwidth, bandwidth + 1):
            if abs(i) <= bandwidth:
                weights[i + bandwidth] = 1.0
    
    return weights


@jit(nopython=True)
def kernel_realized_covariance(returns: np.ndarray, 
                              kernel_type: str, 
                              bandwidth: int) -> np.ndarray:
    """
    Estimates realized covariance using kernel-based methods with Numba optimization.
    
    This function calculates the realized covariance matrix for multivariate
    returns using kernel-based methods to account for various forms of
    microstructure noise and asynchronous trading.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns, either 1D for univariate or 2D for multivariate
    kernel_type : str
        Type of kernel for weighting ('Bartlett', 'Parzen', etc.)
    bandwidth : int
        Kernel bandwidth parameter
        
    Returns
    -------
    np.ndarray
        Kernel-based realized covariance matrix
    """
    # Reshape returns for consistent handling
    if returns.ndim == 1:
        n_vars = 1
        n_obs = len(returns)
        returns_2d = returns.reshape(n_obs, 1)
    else:
        n_obs, n_vars = returns.shape
        returns_2d = returns
    
    # Basic input checks
    if n_obs < 2 or bandwidth < 1:
        # Return zeros with appropriate dimensions
        return np.zeros((n_vars, n_vars))
    
    # Compute kernel weights
    weights = _compute_kernel_weights(kernel_type, bandwidth)
    
    # Initialize covariance matrix
    cov_matrix = np.zeros((n_vars, n_vars))
    
    # Compute main diagonal (variances)
    for i in range(n_vars):
        # For the diagonal, we use the simple sum of squares (weight=1 at lag 0)
        cov_matrix[i, i] = np.sum(returns_2d[:, i]**2)
        
        # Add autocovariance terms
        for lag in range(1, min(bandwidth + 1, n_obs)):
            auto_cov = np.sum(returns_2d[lag:, i] * returns_2d[:-lag, i])
            weight = weights[bandwidth + lag]
            cov_matrix[i, i] += 2.0 * weight * auto_cov
    
    # Compute off-diagonal elements (cross-covariances)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # Contemporary covariance (lag 0)
            cov_matrix[i, j] = np.sum(returns_2d[:, i] * returns_2d[:, j])
            
            # Add cross-covariance terms
            for lag in range(1, min(bandwidth + 1, n_obs)):
                # Positive lag: i leads j
                cross_cov_pos = np.sum(returns_2d[lag:, j] * returns_2d[:-lag, i])
                # Negative lag: j leads i
                cross_cov_neg = np.sum(returns_2d[lag:, i] * returns_2d[:-lag, j])
                
                weight = weights[bandwidth + lag]
                cov_matrix[i, j] += weight * (cross_cov_pos + cross_cov_neg)
            
            # Mirror value for symmetric matrix
            cov_matrix[j, i] = cov_matrix[i, j]
    
    return cov_matrix


@jit(nopython=True)
def noise_variance(prices: np.ndarray, 
                  times: np.ndarray) -> float:
    """
    Estimates microstructure noise variance in high-frequency data using Numba optimization.
    
    This function implements an estimator for the variance of microstructure
    noise in high-frequency price data based on first differences of log prices.
    
    Parameters
    ----------
    prices : np.ndarray
        Array of high-frequency price data
    times : np.ndarray
        Array of time stamps corresponding to prices
        
    Returns
    -------
    float
        Estimated noise variance
    """
    # Check inputs (limited due to Numba constraints)
    n = len(prices)
    if n < 2 or len(times) != n:
        return 0.0
    
    # Compute log prices
    log_prices = np.log(prices)
    
    # First differences
    returns = np.diff(log_prices)
    
    # Estimate noise variance using autocovariance
    # This is a simple estimator based on the negative of the first-order autocovariance
    mean_return = np.mean(returns)
    demean_returns = returns - mean_return
    
    # Compute first-order autocovariance
    auto_cov = 0.0
    for i in range(len(demean_returns) - 1):
        auto_cov += demean_returns[i] * demean_returns[i + 1]
    auto_cov /= (len(demean_returns) - 1)
    
    # Noise variance is -0.5 * first-order autocovariance
    # This is a common estimator in the market microstructure literature
    noise_var = -0.5 * auto_cov
    
    # Ensure positive result (autocovariance could be positive due to sampling)
    return max(noise_var, 0.0)


@dataclass
class RealizedMeasure:
    """
    Base class for realized volatility estimation with various sampling schemes and Numba optimization.
    
    This class provides a unified interface for computing different types of
    realized volatility measures from high-frequency financial data, with
    support for various sampling schemes and noise filtering methods.
    
    Attributes
    ----------
    sampling_type : str
        Method for sampling ('CalendarTime', 'BusinessTime', etc.)
    sampling_interval : Union[int, tuple]
        Sampling interval or parameters for the sampling method
    estimates : np.ndarray
        Array of computed realized measure estimates
        
    Methods
    -------
    compute
        Compute realized measure from high-frequency data
    get_confidence_intervals
        Calculate confidence intervals for realized measures
    """
    sampling_type: str
    sampling_interval: Union[int, tuple]
    estimates: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """
        Initialize realized measure calculator with validation.
        """
        # Validate sampling type
        if self.sampling_type not in SAMPLING_METHODS:
            raise ValueError(f"Sampling type must be one of {SAMPLING_METHODS}")
        
        # Validate sampling interval based on sampling type
        if self.sampling_type == 'Fixed' and not isinstance(self.sampling_interval, int):
            raise ValueError("Fixed sampling requires an integer interval")
        elif self.sampling_type == 'CalendarTime' and not isinstance(self.sampling_interval, tuple):
            raise ValueError("CalendarTime sampling requires a tuple of (min_interval, max_interval)")
        elif self.sampling_type == 'BusinessTime' and not isinstance(self.sampling_interval, tuple):
            raise ValueError("BusinessTime sampling requires a tuple of (min_ticks, max_ticks, max_time)")
        
        logger.debug(f"Initialized RealizedMeasure with {self.sampling_type} sampling")
    
    def compute(self, prices: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Compute realized measure from high-frequency data using Numba optimization.
        
        This method calculates realized volatility measures from high-frequency
        price data using the configured sampling method and parameters.
        
        Parameters
        ----------
        prices : np.ndarray
            Array of high-frequency price data
        times : np.ndarray
            Array of time stamps corresponding to prices
            
        Returns
        -------
        np.ndarray
            Realized measure estimates
        """
        # Validate inputs
        validate_array_input(prices)
        validate_array_input(times)
        
        if len(prices) != len(times):
            raise ValueError("Prices and times arrays must have the same length")
        
        # Compute realized variance
        rv, rv_ss = realized_variance(
            prices, times, 'timestamp', self.sampling_type, self.sampling_interval
        )
        
        # Additional metrics can be computed here (bipower variation, etc.)
        
        # Store results
        self.estimates = np.array([rv, rv_ss])
        
        return self.estimates
    
    def get_confidence_intervals(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for realized measures using statistical methods.
        
        This method computes confidence intervals for the realized volatility
        estimates based on asymptotic theory and the specified confidence level.
        
        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals, by default 0.05
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing lower and upper confidence bounds
        """
        # Validate alpha
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        # Check if estimates are available
        if self.estimates.size == 0:
            raise ValueError("No estimates available. Run compute() first.")
        
        # Simple approximation for confidence intervals
        # In practice, more sophisticated methods could be used
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # Approximate standard errors (simplified asymptotic theory)
        # For realized variance, std error ≈ 2√(2/n) * RV
        # where n is the effective sample size
        std_errors = np.zeros_like(self.estimates)
        
        # Approximate sample size from the ratio of raw and subsampled estimates
        if self.estimates[1] > 0:
            noise_ratio = self.estimates[0] / self.estimates[1]
            effective_n = max(30, int(100 / noise_ratio)) if noise_ratio > 1 else 100
        else:
            effective_n = 100
        
        # Compute standard errors
        std_errors[0] = 2 * np.sqrt(2.0 / effective_n) * self.estimates[0]  # For RV
        std_errors[1] = std_errors[0] / np.sqrt(2)  # Subsampled has lower std error
        
        # Compute confidence intervals
        lower_bound = self.estimates - z_score * std_errors
        upper_bound = self.estimates + z_score * std_errors
        
        # Ensure non-negative lower bounds for variance measures
        lower_bound = np.maximum(lower_bound, 0)
        
        return lower_bound, upper_bound