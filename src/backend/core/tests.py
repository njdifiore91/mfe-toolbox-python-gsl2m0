"""
Statistical testing module for MFE Toolbox.

This module provides comprehensive statistical testing functionality,
including hypothesis tests, diagnostic tests, and test statistics for
time series and econometric analysis. Functions are optimized using
Numba's JIT compilation for high performance.
"""

import numpy as np
from scipy import stats
from numba import jit
from typing import Tuple, Optional, Union
import logging

# Import validation utilities
from ..utils.validation import validate_array_input

# Configure logger
logger = logging.getLogger(__name__)


@jit(nopython=True)
def ljung_box(residuals: np.ndarray, lags: int) -> Tuple[float, float]:
    """
    Compute Ljung-Box Q-statistic for testing serial correlation in residuals with Numba optimization.
    
    The function computes the Ljung-Box Q-statistic and its p-value for detecting 
    autocorrelation in the residuals of a time series model. The test is useful for 
    determining whether there is significant autocorrelation at specific lags.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression or time series model
    lags : int
        Number of lags to include in the test
        
    Returns
    -------
    Tuple[float, float]
        Q-statistic and its p-value
    """
    # Simple validation inside Numba function
    if residuals.ndim != 1:
        return -1.0, 0.0  # Error code for invalid input
        
    n = residuals.shape[0]
    
    if lags <= 0 or lags >= n:
        return -1.0, 0.0  # Error code for invalid lags
    
    # Compute sample autocorrelations
    acf = np.zeros(lags+1)
    variance = np.var(residuals)
    
    if variance <= 1e-10:  # Numerical stability check
        return 0.0, 1.0  # No correlation if variance is near zero
    
    # Compute autocorrelations
    for lag in range(lags+1):
        numerator = 0.0
        for i in range(n-lag):
            numerator += residuals[i] * residuals[i+lag]
        acf[lag] = numerator / ((n-lag) * variance)
    
    # Normalize lag-0 autocorrelation
    if acf[0] > 1e-10:
        acf = acf / acf[0]
    
    # Compute Q-statistic
    q_stat = 0.0
    for lag in range(1, lags+1):
        q_stat += (acf[lag]**2) / (n - lag)
    
    q_stat = n * (n + 2) * q_stat
    
    # Approximate p-value using chi-square distribution
    # For Numba compatibility, we use an approximation
    # In practice, this would be replaced with scipy.stats.chi2.sf(q_stat, lags)
    p_value = _chi2_sf_approx(q_stat, lags)
    
    return q_stat, p_value


@jit(nopython=True)
def _chi2_sf_approx(x: float, df: int) -> float:
    """
    Approximate chi-square survival function (1-CDF) for Numba compatibility.
    
    Parameters
    ----------
    x : float
        Value to evaluate
    df : int
        Degrees of freedom
        
    Returns
    -------
    float
        Approximate survival function value
    """
    if x <= 0:
        return 1.0
    
    if df <= 0:
        return 0.0
    
    # Wilson-Hilferty transformation for chi-square to normal
    z = ((x/df)**(1/3) - (1 - 2/(9*df))) / np.sqrt(2/(9*df))
    
    # Approximate normal survival function
    return _normal_sf_approx(z)


@jit(nopython=True)
def _normal_sf_approx(x: float) -> float:
    """
    Approximate standard normal survival function for Numba compatibility.
    
    Parameters
    ----------
    x : float
        Value to evaluate
        
    Returns
    -------
    float
        Approximate survival function value
    """
    if x < -8.0:
        return 1.0
    if x > 8.0:
        return 0.0
    
    # Polynomial approximation of normal survival function
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * np.exp(-x * x)
    
    if sign < 0.0:
        return 1.0 - 0.5 * y
    else:
        return 0.5 * y


@jit(nopython=True)
def arch_lm(residuals: np.ndarray, lags: int) -> Tuple[float, float]:
    """
    Compute ARCH-LM test for heteroskedasticity in residuals using Numba-accelerated computations.
    
    The ARCH-LM (AutoRegressive Conditional Heteroskedasticity Lagrange Multiplier) test
    detects the presence of ARCH effects in the residuals of a time series model. It tests
    whether the squared residuals exhibit autocorrelation.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression or time series model
    lags : int
        Number of lags to include in the auxiliary regression
        
    Returns
    -------
    Tuple[float, float]
        LM test statistic and its p-value
    """
    # Simple validation inside Numba function
    if residuals.ndim != 1:
        return -1.0, 0.0  # Error code for invalid input
        
    n = residuals.shape[0]
    
    if lags <= 0 or lags >= n:
        return -1.0, 0.0  # Error code for invalid lags
    
    # Compute squared residuals
    squared_resid = residuals**2
    
    # Initialize lagged squared residuals matrix (constant term + lags)
    X = np.ones((n-lags, lags+1))
    
    # Fill matrix with lagged squared residuals
    for lag in range(1, lags+1):
        for t in range(n-lags):
            X[t, lag] = squared_resid[t+lags-lag]
    
    # Compute y vector (squared residuals for the auxiliary regression)
    y = squared_resid[lags:]
    
    # Compute OLS regression with optimized matrix operations
    XtX = np.zeros((lags+1, lags+1))
    Xty = np.zeros(lags+1)
    
    # Compute X'X and X'y with optimized loops
    for i in range(lags+1):
        for j in range(lags+1):
            sum_val = 0.0
            for t in range(n-lags):
                sum_val += X[t, i] * X[t, j]
            XtX[i, j] = sum_val
        
        for t in range(n-lags):
            Xty[i] += X[t, i] * y[t]
    
    # Check for numerical issues
    if np.any(np.isnan(XtX)) or np.any(np.isinf(XtX)):
        return 0.0, 1.0
    
    # Compute matrix inverse with numerical stability checks
    det = np.linalg.det(XtX)
    if abs(det) < 1e-10:
        return 0.0, 1.0  # Near-singular matrix
    
    # Compute OLS estimates
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ Xty
    
    # Compute fitted values and residuals
    fitted = np.zeros(n-lags)
    for i in range(n-lags):
        for j in range(lags+1):
            fitted[i] += X[i, j] * beta[j]
    
    e = y - fitted
    
    # Compute R-squared of auxiliary regression
    y_mean = np.mean(y)
    tss = np.sum((y - y_mean)**2)
    
    if tss <= 1e-10:  # Numerical stability check
        return 0.0, 1.0  # Handle zero variance case
    
    rss = np.sum(e**2)
    r_squared = 1.0 - rss/tss
    
    # Compute LM statistic = n*R^2
    lm_stat = (n-lags) * r_squared
    
    # Approximate p-value using chi-square distribution
    p_value = _chi2_sf_approx(lm_stat, lags)
    
    return lm_stat, p_value


@jit(nopython=True)
def durbin_watson(residuals: np.ndarray) -> float:
    """
    Compute Durbin-Watson statistic for testing first-order serial correlation with Numba optimization.
    
    The Durbin-Watson statistic tests for first-order serial correlation in the residuals
    of a regression model. Values close to 2 indicate no autocorrelation, values towards 0
    indicate positive autocorrelation, and values towards 4 indicate negative autocorrelation.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression model
        
    Returns
    -------
    float
        DW test statistic with numerical stability guarantees
    """
    # Simple validation inside Numba function
    if residuals.ndim != 1:
        return -1.0  # Error code for invalid input
        
    n = residuals.shape[0]
    
    if n <= 1:
        return -1.0  # Error code for insufficient observations
    
    # Compute sum of squared differences
    sum_squared_diff = 0.0
    for i in range(1, n):
        diff = residuals[i] - residuals[i-1]
        sum_squared_diff += diff * diff
    
    # Compute sum of squares
    sum_squared = 0.0
    for i in range(n):
        sum_squared += residuals[i] * residuals[i]
    
    # Avoid division by zero with numerical stability check
    if sum_squared <= 1e-10:
        return 2.0  # Return 2 (no autocorrelation) if variance is near zero
    
    # Compute DW statistic
    dw_stat = sum_squared_diff / sum_squared
    
    return dw_stat


@jit(nopython=True)
def white_test(residuals: np.ndarray, regressors: np.ndarray) -> Tuple[float, float]:
    """
    Perform White's test for heteroskedasticity using Numba-accelerated computations.
    
    White's test is used to detect heteroskedasticity in the residuals from a regression
    model. It regresses squared residuals on the original regressors, their squares, and
    their cross-products.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression model
    regressors : np.ndarray
        Matrix of regressors
        
    Returns
    -------
    Tuple[float, float]
        White's test statistic and its p-value
    """
    # Simple validation inside Numba function
    if residuals.ndim != 1:
        return -1.0, 0.0  # Error code for invalid input
        
    if regressors.ndim != 2:
        return -1.0, 0.0  # Error code for invalid regressors
        
    n = residuals.shape[0]
    k = regressors.shape[1]
    
    if regressors.shape[0] != n:
        return -1.0, 0.0  # Error code for dimension mismatch
    
    # Compute squared residuals
    squared_resid = residuals**2
    
    # Number of columns in auxiliary regression:
    # 1 (constant) + k (original regressors) + k*(k+1)/2 (squares and cross-products)
    num_cols = 1 + k + k*(k+1)//2
    X = np.zeros((n, num_cols))
    
    # Set constant term
    for i in range(n):
        X[i, 0] = 1.0
    
    # Include original regressors
    for i in range(n):
        for j in range(k):
            X[i, j+1] = regressors[i, j]
    
    # Add squares and cross-products
    col_idx = k + 1
    for i in range(k):
        for j in range(i, k):
            for t in range(n):
                X[t, col_idx] = regressors[t, i] * regressors[t, j]
            col_idx += 1
    
    # Compute auxiliary regression with optimized matrix operations
    XtX = np.zeros((num_cols, num_cols))
    Xty = np.zeros(num_cols)
    
    # Compute X'X and X'y
    for i in range(num_cols):
        for j in range(num_cols):
            sum_val = 0.0
            for t in range(n):
                sum_val += X[t, i] * X[t, j]
            XtX[i, j] = sum_val
        
        for t in range(n):
            Xty[i] += X[t, i] * squared_resid[t]
    
    # Check for numerical issues
    if np.any(np.isnan(XtX)) or np.any(np.isinf(XtX)):
        return 0.0, 1.0
    
    # Compute matrix inverse with numerical stability checks
    det = np.linalg.det(XtX)
    if abs(det) < 1e-10:
        return 0.0, 1.0  # Near-singular matrix
    
    # Compute OLS estimates
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ Xty
    
    # Compute fitted values and residuals
    fitted = np.zeros(n)
    for i in range(n):
        for j in range(num_cols):
            fitted[i] += X[i, j] * beta[j]
    
    e = squared_resid - fitted
    
    # Compute R-squared of auxiliary regression
    y_mean = np.mean(squared_resid)
    tss = np.sum((squared_resid - y_mean)**2)
    
    if tss <= 1e-10:  # Numerical stability check
        return 0.0, 1.0  # Handle zero variance case
    
    rss = np.sum(e**2)
    r_squared = 1.0 - rss/tss
    
    # Compute White's statistic = n*R^2
    white_stat = n * r_squared
    
    # Calculate degrees of freedom
    df = num_cols - 1  # Subtract 1 for constant term
    
    # Approximate p-value using chi-square distribution
    p_value = _chi2_sf_approx(white_stat, df)
    
    return white_stat, p_value


@jit(nopython=True)
def breusch_pagan(residuals: np.ndarray, regressors: np.ndarray) -> Tuple[float, float]:
    """
    Perform Breusch-Pagan test for heteroskedasticity with Numba-optimized computations.
    
    The Breusch-Pagan test checks whether the variance of the errors from a regression
    is dependent on the values of the independent variables. It regresses the squared
    residuals on the original regressors.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression model
    regressors : np.ndarray
        Matrix of regressors
        
    Returns
    -------
    Tuple[float, float]
        Breusch-Pagan test statistic and its p-value
    """
    # Simple validation inside Numba function
    if residuals.ndim != 1:
        return -1.0, 0.0  # Error code for invalid input
        
    if regressors.ndim != 2:
        return -1.0, 0.0  # Error code for invalid regressors
        
    n = residuals.shape[0]
    k = regressors.shape[1]
    
    if regressors.shape[0] != n:
        return -1.0, 0.0  # Error code for dimension mismatch
    
    # Compute squared residuals
    squared_resid = residuals**2
    
    # Compute mean squared residuals
    sigma_squared = np.mean(squared_resid)
    
    if sigma_squared <= 1e-10:  # Numerical stability check
        return 0.0, 1.0  # Handle zero variance case
    
    # Normalize squared residuals
    g = squared_resid / sigma_squared
    
    # Compute auxiliary regression with optimized matrix operations
    XtX = np.zeros((k, k))
    Xty = np.zeros(k)
    
    # Compute X'X and X'y
    for i in range(k):
        for j in range(k):
            sum_val = 0.0
            for t in range(n):
                sum_val += regressors[t, i] * regressors[t, j]
            XtX[i, j] = sum_val
        
        for t in range(n):
            Xty[i] += regressors[t, i] * g[t]
    
    # Check for numerical issues
    if np.any(np.isnan(XtX)) or np.any(np.isinf(XtX)):
        return 0.0, 1.0
    
    # Compute matrix inverse with numerical stability checks
    det = np.linalg.det(XtX)
    if abs(det) < 1e-10:
        return 0.0, 1.0  # Near-singular matrix
    
    # Compute OLS estimates
    XtX_inv = np.linalg.inv(XtX)
    gamma = XtX_inv @ Xty
    
    # Compute fitted values
    fitted = np.zeros(n)
    for i in range(n):
        for j in range(k):
            fitted[i] += regressors[i, j] * gamma[j]
    
    # Compute explained sum of squares
    g_mean = np.mean(g)
    ess = np.sum((fitted - g_mean)**2)
    
    # Compute BP statistic = 0.5*ESS
    bp_stat = 0.5 * ess
    
    # Approximate p-value using chi-square distribution
    p_value = _chi2_sf_approx(bp_stat, k)
    
    return bp_stat, p_value


# User-facing wrapper functions to handle proper validation and error reporting
def validate_and_prepare_residuals(residuals: np.ndarray) -> np.ndarray:
    """
    Validate and prepare residuals array for testing.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression or time series model
        
    Returns
    -------
    np.ndarray
        Validated residuals array ready for testing
    """
    # Validate input using the utility function
    validate_array_input(residuals)
    
    # Ensure residuals is a 1D array
    if residuals.ndim > 1:
        if residuals.shape[1] == 1:
            residuals = residuals.flatten()
        else:
            raise ValueError("Residuals must be a 1D array or column vector")
    
    return residuals


def apply_ljung_box(residuals: np.ndarray, lags: int) -> Tuple[float, float]:
    """
    Apply Ljung-Box test with proper validation and error handling.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression or time series model
    lags : int
        Number of lags to include in the test
        
    Returns
    -------
    Tuple[float, float]
        Q-statistic and its p-value from scipy.stats
    """
    try:
        # Validate and prepare inputs
        residuals = validate_and_prepare_residuals(residuals)
        
        if not isinstance(lags, int) or lags <= 0:
            raise ValueError(f"lags must be a positive integer, got {lags}")
        
        if lags >= residuals.shape[0]:
            raise ValueError(f"lags ({lags}) must be less than sample size ({residuals.shape[0]})")
        
        # Call Numba-optimized function
        q_stat, p_value_approx = ljung_box(residuals, lags)
        
        # Check for error codes
        if q_stat < 0:
            raise ValueError("Error computing Ljung-Box statistic")
        
        # Calculate accurate p-value using scipy
        p_value = stats.chi2.sf(q_stat, lags)
        
        logger.debug(f"Ljung-Box test: Q = {q_stat:.4f}, p-value = {p_value:.4f}")
        
        return q_stat, p_value
        
    except Exception as e:
        logger.error(f"Error in Ljung-Box test: {str(e)}")
        raise