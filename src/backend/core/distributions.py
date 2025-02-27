"""
Advanced statistical distributions and testing module for MFE Toolbox.

This module implements robust distribution modeling, parameter estimation,
and statistical testing capabilities with Numba-optimized computations.
It provides specialized distributions like Generalized Error Distribution (GED)
and Hansen's Skewed T-distribution, along with statistical tests such as
Jarque-Bera for normality testing.

The implementations leverage Numba's JIT compilation for performance-critical
operations while maintaining a clean, type-hinted interface.
"""

import numpy as np
import scipy.stats as stats
import scipy.special
from numba import jit
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Internal imports
from ..utils.validation import validate_array_input

# Configure logger
logger = logging.getLogger(__name__)


# Numba-optimized computational functions
@jit(nopython=True)
def _ged_pdf_numba(x, lambda_value, const, nu):
    """
    Numba-optimized GED PDF calculation.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
    lambda_value : float
        Pre-computed lambda constant
    const : float
        Pre-computed normalization constant
    nu : float
        Shape parameter
        
    Returns
    -------
    np.ndarray
        PDF values
    """
    abs_x_scaled = np.abs(x) / lambda_value
    pdf_values = const * np.exp(-np.power(abs_x_scaled, nu))
    return pdf_values


@jit(nopython=True)
def _skewt_pdf_numba(x, a, b, c, nu):
    """
    Numba-optimized Skewed T PDF calculation.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
    a, b, c : float
        Pre-computed constants
    nu : float
        Degrees of freedom
        
    Returns
    -------
    np.ndarray
        PDF values
    """
    pdf = np.zeros_like(x, dtype=np.float64)
    threshold = -a/b
    
    for i in range(len(x)):
        if x[i] < threshold:
            term = 1 + (1 / (nu - 2)) * np.square(b * x[i] + a)
            pdf[i] = b * c * np.power(term, -0.5 * (nu + 1))
        else:
            term = 1 + (1 / (nu - 2)) * np.square(b * x[i] + a)
            pdf[i] = b * c * np.power(term, -0.5 * (nu + 1))
    
    return pdf


@jit(nopython=True)
def _loglikelihood_numba(pdf_values):
    """
    Numba-optimized log-likelihood calculation.
    
    Parameters
    ----------
    pdf_values : np.ndarray
        PDF values for the data
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Add a small constant to prevent log(0)
    return np.sum(np.log(np.maximum(pdf_values, 1e-10)))


@jit(nopython=True)
def _skewness_numba(x):
    """
    Numba-optimized skewness calculation.
    
    Parameters
    ----------
    x : np.ndarray
        Input data
        
    Returns
    -------
    float
        Skewness value
    """
    n = len(x)
    if n < 3:
        return 0.0
    
    # Center the data
    mean = np.mean(x)
    x_centered = x - mean
    
    # Compute centered moments
    m2 = np.sum(x_centered**2) / n
    m3 = np.sum(x_centered**3) / n
    
    # Calculate skewness
    skew = m3 / (m2**1.5)
    
    return skew


@jit(nopython=True)
def _kurtosis_numba(x, excess=True):
    """
    Numba-optimized kurtosis calculation.
    
    Parameters
    ----------
    x : np.ndarray
        Input data
    excess : bool
        If True, return excess kurtosis (kurtosis - 3)
        
    Returns
    -------
    float
        Kurtosis value
    """
    n = len(x)
    if n < 4:
        return 0.0
    
    # Center the data
    mean = np.mean(x)
    x_centered = x - mean
    
    # Compute centered moments
    m2 = np.sum(x_centered**2) / n
    m4 = np.sum(x_centered**4) / n
    
    # Calculate kurtosis
    kurt = m4 / (m2**2)
    
    # Return excess kurtosis if requested
    if excess:
        kurt -= 3.0
    
    return kurt


@jit(nopython=True)
def _jarque_bera_numba(x):
    """
    Numba-optimized Jarque-Bera test statistic calculation.
    
    Parameters
    ----------
    x : np.ndarray
        Input data
        
    Returns
    -------
    Tuple[float, float]
        Tuple containing (test statistic, p-value approximation)
    """
    # Calculate skewness and kurtosis
    skew = _skewness_numba(x)
    kurt = _kurtosis_numba(x, excess=True)
    
    # Compute JB test statistic
    n = len(x)
    jb_stat = n / 6 * (skew**2 + (kurt**2) / 4)
    
    # Approximate p-value (for Numba compatibility)
    # In a real implementation, this would use a more accurate chi-square calculation
    p_value = np.exp(-0.5 * jb_stat)
    
    # Clamp p-value to [0, 1]
    p_value = max(0.0, min(1.0, p_value))
    
    return jb_stat, p_value


@dataclass
class GED:
    """
    Generalized Error Distribution implementation with Numba optimization.
    
    The Generalized Error Distribution (GED) is a symmetric distribution with
    a shape parameter that controls tail thickness. It includes the normal
    distribution as a special case (when nu=2) and can model both heavier and
    lighter tails than the normal distribution.
    
    Parameters
    ----------
    nu : float
        Shape parameter controlling tail behavior. Must be positive.
        nu=2 corresponds to normal distribution, lower values give heavier tails.
    
    Notes
    -----
    PDF given by: f(x) = [nu / (2*lambda*Γ(1/nu))] * exp(-(|x|/lambda)^nu)
    where lambda = [Γ(1/nu)/Γ(3/nu)]^0.5
    """
    
    nu: float
    _lambda: float = None
    _const: float = None
    
    def __post_init__(self):
        """
        Validate parameters and compute distribution constants.
        """
        if self.nu <= 0:
            error_msg = f"Shape parameter nu must be positive, got {self.nu}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Compute constants used in PDF calculation
        self._lambda, self._const = self._compute_constants()
        
    def _compute_constants(self) -> Tuple[float, float]:
        """
        Compute distribution constants needed for PDF calculation.
        
        Returns
        -------
        Tuple[float, float]
            Lambda value and normalization constant
        """
        # Special case for normal distribution
        if abs(self.nu - 2.0) < 1e-10:
            return 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0 * np.pi)
        
        # General case
        gamma_1_nu = scipy.special.gamma(1.0 / self.nu)
        gamma_3_nu = scipy.special.gamma(3.0 / self.nu)
        lambda_value = np.sqrt(gamma_1_nu / gamma_3_nu)
        const = self.nu / (2.0 * lambda_value * gamma_1_nu)
        
        return lambda_value, const
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute probability density function for GED.
        
        Parameters
        ----------
        x : np.ndarray
            Input values where PDF will be evaluated
            
        Returns
        -------
        np.ndarray
            PDF values corresponding to input x
        """
        validate_array_input(x)
        
        # Call Numba-optimized function with pre-computed constants
        return _ged_pdf_numba(x, self._lambda, self._const, self.nu)
    
    def loglikelihood(self, x: np.ndarray) -> float:
        """
        Compute log-likelihood for data under GED distribution.
        
        Parameters
        ----------
        x : np.ndarray
            Input data for log-likelihood computation
            
        Returns
        -------
        float
            Log-likelihood value
        """
        validate_array_input(x)
        
        # Call Numba-optimized function
        pdf_values = self.pdf(x)
        return _loglikelihood_numba(pdf_values)


@dataclass
class SkewedT:
    """
    Hansen's Skewed T distribution implementation with Numba optimization.
    
    This distribution extends the Student's t-distribution to account for skewness.
    It is characterized by two parameters: nu (degrees of freedom) controlling
    tail thickness and lambda (skewness parameter) controlling asymmetry.
    
    Parameters
    ----------
    nu : float
        Degrees of freedom parameter. Must be > 2.
    lambda_ : float
        Skewness parameter. Must be between -1 and 1.
    
    Notes
    -----
    Implements Hansen's (1994) skewed t-distribution which can capture both
    heavy tails and asymmetry in financial returns.
    """
    
    nu: float
    lambda_: float
    _a: float = None
    _b: float = None
    _c: float = None
    
    def __post_init__(self):
        """
        Validate parameters and compute distribution constants.
        """
        if self.nu <= 2:
            error_msg = f"Shape parameter nu must be > 2, got {self.nu}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if self.lambda_ <= -1 or self.lambda_ >= 1:
            error_msg = f"Skewness parameter lambda must be in (-1, 1), got {self.lambda_}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Compute constants used in PDF calculation
        self._a, self._b, self._c = self._compute_constants()
        
    def _compute_constants(self) -> Tuple[float, float, float]:
        """
        Compute distribution constants needed for PDF calculation.
        
        Returns
        -------
        Tuple[float, float, float]
            a, b, and normalization constant c
        """
        a = 4 * self.lambda_ * ((self.nu - 2) / (self.nu - 1))
        b = np.sqrt(1 + a * a)
        c = scipy.special.gamma((self.nu + 1) / 2) / (
            scipy.special.gamma(self.nu / 2) * np.sqrt(np.pi * (self.nu - 2)))
        
        return a, b, c
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute probability density function for Skewed T.
        
        Parameters
        ----------
        x : np.ndarray
            Input values where PDF will be evaluated
            
        Returns
        -------
        np.ndarray
            PDF values corresponding to input x
        """
        validate_array_input(x)
        
        # Call Numba-optimized function with pre-computed constants
        return _skewt_pdf_numba(x, self._a, self._b, self._c, self.nu)
    
    def loglikelihood(self, x: np.ndarray) -> float:
        """
        Compute log-likelihood for data under Skewed T distribution.
        
        Parameters
        ----------
        x : np.ndarray
            Input data for log-likelihood computation
            
        Returns
        -------
        float
            Log-likelihood value
        """
        validate_array_input(x)
        
        # Call Numba-optimized function
        pdf_values = self.pdf(x)
        return _loglikelihood_numba(pdf_values)


def jarque_bera(x: np.ndarray) -> Tuple[float, float]:
    """
    Compute Jarque-Bera test statistic and p-value for normality testing.
    
    The Jarque-Bera test checks whether data have the skewness and kurtosis
    matching a normal distribution.
    
    Parameters
    ----------
    x : np.ndarray
        Input data
        
    Returns
    -------
    Tuple[float, float]
        Tuple containing (test statistic, p-value)
    """
    validate_array_input(x)
    
    # Call Numba-optimized function
    jb_stat, p_value_approx = _jarque_bera_numba(x)
    
    # For more accurate p-value in non-performance critical scenarios, 
    # uncomment the following:
    # p_value = 1.0 - stats.chi2.cdf(jb_stat, 2)
    
    return jb_stat, p_value_approx


def kurtosis(x: np.ndarray, excess: bool = True) -> float:
    """
    Compute excess kurtosis of data with Numba optimization.
    
    Kurtosis measures the "tailedness" of a distribution. The excess kurtosis
    is defined as kurtosis - 3, where 3 is the kurtosis of a normal distribution.
    
    Parameters
    ----------
    x : np.ndarray
        Input data
    excess : bool, optional
        If True, return excess kurtosis (kurtosis - 3), by default True
        
    Returns
    -------
    float
        Kurtosis value
    """
    validate_array_input(x)
    
    # Call Numba-optimized function
    return _kurtosis_numba(x, excess)


def skewness(x: np.ndarray) -> float:
    """
    Compute skewness of data with Numba optimization.
    
    Skewness measures the asymmetry of a distribution around its mean.
    
    Parameters
    ----------
    x : np.ndarray
        Input data
        
    Returns
    -------
    float
        Skewness value
    """
    validate_array_input(x)
    
    # Call Numba-optimized function
    return _skewness_numba(x)