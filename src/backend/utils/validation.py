"""
Input validation module for MFE Toolbox.

This module provides comprehensive validation utilities for ensuring data
integrity across all modeling components of the MFE Toolbox. It implements
robust parameter checking, array validation, and model order verification
using Python's type hints and NumPy's array operations.

Functions in this module are used throughout the toolbox to validate inputs
before performing computations, ensuring that errors are caught early with
clear error messages.
"""

import logging
import numpy as np
from typing import Optional, Union, Tuple, Any, List

# Configure logger
logger = logging.getLogger(__name__)

# Global constants
MAX_ORDER = 30
VALID_DISTRIBUTIONS = ['normal', 'student-t', 'ged', 'skewed-t']
VALID_GARCH_TYPES = ['GARCH', 'EGARCH', 'GJR-GARCH', 'TARCH', 'AGARCH', 'FIGARCH']


def validate_array_input(x: np.ndarray,
                         expected_shape: Optional[tuple] = None,
                         dtype: Optional[str] = None) -> bool:
    """
    Validates NumPy array inputs for numerical computations.
    
    This function performs comprehensive validation on array inputs, checking
    that they are properly formed NumPy arrays with the expected shape,
    data type, and contain only finite values.
    
    Parameters
    ----------
    x : np.ndarray
        The input array to validate
    expected_shape : Optional[tuple]
        The expected shape of the array, if None, any non-empty shape is valid
    dtype : Optional[str]
        The expected data type of the array, if None, any numeric type is valid
        
    Returns
    -------
    bool
        True if validation passes
        
    Raises
    ------
    TypeError
        If input is not a NumPy array
    ValueError
        If array is empty, has incorrect shape, or contains non-finite values
    """
    # Check if input is a NumPy array
    if not isinstance(x, np.ndarray):
        error_msg = f"Input must be a NumPy array, got {type(x).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Check if array is empty
    if x.size == 0:
        error_msg = "Input array cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate shape if specified
    if expected_shape is not None:
        if x.shape != expected_shape:
            error_msg = f"Expected array with shape {expected_shape}, got {x.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Validate data type if specified
    if dtype is not None:
        if not np.issubdtype(x.dtype, np.dtype(dtype).type):
            error_msg = f"Expected array with dtype {dtype}, got {x.dtype}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Check for non-finite values (NaN, Inf)
    if not np.isfinite(x).all():
        error_msg = "Array contains non-finite values (NaN or Inf)"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return True


def validate_parameters(params: Union[float, np.ndarray],
                        bounds: Optional[Tuple[float, float]] = None,
                        param_type: Optional[str] = None) -> bool:
    """
    Validates model parameters against specified bounds and constraints.
    
    This function ensures that model parameters are within acceptable ranges
    and satisfy any model-specific constraints required for valid estimation.
    
    Parameters
    ----------
    params : Union[float, np.ndarray]
        Parameter value(s) to validate
    bounds : Optional[Tuple[float, float]]
        Tuple containing (lower_bound, upper_bound) for parameters
    param_type : Optional[str]
        Type of parameters (e.g., 'GARCH', 'ARMA') for specialized validation
        
    Returns
    -------
    bool
        True if validation passes
        
    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters violate bounds or model-specific constraints
    """
    # Convert single float to array for consistent handling
    if isinstance(params, (float, int)):
        params = np.array([params])
    elif not isinstance(params, np.ndarray):
        error_msg = f"Parameters must be float or NumPy array, got {type(params).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Check for non-finite values
    if not np.isfinite(params).all():
        error_msg = "Parameters contain non-finite values"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check bounds if provided
    if bounds is not None:
        lower, upper = bounds
        if not (lower <= params).all() or not (params <= upper).all():
            error_msg = f"Parameters must be within bounds [{lower}, {upper}]"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Parameter-specific validation
    if param_type is not None:
        if param_type == 'GARCH':
            # GARCH stability condition: alpha + beta < 1
            if params.size >= 2 and params[0] + params[1] >= 1:
                error_msg = f"GARCH stability condition violated: alpha + beta = {params[0] + params[1]} >= 1"
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif param_type == 'ARMA':
            # For ARMA, we might check stationarity or invertibility
            # Simplified check: parameters between -1 and 1
            if not (np.abs(params) < 1).all():
                error_msg = "ARMA parameters must have absolute values less than 1 for stationarity"
                logger.error(error_msg)
                raise ValueError(error_msg)
    
    return True


def validate_model_order(p: int,
                         q: int,
                         model_type: Optional[str] = None) -> bool:
    """
    Validates ARMA/GARCH model orders and ensures they meet requirements.
    
    This function checks that model orders are non-negative, don't exceed
    maximum allowed values, and satisfy model-specific constraints.
    
    Parameters
    ----------
    p : int
        Autoregressive order (p)
    q : int
        Moving average order (q)
    model_type : Optional[str]
        Type of model for specialized validation
        
    Returns
    -------
    bool
        True if validation passes
        
    Raises
    ------
    TypeError
        If orders are not integers
    ValueError
        If orders are negative, exceed maximum allowed, or violate model constraints
    """
    # Check if orders are integers
    if not isinstance(p, int) or not isinstance(q, int):
        error_msg = f"Model orders must be integers, got p: {type(p).__name__}, q: {type(q).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Check if orders are non-negative
    if p < 0 or q < 0:
        error_msg = f"Model orders must be non-negative, got p: {p}, q: {q}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if orders exceed maximum
    if p > MAX_ORDER or q > MAX_ORDER:
        error_msg = f"Model orders exceed maximum ({MAX_ORDER}), got p: {p}, q: {q}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Model-specific validation
    if model_type is not None:
        if model_type == 'ARMA' and p == 0 and q == 0:
            error_msg = "ARMA model must have at least one non-zero order"
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif model_type == 'GARCH' and p + q == 0:
            error_msg = "GARCH model must have at least one non-zero order"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    return True


def validate_distribution_type(distribution: str) -> bool:
    """
    Validates statistical distribution specifications for error terms.
    
    This function ensures that the specified distribution type is supported
    by the toolkit and can be used for model estimation.
    
    Parameters
    ----------
    distribution : str
        The distribution type to validate
        
    Returns
    -------
    bool
        True if validation passes
        
    Raises
    ------
    ValueError
        If distribution type is not supported
    """
    # Check if distribution type is valid
    if distribution.lower() not in [d.lower() for d in VALID_DISTRIBUTIONS]:
        error_msg = f"Distribution '{distribution}' not supported. Valid options: {', '.join(VALID_DISTRIBUTIONS)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return True


def validate_garch_type(model_type: str) -> bool:
    """
    Validates GARCH model type specifications.
    
    This function ensures that the specified GARCH model type is supported
    by the toolkit and can be used for volatility modeling.
    
    Parameters
    ----------
    model_type : str
        The GARCH model type to validate
        
    Returns
    -------
    bool
        True if validation passes
        
    Raises
    ------
    ValueError
        If GARCH model type is not supported
    """
    # Check if GARCH model type is valid (case-insensitive)
    if model_type.upper() not in [t.upper() for t in VALID_GARCH_TYPES]:
        error_msg = f"GARCH model type '{model_type}' not supported. Valid options: {', '.join(VALID_GARCH_TYPES)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return True