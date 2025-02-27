"""
Package initialization module for the MFE Toolbox utilities package.

This module implements Python 3.12 features including strict type hints and 
exposes validation and printing functions while maintaining a clean namespace.
It provides a unified interface to the utility functions used throughout the 
MFE Toolbox.
"""

from typing import Union, List

# Import validation utilities
from .validation import (
    validate_parameters,
    validate_array_input,
    validate_model_order,
    validate_distribution_type,
    VALID_DISTRIBUTIONS,
    VALID_GARCH_TYPES
)

# Import printing utilities
from .printing import (
    format_parameter_table,
    format_model_summary,
    format_diagnostic_tests
)

# Define version
__version__ = '4.0.0'

# Create a wrapper for validate_distribution_params using validate_distribution_type
def validate_distribution_params(distribution: str) -> bool:
    """
    Validates statistical distribution parameters for error terms.
    
    This function ensures that the specified distribution parameters are valid
    and can be used for model estimation.
    
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
        If distribution parameters are not valid
    """
    return validate_distribution_type(distribution)

# Define format_array function
def format_array(array: Union[List[float], 'np.ndarray'], precision: int = 4) -> str:
    """
    Format a numerical array for display.
    
    This function converts an array-like object to a formatted string representation
    with specified precision for consistent output display.
    
    Parameters
    ----------
    array : Union[List[float], np.ndarray]
        The array to format
    precision : int, optional
        Number of decimal places to show, by default 4
        
    Returns
    -------
    str
        Formatted string representation of the array
    """
    import numpy as np
    
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    
    format_str = f"{{:.{precision}f}}"
    return np.array2string(array, formatter={'float_kind': lambda x: format_str.format(x)})