"""
Main initialization module for the MFE (MATLAB Financial Econometrics) Toolbox Python implementation.

Provides version information, core functionality exports, and a unified interface
to the package's modeling and statistical capabilities. Implements comprehensive error
handling, logging, and type safety through Python 3.12 features.
"""

from typing import Optional
import logging

# Version and author information
__version__ = '4.0'
__author__ = 'Kevin Sheppard'

# Configure logger
logger = logging.getLogger(__name__)

# Import and re-export key functionality from submodules
from .models import ARMAModel
from .core import GED, optimize_garch

def configure_logging(level: Optional[int] = None, format: Optional[str] = None) -> None:
    """
    Configures package-wide logging settings.
    
    Parameters
    ----------
    level : Optional[int]
        Logging level (e.g., logging.INFO, logging.DEBUG)
    format : Optional[str]
        Log message format string
    
    Returns
    -------
    None
    """
    # Set default log level if not provided
    if level is None:
        level = logging.INFO
    
    # Set default format if not provided
    if format is None:
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger with specified settings
    logging.basicConfig(level=level, format=format)
    
    logger.debug(f"Logging configured with level={level}")

# Define what gets imported with `from backend import *`
__all__ = [
    'ARMAModel',
    'GED',
    'optimize_garch',
    'configure_logging',
    '__version__',
    '__author__'
]