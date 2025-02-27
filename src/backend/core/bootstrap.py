"""
Bootstrap module for dependent time series data.

This module provides robust bootstrap resampling methods for time series data with
temporal dependence, including block bootstrap and stationary bootstrap techniques.
All core resampling functions are optimized using Numba JIT compilation for high performance.

The module offers both direct access to optimized resampling functions and a high-level
Bootstrap class with asynchronous support for handling long-running bootstrap operations
with progress tracking.

Functions:
    block_bootstrap: Performs block bootstrap with fixed block size
    stationary_bootstrap: Performs stationary bootstrap with random block lengths

Classes:
    Bootstrap: High-level bootstrap manager with async support
"""

import numpy as np
import numba
import logging
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass
import asyncio

from ..utils.validation import validate_array_input

# Configure logger
logger = logging.getLogger(__name__)

# Global constants
DEFAULT_BLOCK_SIZE = 50


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _block_bootstrap_core(data: np.ndarray, 
                          num_bootstraps: int, 
                          block_size: int,
                          start_positions: np.ndarray) -> np.ndarray:
    """
    Core implementation of block bootstrap optimized for Numba.
    
    This is the internal numba-optimized implementation that should be called
    after parameter validation and preparation.
    
    Parameters
    ----------
    data : np.ndarray
        Original time series data, must be a 1D NumPy array
    num_bootstraps : int
        Number of bootstrap samples to generate
    block_size : int
        Size of blocks to use for resampling
    start_positions : np.ndarray
        Pre-generated array of block starting positions
        
    Returns
    -------
    np.ndarray
        Array of bootstrapped samples with shape (num_bootstraps, data.shape[0])
    """
    data_length = data.shape[0]
    num_blocks = start_positions.shape[1]
    
    # Initialize output array for bootstrapped samples
    bootstrapped_samples = np.zeros((num_bootstraps, data_length))
    
    # Generate bootstrap samples
    for i in numba.prange(num_bootstraps):
        # Fill sample with blocks
        current_pos = 0
        for j in range(num_blocks):
            # Get starting position for this block
            start_pos = start_positions[i, j]
            
            # Calculate end positions ensuring we don't go beyond data_length
            end_pos_data = min(start_pos + block_size, data_length)
            end_pos_sample = min(current_pos + (end_pos_data - start_pos), data_length)
            
            # Copy block data
            block_length = end_pos_sample - current_pos
            bootstrapped_samples[i, current_pos:end_pos_sample] = data[start_pos:start_pos+block_length]
            
            # Update current position
            current_pos = end_pos_sample
            
            # Break if we've filled the sample
            if current_pos >= data_length:
                break
    
    return bootstrapped_samples


def block_bootstrap(data: np.ndarray, 
                    num_bootstraps: int, 
                    block_size: Optional[int] = None) -> np.ndarray:
    """
    Performs block bootstrap resampling for time series data.
    
    Block bootstrap resamples from the original data series using fixed-size 
    blocks to preserve the temporal dependence structure in the data.
    This implementation uses Numba JIT compilation for high performance.
    
    Parameters
    ----------
    data : np.ndarray
        Original time series data, must be a 1D NumPy array
    num_bootstraps : int
        Number of bootstrap samples to generate
    block_size : Optional[int], default None
        Size of blocks to use for resampling. If None, defaults to 
        DEFAULT_BLOCK_SIZE or 10% of data length, whichever is smaller.
        
    Returns
    -------
    np.ndarray
        Array of bootstrapped samples with shape (num_bootstraps, data.shape[0])
        
    Notes
    -----
    The function handles edge cases at the end of the series by truncating or
    wrapping blocks as needed to ensure all bootstrap samples have the same 
    length as the original data.
    """
    # Validate input data
    validate_array_input(data)
    
    if not isinstance(num_bootstraps, int) or num_bootstraps <= 0:
        raise ValueError(f"num_bootstraps must be a positive integer, got {num_bootstraps}")
    
    data_length = data.shape[0]
    
    # Set default block size if not provided
    if block_size is None:
        block_size = min(DEFAULT_BLOCK_SIZE, max(int(data_length * 0.1), 1))
    elif not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size}")
    
    # Calculate number of blocks needed
    num_blocks = int(np.ceil(data_length / block_size))
    
    # Generate random starting positions for blocks
    start_positions = np.random.randint(0, data_length, size=(num_bootstraps, num_blocks))
    
    # Call the numba-optimized core function
    return _block_bootstrap_core(data, num_bootstraps, block_size, start_positions)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _stationary_bootstrap_core(data: np.ndarray, 
                              num_bootstraps: int, 
                              block_probability: float) -> np.ndarray:
    """
    Core implementation of stationary bootstrap optimized for Numba.
    
    This is the internal numba-optimized implementation that should be called
    after parameter validation and preparation.
    
    Parameters
    ----------
    data : np.ndarray
        Original time series data, must be a 1D NumPy array
    num_bootstraps : int
        Number of bootstrap samples to generate
    block_probability : float
        Probability parameter for geometric distribution determining block lengths
        
    Returns
    -------
    np.ndarray
        Array of bootstrapped samples with shape (num_bootstraps, data.shape[0])
    """
    data_length = data.shape[0]
    
    # Initialize output array for bootstrapped samples
    bootstrapped_samples = np.zeros((num_bootstraps, data_length))
    
    # Generate bootstrap samples
    for i in numba.prange(num_bootstraps):
        current_pos = 0
        
        while current_pos < data_length:
            # Generate random starting position
            start_pos = np.random.randint(0, data_length)
            
            # Generate random block length from geometric distribution
            # For Numba compatibility, we simulate geometric distribution
            # using exponential approximation
            block_length = int(np.ceil(np.log(np.random.random()) / 
                                      np.log(1 - block_probability)))
            block_length = max(1, block_length)  # Ensure at least length 1
            
            # Copy block data with wrapping if needed
            for j in range(block_length):
                if current_pos >= data_length:
                    break
                    
                # Calculate position in original data with wrapping
                data_pos = (start_pos + j) % data_length
                bootstrapped_samples[i, current_pos] = data[data_pos]
                current_pos += 1
    
    return bootstrapped_samples


def stationary_bootstrap(data: np.ndarray, 
                         num_bootstraps: int, 
                         block_probability: Optional[float] = None) -> np.ndarray:
    """
    Implements stationary bootstrap with random block lengths.
    
    Stationary bootstrap uses random block lengths following a geometric 
    distribution to ensure stationarity in the bootstrapped series.
    This implementation uses Numba JIT compilation for high performance.
    
    Parameters
    ----------
    data : np.ndarray
        Original time series data, must be a 1D NumPy array
    num_bootstraps : int
        Number of bootstrap samples to generate
    block_probability : Optional[float], default None
        Probability parameter for geometric distribution determining block lengths.
        If None, defaults to 1/sqrt(data length).
        
    Returns
    -------
    np.ndarray
        Array of bootstrapped samples with shape (num_bootstraps, data.shape[0])
        
    Notes
    -----
    The function ensures proper handling of block boundaries and wraps around
    to the beginning of the series when a block extends beyond the end of data.
    """
    # Validate input data
    validate_array_input(data)
    
    if not isinstance(num_bootstraps, int) or num_bootstraps <= 0:
        raise ValueError(f"num_bootstraps must be a positive integer, got {num_bootstraps}")
    
    data_length = data.shape[0]
    
    # Set default block probability if not provided
    if block_probability is None:
        block_probability = 1.0 / np.sqrt(data_length)
    elif not isinstance(block_probability, float) or not (0 < block_probability < 1):
        raise ValueError(f"block_probability must be a float between 0 and 1, got {block_probability}")
    
    # Call the numba-optimized core function
    return _stationary_bootstrap_core(data, num_bootstraps, block_probability)


@dataclass
class Bootstrap:
    """
    Manages bootstrap resampling operations with configurable parameters.
    
    This class provides a high-level interface for performing bootstrap analysis
    with support for different resampling methods, async operations, and progress
    tracking for long-running bootstrap operations.
    
    Attributes
    ----------
    method : str
        Bootstrap method to use, either "block" or "stationary"
    options : dict
        Configuration options for the bootstrap method
    
    Methods
    -------
    async_bootstrap(data, num_bootstraps)
        Asynchronously perform bootstrap resampling with progress updates
    """
    method: str
    options: Optional[dict] = None
    
    def __post_init__(self):
        """Validate bootstrap parameters and set defaults after initialization."""
        # Validate method
        self.method = self.method.lower()
        if self.method not in ["block", "stationary"]:
            raise ValueError(f"Invalid bootstrap method: {self.method}. "
                             f"Must be 'block' or 'stationary'")
        
        # Initialize options with defaults if not provided
        if self.options is None:
            self.options = {}
        
        # Set method-specific default options
        if self.method == "block" and "block_size" not in self.options:
            self.options["block_size"] = None
        elif self.method == "stationary" and "block_probability" not in self.options:
            self.options["block_probability"] = None
            
        logger.debug(f"Initialized Bootstrap with method={self.method}, options={self.options}")
    
    async def async_bootstrap(self, data: np.ndarray, num_bootstraps: int) -> np.ndarray:
        """
        Asynchronously perform bootstrap resampling with progress updates.
        
        This method leverages the Numba-optimized core functions to perform
        bootstrap resampling while providing progress updates for long-running
        operations.
        
        Parameters
        ----------
        data : np.ndarray
            Original time series data, must be a 1D NumPy array
        num_bootstraps : int
            Number of bootstrap samples to generate
            
        Returns
        -------
        np.ndarray
            Array of bootstrapped samples with shape (num_bootstraps, data.shape[0])
        """
        # Validate input data
        validate_array_input(data)
        
        if not isinstance(num_bootstraps, int) or num_bootstraps <= 0:
            raise ValueError(f"num_bootstraps must be a positive integer, got {num_bootstraps}")
        
        logger.info(f"Starting async bootstrap with method={self.method}, "
                    f"num_bootstraps={num_bootstraps}, data_length={data.shape[0]}")
        
        # For async operation and progress tracking, divide the work into chunks
        # This is a compromise since we can't easily interleave Numba JIT execution
        # with Python-level progress reporting
        max_chunk_size = 100
        num_chunks = max(1, min(10, num_bootstraps // max_chunk_size))
        chunk_size = num_bootstraps // num_chunks
        remaining = num_bootstraps % num_chunks
        
        # Initialize results array
        results = np.zeros((num_bootstraps, data.shape[0]))
        
        # Process in chunks to provide progress feedback
        start_idx = 0
        for chunk in range(num_chunks):
            # Calculate current chunk size (may be larger for last chunk)
            current_chunk_size = chunk_size + (1 if chunk < remaining else 0)
            end_idx = start_idx + current_chunk_size
            
            # Allow for event loop to process other tasks
            await asyncio.sleep(0)
            
            # Perform bootstrap for this chunk based on selected method
            if self.method == "block":
                block_size = self.options.get("block_size")
                chunk_results = block_bootstrap(data, current_chunk_size, block_size)
            else:  # stationary
                block_probability = self.options.get("block_probability")
                chunk_results = stationary_bootstrap(data, current_chunk_size, block_probability)
            
            # Store chunk results
            results[start_idx:end_idx] = chunk_results
            
            # Update progress
            progress = (chunk + 1) / num_chunks
            logger.debug(f"Bootstrap progress: {progress:.1%} ({end_idx}/{num_bootstraps})")
            
            # Move to next chunk
            start_idx = end_idx
        
        logger.info(f"Completed bootstrap resampling with {num_bootstraps} samples")
        return results