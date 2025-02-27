# MFE Toolbox Style Guide

**Version:** 1.0.0

## 1. Introduction

This document defines the coding style guidelines and standards for the MFE (MATLAB Financial Econometrics) Toolbox Python implementation. Adhering to these guidelines ensures code consistency, maintainability, and alignment with the project's technical requirements.

The MFE Toolbox has been completely reimplemented using Python 3.12, incorporating modern programming constructs such as async/await patterns and strict type hints. The toolbox leverages Python's scientific computing ecosystem, including NumPy, SciPy, Pandas, Statsmodels, and Numba for performance optimization.

This style guide aims to:
- Establish consistent coding practices across the project
- Promote code readability and maintainability
- Ensure proper utilization of Python 3.12 features
- Guide efficient implementation of scientific computing operations
- Define standards for documentation and testing

## 2. Python Coding Style

### 2.1 General Python Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide for Python code
- Use 4 spaces for indentation (no tabs)
- Limit lines to 88 characters (compatible with black formatter)
- Use explicit line continuation with parentheses for long statements
- Use blank lines to separate logical sections
- Use spaces around operators and after commas

### 2.2 Python Version

- All code must be compatible with **Python 3.12**
- Utilize Python 3.12 features where appropriate:
  - Enhanced error messages
  - Improved typing features
  - Performance optimizations for core types
  - Advanced pattern matching capabilities

### 2.3 Type Annotations and Hints

- Use strict type hints for all function parameters and return values
- Add type information to class attributes using annotations or dataclasses
- Employ the `typing` module for complex types
- Ensure type annotations are validated with mypy during development

```python
from typing import Optional, List, Dict, Union, Tuple
import numpy as np

def calculate_returns(prices: np.ndarray, 
                     method: str = 'simple') -> np.ndarray:
    """
    Calculate returns from price series.
    
    Parameters
    ----------
    prices : np.ndarray
        Array of asset prices
    method : str, optional
        Return calculation method ('simple' or 'log'), default 'simple'
        
    Returns
    -------
    np.ndarray
        Array of returns
    """
    if method not in ['simple', 'log']:
        raise ValueError("Method must be 'simple' or 'log'")
        
    if method == 'simple':
        return prices[1:] / prices[:-1] - 1
    else:  # log returns
        return np.diff(np.log(prices))
```

### 2.4 Naming Conventions

- **Modules**: Use lowercase names with underscores if needed (e.g., `volatility_models.py`)
- **Classes**: Use CapWords convention (e.g., `GARCHModel`)
- **Functions/Methods**: Use lowercase with underscores (e.g., `calculate_volatility`)
- **Variables**: Use lowercase with underscores (e.g., `return_series`)
- **Constants**: Use uppercase with underscores (e.g., `MAX_ITERATIONS`)
- **Type Variables**: Use CapWords with short, descriptive names (e.g., `T`, `ArrayLike`)

### 2.5 Dataclasses for Model Parameters

- Use `dataclass` for parameter containers and configuration objects
- Include type annotations for all fields
- Set appropriate default values
- Implement validation logic with `__post_init__`

```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class GARCHParameters:
    p: int = 1
    q: int = 1
    mean_model: Literal["constant", "zero", "arx"] = "constant"
    distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal"
    degrees_of_freedom: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.p < 0 or self.q < 0:
            raise ValueError("GARCH orders must be non-negative")
        
        if self.distribution in ["t", "skewed-t"] and self.degrees_of_freedom is None:
            raise ValueError(f"Distribution '{self.distribution}' requires degrees_of_freedom parameter")
```

### 2.6 Async/Await Patterns

- Use asynchronous programming for long-running operations
- Implement non-blocking operations for GUI responsiveness
- Provide progress updates during lengthy calculations
- Support cancellation of operations when appropriate

```python
import asyncio
from typing import AsyncIterator, Dict, Any

async def estimate_model_async(data: np.ndarray, 
                              max_iterations: int = 1000) -> Dict[str, Any]:
    """
    Asynchronously estimate model parameters with progress updates.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    max_iterations : int, optional
        Maximum number of iterations, default 1000
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of estimated parameters
    """
    result = {}
    
    for i in range(max_iterations):
        # Perform estimation iteration
        
        # Yield control periodically to prevent blocking
        if i % 10 == 0:
            await asyncio.sleep(0)
            
    return result

async def estimate_with_progress(data: np.ndarray) -> AsyncIterator[float]:
    """
    Estimate model with progress updates.
    
    Yields
    ------
    float
        Progress percentage (0-100)
    """
    total_steps = 100
    
    for step in range(total_steps):
        # Perform computation step
        
        # Report progress and yield control
        progress = (step + 1) / total_steps * 100
        yield progress
        await asyncio.sleep(0)
```

### 2.7 Imports Organization

Organize imports in the following order:

1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

Separate each import group with a blank line:

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from numba import jit

# Local application imports
from mfe.core import distributions
from mfe.models import volatility
from .utils import validation
```

### 2.8 Error Handling

- Use specific exception types rather than generic exceptions
- Provide informative error messages
- Use context managers (`with` statements) for resource management
- Handle errors at the appropriate level of abstraction

```python
def validate_returns(returns: np.ndarray) -> None:
    """Validate return data for model estimation."""
    if not isinstance(returns, np.ndarray):
        raise TypeError("Returns must be a NumPy array")
        
    if returns.ndim != 1:
        raise ValueError(f"Returns must be 1-dimensional, got {returns.ndim} dimensions")
        
    if np.isnan(returns).any():
        raise ValueError("Returns contain NaN values")
        
    if np.isinf(returns).any():
        raise ValueError("Returns contain infinite values")
```

## 3. Scientific Computing Conventions

### 3.1 NumPy/SciPy Style

- Follow NumPy/SciPy documentation style for scientific computing code
- Use NumPy-style docstrings for functions and classes
- Prefer NumPy functions over built-in Python functions for numerical operations
- Use SciPy's specialized functions where appropriate

### 3.2 Array Operations

- Use vectorized operations instead of loops where possible
- Leverage NumPy's efficient array operations for large datasets
- Avoid unnecessary array copies
- Use appropriate NumPy data types for memory efficiency

```python
# Inefficient approach with loops
def calculate_returns_slow(prices):
    T = len(prices)
    returns = np.zeros(T-1)
    for t in range(1, T):
        returns[t-1] = (prices[t] / prices[t-1]) - 1
    return returns

# Efficient approach with vectorized operations
def calculate_returns_fast(prices):
    return prices[1:] / prices[:-1] - 1
```

### 3.3 Matrix Operations

- Use appropriate NumPy functions for matrix operations
- Be explicit about matrix dimensions and broadcasting
- Consider memory layout (row vs. column-major) for performance
- Validate matrix dimensions before operations

```python
def covariance_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Calculate covariance matrix of asset returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns with shape (n_samples, n_assets)
        
    Returns
    -------
    np.ndarray
        Covariance matrix with shape (n_assets, n_assets)
    """
    # Validate dimensions
    if returns.ndim != 2:
        raise ValueError(f"Returns must be 2-dimensional, got {returns.ndim} dimensions")
    
    # Center the data
    centered = returns - np.mean(returns, axis=0)
    
    # Compute covariance matrix
    n_samples = returns.shape[0]
    cov_matrix = np.dot(centered.T, centered) / (n_samples - 1)
    
    return cov_matrix
```

### 3.4 Numerical Stability

- Use stable algorithms for numerical computations
- Be aware of floating-point precision issues
- Implement checks for numerical stability
- Use logarithmic transformations for products of small/large numbers

```python
def log_likelihood(params: np.ndarray, data: np.ndarray) -> float:
    """
    Compute log-likelihood with numerical stability.
    
    Parameters
    ----------
    params : np.ndarray
        Model parameters
    data : np.ndarray
        Input data
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Compute log-likelihood in a numerically stable way
    # by working in log space to avoid numerical underflow
    
    log_likes = -0.5 * (np.log(2 * np.pi) + 
                        np.log(params[0]) + 
                        (data - params[1])**2 / params[0])
    
    return np.sum(log_likes)
```

## 4. Documentation Standards

### 4.1 Docstrings

- Use NumPy/SciPy-style docstrings for functions and classes
- Document parameters, return values, and exceptions
- Include examples where appropriate
- Add cross-references to related functions

```python
def calculate_volatility_forecast(
    returns: np.ndarray, 
    model_params: dict, 
    horizon: int = 10
) -> np.ndarray:
    """
    Calculate volatility forecasts using the specified model parameters.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical return series, should be a 1D array
    model_params : dict
        Dictionary containing model parameters:
        - omega : float
            GARCH constant term
        - alpha : float or array_like
            ARCH parameters
        - beta : float or array_like
            GARCH parameters
    horizon : int, optional
        Forecast horizon, default 10
        
    Returns
    -------
    np.ndarray
        Array of volatility forecasts with shape (horizon,)
        
    Raises
    ------
    ValueError
        If returns is empty or contains invalid values
    TypeError
        If model_params is missing required parameters
        
    Notes
    -----
    This function implements multi-step ahead forecasting for GARCH models
    using the standard recursion formula.
    
    See Also
    --------
    estimate_garch : Function to estimate GARCH parameters
    simulate_garch : Function to simulate GARCH processes
    
    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.normal(0, 1, size=1000)
    >>> params = {'omega': 0.01, 'alpha': [0.1], 'beta': [0.8]}
    >>> forecasts = calculate_volatility_forecast(returns, params, horizon=5)
    >>> print(forecasts.shape)
    (5,)
    """
    # Implementation here
```

### 4.2 Code Comments

- Add comments for complex logic
- Explain non-obvious algorithm steps
- Document algorithm references
- Keep comments current with code changes

```python
def estimate_realized_volatility(prices: np.ndarray, 
                                times: np.ndarray, 
                                kernel: str = 'bartlett', 
                                bandwidth: Optional[float] = None) -> float:
    """Estimate realized volatility with kernel-based noise correction."""
    # Implementation based on:
    # Hansen, P. R., & Lunde, A. (2006). Realized Variance and Market 
    # Microstructure Noise. Journal of Business & Economic Statistics, 24(2), 127-161.
    
    # Calculate log-returns
    returns = np.diff(np.log(prices))
    
    # Determine optimal bandwidth if not specified
    if bandwidth is None:
        # Use data-driven bandwidth selection based on Barndorff-Nielsen et al. (2009)
        # Simplified approximation: bandwidth = 4 * (n/100)^0.8
        n = len(returns)
        bandwidth = 4 * ((n / 100) ** 0.8)
    
    # Apply kernel weights based on selected kernel function
    # ...code continues...
```

### 4.3 Module Documentation

- Add module-level docstrings
- Document module purpose, dependencies, and usage
- Include references to relevant literature
- Provide examples of module usage

```python
"""
mfe.models.volatility
=====================

This module implements GARCH-type volatility models for financial time series.
It provides functions for parameter estimation, volatility forecasting, 
simulation, and diagnostic testing.

Models Implemented
-----------------
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- EGARCH (Exponential GARCH)
- GJR-GARCH (Glosten-Jagannathan-Runkle GARCH)
- APARCH (Asymmetric Power ARCH)
- FIGARCH (Fractionally Integrated GARCH)

The implementations use Numba for performance optimization where appropriate.

Examples
--------
>>> from mfe.models.volatility import GARCHModel
>>> import numpy as np
>>> returns = np.random.normal(0, 1, size=1000)
>>> model = GARCHModel(p=1, q=1)
>>> fitted_model = model.fit(returns)
>>> forecasts = fitted_model.forecast(horizon=10)

References
----------
Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
Journal of Econometrics, 31(3), 307-327.

Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: 
A new approach. Econometrica, 59(2), 347-370.

Dependencies
-----------
- numpy (1.26.3+)
- scipy (1.11.4+)
- numba (0.59.0+)
- pandas (2.1.4+) for handling time series data
"""

# Module imports and implementations follow
```

### 4.4 Public API Definition

- Clearly define the public API for each module
- Use `__all__` to specify exported symbols
- Document which functions/classes are intended for public use
- Prefix internal/private functions with a single underscore

```python
"""
mfe.models.garch
===============

GARCH model implementations for volatility modeling.
"""

from typing import Dict, Optional, Union

import numpy as np
from numba import jit

from ..core import distributions

__all__ = ['GARCH', 'EGARCH', 'GJR_GARCH', 'simulate_garch']

# Public classes and functions

class GARCH:
    """GARCH model implementation."""
    # Implementation...

class EGARCH:
    """EGARCH model implementation."""
    # Implementation...

class GJR_GARCH:
    """GJR-GARCH model implementation."""
    # Implementation...

def simulate_garch(n: int, omega: float, alpha: list, beta: list) -> np.ndarray:
    """Simulate a GARCH process."""
    # Implementation...

# Internal helper functions

def _validate_garch_parameters(p: int, q: int, params: np.ndarray) -> bool:
    """Internal function to validate GARCH parameters."""
    # Implementation...
```

## 5. Testing Guidelines

### 5.1 Unit Testing

- Use pytest for unit testing
- Test each function and class independently
- Include edge cases and error conditions
- Maintain high code coverage (minimum 90%)

```python
import pytest
import numpy as np
from mfe.models.garch import GARCH

def test_garch_initialization():
    """Test GARCH model initialization with various parameters."""
    # Test default initialization
    model = GARCH(p=1, q=1)
    assert model.p == 1
    assert model.q == 1
    assert not model.fitted
    
    # Test with different orders
    model = GARCH(p=2, q=3)
    assert model.p == 2
    assert model.q == 3

def test_garch_invalid_parameters():
    """Test GARCH model validation of invalid parameters."""
    # Test negative order parameters
    with pytest.raises(ValueError, match="GARCH orders must be non-negative"):
        GARCH(p=-1, q=1)
        
    with pytest.raises(ValueError, match="GARCH orders must be non-negative"):
        GARCH(p=1, q=-1)

def test_garch_fit():
    """Test GARCH model fitting on simulated data."""
    # Generate simulated returns
    np.random.seed(42)
    returns = np.random.normal(0, 1, size=1000)
    
    # Create and fit model
    model = GARCH(p=1, q=1)
    fitted_model = model.fit(returns)
    
    # Check results
    assert fitted_model is model  # Should return self
    assert model.fitted
    assert model.parameters is not None
    assert len(model.parameters) == 3  # omega, alpha, beta
```

### 5.2 Property-Based Testing

- Use hypothesis for property-based testing
- Verify statistical properties of models
- Test with varied inputs
- Implement robust test strategies

```python
import pytest
from hypothesis import given, strategies as st
import numpy as np
from mfe.core.distributions import normal_pdf, t_pdf

@given(st.floats(min_value=-10, max_value=10),
       st.floats(min_value=0.1, max_value=5))
def test_normal_pdf_properties(mean, std):
    """Test properties of normal PDF implementation."""
    # Generate points to evaluate the PDF
    x = np.linspace(mean - 5*std, mean + 5*std, 1000)
    pdf_values = normal_pdf(x, mean, std)
    
    # PDF should be non-negative
    assert np.all(pdf_values >= 0)
    
    # PDF should integrate to approximately 1
    integral = np.trapz(pdf_values, x)
    assert np.isclose(integral, 1.0, rtol=1e-2)
    
    # Maximum should be at the mean
    max_idx = np.argmax(pdf_values)
    assert np.isclose(x[max_idx], mean, atol=0.1)
```

### 5.3 Performance Testing

- Implement benchmarks for critical functions
- Measure memory usage and execution time
- Compare against baseline implementations
- Verify optimization effectiveness

```python
import pytest
import numpy as np
import time
from mfe.models.realized import realized_variance, realized_variance_numba

@pytest.mark.benchmark
def test_realized_variance_performance():
    """Benchmark performance of standard vs. Numba-optimized implementation."""
    # Generate test data
    np.random.seed(12345)
    n = 10000
    prices = np.cumsum(np.random.normal(0, 0.01, size=n)) + 100
    times = np.linspace(0, 86400, n)  # One day in seconds
    
    # Measure execution time of non-optimized version
    start_time = time.time()
    result_py = realized_variance(prices, times, "seconds", "CalendarTime", 300)
    py_execution_time = time.time() - start_time
    
    # Measure execution time of Numba-optimized version
    start_time = time.time()
    result_numba = realized_variance_numba(prices, times, "seconds", "CalendarTime", 300)
    numba_execution_time = time.time() - start_time
    
    # Assert correctness (results should match)
    assert np.isclose(result_py, result_numba, rtol=1e-10)
    
    # Assert performance improvement
    assert numba_execution_time < py_execution_time
    speedup = py_execution_time / numba_execution_time
    print(f"Numba speedup: {speedup:.2f}x")
    
    # We expect significant speedup with Numba
    assert speedup > 5  # Expect at least 5x speedup
```

### 5.4 Async Testing

- Test asynchronous functions using pytest-asyncio
- Verify progress reporting and cancellation
- Test non-blocking behavior
- Ensure proper error handling in async context

```python
import pytest
import asyncio
import numpy as np
from mfe.models.garch import estimate_garch_async

@pytest.mark.asyncio
async def test_async_estimation():
    """Test asynchronous model estimation."""
    # Generate test data
    np.random.seed(42)
    returns = np.random.normal(0, 1, size=1000)
    
    # Start asynchronous estimation
    progress_values = []
    async for progress, _ in estimate_garch_async(returns, p=1, q=1):
        progress_values.append(progress)
    
    # Verify progress reporting
    assert len(progress_values) > 0
    assert progress_values[0] < progress_values[-1]
    assert 0 <= progress_values[0] <= 100
    assert progress_values[-1] == 100.0

@pytest.mark.asyncio
async def test_async_cancellation():
    """Test cancellation of asynchronous operations."""
    # Generate test data
    np.random.seed(42)
    returns = np.random.normal(0, 1, size=10000)
    
    # Create a task
    task = asyncio.create_task(estimate_garch_async(returns, p=2, q=2))
    
    # Wait briefly to let it start, then cancel
    await asyncio.sleep(0.1)
    task.cancel()
    
    # Verify cancellation is handled properly
    with pytest.raises(asyncio.CancelledError):
        await task
```

## 6. Performance Optimization Patterns

### 6.1 Numba JIT Optimization

- Use `@jit` decorator for performance-critical numerical functions
- Prefer `nopython=True` mode for maximum performance
- Ensure type stability within JIT-compiled functions
- Avoid Python objects inside JIT-compiled code

```python
import numpy as np
from numba import jit

@jit(nopython=True)
def calculate_garch_likelihood(params: np.ndarray, 
                              returns: np.ndarray, 
                              sigma2: np.ndarray) -> float:
    """
    GARCH likelihood calculation optimized with Numba.
    
    Parameters
    ----------
    params : np.ndarray
        Model parameters [omega, alpha, beta]
    returns : np.ndarray
        Return data
    sigma2 : np.ndarray
        Conditional variances (output array)
        
    Returns
    -------
    float
        Negative log-likelihood value
    """
    T = returns.shape[0]
    omega, alpha, beta = params[0], params[1], params[2]
    
    # Initialize first variance with unconditional variance
    sigma2[0] = np.var(returns)
    
    # Recursively compute conditional variances
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Compute log-likelihood
    llh = 0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
    
    return llh
```

### 6.2 When to Use Numba

- Use Numba for:
  - Computationally intensive numerical algorithms
  - Inner loops that cannot be vectorized
  - Performance-critical mathematical operations
  - Functions called repeatedly in model estimation

- Don't use Numba for:
  - I/O operations
  - Functions with complex Python objects
  - Code with many dependencies on Python libraries
  - Simple vectorized operations (already efficient in NumPy)

### 6.3 NumPy Vectorized Operations

- Use vectorized operations instead of loops where possible
- Leverage NumPy's efficient array operations for large datasets
- Avoid unnecessary array copies
- Use appropriate NumPy data types for memory efficiency

```python
import numpy as np

# Inefficient approach with loops
def calculate_returns_slow(prices: np.ndarray) -> np.ndarray:
    """Calculate returns using loops (inefficient)."""
    T = len(prices)
    returns = np.zeros(T-1)
    for t in range(1, T):
        returns[t-1] = (prices[t] / prices[t-1]) - 1
    return returns

# Efficient approach with vectorized operations
def calculate_returns_fast(prices: np.ndarray) -> np.ndarray:
    """Calculate returns using vectorized operations (efficient)."""
    return prices[1:] / prices[:-1] - 1
```

### 6.4 Memory Efficiency

- Use appropriate data types (e.g., float32 vs. float64) based on precision needs
- Implement generators for large data processing
- Release memory explicitly when working with large datasets
- Use memory profiling tools to identify bottlenecks

```python
import numpy as np
from typing import Generator, Tuple

def process_large_array(data: np.ndarray, 
                       chunk_size: int = 10000) -> np.ndarray:
    """
    Process a large array in chunks to minimize memory usage.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    chunk_size : int, optional
        Size of each processing chunk, default 10000
        
    Returns
    -------
    np.ndarray
        Processed data
    """
    # Pre-allocate output array with proper type
    result = np.empty_like(data)
    
    # Process in chunks to reduce memory usage
    for i in range(0, len(data), chunk_size):
        end_idx = min(i + chunk_size, len(data))
        chunk = data[i:end_idx]
        
        # Process the chunk (example operation)
        processed_chunk = np.sqrt(np.abs(chunk))
        
        # Store result
        result[i:end_idx] = processed_chunk
        
        # Explicitly delete to help garbage collection
        del chunk, processed_chunk
        
    return result

def generate_batches(data: np.ndarray, batch_size: int) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generate batches from a large array with indexes.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    batch_size : int
        Size of each batch
        
    Yields
    ------
    Tuple[int, np.ndarray]
        Batch index and data batch
    """
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(data))
        yield i, data[start_idx:end_idx]
```

### 6.5 Asynchronous Processing

- Use asyncio for I/O-bound operations
- Implement concurrent processing for CPU-bound tasks with appropriate tools
- Balance between concurrency and overhead
- Provide cancellation mechanisms for long-running operations

```python
import asyncio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable, Any, TypeVar

T = TypeVar('T')
R = TypeVar('R')

async def process_data_async(data_chunks: List[T], 
                           process_func: Callable[[T], R]) -> List[R]:
    """
    Process data chunks asynchronously.
    
    Parameters
    ----------
    data_chunks : List[T]
        List of data chunks to process
    process_func : Callable[[T], R]
        Processing function to apply to each chunk
        
    Returns
    -------
    List[R]
        List of processed results
    """
    loop = asyncio.get_running_loop()
    
    with ProcessPoolExecutor() as executor:
        # Schedule tasks
        tasks = [
            loop.run_in_executor(executor, process_func, chunk)
            for chunk in data_chunks
        ]
        
        # Gather results
        results = await asyncio.gather(*tasks)
        
    return results
```

### 6.6 Parallel Numba Implementation

- Use `@jit(parallel=True)` for embarrassingly parallel operations
- Replace sequential loops with `prange` for automatic parallelization
- Be aware of thread safety concerns

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def parallel_monte_carlo(n_simulations: int, n_steps: int) -> np.ndarray:
    """
    Parallel Monte Carlo simulation of geometric Brownian motion.
    
    Parameters
    ----------
    n_simulations : int
        Number of price paths to simulate
    n_steps : int
        Number of time steps per path
        
    Returns
    -------
    np.ndarray
        Array of simulated price paths
    """
    # Parameters for GBM
    mu = 0.05
    sigma = 0.2
    dt = 1.0 / 252  # Daily steps
    
    # Preallocate output array
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = 100.0  # Starting price
    
    # Parallel simulation
    for i in prange(n_simulations):
        for t in range(1, n_steps + 1):
            z = np.random.normal(0, 1)
            paths[i, t] = paths[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + 
                                               sigma * np.sqrt(dt) * z)
    
    return paths
```

## 7. Package Organization

### 7.1 Directory Structure

Follow the MFE package organization structure:

```
mfe/
├── __init__.py             # Package initialization with version
├── core/                   # Core statistical modules
│   ├── __init__.py
│   ├── bootstrap.py        # Bootstrap resampling methods
│   ├── distributions.py    # Statistical distributions
│   ├── optimization.py     # Numba-optimized routines
│   └── tests.py            # Statistical tests
├── models/                 # Time series & volatility models
│   ├── __init__.py
│   ├── garch.py            # GARCH model variants
│   ├── realized.py         # Realized volatility measures
│   ├── timeseries.py       # ARMA/ARMAX models
│   └── volatility.py       # Additional volatility models
├── ui/                     # User interface components
│   ├── __init__.py
│   ├── widgets.py          # PyQt6 widgets
│   └── armax_viewer.py     # ARMAX model viewer
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── validation.py       # Input validation
│   └── printing.py         # Formatted output utilities
├── pyproject.toml          # Build configuration
├── setup.py                # Legacy build support
└── README.md               # Documentation
```

### 7.2 Import Conventions

- Follow consistent import patterns
- Use relative imports within the package
- Define clear public APIs with `__all__`
- Keep imports organized and structured

```python
# Standard library imports first
import os
import sys
from typing import Dict, List, Optional, Union, Tuple, Any

# Third-party library imports next
import numpy as np
import pandas as pd
from scipy import stats, optimize
from numba import jit, prange

# Relative imports last
from ..core import distributions
from ..utils import validation
from . import timeseries

# Define public API
__all__ = ['GARCHModel', 'EGARCHModel', 'APARCHModel']
```

### 7.3 Version Management

- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Document version changes in a CHANGELOG
- Include version in `__init__.py`
- Tag releases in version control

```python
"""
MFE Toolbox - Python Financial Econometrics
===========================================

A comprehensive suite of Python modules for modeling financial time series
and conducting advanced econometric analyses.
"""

__version__ = '4.0.0'
__author__ = 'Kevin Sheppard'
```

## 8. Development Environment

### 8.1 Recommended Tools

- **IDE/Editor**: 
  - Visual Studio Code with Python extension
  - PyCharm with scientific mode enabled
  
- **Development Tools**:
  - **Linting & Formatting**: 
    - mypy for static type checking
    - black for code formatting
    - flake8 for linting
    - isort for import sorting
  
  - **Testing**: 
    - pytest for unit and integration testing
    - pytest-asyncio for async tests
    - pytest-cov for coverage reports
    - pytest-benchmark for performance testing
    - hypothesis for property-based testing

  - **Documentation**:
    - Sphinx for API documentation
    - numpydoc for NumPy-style docstrings

### 8.2 Environment Setup

- Use virtual environments for isolated development:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```

- Install development dependencies:
  ```bash
  pip install -e ".[dev]"
  ```

- Set up pre-commit hooks for consistent code quality:
  ```bash
  pre-commit install
  ```

### 8.3 Dependency Management

- Define dependencies in pyproject.toml with version constraints:
  ```toml
  [project]
  dependencies = [
      "numpy>=1.26.3",
      "scipy>=1.11.4",
      "pandas>=2.1.4",
      "statsmodels>=0.14.1",
      "numba>=0.59.0",
      "pyqt6>=6.6.1",
  ]
  
  [project.optional-dependencies]
  dev = [
      "pytest>=7.4.3",
      "pytest-asyncio>=0.21.1",
      "pytest-cov>=4.1.0",
      "pytest-benchmark>=4.0.0",
      "hypothesis>=6.92.1",
      "mypy>=1.7.1",
      "black>=23.11.0",
      "flake8>=6.1.0",
      "isort>=5.12.0",
      "pre-commit>=3.5.0",
      "sphinx>=7.2.6",
      "numpydoc>=1.6.0",
  ]
  ```

## 9. Conclusion

This Style Guide defines the standards and conventions to be followed in the MFE Toolbox Python implementation. Adhering to these guidelines ensures:

1. Code consistency and maintainability
2. Proper leverage of Python 3.12 features
3. Effective use of scientific computing libraries
4. Robust documentation and testing
5. Optimized performance through appropriate patterns

Developers should consult this guide when implementing new components or modifying existing code. The guidelines are meant to promote high-quality, maintainable, and efficient code that meets the project's technical requirements.

## 10. Resources

### 10.1 Python References

- [Python 3.12 Documentation](https://docs.python.org/3.12/)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Numba Documentation](https://numba.pydata.org/numba-doc/latest/index.html)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)

### 10.2 Econometrics References

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007.
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. Econometrica, 59(2), 347-370.
- Hansen, P. R., & Lunde, A. (2006). Realized variance and market microstructure noise. Journal of Business & Economic Statistics, 24(2), 127-161.