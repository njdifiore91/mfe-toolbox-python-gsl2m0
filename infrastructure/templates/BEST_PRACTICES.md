# MFE Toolbox Best Practices Guide

Version: 1.0.0

## 1. Introduction

This document outlines the best practices and coding standards for the MFE (Financial Econometrics) Toolbox Python implementation. Following these guidelines ensures code consistency, maintainability, performance optimization, and adherence to modern Python development principles.

The MFE Toolbox has been reimplemented using Python 3.12, incorporating modern programming constructs while leveraging Python's scientific computing ecosystem. This guide will help developers adhere to the architectural vision and technical requirements of the project.

## 2. Python Coding Standards

### 2.1 Python Version

- The MFE Toolbox requires **Python 3.12** or later
- Leverage Python 3.12's specific features:
  - Enhanced error messages with precise locations
  - Improved typing features
  - Performance optimizations for core types
  - Advanced pattern matching capabilities

### 2.2 Type Annotations and Hints

- Use strict type hints for all function parameters and return values
- Add type information to class attributes using annotations or dataclasses
- Employ the `typing` module for complex types (e.g., `List`, `Dict`, `Optional`, `Union`)
- Ensure type annotations are validated with mypy during development

```python
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import pandas as pd

def calculate_volatility(returns: np.ndarray, 
                        window: int = 20, 
                        annualize: bool = True) -> np.ndarray:
    """
    Calculate rolling volatility of returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns
    window : int, optional
        Rolling window size, default 20
    annualize : bool, optional
        Whether to annualize the result, default True
        
    Returns
    -------
    np.ndarray
        Rolling volatility estimates
    """
    if not isinstance(returns, np.ndarray):
        raise TypeError("Returns must be a NumPy array")
    
    if window < 1:
        raise ValueError("Window size must be positive")
    
    # Calculate rolling standard deviation
    vol = np.zeros_like(returns)
    for i in range(window, len(returns) + 1):
        vol[i-1] = np.std(returns[i-window:i], ddof=1)
        
    # Annualize if requested
    if annualize:
        vol *= np.sqrt(252)  # Trading days in a year
        
    return vol
```

### 2.3 Dataclasses for Model Parameters

- Use `dataclass` for parameter containers and configuration objects
- Include type annotations for all fields
- Set appropriate default values
- Implement validation logic with `__post_init__`

```python
from dataclasses import dataclass, field
from typing import Optional, List, Literal

@dataclass
class GARCHParameters:
    p: int = 1
    q: int = 1
    mean_model: Literal["constant", "zero", "arx"] = "constant"
    scale: float = 1.0
    distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal"
    degrees_of_freedom: Optional[float] = None
    error_terms: Optional[List[float]] = None
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.p < 0 or self.q < 0:
            raise ValueError("GARCH orders must be non-negative")
        
        if self.distribution in ["t", "skewed-t"] and self.degrees_of_freedom is None:
            raise ValueError(f"Distribution '{self.distribution}' requires degrees_of_freedom parameter")
            
        if self.scale <= 0:
            raise ValueError("Scale parameter must be positive")
```

### 2.4 Async/Await Patterns

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
    likelihood = float('-inf')
    
    for i in range(max_iterations):
        # Perform estimation iteration (simplified example)
        new_likelihood = -np.sum(data ** 2) / len(data) - i * 0.001
        
        # Check for convergence
        if abs(new_likelihood - likelihood) < 1e-6:
            result["converged"] = True
            break
            
        likelihood = new_likelihood
        result["likelihood"] = likelihood
        result["iteration"] = i
        
        # Yield control periodically to prevent blocking
        if i % 10 == 0:
            await asyncio.sleep(0)
            
    return result

async def estimate_with_progress(data: np.ndarray) -> AsyncIterator[float]:
    """
    Estimate model with progress updates.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
        
    Yields
    ------
    float
        Progress percentage (0-100)
    """
    total_steps = 100
    
    for step in range(total_steps):
        # Perform computation step
        # (simplified example)
        
        # Report progress and yield control
        progress = (step + 1) / total_steps * 100
        yield progress
        await asyncio.sleep(0)
```

### 2.5 Class-Based Architecture

- Organize code using classes with clear responsibilities
- Follow single responsibility principle
- Use inheritance where appropriate for model hierarchies
- Implement interfaces and abstract base classes for common patterns

```python
from abc import ABC, abstractmethod

class VolatilityModel(ABC):
    """Abstract base class for volatility models."""
    
    @abstractmethod
    def fit(self, returns: np.ndarray) -> 'VolatilityModel':
        """
        Fit the volatility model to return data.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of asset returns
            
        Returns
        -------
        VolatilityModel
            Fitted model instance
        """
        pass
    
    @abstractmethod
    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Generate volatility forecasts.
        
        Parameters
        ----------
        horizon : int, optional
            Forecast horizon, default 1
            
        Returns
        -------
        np.ndarray
            Volatility forecasts
        """
        pass

class GARCHModel(VolatilityModel):
    """GARCH volatility model implementation."""
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.parameters = None
        self.fitted = False
        self._volatility = None
    
    def fit(self, returns: np.ndarray) -> 'GARCHModel':
        # Implementation details
        self.fitted = True
        return self
    
    def forecast(self, horizon: int = 1) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Implementation details
        forecasts = np.zeros(horizon)
        # Forecast calculation logic
        return forecasts
```

### 2.6 Module Structure

- Follow the package layout conventions established for MFE:
  - `mfe.core`: Fundamental statistical and computational components
  - `mfe.models`: Time series and volatility modeling implementations
  - `mfe.ui`: User interface components using PyQt6
  - `mfe.utils`: Utility functions and helper routines

- Each module should:
  - Have a clear, focused purpose
  - Include comprehensive docstrings
  - Export a well-defined public API
  - Handle errors appropriately

```python
"""
mfe.models.garch
================

This module implements GARCH-type volatility models for financial time series.

Models
------
GARCH : Standard GARCH models
EGARCH : Exponential GARCH models
GJR_GARCH : Threshold GARCH models with asymmetry
APARCH : Asymmetric Power ARCH models

References
----------
Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
Journal of Econometrics, 31(3), 307-327.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
from scipy import optimize
from numba import jit

from mfe.core.distributions import t_distribution, ged_distribution
from mfe.utils.validation import validate_array

# Export public API
__all__ = ['GARCH', 'EGARCH', 'GJR_GARCH', 'APARCH']

class GARCH:
    """
    Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model.
    
    Attributes
    ----------
    p : int
        Order of the ARCH component
    q : int
        Order of the GARCH component
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        # Implementation
        pass
```

## 3. Performance Optimization

### 3.1 Numba JIT Optimization

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

### 3.2 NumPy Array Operations

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

# Memory-efficient approach for large datasets
def calculate_rolling_statistics(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling statistics using strided operations."""
    # Use strided operations instead of creating many subarrays
    return np.lib.stride_tricks.sliding_window_view(data, window)
```

### 3.3 Memory Efficiency

- Use appropriate data types (e.g., float32 vs. float64) based on precision needs
- Implement generators for large data processing
- Release memory explicitly when working with large datasets
- Use memory profiling tools to identify bottlenecks

```python
import numpy as np
from typing import Iterator, Tuple, Optional, Generator

def process_large_array(data: np.ndarray, 
                       chunk_size: int = 10000,
                       dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Process a large array in chunks to minimize memory usage.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    chunk_size : int, optional
        Size of each processing chunk, default 10000
    dtype : np.dtype, optional
        Output data type, defaults to input dtype
        
    Returns
    -------
    np.ndarray
        Processed data
    """
    if dtype is None:
        dtype = data.dtype
        
    # Pre-allocate output array with proper type
    result = np.empty_like(data, dtype=dtype)
    
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

### 3.4 Asynchronous Processing

- Use asyncio for I/O-bound operations
- Implement concurrent processing for CPU-bound tasks with appropriate tools
- Balance between concurrency and overhead
- Provide cancellation mechanisms for long-running operations

```python
import asyncio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable, Any, TypeVar, Generic

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

async def estimate_models_async(model_class: type, 
                              datasets: List[np.ndarray], 
                              params: List[dict]) -> List[Any]:
    """
    Estimate multiple models asynchronously.
    
    Parameters
    ----------
    model_class : type
        Model class to instantiate
    datasets : List[np.ndarray]
        List of datasets to fit models to
    params : List[dict]
        Model parameters for each dataset
        
    Returns
    -------
    List[Any]
        List of fitted models
    """
    async def fit_model(model, data):
        return await model.fit_async(data)
    
    # Create model instances
    models = [model_class(**p) for p in params]
    
    # Create tasks for each model
    tasks = [fit_model(model, data) for model, data in zip(models, datasets)]
    
    # Run all tasks concurrently
    return await asyncio.gather(*tasks)
```

## 4. Testing Best Practices

### 4.1 Unit Testing

- Use pytest for comprehensive test coverage
- Test each module and class independently
- Include edge cases and error conditions
- Maintain high code coverage (aim for >90%)

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
    
def test_garch_forecast_without_fit():
    """Test behavior when forecasting without fitting first."""
    model = GARCH(p=1, q=1)
    with pytest.raises(ValueError, match="Model must be fitted before forecasting"):
        model.forecast(horizon=5)
```

### 4.2 Property-Based Testing

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

@given(st.floats(min_value=2.1, max_value=30))
def test_t_distribution_properties(df):
    """Test properties of t-distribution implementation."""
    # Generate points to evaluate the PDF
    x = np.linspace(-5, 5, 1000)
    pdf_values = t_pdf(x, df)
    
    # PDF should be non-negative
    assert np.all(pdf_values >= 0)
    
    # PDF should be symmetric around zero
    center = len(x) // 2
    left_half = pdf_values[:center]
    right_half = pdf_values[-center:][::-1]  # Reverse the right half
    assert np.allclose(left_half, right_half, rtol=1e-10)
    
    # PDF should integrate to approximately 1
    integral = np.trapz(pdf_values, x)
    assert np.isclose(integral, 1.0, rtol=1e-2)
```

### 4.3 Performance Testing

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

### 4.4 Async Testing

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

## 5. Documentation Standards

### 5.1 Docstrings

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
    # Implementation
    pass
```

### 5.2 Code Comments

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
    weights = np.zeros(int(bandwidth))
    if kernel == 'bartlett':
        # Bartlett (triangle) kernel - linear decay
        for h in range(int(bandwidth)):
            weights[h] = 1 - (h / (bandwidth + 1))
    elif kernel == 'flat-top':
        # Flat-top kernel: k(x) = 1 for |x| ≤ 0.5, smoothly declining after
        for h in range(int(bandwidth)):
            x = h / bandwidth
            if x <= 0.5:
                weights[h] = 1
            else:
                weights[h] = 2 - 2*x  # Linear decline
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Compute realized variance (sum of squared returns)
    gamma0 = np.sum(returns**2)
    
    # Add weighted autocovariances to correct for noise
    rv = gamma0
    for h in range(1, len(weights)):
        # Calculate h-th order autocovariance
        gamma_h = np.sum(returns[h:] * returns[:-h])
        
        # Add weighted autocovariance (factor of 2 because we sum both sides)
        rv += 2 * weights[h] * gamma_h
    
    # Normalize by sample size
    return rv / len(returns)
```

### 5.3 Module Documentation

- Add module-level docstrings
- Document module purpose, dependencies, and usage
- Include references to relevant literature
- Provide examples of module usage

```python
"""
Volatility Models for Financial Time Series
===========================================

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

### 5.4 README and User Documentation

- Provide clear installation instructions
- Include getting started examples
- Document API usage
- Add links to relevant resources

```markdown
# MFE Toolbox - Financial Econometrics in Python

A comprehensive suite of Python modules for modeling financial time series and conducting advanced econometric analyses. This toolbox provides researchers, analysts, and practitioners with robust tools for financial modeling, volatility forecasting, and statistical analysis.

## Installation

```bash
pip install mfe
```

## Features

- Time series modeling (ARMA/ARMAX)
- Volatility models (GARCH, EGARCH, GJR-GARCH, etc.)
- High-frequency financial data analysis
- Bootstrap methods for time series
- Advanced statistical distributions
- Cross-sectional econometric analysis

## Quick Start

```python
import numpy as np
from mfe.models import GARCHModel
from mfe.utils import simulate_garch

# Simulate GARCH(1,1) data
np.random.seed(42)
returns = simulate_garch(n=1000, omega=0.05, alpha=[0.1], beta=[0.8])

# Fit model
model = GARCHModel(p=1, q=1)
result = model.fit(returns)

# Display results
print(result.summary())

# Generate forecasts
forecasts = model.forecast(horizon=10)
print("Volatility forecasts:", forecasts)
```

## Documentation

Full documentation is available at [https://mfe.readthedocs.io/](https://mfe.readthedocs.io/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

## 6. Numba Optimization Guidelines

### 6.1 When to Use Numba

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

### 6.2 Numba Optimization Patterns

- Use `nopython=True` mode whenever possible
- Ensure type stability throughout Numba functions
- Prefer NumPy arrays with defined types
- Avoid Python objects and complex data structures

```python
from numba import jit, float64, int64
import numpy as np

# Basic Numba optimization
@jit(nopython=True)
def fast_loop(x: np.ndarray) -> np.ndarray:
    """Simple Numba-optimized array processing."""
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = x[i] * x[i] + 2.0 * x[i] + 1.0
    return result

# More complex with specified signature
@jit(float64[:](float64[:], float64, float64), nopython=True)
def garch_recursion(returns: np.ndarray, 
                   alpha: float, 
                   beta: float) -> np.ndarray:
    """
    Compute GARCH conditional variances with explicit type signature.
    """
    T = returns.shape[0]
    sigma2 = np.zeros(T, dtype=np.float64)
    
    # Initialize with sample variance
    sigma2[0] = np.mean(returns**2)
    
    # GARCH recursion
    for t in range(1, T):
        sigma2[t] = (1 - alpha - beta) * sigma2[0] + \
                   alpha * returns[t-1]**2 + \
                   beta * sigma2[t-1]
    
    return sigma2
```

### 6.3 Parallel Acceleration

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

### 6.4 Numba Testing and Debugging

- Test both pure Python and Numba-optimized versions
- Compare results for correctness
- Measure performance gains precisely
- Provide fallback paths for debugging

```python
import numpy as np
from numba import jit
import time

# Pure Python implementation for testing
def pure_python_function(x, y):
    """Reference implementation in pure Python."""
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = x[i] * y[i] + np.sin(x[i])
    return result

# Numba-optimized version
@jit(nopython=True)
def numba_optimized_function(x, y):
    """Numba-optimized implementation."""
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = x[i] * y[i] + np.sin(x[i])
    return result

# Testing and timing function
def test_numba_performance(size=1000000, debug=False):
    """Test performance of Numba vs pure Python implementation."""
    # Generate test data
    x = np.random.random(size)
    y = np.random.random(size)
    
    # Warm up the JIT compiler
    if not debug:
        _ = numba_optimized_function(x[:100], y[:100])
    
    # Measure pure Python performance
    start = time.time()
    python_result = pure_python_function(x, y)
    python_time = time.time() - start
    
    # Measure Numba performance
    start = time.time()
    if debug:
        numba_result = pure_python_function(x, y)  # Use Python version in debug mode
    else:
        numba_result = numba_optimized_function(x, y)
    numba_time = time.time() - start
    
    # Check correctness
    assert np.allclose(python_result, numba_result)
    
    # Report performance
    speedup = python_time / numba_time
    print(f"Pure Python: {python_time:.4f} seconds")
    print(f"Numba: {numba_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    
    return python_time, numba_time, speedup
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

### 7.3 Versioning

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

- **Code Editors/IDEs**:
  - VS Code with Python extension
  - PyCharm with MFE-specific configuration
  
- **Development Tools**:
  - mypy for static type checking
  - black for code formatting
  - isort for import sorting
  - flake8 for linting
  - pytest for testing

### 8.2 Environment Setup

- Use virtual environments:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```

- Install development dependencies:
  ```bash
  pip install -e ".[dev]"
  ```

- Configure pre-commit hooks:
  ```bash
  pre-commit install
  ```

### 8.3 Documentation Build

- Build documentation using Sphinx:
  ```bash
  cd docs
  make html
  ```

- Preview locally at `docs/_build/html/index.html`

## 9. Continuous Integration

- Run unit tests on pull requests
- Perform type checking with mypy
- Generate coverage reports
- Verify Numba compilation
- Validate documentation building
- Check package installability

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