# MFE Toolbox API Documentation Template

## Overview

This document provides a comprehensive template for documenting the MFE Toolbox API (version 4.0). The MFE Toolbox is a Python-based suite of financial econometric tools for time series analysis, volatility modeling, and high-frequency analytics.

## Documentation Structure

Each component of the API should be documented following these templates, organized into the following main categories:

1. **Model Classes**: Time series, volatility, and high-frequency models
2. **Estimation Functions**: Parameter estimation, forecasting, and diagnostic routines
3. **Utility Functions**: Optimization, validation, and statistical utilities

## Model Class Documentation

```markdown
# {ModelName}

Class documentation for {ModelName}.

## Description

{Comprehensive description of the model's purpose, theoretical background, and capabilities}

## Parameters

- **{param_name}** (*{param_type}*): {Description}
  - Default: {default_value}
  - Constraints: {Any constraints on parameter values}

## Attributes

- **{attribute_name}** (*{attribute_type}*): {Description}

## Methods

### {method_name}

{Method description}

#### Parameters

- **{param_name}** (*{param_type}*): {Description}

#### Returns

- **{return_name}** (*{return_type}*): {Description}

#### Examples

```python
# Example usage of this method
```
```

## Asynchronous Method Documentation

```markdown
# async {method_name}

Asynchronous method documentation for {method_name}.

## Description

{Comprehensive description of the method's purpose and behavior}

## Parameters

- **{param_name}** (*{param_type}*): {Description}

## Returns

- **{return_name}** (*{return_type}*): {Description}

## Notes

This method is designed to be called with `await` in an asynchronous context.
Use in a synchronous context will require running an event loop.

## Examples

```python
# Asynchronous usage
async def example():
    result = await model.{method_name}(...)
    
# Synchronous usage with event loop
import asyncio
result = asyncio.run(model.{method_name}(...))
```
```

## Numba-Optimized Function Documentation

```markdown
# @numba.jit {function_name}

Documentation for Numba-optimized function {function_name}.

## Description

{Comprehensive description of the function's purpose and behavior}

## Parameters

- **{param_name}** (*{param_type}*): {Description}

## Returns

- **{return_name}** (*{return_type}*): {Description}

## Performance Notes

This function is optimized using Numba's JIT compilation with the following settings:
- nopython: {True/False} - {Explanation}
- parallel: {True/False} - {Explanation}
- cache: {True/False} - {Explanation}

The implementation achieves performance optimization through:
- {Specific optimization technique}
- {Memory access pattern}
- {Algorithm optimization}

## Examples

```python
# Example usage
```
```

## Utility Function Documentation

```markdown
# {function_name}

Documentation for utility function {function_name}.

## Description

{Comprehensive description of the function's purpose and behavior}

## Parameters

- **{param_name}** (*{param_type}*): {Description}

## Returns

- **{return_name}** (*{return_type}*): {Description}

## Raises

- **{exception_name}**: {Description of when this exception is raised}

## Examples

```python
# Example usage
```
```

## Documentation Examples

Below are examples of how to document specific components using these templates:

### Example: ARMAX Model Documentation

```markdown
# ARMAX

Class documentation for ARMAX (AutoRegressive Moving Average with eXogenous variables) model.

## Description

The ARMAX model implements a comprehensive time series modeling framework with parameter estimation, diagnostics, and forecasting capabilities. It supports both ARMA and ARMAX specifications with robust optimization and asynchronous estimation.

## Parameters

- **p** (*int*): Autoregressive order
  - Constraints: 0 ≤ p ≤ 30
- **q** (*int*): Moving average order
  - Constraints: 0 ≤ q ≤ 30
- **include_constant** (*bool*): Whether to include a constant term in the model
  - Default: True

## Attributes

- **params** (*ndarray*): Estimated model parameters
- **residuals** (*ndarray*): Model residuals
- **loglikelihood** (*float*): Log-likelihood value of the fitted model
- **standard_errors** (*ndarray*): Standard errors of parameter estimates

## Methods

### async_fit

Asynchronously estimate model parameters using maximum likelihood.

#### Parameters

- **data** (*ndarray*): Time series data for model estimation
- **exog** (*ndarray, optional*): Exogenous variables for ARMAX model, if any

#### Returns

- **bool**: True if estimation converged, False otherwise

#### Examples

```python
import asyncio
import numpy as np
from mfe.models import ARMAX

# Create synthetic data
np.random.seed(42)
data = np.random.normal(0, 1, 200)

# Initialize model
model = ARMAX(p=1, q=1)

# Estimate parameters asynchronously
async def estimate_model():
    success = await model.async_fit(data)
    if success:
        print("Model estimation converged!")
        print(f"Parameters: {model.params}")
    else:
        print("Model estimation did not converge")

# Run the async function
asyncio.run(estimate_model())
```
```

### Example: Numba-Optimized Function Documentation

```markdown
# @numba.jit compute_residuals

Documentation for Numba-optimized function compute_residuals.

## Description

Computes model residuals from fitted ARMAX model parameters using efficient Numba-optimized implementation.

## Parameters

- **data** (*np.ndarray*): Time series data array
- **params** (*np.ndarray*): Model parameters [ar_params, ma_params, constant, exog_params]
- **exog** (*np.ndarray, optional*): Exogenous variables, if any

## Returns

- **np.ndarray**: Model residuals

## Performance Notes

This function is optimized using Numba's JIT compilation with the following settings:
- nopython: True - Ensures maximum performance by compiling without Python API calls
- parallel: False - Serial implementation appropriate for the sequential nature of ARMA filtering
- cache: False - Default caching behavior

The implementation achieves performance optimization through:
- Contiguous array access for maximum memory efficiency
- Preallocated output arrays to minimize memory allocations
- Loop-based implementation avoiding unnecessary function calls

## Examples

```python
import numpy as np
from mfe.models.armax import compute_residuals

# Example AR(1) model with parameters [0.5]
data = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
params = np.array([1, 0, 0, 0.5])  # [p=1, q=0, no_constant=0, ar_param=0.5]

# Compute residuals
residuals = compute_residuals(data, params)
print(residuals)
```
```

### Example: Optimizer Documentation

```markdown
# Optimizer

Class documentation for asynchronous optimization manager.

## Description

The Optimizer class provides a high-level interface for model parameter optimization using numerical methods. It supports asynchronous execution through Python's async/await pattern, allowing for responsive user interfaces during long-running optimizations.

## Parameters

- **options** (*dict, optional*): Configuration options for the optimizer

## Attributes

- **optimization_options** (*dict*): Dictionary of options configuring the optimization process
- **converged** (*bool*): Flag indicating whether the last optimization converged successfully

## Methods

### async_optimize

Asynchronously optimize model parameters using Numba-accelerated routines.

#### Parameters

- **data** (*np.ndarray*): Time series data for model estimation
- **initial_params** (*np.ndarray*): Initial parameter values to start optimization
- **model_type** (*str*): Type of model ('GARCH', 'EGARCH', etc.)
- **distribution** (*str*): Error distribution specification ('normal', 'student-t', etc.)

#### Returns

- **Tuple[np.ndarray, float]**: Tuple containing optimized parameter values and maximized log-likelihood value

#### Examples

```python
import asyncio
import numpy as np
from mfe.core.optimization import Optimizer

# Create synthetic return data
np.random.seed(42)
returns = np.random.normal(0, 1, 1000)

# Initialize optimizer
optimizer = Optimizer()

# Initial GARCH parameters [omega, alpha, beta]
initial_params = np.array([0.01, 0.1, 0.8])

# Optimize parameters asynchronously
async def optimize_garch():
    params, likelihood = await optimizer.async_optimize(
        returns, 
        initial_params,
        'GARCH',
        'normal'
    )
    print(f"Optimized parameters: {params}")
    print(f"Log-likelihood: {likelihood}")

# Run the async function
asyncio.run(optimize_garch())
```
```