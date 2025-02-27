# MFE Toolbox Backend

## Overview

The backend of the MFE Toolbox is a comprehensive suite of Python modules designed for modeling financial time series and conducting advanced econometric analyses. Re-implemented in Python 3.12 from its legacy version 4.0, the backend provides high-performance implementations of financial econometric models through Numba optimization.

## Key Features

- **ARMA/ARMAX Modeling**: Time series modeling with exogenous variables and robust forecasting
- **GARCH Volatility Models**: Multiple GARCH variants (GARCH, EGARCH, GJR-GARCH, TARCH, AGARCH, FIGARCH)
- **Statistical Distributions**: Advanced distribution implementations (GED, Hansen's Skewed T)
- **Numba Optimization**: Just-in-time compilation for performance-critical operations
- **Asynchronous Execution**: Modern async/await patterns for efficient computation
- **Type Safety**: Comprehensive type hints throughout the codebase

## Module Structure

The backend is organized into the following main namespaces:

### Core Modules (`mfe.core`)

- **bootstrap**: Robust resampling for dependent time series
- **distributions**: Advanced statistical distributions and testing
- **optimization**: Numba-optimized parameter estimation with JIT compilation

### Model Modules (`mfe.models`)

- **timeseries**: ARMA/ARMAX modeling and forecasting
- **garch**: Comprehensive GARCH volatility model implementations
- **realized**: High-frequency financial data analysis

### Utility Modules (`mfe.utils`)

- **validation**: Input validation and parameter verification utilities
- **printing**: Formatted output and result presentation tools

## Installation

### Requirements

- Python 3.12+
- NumPy 1.26.3+
- SciPy 1.11.4+
- Pandas 2.1.4+
- Statsmodels 0.14.1+
- Numba 0.59.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/username/mfe-toolbox.git

# Change to the project directory
cd mfe-toolbox

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Numba Optimization

The MFE Toolbox backend leverages Numba's just-in-time (JIT) compilation to achieve near-native performance for computationally intensive operations:

### JIT Decorators

Performance-critical functions are decorated with `@numba.jit` to enable automatic compilation to optimized machine code:

```python
@numba.jit(nopython=True)
def compute_garch_likelihood(returns, parameters, model_type_id, distribution_id):
    # Optimized implementation...
    return likelihood
```

### Key Optimized Components

- **Likelihood Calculation**: Fast parameter estimation for ARMAX and GARCH models
- **Distribution Functions**: Efficient PDF calculation for GED and Skewed-T distributions
- **Simulation Routines**: High-performance Monte Carlo simulations
- **Statistical Functions**: Accelerated calculation of ACF, PACF, and other statistics

### Performance Benefits

- **Execution Speed**: Orders of magnitude faster than pure Python implementations
- **Memory Efficiency**: Optimized memory access patterns
- **Hardware Utilization**: Automatic vectorization and parallel execution

## Asynchronous Execution

The MFE Toolbox implements modern async/await patterns for efficient execution of long-running computations:

### Benefits

- **Responsiveness**: Non-blocking execution for UI-integrated applications
- **Progress Monitoring**: Real-time progress tracking during computation
- **Concurrency**: Effective utilization of system resources

### Implementation

Key async methods include:

```python
# Asynchronous model estimation
await model.async_fit(data)

# Asynchronous optimization
await optimizer.async_optimize(data, initial_params, model_type, distribution)
```

## Usage Examples

### Time Series Modeling with ARMAX

```python
import numpy as np
from mfe.models.timeseries import ARMAX

# Generate sample data
np.random.seed(42)
data = np.random.randn(1000)

# Create and fit ARMAX model
model = ARMAX(p=1, q=1, trend='c')
await model.async_fit(data)

# Make forecasts
forecasts = model.forecast(10)
print(forecasts)

# Run diagnostic tests
diagnostics = model.diagnostic_tests()
print(diagnostics)
```

### Volatility Modeling with GARCH

```python
import numpy as np
from mfe.models.garch import GARCHModel

# Generate sample returns
np.random.seed(42)
returns = np.random.randn(1000) * 0.01

# Create and fit GARCH model
model = GARCHModel(p=1, q=1, model_type='GARCH', distribution='normal')
await model.async_fit(returns)

# Forecast volatility
volatility_forecasts = model.forecast(10)
print(volatility_forecasts)

# Simulate returns from the model
simulated_returns, simulated_volatility = model.simulate(100)
```

## API Reference

### Core Modules

#### Optimizer

The `Optimizer` class provides asynchronous optimization for parameter estimation:

```python
from mfe.core.optimization import Optimizer

optimizer = Optimizer()
optimal_params, likelihood = await optimizer.async_optimize(
    data, initial_params, model_type, distribution)
```

#### Distributions

Statistical distributions for error terms:

```python
from mfe.core.distributions import GED, SkewedT, jarque_bera

# Create GED distribution
ged = GED(nu=1.5)
pdf_values = ged.pdf(data)
log_likelihood = ged.loglikelihood(data)

# Test for normality
jb_stat, p_value = jarque_bera(data)
```

### Model Modules

#### ARMAX

Time series modeling with exogenous variables:

```python
from mfe.models.timeseries import ARMAX

# Create model
model = ARMAX(p=1, q=1, exog=exog_data, trend='c')

# Estimate parameters
await model.async_fit(data)

# Forecast
forecasts = model.forecast(steps=10, exog_future=future_exog)
```

#### GARCHModel

Volatility modeling with various GARCH variants:

```python
from mfe.models.garch import GARCHModel

# Create model
model = GARCHModel(p=1, q=1, model_type='EGARCH', distribution='student-t')

# Estimate parameters
await model.async_fit(returns)

# Forecast volatility
volatility = model.forecast(horizon=10)
```

## Contributing

Contributions to the MFE Toolbox backend are welcome. Please ensure that any contributions maintain:

1. Comprehensive type hints for all functions and classes
2. Numba optimization for performance-critical operations
3. Proper error handling and input validation
4. Thorough documentation with examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.