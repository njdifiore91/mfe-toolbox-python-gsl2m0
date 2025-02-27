# MFE Toolbox (Python Implementation)

[![PyPI version](https://badge.fury.io/py/mfe-toolbox.svg)](https://badge.fury.io/py/mfe-toolbox)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

The MFE (Financial Econometrics) Toolbox is a comprehensive suite of Python modules designed for modeling financial time series and conducting advanced econometric analyses. While retaining its legacy version 4.0 identity, the toolbox has been completely re-implemented using Python 3.12, incorporating modern programming constructs such as async/await patterns and strict type hints.

The toolbox leverages Python's scientific computing ecosystem, built upon foundational libraries including:
- **NumPy** for matrix operations
- **SciPy** for optimization and statistical functions
- **Pandas** for time series handling
- **Statsmodels** for econometric modeling
- **Numba** for performance optimization

### Key Features

- Financial time series modeling and forecasting
- Volatility and risk modeling using univariate and multivariate approaches
- High-frequency financial data analysis
- Cross-sectional econometric analysis
- Bootstrap-based statistical inference
- Advanced distribution modeling and simulation
- Interactive modeling environment built with PyQt6

### Package Structure

The system architecture follows a modern Python package structure organized into four main namespaces:

1. **Core Statistical Modules** (`mfe.core`):
   - Bootstrap: Robust resampling for dependent time series
   - Cross-section: Regression and principal component analysis
   - Distributions: Advanced statistical distributions
   - Tests: Comprehensive statistical testing suite

2. **Time Series & Volatility Modules** (`mfe.models`):
   - Timeseries: ARMA/ARMAX modeling and diagnostics
   - Univariate: Single-asset volatility models (AGARCH, APARCH, etc.)
   - Multivariate: Multi-asset volatility models (BEKK, CCC, DCC)
   - Realized: High-frequency financial econometrics

3. **Support Modules** (`mfe.utils`, `mfe.ui`):
   - GUI: Interactive modeling environment built with PyQt6
   - Utility: Data transformation and helper functions
   - Performance: Numba-optimized computational kernels

## Installation

### Requirements

- Python 3.12 or newer
- NumPy 1.26.3 or newer
- SciPy 1.11.4 or newer
- Pandas 2.1.4 or newer
- Statsmodels 0.14.1 or newer
- Numba 0.59.0 or newer
- PyQt6 6.6.1 or newer (for GUI components)

### Using pip

The simplest way to install the MFE Toolbox is via pip:

```bash
pip install mfe-toolbox
```

### Development Installation

For development purposes, you can install the package in editable mode:

```bash
git clone https://github.com/username/mfe-toolbox.git
cd mfe-toolbox
pip install -e .
```

### Virtual Environment Setup (Recommended)

We recommend using a virtual environment for installation:

```bash
# Create a virtual environment
python -m venv mfe-env

# Activate the environment (Windows)
mfe-env\Scripts\activate

# Activate the environment (macOS/Linux)
source mfe-env/bin/activate

# Install the package
pip install mfe-toolbox
```

### Verifying Installation

To verify your installation, run the following in Python:

```python
import mfe
print(mfe.__version__)
```

## Usage Examples

### Basic Time Series Modeling

```python
import numpy as np
import pandas as pd
from mfe.models.timeseries import ARMAX

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 500).cumsum() + 10
returns = np.diff(data)

# Create and fit ARMAX model
model = ARMAX(p=1, q=1)
result = model.fit(returns)

# Print results
print(result.summary())

# Generate forecasts
forecasts = result.forecast(steps=10)
print(forecasts)
```

### Volatility Modeling with GARCH

```python
import numpy as np
from mfe.models.univariate import GARCH

# Generate sample returns
np.random.seed(42)
returns = np.random.normal(0, 1, 1000) * np.sqrt(np.random.gamma(1, 0.2, 1000))

# Create and fit GARCH model
model = GARCH(p=1, q=1)
result = model.fit(returns)

# Print results
print(result.summary())

# Forecast volatility
vol_forecast = result.forecast_variance(steps=10)
print(vol_forecast)
```

### Asynchronous Model Estimation

```python
import numpy as np
import asyncio
from mfe.models.univariate import EGARCH

async def estimate_model():
    # Generate sample data
    np.random.seed(42)
    returns = np.random.normal(0, 1, 1000) * np.sqrt(np.random.gamma(1, 0.2, 1000))
    
    # Create model
    model = EGARCH(p=1, q=1)
    
    # Asynchronous estimation with progress updates
    async for progress, state in model.fit_async(returns):
        print(f"Estimation progress: {progress:.2%}")
    
    result = await model.get_results()
    return result

# Run the async function
result = asyncio.run(estimate_model())
print(result.summary())
```

### Numba-Optimized Computations

```python
import numpy as np
from mfe.core.bootstrap import stationary_bootstrap
from mfe.models.realized import realized_volatility

# Generate high-frequency data
np.random.seed(42)
prices = 100 * np.exp(np.random.normal(0, 0.0001, 1000).cumsum())
times = np.arange(1000) / 1000

# Compute realized volatility with Numba optimization
rv = realized_volatility(prices, times, timeType='seconds', 
                        samplingType='CalendarTime', 
                        samplingInterval=5)
print(f"Realized volatility: {rv}")

# Run stationary bootstrap
bootstrap_samples = stationary_bootstrap(returns=np.diff(np.log(prices)), 
                                        block_size=50, 
                                        replications=1000)
bootstrap_means = bootstrap_samples.mean(axis=1)
print(f"Bootstrap 95% confidence interval: [{np.percentile(bootstrap_means, 2.5)}, {np.percentile(bootstrap_means, 97.5)}]")
```

### Using the PyQt6 GUI

```python
import sys
from PyQt6.QtWidgets import QApplication
from mfe.ui.armax_viewer import ARMAXViewer

# Create PyQt application
app = QApplication(sys.argv)

# Create and show ARMAX viewer
viewer = ARMAXViewer()
viewer.show()

# Run the application
sys.exit(app.exec())
```

## API Documentation

### Core Modules (mfe.core)

#### Bootstrap Module

- `block_bootstrap`: Bootstrap for time series data using fixed block sizes
- `stationary_bootstrap`: Bootstrap with random block sizes for time series data
- `circular_block_bootstrap`: Bootstrap with circular blocks to handle edge effects

#### Distribution Module

- `skewed_t_distribution`: Hansen's skewed t-distribution for asymmetric returns
- `ged_distribution`: Generalized Error Distribution for fat-tailed data
- `jarque_bera`: Test for normality based on skewness and kurtosis

#### Optimization Module

- `optimize_with_constraints`: Constrained optimization routines for model fitting
- `quasi_newton`: Quasi-Newton optimization methods for likelihood maximization

### Model Modules (mfe.models)

#### Timeseries Module

- `ARMAX`: Class for ARMA models with exogenous variables
- `lag_selection`: Automatic lag order selection using information criteria
- `serial_correlation`: Tests for serial correlation in time series

#### Univariate Volatility Module

- `GARCH`: Standard GARCH model for volatility
- `EGARCH`: Exponential GARCH for asymmetric volatility
- `FIGARCH`: Fractionally Integrated GARCH for long memory
- `APARCH`: Asymmetric Power ARCH for flexible power transformations

#### Multivariate Volatility Module

- `BEKK`: Multivariate GARCH with positive definite covariance matrices
- `DCC`: Dynamic Conditional Correlation model
- `CCC`: Constant Conditional Correlation model

#### Realized Volatility Module

- `realized_variance`: Estimators for realized variance from high-frequency data
- `realized_kernel`: Noise-robust realized volatility estimators
- `bipower_variation`: Jump-robust volatility estimator

### UI Components (mfe.ui)

- `ARMAXViewer`: Interactive GUI for ARMAX model estimation and diagnostics
- `GARCHViewer`: GUI for GARCH model fitting and visualization
- `ResultsViewer`: General-purpose results visualization interface

### Utility Functions (mfe.utils)

- `data_transformations`: Functions for data preprocessing and transformation
- `statistical_tests`: Common statistical tests used in financial analysis
- `plotting`: High-level plotting functions for model diagnostics

## Performance Optimization

The MFE Toolbox leverages Numba for performance-critical computations, providing:

- Just-in-time compilation of performance-critical Python functions
- Near-C-level performance for intensive numerical operations
- Hardware-specific optimizations
- Parallel execution support where applicable

Example of a Numba-optimized function:

```python
from numba import jit
import numpy as np

@jit(nopython=True)
def garch_likelihood(parameters, returns, sigma2, p, q):
    """GARCH likelihood function optimized with Numba JIT compilation."""
    T = returns.shape[0]
    omega = parameters[0]
    alpha = parameters[1:p+1]
    beta = parameters[p+1:p+q+1]
    
    for t in range(max(p, q), T):
        sigma2[t] = omega
        for i in range(p):
            sigma2[t] += alpha[i] * returns[t-i-1]**2
        for j in range(q):
            sigma2[t] += beta[j] * sigma2[t-j-1]
    
    logliks = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return logliks
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Acknowledgments

- Original MATLAB MFE Toolbox by Kevin Sheppard
- Contributors to the Python scientific ecosystem
- Financial econometrics community

## Contact

For issues and feature requests, please use the [issue tracker](https://github.com/username/mfe-toolbox/issues).