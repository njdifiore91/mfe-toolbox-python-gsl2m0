# MFE Toolbox

A comprehensive suite of Python modules for financial time series modeling and advanced econometric analysis.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.3-green.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11.4-green.svg)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.4-green.svg)](https://pandas.pydata.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14.1-green.svg)](https://www.statsmodels.org/)
[![Numba](https://img.shields.io/badge/Numba-0.59.0-green.svg)](https://numba.pydata.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.6.1-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

## Overview

The MFE (Financial Econometrics) Toolbox provides researchers, analysts, and practitioners with robust tools for financial modeling and econometric analysis. While retaining its legacy version 4.0 identity, the toolbox has been completely re-implemented using Python 3.12, incorporating modern programming constructs such as async/await patterns, dataclasses, and strict type hints.

The toolbox leverages Python's scientific computing ecosystem, built upon foundational libraries including NumPy for matrix operations, SciPy for optimization and statistical functions, Pandas for time series handling, Statsmodels for econometric modeling, and Numba for performance optimization.

## Key Features

- **Time Series Analysis**: ARMA/ARMAX modeling with robust parameter optimization
- **Volatility Modeling**: Comprehensive suite of GARCH variants (AGARCH, EGARCH, FIGARCH) and multivariate specifications (BEKK, DCC)
- **High-Frequency Analytics**: Advanced realized volatility estimation and noise filtering
- **Bootstrap Methods**: Robust resampling for dependent time series
- **Cross-sectional Tools**: Regression and principal component analysis
- **Statistical Framework**: Comprehensive distribution and testing framework
- **Interactive Interface**: PyQt6-based GUI for interactive modeling

## Installation

### Quick Start

```bash
pip install mfe
```

### Requirements

- Python 3.12 or higher
- Core dependencies:
  - NumPy (1.26.3+)
  - SciPy (1.11.4+)
  - Pandas (2.1.4+)
  - Statsmodels (0.14.1+)
  - Numba (0.59.0+)
  - PyQt6 (6.6.1+) for GUI components

For detailed installation instructions, see the [Installation Guide](docs/INSTALLATION.md).

## Architecture

The MFE Toolbox is organized into four main namespaces:

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

### Modern Python Design

- **Asynchronous Operations**: Uses async/await patterns for improved responsiveness
- **Type Safety**: Implements strict type hints throughout the codebase
- **Performance Optimization**: Leverages Numba's JIT compilation for near-C performance
- **Object-Oriented Design**: Employs class-based architecture with dataclasses

## Usage Example

```python
import numpy as np
from mfe.models import ARMAX
import asyncio

# Generate sample data
n = 1000
ar_coef = 0.7
data = np.zeros(n)
for t in range(1, n):
    data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)

# Create and fit the model using async/await pattern
model = ARMAX(p=1, q=0)  # AR(1) model

async def estimate():
    # Fit the model
    converged = await model.async_fit(data)
    if converged:
        print(f"Estimated AR coefficient: {model._model_params['ar_params'][0]:.4f}")
        
        # Run diagnostic tests
        diagnostics = model.diagnostic_tests()
        print(f"Log-likelihood: {model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        
        # Generate forecasts
        forecasts = model.forecast(steps=10)
        print("Forecasts:", forecasts)

# Run the estimation
asyncio.run(estimate())
```

### Launch GUI Interface

```python
from mfe.ui import launch_gui

# Launch the interactive modeling interface
launch_gui()
```

For comprehensive examples and usage instructions, see the [Usage Guide](docs/USAGE.md).

## Technical Features

### Performance Optimization

- Numba-optimized routines with @jit decorators for near-C performance
- Efficient NumPy array operations for large datasets
- Asynchronous processing for improved responsiveness

### Type Safety

- Comprehensive type hints throughout the codebase
- Runtime parameter validation
- Static type checking support

### Cross-Platform Compatibility

- Works on Windows, macOS, and Linux
- Consistent behavior across operating systems
- Platform-agnostic Python implementation

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Usage Guide](docs/USAGE.md)
- [API Reference](docs/API.md)
- [Examples](docs/examples/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite it as:

```
Kevin Sheppard, MFE Toolbox for Python, Version 4.0, 2023
```

## Acknowledgments

- Original MATLAB version by Kevin Sheppard
- Contributors to the Python scientific ecosystem