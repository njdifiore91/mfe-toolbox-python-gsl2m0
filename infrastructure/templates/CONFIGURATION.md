# MFE Toolbox Configuration Guide

This document provides comprehensive configuration information for the MFE Toolbox (version 4.0.0), covering environment setup, build configuration, testing settings, and type checking.

## Environment Configuration

### Python Environment Requirements

The MFE Toolbox requires Python 3.12 or later with the following scientific computing libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | 1.26.3+ | Array operations and numerical computing |
| SciPy | 1.11.4+ | Scientific and statistical functions |
| Pandas | 2.1.4+ | Time series data handling and manipulation |
| Statsmodels | 0.14.1+ | Statistical models and econometric tools |
| Numba | 0.59.0+ | JIT compilation for performance optimization |
| PyQt6 | 6.6.1+ | GUI framework for interactive components |

### Development Environment Setup

Follow these steps to set up your development environment:

#### 1. Install Python 3.12

Download and install Python 3.12 from [python.org](https://www.python.org/downloads/) or use your operating system's package manager.

#### 2. Create a Virtual Environment

```bash
# Create a new virtual environment
python -m venv mfe-env

# Activate the virtual environment
# On Windows
mfe-env\Scripts\activate
# On Unix or macOS
source mfe-env/bin/activate
```

#### 3. Install Required Dependencies

```bash
# Install core dependencies
pip install numpy>=1.26.3 scipy>=1.11.4 pandas>=2.1.4 statsmodels>=0.14.1 numba>=0.59.0 pyqt6>=6.6.1

# Install development dependencies
pip install pytest>=7.4.3 pytest-asyncio>=0.21.1 pytest-cov>=4.1.0 \
            pytest-benchmark>=4.0.0 pytest-memray>=1.5.0 hypothesis>=6.92.1 \
            sphinx>=7.1.2 mypy>=1.7.1
```

#### 4. Verify Installation

```python
# Run this in a Python interpreter to verify your setup
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import numba
from PyQt6.QtCore import QT_VERSION_STR

print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {sp.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Statsmodels version: {sm.__version__}")
print(f"Numba version: {numba.__version__}")
print(f"PyQt6 version: {QT_VERSION_STR}")
```

### Platform-Specific Considerations

#### Windows
- Ensure Microsoft Visual C++ Build Tools are installed for Numba compilation
- Use `\\` or raw strings `r"path\to\file"` for file paths

#### macOS
- For Apple Silicon (M1/M2), ensure libraries are compatible with arm64 architecture
- XCode Command Line Tools are required for compilation

#### Linux
- Required development libraries: `python3-dev`, `build-essential`
- For GUI functionality, ensure Qt dependencies are installed

## Build System Configuration

The MFE Toolbox uses modern Python packaging tools defined in `pyproject.toml`.

### Project Metadata

The core project configuration from `pyproject.toml`:

```toml
[project]
name = "mfe"
version = "4.0.0"
description = "MATLAB Financial Econometrics Toolbox re-implemented in Python"
authors = [
    {name = "Kevin Sheppard"}
]
requires-python = ">=3.12"
readme = "README.md"
license = {text = "MIT"}
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Mathematics"
]
dependencies = [
    "numpy>=1.26.3",
    "scipy>=1.11.4",
    "pandas>=2.1.4",
    "statsmodels>=0.14.1",
    "numba>=0.59.0",
    "pyqt6>=6.6.1"
]
```

### Build System Configuration

```toml
[build-system]
requires = ["setuptools>=69.0.2", "wheel>=0.42.0"]
build-backend = "setuptools.build_meta"
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "pytest-cov>=4.1.0",
    "pytest-benchmark>=4.0.0",
    "pytest-memray>=1.5.0",
    "hypothesis>=6.92.1",
    "sphinx>=7.1.2",
    "mypy>=1.7.1"
]
```

### Project URLs

```toml
[project.urls]
Homepage = "https://github.com/bashtage/arch"
Documentation = "https://bashtage.github.io/arch/"
Source = "https://github.com/bashtage/arch"
```

### Building the Package

To build the package:

```bash
# Install build dependencies
pip install build

# Build the package
python -m build
```

This will create:
- A source distribution (.tar.gz) in the `dist/` directory
- A wheel package (.whl) in the `dist/` directory

### Installing the Package

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install from the built distribution
pip install dist/mfe-4.0.0-py3-none-any.whl
```

### Customizing the Build Process

For advanced customization, you can create a `setup.py` file:

```python
from setuptools import setup

if __name__ == "__main__":
    setup()
```

And extend the build configuration in `pyproject.toml`:

```toml
[tool.setuptools]
packages = ["mfe"]
package-dir = {"" = "src"}

[tool.setuptools.package-data]
mfe = ["py.typed", "*.pyi"]
```

## Testing Configuration

The MFE Toolbox uses pytest with multiple plugins for comprehensive testing, benchmarking, and profiling.

### Pytest Configuration

The test configuration from `pytest.ini`:

```ini
[pytest]
testpaths = src/backend/tests src/web/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts = --strict-markers -v --cov=mfe --cov-report=term-missing --cov-report=html --benchmark-only --benchmark-storage=.benchmarks --benchmark-autosave --memray --memray-threshold=100MB

markers =
    asyncio: mark test as async/await test
    benchmark: mark test as performance benchmark
    slow: mark test as slow running (>30s)
    numba: mark test as requiring Numba optimization
    numba_parallel: mark test as using parallel Numba optimization
    hypothesis: mark test as property-based test
    distribution: mark test as distribution property test
    memray: mark test for memory profiling
    high_memory: mark test as memory intensive
```

### Plugin Configurations

```ini
# Coverage configuration
cov_fail_under = 90

# Benchmark configuration
benchmark_min_rounds = 100
benchmark_warmup = True
benchmark_timer = time.perf_counter
benchmark_disable_gc = True

# Asyncio configuration
asyncio_mode = auto

# Memory profiling configuration
memray_threshold = 100MB
memray_output = html
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "numba"
pytest -m "not slow"
pytest -m "benchmark"

# Run tests in a specific file
pytest tests/test_garch.py

# Run with coverage report
pytest --cov=mfe --cov-report=html
```

### Test Categories

| Marker | Description |
|--------|-------------|
| `asyncio` | Tests for asynchronous functionality |
| `benchmark` | Performance benchmark tests |
| `slow` | Tests that take more than 30 seconds to run |
| `numba` | Tests for Numba-optimized functions |
| `numba_parallel` | Tests for parallel Numba implementations |
| `hypothesis` | Property-based tests using the hypothesis framework |
| `distribution` | Tests for statistical distribution properties |
| `memray` | Tests for memory profiling |
| `high_memory` | Tests that require significant memory |

### Performance Benchmarking

```bash
# Run benchmark tests
pytest -m benchmark

# Run specific benchmark tests
pytest -m benchmark tests/test_garch.py

# Compare against saved benchmarks
pytest --benchmark-compare
```

### Memory Profiling

```bash
# Run tests with memory profiling
pytest --memray

# Set custom memory threshold
pytest --memray --memray-threshold=200MB
```

### Writing Tests

Example of a test with appropriate markers:

```python
import pytest
import numpy as np
from mfe.models import garch

@pytest.mark.numba
def test_garch_optimization():
    """Test that GARCH optimization works with Numba."""
    data = np.random.normal(0, 1, 1000)
    model = garch.GARCH(p=1, q=1)
    result = model.fit(data)
    assert result.converged
    
@pytest.mark.benchmark
def test_garch_performance(benchmark):
    """Benchmark GARCH estimation performance."""
    data = np.random.normal(0, 1, 1000)
    model = garch.GARCH(p=1, q=1)
    
    # Benchmark the fit operation
    result = benchmark(model.fit, data)
    assert result.converged
    
@pytest.mark.asyncio
async def test_garch_async():
    """Test asynchronous GARCH estimation."""
    data = np.random.normal(0, 1, 1000)
    model = garch.GARCH(p=1, q=1)
    result = await model.fit_async(data)
    assert result.converged
```

## Type Checking Configuration

The MFE Toolbox uses mypy for static type checking with strict settings to ensure type safety.

### Mypy Configuration

The type checking configuration from `mypy.ini`:

```ini
[mypy]
python_version = 3.12
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_return_any = True
warn_unreachable = True
strict_optional = True
strict_equality = True
```

### Library-Specific Configuration

```ini
[mypy.plugins.numpy.*]
plugin_modules = numpy.typing.mypy_plugin

[mypy-numba.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True

[mypy-statsmodels.*]
ignore_missing_imports = True

[mypy-PyQt6.*]
ignore_missing_imports = True
```

### Running Type Checking

```bash
# Check all files
mypy src

# Check specific files
mypy src/mfe/core/bootstrap.py

# Show detailed error messages
mypy --show-error-codes src
```

### Type Checking Standards

- All function definitions must have complete type annotations
- All parameters and return values must be typed
- All class attributes must have type annotations
- Custom types should be defined in `mfe/types.py`

### Example with Proper Type Annotations

```python
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray, ArrayLike

def process_returns(
    returns: NDArray[np.float64],
    window_size: int,
    alpha: Optional[float] = 0.05
) -> Tuple[NDArray[np.float64], float]:
    """
    Process return series with proper type annotations.
    
    Parameters
    ----------
    returns : NDArray[np.float64]
        Array of return values
    window_size : int
        Size of rolling window
    alpha : Optional[float], optional
        Significance level, by default 0.05
        
    Returns
    -------
    Tuple[NDArray[np.float64], float]
        Processed returns and test statistic
    """
    # Implementation
    processed_returns = np.zeros_like(returns)
    statistic = 0.0
    return processed_returns, statistic
```

### Using Type Aliases

Define common types in a central location:

```python
# mfe/types.py
from typing import Dict, List, Tuple, Union, Optional, Callable, TypeVar
import numpy as np
from numpy.typing import NDArray

# Type aliases
ReturnSeries = NDArray[np.float64]
PriceData = NDArray[np.float64]
TimeIndex = NDArray[np.int64]
ParamVector = NDArray[np.float64]
CovarianceMatrix = NDArray[np.float64]

# Function types
OptimizationCallback = Callable[[ParamVector, float], None]
```

### Numba and Type Checking

When using Numba with type annotations:

```python
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import numba

@numba.jit(nopython=True)
def compute_volatility(
    returns: NDArray[np.float64],
    alpha: float,
    beta: float
) -> NDArray[np.float64]:
    """
    Compute volatility with numba optimization.
    
    This function will be JIT compiled by Numba.
    The type annotations are for mypy, not Numba.
    Numba infers types at runtime.
    """
    n = len(returns)
    variance = np.zeros(n, dtype=np.float64)
    variance[0] = returns[0] ** 2
    
    for t in range(1, n):
        variance[t] = alpha * returns[t-1]**2 + beta * variance[t-1]
        
    return np.sqrt(variance)
```

## Troubleshooting

### Common Issues and Solutions

#### Import Errors
If you encounter import errors:
```
ModuleNotFoundError: No module named 'mfe'
```
- Ensure the package is installed: `pip install -e .`
- Check your PYTHONPATH: `echo $PYTHONPATH`
- Verify your virtual environment is activated

#### Numba Compilation Errors
If Numba fails to compile:
```
Failed in nopython mode pipeline
```
- Update to the latest Numba version: `pip install -U numba`
- Use `@numba.jit(debug=True)` to get detailed error information
- Check for unsupported Python features in the JIT-compiled function

#### Type Checking Errors
If mypy reports errors:
```
error: Function is missing a type annotation
```
- Add proper type annotations to all functions
- Use `# type: ignore` sparingly and only when necessary
- Check for inconsistent type usage across modules

#### Test Failures
If tests fail:
```
FAILED tests/test_garch.py::test_garch_estimation
```
- Run with verbose output: `pytest -v tests/test_garch.py`
- Check for platform-specific issues
- Verify all dependencies are correctly installed

### Numba Optimization Issues

If Numba-optimized functions are not performing as expected:

```bash
# Set environment variables for debugging
export NUMBA_DEBUG=1
export NUMBA_DEBUG_ARRAY_OPT=1

# Run the code again
python your_script.py
```

Key environment variables for Numba:

```bash
# Performance configuration
export NUMBA_NUM_THREADS=8  # Set to number of CPU cores
export NUMBA_THREADING_LAYER=tbb  # Options: tbb, omp, workqueue

# Debug information
export NUMBA_DEBUG=1
export NUMBA_DUMP_CFG=1
export NUMBA_DEBUG_ARRAY_OPT=1

# Compilation options
export NUMBA_DISABLE_JIT=0  # Set to 1 to disable JIT for testing
export NUMBA_CACHE_DIR=/path/to/cache  # Custom cache directory
```

## References

- [Python Packaging User Guide](https://packaging.python.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Numba Documentation](https://numba.pydata.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [PyQt6 Documentation](https://www.riverbankcomputing.com/software/pyqt/)