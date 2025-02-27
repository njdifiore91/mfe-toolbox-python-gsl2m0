# Contributing to MFE Toolbox

Thank you for your interest in contributing to the MFE Toolbox project! This document outlines the process for contributing to the project, including development workflow, code standards, testing requirements, and submission guidelines.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Submission Process](#submission-process)

## Getting Started

### Python 3.12 environment setup

The MFE Toolbox requires Python 3.12 or later. We recommend using a virtual environment for development:

```bash
# Create a virtual environment
python3.12 -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Scientific computing package installation

Install the required packages:

```bash
# Install base requirements
pip install numpy==1.26.3 scipy==1.11.4 pandas==2.1.4 statsmodels==0.14.1

# Install Numba for JIT compilation
pip install numba==0.59.0

# Install GUI components
pip install PyQt6==6.6.1
```

### Development tools configuration

Install development tools:

```bash
# Install development requirements
pip install pytest==7.4.3 hypothesis==6.92.1 pytest-asyncio==0.21.1 pytest-benchmark==4.0.0
pip install black==23.11.0 mypy==1.7.0 flake8==6.1.0 isort==5.12.0
pip install pre-commit==3.5.0
```

Configure pre-commit hooks:

```bash
pre-commit install
```

### Numba setup and validation

Verify your Numba installation and JIT compilation capabilities:

```python
import numba
import numpy as np

@numba.jit(nopython=True)
def test_function(x):
    return x * x

# Test with array
x = np.arange(10)
result = test_function(x)
print("Numba test successful:", result)
```

## Development Workflow

### Git branching strategy

We follow a feature branch workflow:

1. Fork the repository (for external contributors)
2. Create a new branch from `main` for your feature or bugfix:
   ```bash
   git checkout -b feature/descriptive-name
   # or
   git checkout -b fix/issue-description
   ```
3. Make your changes and commit them
4. Push your branch and create a pull request against `main`

### Commit message format

Follow these guidelines for commit messages:

```
type(scope): concise description

Detailed explanation if needed
```

Where `type` is one of:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc; no code change
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding tests
- `chore`: Maintenance tasks, dependency updates, etc.

And `scope` is the module or component being modified.

### Code review process

All submissions require review:

1. A maintainer will review your pull request
2. Automated checks must pass (tests, linting, type checking)
3. Address any review comments and update your pull request
4. Once approved, a maintainer will merge your changes

### Async development patterns

When implementing asynchronous code:

1. Use Python's `async/await` patterns consistently
2. Provide progress reporting for long-running operations
3. Support cancellation where appropriate
4. Test async code using `pytest-asyncio`

Example:

```python
async def process_data_async(data):
    total_steps = len(data)
    for i, item in enumerate(data):
        # Process item
        processed = await compute_item(item)
        
        # Periodically yield control back to event loop
        if i % 10 == 0:
            await asyncio.sleep(0)
            
        # Report progress
        progress = (i + 1) / total_steps * 100
        
    return results
```

### Performance optimization workflow

For performance-critical code:

1. First implement a pure Python version that is correct and well-tested
2. Profile to identify bottlenecks
3. Apply Numba optimization with `@jit` decorator
4. Benchmark to verify performance improvement
5. Write tests that verify both correctness and performance

## Code Standards

### Python 3.12 features usage

Leverage modern Python 3.12 features:

- Use improved error messages with precise locations
- Leverage enhanced typing features
- Utilize performance optimizations for core types
- Apply pattern matching where appropriate

### Type hint requirements

The MFE Toolbox uses strict type hints throughout:

- Add type hints to all function parameters and return values
- Use the `typing` module for complex types: `List`, `Dict`, `Optional`, etc.
- Apply type annotations to class attributes
- Leverage dataclasses with type hints for parameter containers
- Run `mypy` before submitting code to verify type consistency

Example:

```python
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

def calculate_volatility(returns: np.ndarray, 
                        window: int = 20,
                        annualize: bool = True) -> np.ndarray:
    """Calculate rolling volatility with proper type hints."""
    # Implementation
```

### Docstring standards

Follow NumPy/SciPy-style docstrings:

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    Brief description of function.
    
    Detailed description of function behavior and implementation.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2, by default default
        
    Returns
    -------
    return_type
        Description of return value
        
    Raises
    ------
    ExceptionType
        Description of when this exception is raised
        
    Examples
    --------
    >>> function_name(1, 2)
    3
    """
```

### Numba optimization patterns

When applying Numba optimization:

- Use `@jit(nopython=True)` for maximum performance
- Ensure type stability within JIT-compiled functions
- Avoid Python objects inside JIT-compiled code
- Use `@jit(parallel=True)` with `prange` for parallelizable operations

Example:

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True)
def optimized_function(data: np.ndarray) -> np.ndarray:
    """Numba-optimized function."""
    result = np.empty_like(data)
    for i in range(len(data)):
        result[i] = data[i] * data[i]
    return result

@jit(nopython=True, parallel=True)
def parallel_function(data: np.ndarray) -> np.ndarray:
    """Parallelized Numba function."""
    result = np.empty_like(data)
    for i in prange(len(data)):
        result[i] = data[i] * data[i]
    return result
```

### Scientific computing conventions

Follow these conventions for scientific computing:

- Use NumPy array operations for vectorized computations when possible
- Implement Numba JIT compilation for performance-critical loops
- Leverage SciPy for optimization and statistical functions
- Apply Pandas for time series handling
- Use Statsmodels for econometric modeling

## Testing Requirements

### Pytest configuration

Create comprehensive tests using pytest:

- Place tests in the `tests/` directory following the project structure
- Name test files with the `test_` prefix
- Name test functions with the `test_` prefix
- Run tests with `pytest` before submitting code

Example directory structure:

```
tests/
├── test_core/
│   ├── test_bootstrap.py
│   ├── test_distributions.py
│   └── test_optimization.py
├── test_models/
│   ├── test_garch.py
│   ├── test_realized.py
│   └── test_volatility.py
└── conftest.py
```

### Property-based testing

Use hypothesis for property-based testing:

```python
import pytest
from hypothesis import given, strategies as st
import numpy as np

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
```

### Performance benchmarking

Use pytest-benchmark to measure performance:

```python
import pytest
import numpy as np
from mfe.models.realized import realized_variance, realized_variance_numba

@pytest.mark.benchmark
def test_realized_variance_performance(benchmark):
    """Benchmark performance of standard vs. Numba-optimized implementation."""
    # Generate test data
    np.random.seed(12345)
    n = 10000
    prices = np.cumsum(np.random.normal(0, 0.01, size=n)) + 100
    times = np.linspace(0, 86400, n)  # One day in seconds
    
    # Benchmark execution
    result = benchmark(realized_variance_numba, prices, times, "seconds", "CalendarTime", 300)
    
    # Verify result is valid
    assert isinstance(result, float)
    assert result > 0
```

### Type checking validation

Use mypy to validate type hints:

```bash
# Run mypy on the entire package
mypy mfe/

# Run mypy on a specific module
mypy mfe/models/garch.py
```

Fix any type errors before submitting code.

### Async test patterns

Test asynchronous code with pytest-asyncio:

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
```

## Submission Process

### Pull request requirements

When submitting a pull request:

1. Ensure your branch is up-to-date with the latest main branch
2. Include a clear, concise description of changes
3. Reference any related issues using the syntax `Fixes #123` or `Relates to #123`
4. Ensure all tests pass
5. Verify code passes type checking with mypy
6. Confirm code follows style guidelines using black and flake8
7. Include appropriate tests for new functionality

### Code review checklist

Before submitting for review, verify:

- [ ] Code follows Python 3.12 best practices
- [ ] Proper type hints are applied throughout
- [ ] Docstrings are complete and follow NumPy/SciPy style
- [ ] Tests are comprehensive and pass
- [ ] Numba optimizations are applied where appropriate
- [ ] Performance benchmarks show expected improvements
- [ ] Async code follows established patterns
- [ ] No unnecessary dependencies are introduced

### Performance review criteria

Performance-critical code must meet these criteria:

- Numba optimization for compute-intensive operations
- Benchmarks showing significant speedup over pure Python implementation
- Memory efficiency for large datasets
- Appropriate use of parallel execution where applicable
- Validation of numerical stability

### Security review requirements

Security considerations include:

- Input validation for all public functions
- Bounds checking for array operations
- Proper error handling
- No use of `eval()` or similar unsafe constructs
- Secure handling of file operations
- No hardcoded credentials or sensitive information

### Documentation standards

Include appropriate documentation:

- Module-level docstrings explaining purpose and usage
- Function and class docstrings following NumPy/SciPy style
- Examples demonstrating typical usage
- Notes on implementation details where relevant
- References to relevant literature or algorithms
- Cross-references to related functionality

---

## Additional Resources

- [Style Guide](STYLE_GUIDE.md): Detailed Python coding standards
- [Code of Conduct](CODE_OF_CONDUCT.md): Community standards and expectations

## Questions?

If you have questions about contributing, please [open an issue](https://github.com/username/mfe-toolbox/issues/new) or contact the maintainers.