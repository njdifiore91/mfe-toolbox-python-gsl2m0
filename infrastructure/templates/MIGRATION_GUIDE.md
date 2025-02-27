# Migration Guide

This guide provides comprehensive instructions for migrating from MATLAB 4.0 to Python 3.12 implementation of the MFE Toolbox.

---

## Overview

The MFE Toolbox has been completely re-implemented in Python 3.12. This guide covers:
- MATLAB to Python migration considerations
- Ecosystem and dependency changes
- Code migration patterns
- Performance optimization strategies

Migrating from the legacy MATLAB implementation (version 4.0) to the modern Python ecosystem brings a host of benefits:
- Cleaner, modular code structure using Python packages and dataclasses
- Enhanced performance through Numba-optimized routines using @jit decorators
- Improved code readability and maintainability with strict type hints and async/await patterns
- Seamless integration with the scientific computing stack (NumPy, SciPy, Pandas, Statsmodels)

---

## Ecosystem Changes

### Development Environment
- **Python 3.12 Installation:** Ensure you have Python 3.12 or higher installed. This is required to leverage modern language features such as async/await, dataclasses, and strict type hints.
- **Scientific Computing Dependencies:** The toolbox now relies on core packages:
  - **NumPy:** For efficient array and matrix operations (e.g., using numpy_arrays).
  - **SciPy:** For numerical optimization and statistical functions.
  - **Pandas:** For handling time series data.
  - **Statsmodels:** For advanced econometric modeling.
- **Development Tools and Virtual Environments:** Modern tools such as virtualenv or venv are recommended for dependency isolation. The build system is now based on pyproject.toml and setup.py, ensuring smooth package management.

### Package Structure
- **Python Package Organization:** The toolbox is organized into distinct namespaces (e.g., `mfe.core`, `mfe.models`, `mfe.ui`, `mfe.utils`) to promote separation of concerns.
- **Module Hierarchy and Imports:** Modules are now imported using Python’s standard import system. Internal dependencies such as version compatibility (see VERSIONING.md) and version history (see CHANGELOG.md) are maintained in separate markdown documents.
- **Build System Changes:** The migration replaces legacy build processes with modern Python packaging conventions. The new build system supports both source (sdist) and binary (wheel) distributions, enabling easier deployment and dependency resolution.

---

## Code Migration

### Syntax Changes
- **MATLAB to Python Syntax Conversion:** MATLAB scripts and functions have been re-written in Python. For example, MATLAB’s 1-indexed arrays now use 0-indexing in Python.
- **Array Indexing Differences:** All MATLAB arrays have been translated to NumPy arrays. Be aware of the shift from MATLAB’s 1-indexing to Python’s 0-indexing, which may affect loop boundaries and slicing.
- **Function Definition Patterns:** MATLAB functions are now implemented as Python functions or methods within classes. The new design leverages Python’s modern constructs such as async functions and decorators.

### Modern Python Features
- **Type Hints and Dataclasses:** Code is now type-safe and self-documenting using Python type hints. Data models are defined as dataclasses to ensure immutable and clear data structures.
- **Async/Await Patterns:** Long-running operations such as model estimation and bootstrap resampling have been converted to asynchronous functions using Python’s async/await syntax. This improves responsiveness, especially in GUI applications.
- **Error Handling Approaches:** Traditional MATLAB error handling has been replaced by Python’s try/except blocks. Comprehensive input validation and exception logging are now integrated across the toolbox.

#### Code Migration Example

Below is an example illustrating typical changes during migration:

```matlab
% MATLAB code snippet (1-indexed)
function y = add_one(x)
    y = x + 1;
end
```

has been migrated to Python as:

```python
def add_one(x: float) -> float:
    """Add one to the input value."""
    return x + 1
```

Similarly, legacy MEX functions for performance-critical operations are now replaced with Numba-optimized functions using the `@jit` decorator:

```python
from numba import jit
import numpy as np

@jit(nopython=True)
def compute_acf(data: np.ndarray, nlags: int) -> np.ndarray:
    # Compute the autocorrelation function using optimized loops
    n = len(data)
    acf = np.zeros(nlags + 1)
    data_mean = np.mean(data)
    y = data - data_mean
    denominator = np.sum(y * y)
    for lag in range(nlags + 1):
        numerator = 0.0
        for t in range(lag, n):
            numerator += y[t] * y[t - lag]
        acf[lag] = numerator / denominator
    return acf
```

*Note: Always refer to the in-code documentation for detailed parameter validation and error handling corrections.*

#### Upgrade Paths
The following upgrade paths have been defined to aid in the transition:

- **MATLAB 4.0 → Python 3.12**
- **MEX → Numba:** MEX optimizations have been replaced with Numba's JIT compilation using `@jit` decorators.
- **mex_files → jit_decorators**
- **matlab_arrays → numpy_arrays**

These upgrades form the backbone of the performance and maintainability improvements in the new implementation.

---

## Performance Optimization

### MEX to Numba Migration
- **JIT Compilation with @jit Decorators:** Legacy MEX functions have been completely re-written in Python and optimized using Numba’s JIT compilation. This approach provides near-C performance on critical routines.
- **Numba Type Specialization:** Functions have been annotated with type hints and compiled in nopython mode to enable hardware-specific optimizations.
- **Performance Considerations:** Extensive testing and benchmarking have been performed to ensure that the new implementation meets or exceeds the performance of the legacy MATLAB code.

### Vectorization
- **NumPy Array Operations:** Core numerical routines now leverage NumPy’s vectorized operations to enhance performance and reduce loop overhead.
- **Efficient Matrix Computations:** Many linear algebra tasks are handled by optimized NumPy and SciPy routines, ensuring efficient manipulation of large datasets.
- **Memory Management:** Memory efficiency is achieved by using in-memory NumPy arrays and minimizing unnecessary data copying.

---

## Testing Migration

### Test Framework
- **Python Testing Tools (pytest):** The entire toolbox now uses pytest for unit, integration, and performance testing. Property-based testing with Hypothesis further ensures code robustness.
- **Test Organization:** Tests are organized in a hierarchical directory structure, aligning with the package layout to facilitate maintenance.
- **Coverage Requirements:** The migration mandate includes high code coverage to ensure that all critical functionality is rigorously tested.

### Validation
- **Numerical Accuracy Verification:** Unit tests validate that migrated algorithms produce results consistent with previous MATLAB implementations.
- **Performance Benchmarking:** Automated benchmarking tests with pytest-benchmark ensure that performance improvements are maintained.
- **Integration Testing:** End-to-end tests validate the interaction between modules across the toolbox.

---

## Breaking Changes

This migration introduces the following breaking changes to ensure better future compatibility and performance:
- Complete transition to the Python ecosystem (MATLAB code is no longer supported).
- MEX optimizations have been replaced with Numba-based JIT optimization.
- Array indexing must now use 0-indexing instead of MATLAB’s 1-indexing.
- Function calling conventions and parameter passing have changed to conform to Python standards.
- Error handling has been overhauled, with robust try/except blocks replacing MATLAB error handling patterns.

---

## Footer

For detailed version history and changes, see [CHANGELOG.md](CHANGELOG.md). For version compatibility and release procedures, refer to [VERSIONING.md](VERSIONING.md).

---

*This migration guide is intended to serve as a reference for developers and researchers transitioning to the new Python implementation of the MFE Toolbox. It documents not only the necessary code changes but also the underlying design philosophy aimed at achieving high performance and improved maintainability in modern scientific computing environments.*