# Upgrade Guide

This guide provides instructions for upgrading between versions of the MFE Toolbox Python implementation.

## Overview

The MFE Toolbox follows semantic versioning. This guide covers:
- Version compatibility requirements
- Breaking changes between versions
- Upgrade instructions and best practices
- Migration considerations
- Performance optimization guidelines
- Cross-platform deployment notes

## Version 4.0.0 Upgrade Guide

### Compatibility Requirements
- Python 3.12 or higher
- NumPy 1.26.3 or higher
- SciPy 1.11.4 or higher
- Pandas 2.1.4 or higher
- Statsmodels 0.14.1 or higher
- Numba 0.59.0 or higher
- PyQt6 6.6.1 or higher (for GUI components)

### Breaking Changes
- Python 3.12 requirement: This version requires Python 3.12 due to its use of modern language features.
- Numba optimization changes: Performance-critical operations now use Numba's @jit decorators instead of MEX files.
- Package structure updates: The toolbox follows a modern Python package structure with clearly defined namespaces.
- API modifications: Function signatures and class interfaces have been updated to align with Python conventions.
- Async/await pattern adoption: Long-running operations now use async/await patterns for improved responsiveness.
- Type hint requirements: Strict type hints are used throughout the codebase for better safety and documentation.
- Cross-platform deployment changes: The toolbox is now deployed as a standard Python package using modern packaging tools.

### Upgrade Instructions
1. Ensure Python 3.12 or higher is installed on your system
2. Install the package using pip:
   ```
   pip install mfe==4.0.0
   ```
3. Update import statements to reflect the new package structure:
   ```python
   # Old: N/A (initial Python version)
   # New:
   from mfe.core import GED, optimize_garch
   from mfe.models import ARMAModel
   from mfe.utils import validation
   ```
4. Adapt any existing code to utilize the new async/await patterns:
   ```python
   # Example of async usage:
   import asyncio
   from mfe.models import GARCHModel
   
   async def estimate_model():
       model = GARCHModel(p=1, q=1)
       data = load_your_data()
       await model.async_fit(data)
       print(model.parameters)
   
   # Run the async function
   asyncio.run(estimate_model())
   ```

### Validation Steps
After upgrading, verify your installation:
1. Run a simple test to ensure core functionality:
   ```python
   import numpy as np
   from mfe.core import GED
   
   # Create a GED distribution and verify it works
   data = np.random.randn(1000)
   ged = GED(nu=1.5)
   loglik = ged.loglikelihood(data)
   print(f"Log-likelihood: {loglik}")
   ```
2. Verify Numba optimization is working correctly:
   ```python
   from mfe.models import GARCHModel
   import numpy as np
   
   # Generate sample data
   np.random.seed(42)
   returns = np.random.normal(0, 1, 1000)
   
   # Create and estimate a GARCH model
   model = GARCHModel(p=1, q=1)
   import asyncio
   asyncio.run(model.async_fit(returns))
   
   # Verify results
   print(f"Parameters: {model.parameters}")
   print(f"Log-likelihood: {model.likelihood}")
   ```

### Performance Optimization
- Numba-optimized functions should compile on first use, which may cause a slight delay
- Subsequent calls will benefit from JIT compilation with near-native performance
- For high-performance applications, consider pre-warming the JIT cache by executing core functions once before timing-critical operations
- Use the provided async interfaces for long-running operations to maintain UI responsiveness

### Platform-Specific Notes
- Windows: Ensure Microsoft Visual C++ Redistributable is installed for Numba compilation
- Linux: Verify development tools are installed for optimal Numba performance
- macOS: Both Intel and Apple Silicon are supported, with optimized performance on Apple Silicon via Numba

## Compatibility Matrix

The table below outlines the compatibility between MFE Toolbox versions and required dependencies:

| MFE Version | Python | NumPy | SciPy | Pandas | Statsmodels | Numba | PyQt6 |
|-------------|--------|-------|-------|--------|-------------|-------|-------|
| 4.0.0       | ≥3.12  | ≥1.26.3 | ≥1.11.4 | ≥2.1.4 | ≥0.14.1 | ≥0.59.0 | ≥6.6.1 |

### Dependency Compatibility Notes

#### Python
- Python 3.12 is required for MFE Toolbox 4.0.0 due to its use of modern language features
- The toolbox leverages typing improvements, performance enhancements, and async/await patterns in Python 3.12
- Earlier Python versions are not supported

#### Scientific Computing Stack
- NumPy, SciPy, Pandas, and Statsmodels versions are selected for compatibility with Python 3.12
- These libraries provide essential functionality for matrix operations, optimization, time series handling, and statistical modeling
- When upgrading dependencies, test thoroughly to ensure compatibility

#### Numba
- Numba provides just-in-time compilation for performance-critical operations
- Version 0.59.0 or higher is required for Python 3.12 compatibility
- Numba optimizations replace the legacy MEX files from the MATLAB implementation

#### PyQt6
- PyQt6 is required for the GUI components
- Can be omitted if only using the toolbox programmatically
- Version 6.6.1 or higher ensures compatibility with modern operating systems

## Validation Checklist

Use the following checklist to validate your MFE Toolbox installation and integration:

### Environment Validation
- [ ] Verify Python version: `python --version` should show 3.12.x or higher
- [ ] Confirm MFE Toolbox installation: `pip show mfe` should display version 4.0.0
- [ ] Validate NumPy, SciPy, Pandas, Statsmodels versions meet minimum requirements
- [ ] Check Numba installation: `pip show numba` should show version 0.59.0 or higher
- [ ] Verify PyQt6 installation if using GUI components: `pip show pyqt6`

### Functionality Validation
- [ ] Import core modules: `from mfe.core import GED, optimize_garch`
- [ ] Import model modules: `from mfe.models import ARMAModel, GARCHModel`
- [ ] Create and estimate a simple model with sample data
- [ ] Verify async operations function correctly
- [ ] Test error handling by intentionally passing invalid parameters

### Performance Validation
- [ ] Measure execution time of Numba-optimized functions
- [ ] Compare performance to expectations or previous versions
- [ ] Verify JIT compilation works correctly by examining first-run vs. subsequent-run times
- [ ] Test with realistic data sizes to ensure scalability

### Integration Validation
- [ ] Verify integration with your existing codebase
- [ ] Check compatibility with your data processing pipeline
- [ ] Test cross-platform functionality if deploying on multiple operating systems
- [ ] Validate GUI components if using the interactive interface

### Documentation Resources
- CHANGELOG.md: Detailed version history and changes
- MIGRATION_GUIDE.md: Comprehensive migration instructions from MATLAB
- README.md: Quick start guide and overview
- mfe/docs/: API documentation and usage examples

## Troubleshooting

### Common Issues

#### Python Version Conflicts
- **Issue**: "ImportError: This package requires Python 3.12 or higher."
- **Solution**: Install Python 3.12+ or create a virtual environment with the correct Python version.

#### Package Dependency Issues
- **Issue**: "ImportError: No module named 'numba'" or similar dependency errors.
- **Solution**: Install missing dependencies: `pip install numba>=0.59.0`

#### Numba Optimization Failures
- **Issue**: Warnings about function compilation failures or falling back to object mode.
- **Solution**: Ensure compatible hardware/software environment and Numba version.

#### Type Hint Errors
- **Issue**: Type checking errors when using static type checkers.
- **Solution**: Update your code to match the type signatures defined in the toolbox.

### Support Resources
- GitHub Issues: Report problems at [GitHub repository](https://github.com/bashtage/arch)
- Documentation: Refer to [online documentation](https://bashtage.github.io/arch/)
- Community Support: Join discussions in the GitHub repository

For detailed version history and changes, see CHANGELOG.md