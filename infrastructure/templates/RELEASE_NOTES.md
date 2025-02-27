# Release Notes

Detailed release notes for the MFE Toolbox Python implementation.
For a summary of changes, see CHANGELOG.md

## Version 4.0.0 (YYYY-MM-DD)

### Overview
Complete re-implementation of the MFE Toolbox in Python 3.12, modernizing the codebase while maintaining core functionality.

### New Features
- Python 3.12 Implementation
  - Modern async/await patterns for improved performance
  - Strict type hints for enhanced code safety
  - Dataclasses for clean model parameter handling

- Performance Optimization
  - Numba JIT compilation replacing MEX files
  - Hardware-specific optimizations
  - Vectorized operations via NumPy

- Enhanced UI
  - PyQt6-based graphical interface
  - Interactive plotting capabilities
  - Real-time parameter updates

### API Changes
- Transition from MATLAB to Python syntax
- New class-based model implementations
- Updated function signatures with type hints

### Migration Guide
- Install Python 3.12 or later
- Use pip to install the package
- Update scripts to use Python syntax
- Review updated API documentation

### Performance
- Near-C performance through Numba optimization
- Efficient array operations via NumPy
- Improved memory management

### Dependencies
- Python >= 3.12
- NumPy >= 1.26.3
- SciPy >= 1.11.4
- Pandas >= 2.1.4
- Statsmodels >= 0.14.1
- Numba >= 0.59.0
- PyQt6 >= 6.6.1

### Bug Fixes
- Initial release

### Security Updates
- Modern Python security practices
- Type-safe implementations

For version history and summary of changes, see CHANGELOG.md