# Deprecation Notice

This document tracks deprecated features, functionality and APIs in the MFE Toolbox.

Each entry includes the version in which the feature was deprecated, when it will be removed, and migration guidance.

## MATLAB Implementation - Deprecated in version 4.0.0

### Removed Features
- Complete MATLAB codebase replaced with Python implementation
- MEX file optimizations replaced with Numba JIT compilation
- MATLAB-style path configuration replaced with Python package structure

### Migration Path
- Transition to Python 3.12 implementation
- Use Numba-decorated functions for performance optimization
- Follow Python package import conventions
- Leverage Python scientific stack (NumPy, SciPy, Pandas, Statsmodels)
- Utilize PyQt6 for GUI components

## [Feature Name] - Deprecated in version X.Y.Z

### Description
Description of the deprecated feature

### Reason
Reason for deprecation

### Alternative
Recommended alternative approach

### Timeline
Will be removed in version X.Y.Z

### References
- Link to relevant documentation
- Link to migration guide

For detailed version history, see CHANGELOG.md
For versioning scheme information, see VERSIONING.md