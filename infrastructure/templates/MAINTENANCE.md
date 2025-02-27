# MFE Toolbox Maintenance Guide

This document provides comprehensive maintenance procedures and guidelines for the MFE Toolbox Python package. It covers package updates, dependency management, testing, documentation, version control, and error recovery procedures to ensure the ongoing stability and reliability of the system.

## Table of Contents

1. [Package Updates](#package-updates)
2. [Dependency Management](#dependency-management)
3. [Testing Procedures](#testing-procedures)
4. [Documentation Updates](#documentation-updates)
5. [Version Management](#version-management)
6. [Error Recovery](#error-recovery)

## Package Updates

### Regular Release Cycle

The MFE Toolbox follows a structured release cycle to ensure stability and incorporate improvements:

1. **Minor Updates (x.y.Z)**: Bug fixes and non-breaking improvements
   - Typically released monthly or as needed
   - Minimal risk to existing functionality
   - Backward compatible

2. **Feature Updates (x.Y.z)**: New functionality and enhancements
   - Released quarterly
   - Requires comprehensive testing
   - Maintains backward compatibility where possible

3. **Major Updates (X.y.z)**: Significant architectural changes
   - Released annually
   - May include breaking changes (with documented migration paths)
   - Requires complete test suite execution

### Update Process

To update the MFE Toolbox package:

1. Update the local repository:
   ```bash
   git pull origin main
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the test suite to verify functionality:
   ```bash
   pytest
   ```

4. Build the updated package:
   ```bash
   python -m build
   ```

5. Upload to PyPI (maintainers only):
   ```bash
   python -m twine upload dist/*
   ```

### Hotfix Procedure

For critical issues requiring immediate fixes:

1. Create a hotfix branch from the main branch:
   ```bash
   git checkout -b hotfix/issue-description
   ```

2. Implement the minimal necessary fix
3. Update version number in `__init__.py` and `pyproject.toml`
4. Run targeted tests to verify the fix
5. Merge back to main after review
6. Create a patch release

## Dependency Management

### Core Dependencies

The MFE Toolbox relies on the following core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12+ | Runtime environment |
| NumPy | 1.26.3+ | Matrix operations |
| SciPy | 1.11.4+ | Statistical functions |
| Pandas | 2.1.4+ | Time series handling |
| Statsmodels | 0.14.1+ | Econometric modeling |
| Numba | 0.59.0+ | Performance optimization |
| PyQt6 | 6.6.1+ | GUI components |

### Managing Dependencies

#### 1. Dependency Specification

Dependencies are specified in multiple locations:

- `pyproject.toml`: Primary specification for build and runtime dependencies
- `requirements.txt`: Explicit pinned requirements for reproducible environments
- `requirements-dev.txt`: Development-specific requirements

#### 2. Virtual Environment Setup

Always use virtual environments for development and testing:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### 3. Dependency Updates

To safely update dependencies:

1. Create a dedicated branch for dependency updates
2. Update version requirements in `pyproject.toml`
3. Update pinned versions in `requirements.txt`
4. Generate a new `requirements.txt` if needed:
   ```bash
   pip-compile pyproject.toml --output-file requirements.txt
   ```
5. Run the full test suite to verify compatibility
6. Document any breaking changes or required code adjustments

#### 4. Dependency Conflicts

If dependency conflicts arise:

1. Identify the conflicting packages using `pip check`
2. Consult each package's changelog for compatibility information
3. Test specific version combinations to find compatible versions
4. If necessary, fork and patch dependencies with critical conflicts
5. Document workarounds in the project wiki

## Testing Procedures

### Test Suite Overview

The MFE Toolbox employs a comprehensive testing strategy using pytest:

1. **Unit Tests**: Testing individual functions and classes
2. **Integration Tests**: Verifying cross-module interactions
3. **Statistical Tests**: Validating statistical properties using hypothesis
4. **Performance Tests**: Ensuring computational efficiency

### Running Tests

#### Basic Test Execution

```bash
# Run the complete test suite
pytest

# Run tests with coverage reporting
pytest --cov=mfe

# Run tests for a specific module
pytest tests/test_models/test_garch.py

# Run a specific test
pytest tests/test_core/test_distributions.py::test_ged_pdf
```

#### Performance Testing

```bash
# Run performance benchmarks
pytest --benchmark-only

# Compare against baseline performance
pytest-benchmark compare
```

### Continuous Integration

The MFE Toolbox uses GitHub Actions for continuous integration:

1. **Pull Request Checks**: Automatically triggered on PRs
   - Runs the complete test suite
   - Validates code style and type hints
   - Generates coverage reports

2. **Nightly Builds**:
   - Run extended test suite
   - Verify compatibility with latest dependency versions
   - Generate performance benchmarks

### Test Maintenance

To maintain the test suite:

1. **Add Tests for New Features**: All new functionality must include tests
2. **Update Existing Tests**: When modifying existing functionality
3. **Test Data Management**: Store test data in `tests/data/`
4. **Test Configuration**: Maintain pytest configuration in `pytest.ini`

### Coverage Requirements

- Minimum code coverage requirement: 90%
- Critical modules require 95% coverage:
  - `mfe.core.distributions`
  - `mfe.models.garch`
  - `mfe.models.timeseries`

Monitor coverage with:
```bash
pytest --cov=mfe --cov-report=html
```

## Documentation Updates

### Documentation Structure

The MFE Toolbox documentation includes:

1. **API Reference**: Generated from docstrings
2. **User Guides**: Tutorials and examples
3. **Developer Guides**: Implementation details
4. **Maintenance Guides**: This document and related procedures

### Updating Documentation

#### 1. Docstring Updates

- All public functions, classes, and methods must have docstrings
- Follow the NumPy docstring format
- Include type annotations
- Provide examples for complex functionality

Example:

```python
def compute_standard_errors(params: np.ndarray, 
                           data: np.ndarray, 
                           model_type_id: int) -> np.ndarray:
    """
    Computes parameter standard errors using numerical Hessian approximation.
    
    This function calculates the standard errors of model parameters by 
    numerically approximating the Hessian matrix of the log-likelihood function
    and deriving the parameter covariance matrix from its inverse.
    
    Parameters
    ----------
    params : np.ndarray
        Optimized parameter values
    data : np.ndarray
        Time series data used for model estimation
    model_type_id : int
        Integer ID for model type
    
    Returns
    -------
    np.ndarray
        Standard errors for model parameters
        
    Examples
    --------
    >>> params = np.array([0.01, 0.1, 0.8])
    >>> data = np.random.randn(1000)
    >>> model_type_id = 0  # GARCH model
    >>> std_errors = compute_standard_errors(params, data, model_type_id)
    """
    # Implementation...
```

#### 2. Building Documentation

Documentation is built using Sphinx:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html
```

#### 3. Documentation Review Process

Before merging documentation changes:

1. Build the documentation locally
2. Verify all links work correctly
3. Check cross-references between documents
4. Ensure examples execute without errors
5. Review formatting consistency

### Documentation Deployment

Documentation is automatically deployed when:

1. Changes are merged to the main branch
2. A new release is created
3. Manual deployment is triggered via GitHub Actions

## Version Management

### Version Numbering Scheme

The MFE Toolbox follows Semantic Versioning (SemVer) with format X.Y.Z:

- **X (Major)**: Incompatible API changes
- **Y (Minor)**: Backwards-compatible functionality
- **Z (Patch)**: Backwards-compatible bug fixes

### Version Location

Version information is maintained in multiple locations:

1. **Primary Source**: `__version__` in `mfe/__init__.py`
2. **Build Configuration**: `version` field in `pyproject.toml`
3. **Documentation**: Version in `docs/conf.py`

Sample version specification in `__init__.py`:
```python
"""
MFE Toolbox: Financial Econometrics in Python
"""

__version__ = "4.0.0"
```

### Version Update Process

To update the version:

1. Update the version string in `mfe/__init__.py`
2. Update corresponding version in `pyproject.toml`
3. Update documentation version if needed
4. Commit the version changes with message "Bump version to X.Y.Z"
5. Create a git tag for the release:
   ```bash
   git tag -a vX.Y.Z -m "Version X.Y.Z"
   git push origin vX.Y.Z
   ```

### Release Documentation

For each release, create comprehensive release notes in:

1. **CHANGELOG.md**: Maintain a chronological history of changes
2. **GitHub Releases**: Create a new release with detailed notes
3. **Documentation**: Update the "What's New" section

Format for CHANGELOG.md entries:
```markdown
## [4.0.1] - 2023-07-15

### Added
- New simulation options for GARCH models

### Fixed
- Corrected numerical precision issue in EGARCH likelihood calculation
- Fixed threading issues in async optimization

### Changed
- Improved error messages for invalid parameters
- Updated NumPy dependency to 1.26.3
```

## Error Recovery

### Common Error Scenarios

#### 1. Package Import Errors

Errors related to missing or incompatible dependencies:

```python
try:
    import numba
except ImportError:
    print("Error: Numba not found. Install with: pip install numba>=0.59.0")
    # Fall back to non-optimized version if possible
```

Recovery steps:
1. Verify all dependencies are installed with correct versions
2. Check for conflicting packages with `pip check`
3. Reinstall the package in a clean virtual environment

#### 2. Numerical Computation Errors

Handle numerical issues gracefully:

```python
try:
    result = optimizer.optimize(data, initial_params)
except np.linalg.LinAlgError:
    logger.warning("Optimization failed due to numerical issues. Trying alternative method.")
    result = optimizer.optimize_robust(data, initial_params)
```

Recovery steps:
1. Check input data for extreme values or non-stationarity
2. Try alternative initialization methods
3. Use more robust optimization techniques

#### 3. Path Configuration Errors

Ensure proper file path handling:

```python
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    logger.error(f"Configuration file not found at {config_path}")
    # Use default configuration
    config = default_config
```

Recovery steps:
1. Verify path configuration in environment
2. Check file permissions
3. Fall back to default paths if necessary

### Exception Handling Strategy

Follow these guidelines for robust error handling:

1. **Specific Exceptions**: Catch specific exceptions rather than generic ones
2. **Hierarchical Recovery**: Try specific recovery methods before generic ones
3. **Logging**: Always log exceptions with context and traceback
4. **User Feedback**: Provide clear error messages for end users

Example of proper exception handling:

```python
import logging
logger = logging.getLogger(__name__)

def estimate_model(data, params):
    try:
        # Try primary estimation method
        result = _estimate_implementation(data, params)
        return result
    except ValueError as e:
        # Handle specific value errors
        logger.warning(f"Value error in estimation: {str(e)}")
        if "non-finite" in str(e):
            # Try cleaning the data
            clean_data = preprocess_data(data)
            return _estimate_implementation(clean_data, params)
        raise
    except np.linalg.LinAlgError as e:
        # Handle numerical issues
        logger.warning(f"Linear algebra error: {str(e)}")
        # Try more robust algorithm
        return _estimate_robust(data, params)
    except Exception as e:
        # Last resort error handling
        logger.error(f"Unexpected error in model estimation: {str(e)}", exc_info=True)
        raise RuntimeError(f"Model estimation failed: {str(e)}") from e
```

### Recovery Procedures

For severe system-level issues:

1. **Package Reinstallation**:
   ```bash
   pip uninstall mfe
   pip install mfe
   ```

2. **Clean Installation**:
   ```bash
   pip uninstall -y mfe
   pip cache purge
   pip install mfe
   ```

3. **Environment Recreation**:
   ```bash
   deactivate  # Exit current environment
   rm -rf .venv  # Remove existing environment
   python -m venv .venv  # Create new environment
   source .venv/bin/activate
   pip install mfe
   ```

4. **Configuration Reset**:
   ```python
   from mfe import reset_configuration
   reset_configuration()  # Restore default settings
   ```

### Path Validation

To validate and fix Python path configuration:

```python
import sys
import os
import mfe

def validate_mfe_paths():
    """Validate and print MFE package paths"""
    print(f"MFE package location: {os.path.dirname(mfe.__file__)}")
    print(f"Python path: {sys.path}")
    
    # Check if MFE modules are importable
    try:
        from mfe.core import distributions
        from mfe.models import garch
        print("Core modules successfully imported")
    except ImportError as e:
        print(f"Error importing modules: {str(e)}")
        print("Try adding the package directory to your Python path:")
        print("    import sys; sys.path.append('/path/to/mfe')")
```

## Additional Resources

- [GitHub Repository](https://github.com/username/mfe-toolbox)
- [Bug Tracker](https://github.com/username/mfe-toolbox/issues)
- [Online Documentation](https://mfe-toolbox.readthedocs.io/)
- [Community Forum](https://community.mfe-toolbox.org)

## License

The MFE Toolbox is licensed under the MIT License. See the LICENSE file for details.