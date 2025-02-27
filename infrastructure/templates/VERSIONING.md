# MFE Toolbox Versioning

This document describes the versioning scheme, version management practices, and release procedures for the MFE (MATLAB Financial Econometrics) Toolbox Python implementation.

## 1. Versioning Philosophy

The MFE Toolbox follows [Semantic Versioning](https://semver.org/) principles to provide clear and consistent version information that communicates compatibility and changes to users. As a Python reimplementation of a legacy MATLAB toolbox, our versioning approach also acknowledges the history of the original implementation.

## 2. Semantic Versioning Scheme

The version number format is:

```
MAJOR.MINOR.PATCH
```

Where:
- **MAJOR**: Incremented for incompatible API changes
- **MINOR**: Incremented for backward-compatible functionality additions
- **PATCH**: Incremented for backward-compatible bug fixes

## 3. Version Format

The current version is maintained in two locations:
- `__version__` in `src/backend/__init__.py`
- `version` in `pyproject.toml`

Both must be synchronized when making a release.

### 3.1 Current Version

The current version is **4.0.0**.

This version represents the first complete Python reimplementation of the toolbox, which was previously at version 4.0 in MATLAB.

### 3.2 Python Compatibility

MFE Toolbox requires Python 3.12 or higher due to the use of modern Python features including async/await patterns and strict type hints.

## 4. Version Management

### 4.1 Version Validation

The VersionInfo class provides functionality to validate version compatibility:

```python
from mfe.versioning import VersionInfo

version_info = VersionInfo(version="4.0.0", python_version=">=3.12")
is_compatible = version_info.check_compatibility()
```

### 4.2 Version Formatting

For consistent version display, the `format_version_number` function ensures proper formatting:

```python
from mfe.versioning import format_version_number

formatted_version = format_version_number("4.0.0")
# Returns: "4.0.0"
```

## 5. Release Procedures

### 5.1 Preparing a Release

1. Update version numbers in:
   - `src/backend/__init__.py`
   - `pyproject.toml`

2. Update changelog with notable changes

3. Create a new git tag:
   ```
   git tag -a v4.0.0 -m "Version 4.0.0 release"
   ```

### 5.2 Building a Release

1. Clean the build directory:
   ```
   rm -rf dist/ build/ *.egg-info
   ```

2. Build the distribution packages:
   ```
   python -m build
   ```

   This creates both source distribution (.tar.gz) and wheel (.whl) packages.

### 5.3 Publishing a Release

1. Upload to PyPI:
   ```
   python -m twine upload dist/*
   ```

2. Push the release tag:
   ```
   git push origin v4.0.0
   ```

3. Create a release on GitHub with release notes

## 6. Version History Management

### 6.1 Versioning Branches

- `main`: Contains the latest stable release
- `develop`: Active development branch
- `feature/*`: For new features
- `release/*`: For preparing releases

### 6.2 Dependency Management

When upgrading dependencies, follow these guidelines:

- Require minimum versions that provide necessary functionality
- Specify minimum Python version with `>=3.12`
- Maintain a table of verified compatible dependency versions

## 7. Technical Implementation

### 7.1 Version Formatting

```python
def format_version_number(version: str) -> str:
    """
    Formats version numbers according to semantic versioning specification.
    
    Parameters
    ----------
    version : str
        Version string to format
        
    Returns
    -------
    str
        Formatted version string following MAJOR.MINOR.PATCH format
        
    Examples
    --------
    >>> format_version_number("4.0")
    "4.0.0"
    >>> format_version_number("4.0.0")
    "4.0.0"
    >>> format_version_number("4")
    "4.0.0"
    """
    components = version.split('.')
    
    # Ensure we have three components (major, minor, patch)
    while len(components) < 3:
        components.append('0')
        
    # Validate each component is a valid integer
    for component in components:
        try:
            int(component)
        except ValueError:
            raise ValueError(f"Version component '{component}' is not a valid integer")
    
    # Return formatted version
    return '.'.join(components[:3])
```

### 7.2 Version Compatibility Validation

```python
def validate_version_compatibility(version: str, python_version: str) -> bool:
    """
    Validates version compatibility with Python requirements.
    
    Parameters
    ----------
    version : str
        Version string to validate
    python_version : str
        Python version requirement string (e.g., ">=3.12")
        
    Returns
    -------
    bool
        True if version is compatible with Python requirements
        
    Examples
    --------
    >>> validate_version_compatibility("4.0.0", ">=3.12")
    True
    """
    import re
    from packaging import version as packaging_version
    
    # Format version
    formatted_version = format_version_number(version)
    
    # Extract Python version requirement operator and version
    match = re.match(r'([<>=!]+)?(.*)', python_version)
    if not match:
        return False
        
    operator, required_version = match.groups()
    
    if not operator:
        # If no operator, assume exact match
        return packaging_version.parse(formatted_version) == packaging_version.parse(required_version)
    
    # Get current Python version
    import sys
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Check Python version compatibility
    if operator == ">=":
        return packaging_version.parse(current_python) >= packaging_version.parse(required_version)
    elif operator == ">":
        return packaging_version.parse(current_python) > packaging_version.parse(required_version)
    elif operator == "==":
        return packaging_version.parse(current_python) == packaging_version.parse(required_version)
    elif operator == "<=":
        return packaging_version.parse(current_python) <= packaging_version.parse(required_version)
    elif operator == "<":
        return packaging_version.parse(current_python) < packaging_version.parse(required_version)
    else:
        # For complex version specifiers, use packaging
        from packaging import specifiers
        spec = specifiers.SpecifierSet(python_version)
        return spec.contains(current_python)
```

### 7.3 Version Information Class

```python
class VersionInfo:
    """
    Class for managing version information and compatibility.
    
    Parameters
    ----------
    version : str
        Version string
    python_version : str
        Python version requirement string (e.g., ">=3.12")
        
    Attributes
    ----------
    version : str
        Formatted version string
    python_version : str
        Python version requirement
    dependencies : dict
        Dictionary of dependencies and their versions
        
    Methods
    -------
    check_compatibility()
        Check version compatibility with Python and dependencies
        
    Examples
    --------
    >>> version_info = VersionInfo("4.0.0", ">=3.12")
    >>> version_info.check_compatibility()
    True
    """
    def __init__(self, version: str, python_version: str):
        """
        Initialize version information.
        
        Parameters
        ----------
        version : str
            Version string
        python_version : str
            Python version requirement string
        """
        self.version = format_version_number(version)
        self.python_version = python_version
        self.dependencies = {
            "numpy": ">=1.26.3",
            "scipy": ">=1.11.4",
            "pandas": ">=2.1.4",
            "statsmodels": ">=0.14.1",
            "numba": ">=0.59.0",
            "pyqt6": ">=6.6.1"
        }
        
    def check_compatibility(self) -> bool:
        """
        Check version compatibility with Python and dependencies.
        
        Returns
        -------
        bool
            True if version is compatible with requirements
        """
        # Validate version format
        try:
            formatted_version = format_version_number(self.version)
        except ValueError:
            return False
            
        # Check Python compatibility
        if not validate_version_compatibility(formatted_version, self.python_version):
            return False
            
        # Verify dependency versions
        try:
            import importlib.metadata as metadata
        except ImportError:
            # Fallback for Python <3.8
            try:
                import importlib_metadata as metadata
            except ImportError:
                # Skip dependency checks if metadata is unavailable
                return True
                
        for package, version_req in self.dependencies.items():
            try:
                installed_version = metadata.version(package)
                from packaging import specifiers
                spec = specifiers.SpecifierSet(version_req)
                if not spec.contains(installed_version):
                    return False
            except (metadata.PackageNotFoundError, ImportError):
                # Dependency not installed
                return False
                
        return True
```