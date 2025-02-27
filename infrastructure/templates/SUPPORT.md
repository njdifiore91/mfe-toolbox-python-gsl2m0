# MFE Toolbox Support Guide

This document provides guidance on getting help, reporting issues, and accessing resources for the MFE Toolbox version 4.0.

## Overview

The MFE Toolbox is a comprehensive suite of Python modules for financial time series modeling and advanced econometric analysis, implemented in Python 3.12 with modern programming constructs such as async/await patterns and strict type hints.

## Getting Help

### Documentation Resources

- **Installation Guide**: See [INSTALLATION.md](./docs/INSTALLATION.md) for comprehensive installation instructions
- **Usage Guide**: See [USAGE.md](./docs/USAGE.md) for detailed usage examples
- **API Reference**: See [API.md](./docs/API.md) for complete API documentation

### Python Environment Support

The MFE Toolbox requires Python 3.12 or higher and depends on the following packages:

- NumPy (1.26.3 or later)
- SciPy (1.11.4 or later)
- Pandas (2.1.4 or later)
- Statsmodels (0.14.1 or later)
- Numba (0.59.0 or later)
- PyQt6 (6.6.1 or later) - for GUI components

For environment setup assistance:
- Use the official Python documentation at [python.org](https://www.python.org/doc/)
- Consult package-specific documentation:
  - [NumPy Documentation](https://numpy.org/doc/)
  - [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
  - [Pandas Documentation](https://pandas.pydata.org/docs/)
  - [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
  - [Numba Documentation](https://numba.pydata.org/numba-doc/latest/index.html)
  - [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)

## Reporting Issues

If you encounter problems with the MFE Toolbox, please report them through the following channels:

### GitHub Issues

1. Go to the [Issues page](https://github.com/username/mfe-toolbox/issues) on GitHub
2. Click "New Issue"
3. Provide as much detail as possible, including:
   - MFE Toolbox version
   - Python version
   - Operating system
   - Complete error message or screenshot
   - Steps to reproduce the issue
   - Expected vs. actual behavior

### Issue Template

```
### Environment
- MFE Toolbox version: [e.g., 4.0]
- Python version: [e.g., 3.12.0]
- OS: [e.g., Windows 11, macOS 13.5, Ubuntu 22.04]
- Dependencies: [List any relevant package versions]

### Description
[Describe the issue here]

### Steps to Reproduce
1. [First step]
2. [Second step]
3. [And so on...]

### Expected Behavior
[What you expected to happen]

### Actual Behavior
[What actually happened]

### Error Messages
[Include complete error output]
```

## Common Problems and Solutions

### ImportError: No module named 'numba'

**Problem**: Python cannot find the Numba package, which is required for performance-critical operations.

**Solution**:
```bash
pip install numba==0.59.0
```

If that doesn't work, check:
- Your Python version (must be 3.12 or higher)
- Your CPU compatibility with Numba
- Reinstall with verbose output:
  ```bash
  pip install -v numba==0.59.0
  ```

### Performance Issues with Numba JIT Compilation

**Problem**: Performance optimizations aren't working as expected with Numba.

**Solutions**:
1. Check that Numba is properly installed:
   ```python
   import numba
   print(numba.__version__)
   ```

2. Verify that your CPU supports the features Numba uses:
   ```python
   from numba.core.cpu_features import get_cpu_features
   print(get_cpu_features())
   ```

3. Try forcing Numba to recompile cached functions:
   ```bash
   # Remove Numba cache
   rm -rf ~/.numba/cache
   ```

### Dependency Conflicts

**Problem**: Conflicts between package versions.

**Solution**: Use a virtual environment with exact versions:
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/macOS
# Or
fresh_env\Scripts\activate     # Windows

pip install numpy==1.26.3 scipy==1.11.4 pandas==2.1.4 statsmodels==0.14.1 numba==0.59.0 pyqt6==6.6.1
pip install mfe
```

## Platform-Specific Support

### Windows

**Environment Setup**:
- Install Python 3.12 from [python.org](https://www.python.org/downloads/windows/)
- Ensure Python is added to PATH during installation
- Use PowerShell or Command Prompt for package installation

**Common Windows Issues**:
- **Long Path Issues**: Enable long path support in Windows 10/11
- **Permission Errors**: Run Command Prompt/PowerShell as Administrator
- **PyQt6 Installation Failures**: Install Visual C++ Build Tools

### macOS

**Environment Setup**:
- Install Python 3.12 using Homebrew: `brew install python@3.12`
- Alternatively, use the official installer from [python.org](https://www.python.org/downloads/macos/)

**Common macOS Issues**:
- **Apple Silicon (M1/M2) Compatibility**: Use a Python version built for arm64
- **Numba Performance on Apple Silicon**: Ensure using the latest Numba with arm64 support
- **Installation Permission Errors**: Use `pip install --user` or set up a virtual environment

### Linux

**Environment Setup**:
- Most distributions: Use package manager to install Python 3.12
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install python3.12 python3.12-venv python3.12-dev`
- Fedora: `sudo dnf install python3.12 python3.12-devel`

**Common Linux Issues**:
- **Missing Development Headers**: Install -dev/-devel packages
- **GUI Dependencies**: Install required Qt libraries for PyQt6
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libqt6core6 libqt6gui6 libqt6widgets6
  ```

## Package Reinstallation and Recovery

### Reinstallation Procedure

If you need to reinstall the MFE Toolbox:

1. Uninstall the existing package:
   ```bash
   pip uninstall mfe
   ```

2. Clean any residual files (optional):
   ```bash
   # Find site-packages location
   python -c "import site; print(site.getsitepackages())"
   # Remove any remaining MFE files
   ```

3. Reinstall the package:
   ```bash
   pip install mfe
   ```

### Environment Recreation

If your Python environment has become corrupted:

1. Create a fresh virtual environment:
   ```bash
   python -m venv new_env
   ```

2. Activate the new environment:
   ```bash
   # Linux/macOS
   source new_env/bin/activate
   # Windows
   new_env\Scripts\activate
   ```

3. Install the MFE Toolbox and dependencies:
   ```bash
   pip install mfe
   ```

### Dependency Resolution

For dependency conflicts:

1. Install specific versions that are known to work:
   ```bash
   pip install numpy==1.26.3 scipy==1.11.4 pandas==2.1.4 statsmodels==0.14.1 numba==0.59.0 pyqt6==6.6.1
   pip install mfe
   ```

2. Use pip's dependency resolver in a fresh environment:
   ```bash
   pip install --upgrade pip
   pip install --use-feature=2020-resolver mfe
   ```

## Community Resources

- **Scientific Python Community**: 
  - [NumPy Community](https://numpy.org/community/)
  - [SciPy Community](https://scipy.org/community/)
  - [PyData Community](https://pydata.org/)
  
- **Financial Python Resources**:
  - [Python for Finance](https://www.pythonforfinance.net/)
  - [Quantitative Economics with Python](https://quantecon.org/)

- **Econometrics Resources**:
  - [Statsmodels Documentation](https://www.statsmodels.org/)
  - [Econometrics in Python](https://pyecon.org/)

## Contact Information

For additional support, contact:
- Email: support@mfe-toolbox.org
- GitHub: https://github.com/username/mfe-toolbox

---

*This support document is for MFE Toolbox version 4.0, running on Python 3.12.*