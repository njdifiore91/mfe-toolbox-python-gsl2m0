# Python Requirements

## Runtime Environment
- Python 3.12 or higher
- Support for async/await and type hints
- Virtual environment recommended

## Platform Support
- Windows (x86_64)
- Linux (x86_64)
- macOS (x86_64, arm64)

# Core Dependencies

## Scientific Computing
- NumPy >= 1.26.3
- SciPy >= 1.11.4
- Pandas >= 2.1.4
- Statsmodels >= 0.14.1

## Performance Optimization
- Numba >= 0.59.0

## GUI Framework
- PyQt6 >= 6.6.1

# Optional Dependencies

## Development Tools
- pytest >= 7.4.3
- pytest-asyncio >= 0.21.1
- pytest-cov >= 4.1.0
- mypy >= 1.7.1

## Documentation
- Sphinx >= 7.2.6
- sphinx-rtd-theme >= 1.3.0

# Installation Instructions

## Virtual Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

## Package Installation
```bash
# Install from PyPI
pip install mfe

# Development installation
pip install -e .
```

## Troubleshooting
- Ensure Python 3.12 is installed
- Check for platform-specific dependencies
- Verify NumPy/SciPy compatibility
- Confirm Numba support for your CPU