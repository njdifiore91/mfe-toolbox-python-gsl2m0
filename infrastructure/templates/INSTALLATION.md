# MFE Toolbox Installation Guide

This document provides comprehensive installation instructions for the MFE Toolbox, a Python-based financial econometrics package.

# System Requirements

## Software Requirements
- Python 3.12 or higher
- pip package manager
- Virtual environment tool (recommended)

## Hardware Requirements
- Memory: Minimum 50MB
- Disk Space: Minimum 50MB
- Processor: Multi-core CPU recommended for parallel processing

## Operating System Support
- Windows (x86_64)
- Linux (x86_64)
- macOS (x86_64, arm64)

# Installation Steps

## Quick Installation
```bash
pip install mfe
```

## Recommended Installation
1. Create virtual environment:
```bash
python -m venv .venv
```

2. Activate virtual environment:
- Windows: `.venv\Scripts\activate`
- Unix/macOS: `source .venv/bin/activate`

3. Install package:
```bash
pip install mfe
```

## Verify Installation
```python
import mfe
print(mfe.__version__)
```

# Platform-Specific Instructions

## Windows
1. Install Python 3.12 from python.org
2. Add Python to PATH
3. Open Command Prompt as Administrator
4. Follow installation steps above

## Linux
1. Install Python 3.12:
```bash
sudo apt-get update  # Ubuntu/Debian
sudo apt-get install python3.12
```
2. Follow installation steps above

## macOS
1. Install Python 3.12 using Homebrew:
```bash
brew install python@3.12
```
2. Follow installation steps above

# Development Setup

## Clone Repository
```bash
git clone https://github.com/username/mfe-toolbox.git
cd mfe-toolbox
```

## Development Installation
1. Create development environment:
```bash
python -m venv .venv
```

2. Activate environment (see above)

3. Install in editable mode:
```bash
pip install -e .
```

## Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

# Troubleshooting

## Common Issues

### ImportError: No module named 'numba'
- Ensure Numba is installed: `pip install numba`
- Verify Python version compatibility

### Performance Issues
- Check Numba JIT compilation status
- Verify hardware compatibility
- Monitor resource usage

### Installation Errors
- Update pip: `pip install --upgrade pip`
- Check Python version compatibility
- Verify system requirements

## Getting Help
- Check documentation
- Submit issue on GitHub
- Contact support team