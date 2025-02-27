# Contributing to MFE Toolbox

Thank you for your interest in contributing to the MFE Toolbox project! This document provides guidelines and instructions for contributing to ensure consistent and high-quality code that meets the project's technical standards.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Standards](#code-standards)
4. [Testing Requirements](#testing-requirements)
5. [Submission Process](#submission-process)

## Getting Started

### Prerequisites

- Python 3.12 or later
- Git
- Understanding of financial econometrics concepts (for substantial contributions)

### Setup Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/mfe-toolbox.git
   cd mfe-toolbox
   ```

3. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Development Dependencies

The project requires these key development tools:

- **pytest** (7.4.3+): For unit and integration testing
- **black** (23.11.0+): For code formatting
- **mypy** (1.7.0+): For static type checking
- **flake8**: For code linting
- **isort**: For import sorting
- **pytest-cov**: For test coverage reports
- **pytest-asyncio**: For testing async functions
- **pytest-benchmark**: For performance testing
- **hypothesis**: For property-based testing

## Development Workflow

### Git Workflow

We follow a standard GitHub Flow workflow:

1. Create a new branch from `main` for each feature or bugfix
2. Make focused, incremental changes with clear commit messages
3. Submit a pull request for review
4. Address review feedback and update your PR
5. Once approved, your changes will be merged into `main`

### Branching Strategy

- **main**: Main production branch, always stable
- **feature/XXX**: For new features
- **bugfix/XXX**: For bug fixes
- **hotfix/XXX**: For critical fixes to production
- **docs/XXX**: For documentation changes

### Commit Message Format

Follow this format for clear and consistent commit messages:

```
[component]: Short summary (max 50 chars)

More detailed explanation, if necessary. Wrap text at
approximately 72 characters. Use paragraphs as needed.

- Bullet points are also acceptable
- Typically a hyphen or asterisk is used for the bullets

Closes #123, #456
```

Components can be:
- `core`: Core statistical modules
- `models`: Time series & volatility models
- `ui`: User interface components
- `utils`: Utility functions
- `docs`: Documentation changes
- `tests`: Test-related changes
- `build`: Build system changes

### Code Review Process

1. Pull requests require review from at least one project maintainer
2. Automated checks must pass (tests, linting, type checking)
3. Reviewers may request changes before merging
4. Focus on constructive feedback and discussion

## Code Standards

The MFE Toolbox adheres to strict code quality standards. All contributions must:

### Python Standards

- Use **Python 3.12+ features** appropriately
- Follow the [MFE Toolbox Style Guide](STYLE_GUIDE.md) for detailed formatting guidelines
- Conform to **PEP 8** style guide with Black formatter
- Use Black with default settings (88 character line length)

### Type Hints

- Use **strict type hints** for all function parameters and return values
- Add type information to class attributes using annotations or dataclasses
- Employ the `typing` module for complex types
- Ensure type annotations are validated with mypy

Example:
```python
from typing import Optional, List, Dict, Any
import numpy as np

def calculate_volatility(returns: np.ndarray, 
                        window: int = 20, 
                        annualize: bool = True) -> np.ndarray:
    """
    Calculate rolling volatility of returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns
    window : int, optional
        Rolling window size, default 20
    annualize : bool, optional
        Whether to annualize the result, default True
        
    Returns
    -------
    np.ndarray
        Rolling volatility estimates
    """
    # Implementation
    pass
```

### Documentation

- Use **NumPy/SciPy-style docstrings** for all public functions and classes
- Document parameters, return values, and exceptions
- Include examples for complex functions
- Add module-level docstrings explaining purpose and usage
- Keep docstrings updated when changing code

### Numba Optimization

- Use `@jit` decorator for performance-critical numerical functions
- Prefer `nopython=True` mode for maximum performance
- Ensure type stability within JIT-compiled functions
- Avoid Python objects inside JIT-compiled code

## Testing Requirements

All contributions must include appropriate tests:

### Unit Tests

- Write **pytest**-based unit tests for all new features
- Test each function and class independently
- Include tests for edge cases and error conditions
- Maintain high code coverage (minimum 90% for new code)

Example:
```python
import pytest
import numpy as np
from mfe.models.garch import GARCH

def test_garch_initialization():
    """Test GARCH model initialization with various parameters."""
    # Test default initialization
    model = GARCH(p=1, q=1)
    assert model.p == 1
    assert model.q == 1
    assert not model.fitted
    
    # Test with different orders
    model = GARCH(p=2, q=3)
    assert model.p == 2
    assert model.q == 3

def test_garch_invalid_parameters():
    """Test GARCH model validation of invalid parameters."""
    # Test negative order parameters
    with pytest.raises(ValueError, match="GARCH orders must be non-negative"):
        GARCH(p=-1, q=1)
```

### Asynchronous Testing

- Use **pytest-asyncio** for testing async functions
- Verify progress reporting and cancellation
- Test non-blocking behavior
- Ensure proper error handling in async context

Example:
```python
import pytest
import asyncio
import numpy as np

@pytest.mark.asyncio
async def test_async_estimation():
    """Test asynchronous model estimation."""
    # Implementation
    pass
```

### Performance Testing

- Implement benchmarks for performance-critical functions
- Compare against baseline implementations
- Verify optimization effectiveness
- Use **pytest-benchmark** for standardized benchmarking

### Test Running

Run tests with:
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=mfe

# Run with benchmark tests
pytest --benchmark-only

# Run specific test file
pytest tests/test_models/test_garch.py
```

## Submission Process

### Pull Request Guidelines

1. Ensure your code follows all standards and passes all tests
2. Create a clearly labeled pull request with a descriptive title
3. Fill in the PR template with:
   - Summary of changes
   - Related issue numbers
   - Testing performed
   - Documentation updates

### Pull Request Template

```
## Description
A clear and concise description of the changes implemented in this PR.

## Related Issues
Fixes #Issue_Number

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Testing
Describe the tests you've added/modified and how to run them.

## Checklist
- [ ] My code follows the style guidelines of the project
- [ ] I have performed a self-review of my own code
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally with my changes
- [ ] I have updated the documentation accordingly
```

### Code Review Expectations

During code review, expect feedback on:
- Code correctness and accuracy
- Performance and efficiency
- Type safety and error handling
- Test coverage and quality
- Documentation clarity and completeness
- Adherence to project standards and style

### Contribution Acceptance Criteria

Contributions will be accepted when they:
1. Pass all automated tests and checks
2. Meet code quality standards
3. Include appropriate documentation
4. Have sufficient test coverage
5. Receive approval from project maintainers

## Questions?

If you have questions or need help, please:
- Open an issue on GitHub
- Contact the project maintainers

Thank you for contributing to the MFE Toolbox project!