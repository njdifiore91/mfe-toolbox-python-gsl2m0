---
name: Bug Report
about: Create a report to help improve the MFE Toolbox
title: '[BUG] '
labels: ['bug']
assignees: []
---

## Description
Provide a clear and concise description of the bug

## Reproduction Steps
Provide detailed steps to reproduce the bug. Include code snippets in pytest format if possible

```python
# Example:
import pytest
from mfe.models import garch

def test_bug_reproduction():
    # Your reproduction code here
    pass
```

## Expected Behavior
Describe what you expected to happen

## Actual Behavior
Describe what actually happened

## Error Type
- [ ] Parameter Error (invalid input, wrong dimensions, etc.)
- [ ] Memory Error (allocation issues, buffer overflow, etc.)
- [ ] Numerical Error (overflow, underflow, precision loss, etc.)
- [ ] Numba Error (compilation failure, type inference issues, etc.)
- [ ] GUI Error (PyQt6-related issues)
- [ ] Other (please specify)

## Error Message/Stack Trace
Paste the full error message or stack trace here

## Python Version
e.g., 3.12.0

## Operating System
e.g., Windows 10, Ubuntu 22.04, macOS 13.0

## Library Versions
NumPy:
SciPy:
Pandas:
Statsmodels:
Numba:
PyQt6:

## Recovery Attempts
Describe any steps you've taken to resolve the issue

## Additional Context
Add any other context about the problem here

## Related Feature Requests
e.g., #123 - Add new GARCH variant