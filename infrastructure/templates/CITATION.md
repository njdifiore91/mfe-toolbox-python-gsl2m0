# MFE Toolbox Citation Guidelines

## Overview

The MFE (MATLAB Financial Econometrics) Toolbox has been reimplemented in Python 3.12, maintaining its legacy version 4.0 identity while incorporating modern Python programming constructs. This document provides standardized formats for citing the MFE Toolbox in academic papers, technical reports, and other publications.

## Academic Citation Formats

### APA Style (7th Edition)
```
Sheppard, K., et al. (2024). MFE Toolbox: Python Financial Econometrics Library (Version 4.0) [Software]. University of Oxford. https://github.com/organization/mfe-toolbox
```

### MLA Style (9th Edition)
```
Sheppard, Kevin, et al. MFE Toolbox: Python Financial Econometrics Library, version 4.0, University of Oxford, 2024, https://github.com/organization/mfe-toolbox.
```

### Chicago Style (17th Edition)
```
Sheppard, Kevin, et al. 2024. "MFE Toolbox: Python Financial Econometrics Library." Version 4.0. University of Oxford. https://github.com/organization/mfe-toolbox.
```

### BibTeX Entry
```bibtex
@software{mfe_toolbox,
  author       = {Sheppard, Kevin and
                  {Contributors}},
  title        = {MFE Toolbox: Python Financial Econometrics Library},
  version      = {4.0},
  year         = {2024},
  publisher    = {University of Oxford},
  url          = {https://github.com/organization/mfe-toolbox},
  note         = {Python implementation of the MATLAB Financial Econometrics Toolbox}
}
```

## Software Citation Formats

### Python Package Citation
When citing the MFE Toolbox in Python code comments or documentation:

```python
# Citation: MFE Toolbox v4.0 (Python) by Sheppard, K., et al. (2024)
# https://github.com/organization/mfe-toolbox
```

### Requirements File Citation
When including in requirements.txt or similar dependency files:

```
# MFE Toolbox v4.0 (Python) - Financial Econometrics Library
# Citation: Sheppard, K., et al. (2024)
mfe==4.0.0
```

## Version and DOI Information

For reproducibility, please always specify the exact version of the MFE Toolbox used in your research.

- **Current Version:** 4.0.0
- **Release Date:** 2024
- **DOI:** [DOI placeholder]
- **PyPI:** [PyPI link placeholder]

## Author Attribution

The Python implementation of the MFE Toolbox builds upon the original MATLAB version created by Kevin Sheppard. When citing specific methods or models, please attribute the original authors as appropriate.

For detailed information about contributors and their specific contributions, please refer to the [AUTHORS.md](AUTHORS.md) and [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) files in the project repository.

## Citing Specific Functionality

When citing specific functionality within the MFE Toolbox, please use the following format:

```
For the [specific feature/model] implementation, we used the MFE Toolbox v4.0 (Python) by Sheppard, K., et al. (2024).
```

Example:
```
For the GARCH model estimation, we used the MFE Toolbox v4.0 (Python) by Sheppard, K., et al. (2024).
```

## References and Further Reading

For additional information about the methods implemented in the MFE Toolbox, please consult the following references:

1. Sheppard, K. (Year). Original documentation or papers related to MFE Toolbox.
2. [Additional references as needed]

---

*This document was last updated: 2024-XX-XX*