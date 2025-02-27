# MFE Toolbox Bibliography

## Overview

This document provides a comprehensive bibliography for the MFE (MATLAB Financial Econometrics) Toolbox, which has been reimplemented in Python 3.12 while maintaining its legacy version 4.0 identity. The bibliography includes references to statistical methods, econometric papers, and implementation details that form the foundation of the toolbox.

The MFE Toolbox leverages Python's scientific computing ecosystem, built upon foundational libraries including NumPy for matrix operations, SciPy for optimization and statistical functions, Pandas for time series handling, Statsmodels for econometric modeling, and Numba for performance optimization. This comprehensive bibliography serves as a reference for researchers, analysts, and practitioners using the toolbox.

## Citation Guidelines

When citing the MFE Toolbox in academic work, please use the following formats:

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

## Academic References

### Time Series Analysis

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.

2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

3. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag.

4. Tsay, R. S. (2010). *Analysis of Financial Time Series*. John Wiley & Sons.

### GARCH and Volatility Models

5. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

6. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.

7. Engle, R. F., & Kroner, K. F. (1995). Multivariate simultaneous generalized ARCH. *Econometric Theory*, 11(1), 122-150.

8. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance*, 48(5), 1779-1801.

9. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347-370.

### Bootstrap Methods

10. Efron, B., & Tibshirani, R. (1994). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

11. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.

12. Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241.

### High-Frequency Financial Data Analysis

13. Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579-625.

14. Barndorff-Nielsen, O. E., & Shephard, N. (2004). Econometric analysis of realized covariation: High frequency-based covariance, regression, and correlation in financial economics. *Econometrica*, 72(3), 885-925.

15. Hansen, P. R., & Lunde, A. (2006). Realized variance and market microstructure noise. *Journal of Business & Economic Statistics*, 24(2), 127-161.

### Statistical Distributions and Tests

16. Jarque, C. M., & Bera, A. K. (1987). A test for normality of observations and regression residuals. *International Statistical Review*, 55(2), 163-172.

17. Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

18. White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.

## Statistical Methods References

### Time Series Models

- **ARMA/ARMAX Models**: Autoregressive Moving Average models with exogenous variables
  - Implementation: `mfe.models.timeseries.arma_estimator`
  - Key references: [1], [2], [4]
  - Features: Class-based Python modules using SciPy's numerical optimization capabilities

- **Unit Root Testing**: Tests for stationarity in time series
  - Implementation: `mfe.core.tests.unit_root_tests`
  - Key references: [2], [3]
  - Features: Integration with Statsmodels for comprehensive testing

### Volatility Models

- **GARCH Family Models**: Univariate volatility models
  - Implementation: `mfe.models.univariate`
  - Variants: GARCH [5], EGARCH [9], GJR-GARCH [8], APARCH, FIGARCH
  - Key references: [5], [8], [9]
  - Features: Numba-accelerated parameter optimization

- **Multivariate GARCH Models**: Volatility models for multiple time series
  - Implementation: `mfe.models.multivariate`
  - Variants: BEKK [7], CCC, DCC
  - Key references: [7]
  - Features: Efficient matrix operations through NumPy and Numba

### Bootstrap Methods

- **Block Bootstrap**: Resampling method for dependent data
  - Implementation: `mfe.core.bootstrap.block_bootstrap`
  - Key references: [11], [12]
  - Features: Python implementation with Numba optimization

- **Stationary Bootstrap**: Probabilistic block resampling
  - Implementation: `mfe.core.bootstrap.stationary_bootstrap`
  - Key references: [11]
  - Features: Efficient implementation using NumPy and Numba

### High-Frequency Analysis

- **Realized Volatility Measures**: Non-parametric volatility estimation
  - Implementation: `mfe.models.realized.realized_volatility`
  - Key references: [13], [14], [15]
  - Features: SciPy-based kernel methods and Python-native price filtering

- **Microstructure Noise Filters**: Methods for handling market microstructure noise
  - Implementation: `mfe.models.realized.noise_filtering`
  - Key references: [15]
  - Features: Robust estimation techniques leveraging Pandas for time handling

### Statistical Distributions and Tests

- **Distribution Functions**: Advanced statistical distributions
  - Implementation: `mfe.core.distributions`
  - Variants: Generalized Error Distribution (GED), Hansen's Skewed T
  - Key references: [16]
  - Features: Integration with SciPy's statistical functions

- **Diagnostic Tests**: Tests for model adequacy
  - Implementation: `mfe.core.tests`
  - Variants: Jarque-Bera [16], White [18], Newey-West [17]
  - Key references: [16], [17], [18]
  - Features: Comprehensive implementation using Statsmodels

## Implementation References

### Core Python Libraries

- **NumPy** (Version 1.26.3)
  - Purpose: Fundamental package for scientific computing in Python
  - GitHub: [https://github.com/numpy/numpy](https://github.com/numpy/numpy)
  - Documentation: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)
  - Usage in MFE: Primary data structure for matrix operations and numerical computations

- **SciPy** (Version 1.11.4)
  - Purpose: Scientific Python library for mathematics, science, and engineering
  - GitHub: [https://github.com/scipy/scipy](https://github.com/scipy/scipy)
  - Documentation: [https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/)
  - Usage in MFE: Optimization algorithms, statistical functions, and advanced mathematical operations

- **Pandas** (Version 2.1.4)
  - Purpose: Data analysis and manipulation library
  - GitHub: [https://github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas)
  - Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
  - Usage in MFE: Time series data structures and efficient time-based operations

- **Statsmodels** (Version 0.14.1)
  - Purpose: Statistical models, hypothesis tests, and data exploration
  - GitHub: [https://github.com/statsmodels/statsmodels](https://github.com/statsmodels/statsmodels)
  - Documentation: [https://www.statsmodels.org/stable/index.html](https://www.statsmodels.org/stable/index.html)
  - Usage in MFE: Econometric modeling and statistical testing

### Performance Optimization

- **Numba** (Version 0.59.0)
  - Purpose: JIT compiler that translates Python and NumPy code to optimized machine code
  - GitHub: [https://github.com/numba/numba](https://github.com/numba/numba)
  - Documentation: [https://numba.pydata.org/numba-doc/latest/index.html](https://numba.pydata.org/numba-doc/latest/index.html)
  - Usage in MFE: Performance-critical functions decorated with @jit for near-C performance

### User Interface

- **PyQt6** (Version 6.6.1)
  - Purpose: Python bindings for the Qt application framework
  - Documentation: [https://www.riverbankcomputing.com/static/Docs/PyQt6/](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
  - Usage in MFE: Interactive modeling environment and visualization tools

### Development Tools

- **pytest** (Version 7.4.3)
  - Purpose: Testing framework
  - GitHub: [https://github.com/pytest-dev/pytest](https://github.com/pytest-dev/pytest)
  - Documentation: [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/)
  - Usage in MFE: Comprehensive unit and integration testing

- **mypy** (Version 1.7.1)
  - Purpose: Static type checker for Python
  - GitHub: [https://github.com/python/mypy](https://github.com/python/mypy)
  - Documentation: [https://mypy.readthedocs.io/en/stable/](https://mypy.readthedocs.io/en/stable/)
  - Usage in MFE: Type validation and code safety

- **Sphinx** (Version 7.2.6)
  - Purpose: Documentation generator
  - GitHub: [https://github.com/sphinx-doc/sphinx](https://github.com/sphinx-doc/sphinx)
  - Documentation: [https://www.sphinx-doc.org/en/master/](https://www.sphinx-doc.org/en/master/)
  - Usage in MFE: Comprehensive API documentation and examples

## Software Design References

### Python Best Practices

- **Python Enhancement Proposals (PEPs)**
  - PEP 8 - Style Guide for Python Code
  - PEP 484 - Type Hints
  - PEP 526 - Syntax for Variable Annotations
  - PEP 557 - Data Classes
  - PEP 585 - Type Hinting Generics In Standard Collections
  - PEP 604 - Complementary Syntax for Union[]
  - PEP 3119 - Introducing Abstract Base Classes

### Asynchronous Programming

- **Python asyncio**
  - Documentation: [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)
  - Implementation in MFE: Used for non-blocking long-running computations in model estimation
  - Features: Leverages Python's async/await patterns for responsive operations

### Scientific Python Development

- **NumPy Enhancement Proposals (NEPs)**
  - NEP 29 - Recommend Python and NumPy version support as a community policy standard
  - NEP 18 - A dispatch mechanism for NumPy's high level array functions

- **Scientific Python Development Guide**
  - Documentation: [https://scientific-python.org/specs/](https://scientific-python.org/specs/)
  - Relevance: Guidelines for scientific Python package development

## Bibliography Metadata

- **Project Name**: MFE Toolbox
- **Version**: 4.0
- **Original Author**: Kevin Sheppard, University of Oxford
- **Implementation**: Python 3.12 with NumPy, SciPy, Pandas, Statsmodels, and Numba
- **Repository**: [https://github.com/organization/mfe-toolbox](https://github.com/organization/mfe-toolbox)
- **Documentation**: [https://mfe-toolbox.readthedocs.io/](https://mfe-toolbox.readthedocs.io/)

## Bibliography Format Templates

### APA Style
```
{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}.
```

### MLA Style
```
{authors}. "{title}." {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}.
```

### Chicago Style
```
{authors}. "{title}." {journal} {volume}, no. {issue} ({year}): {pages}.
```

---

*This document was last updated: 2024*

<!-- 
This file was generated as a template for the MFE Toolbox bibliography.
It can be processed with Python functions for formatting bibliography entries:

load_bibliography_metadata(yaml_path): Loads bibliography metadata from YAML
format_bibliography_entry(entry_info, style='APA'): Formats entries by style
generate_bibliography_document(bibliography_data): Generates complete bibliography
-->