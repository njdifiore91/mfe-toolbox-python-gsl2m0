# MFE Toolbox Reference Materials

## 1. Overview

This document provides a comprehensive list of references for the MFE (MATLAB Financial Econometrics) Toolbox version 4.0, which has been reimplemented in Python 3.12. The references are organized into the following categories:

1. **Academic References**: Key papers and books on financial econometrics methods implemented in the toolbox
2. **Statistical Methods References**: Documentation of statistical methods and algorithms used
3. **Implementation References**: Python packages, libraries, and tools used in the implementation
4. **Software Design References**: Resources on software architecture and design patterns

Please cite the MFE Toolbox and relevant references when using this software in academic work.

## 2. Academic References

### 2.1 Time Series Analysis

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.

2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

3. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag.

4. Tsay, R. S. (2010). *Analysis of Financial Time Series*. John Wiley & Sons.

### 2.2 GARCH and Volatility Models

5. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

6. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.

7. Engle, R. F., & Kroner, K. F. (1995). Multivariate simultaneous generalized ARCH. *Econometric Theory*, 11(1), 122-150.

8. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance*, 48(5), 1779-1801.

9. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347-370.

### 2.3 Bootstrap Methods

10. Efron, B., & Tibshirani, R. (1994). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

11. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.

12. Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241.

### 2.4 High-Frequency Financial Data Analysis

13. Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579-625.

14. Barndorff-Nielsen, O. E., & Shephard, N. (2004). Econometric analysis of realized covariation: High frequency-based covariance, regression, and correlation in financial economics. *Econometrica*, 72(3), 885-925.

15. Hansen, P. R., & Lunde, A. (2006). Realized variance and market microstructure noise. *Journal of Business & Economic Statistics*, 24(2), 127-161.

### 2.5 Statistical Distributions and Tests

16. Jarque, C. M., & Bera, A. K. (1987). A test for normality of observations and regression residuals. *International Statistical Review*, 55(2), 163-172.

17. Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

18. White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.

## 3. Statistical Methods References

### 3.1 Time Series Models

- **ARMA/ARMAX Models**: Autoregressive Moving Average models with exogenous variables
  - Implementation: `mfe.models.timeseries.arma_estimator`
  - Key references: [1], [2], [4]

- **Unit Root Testing**: Tests for stationarity in time series
  - Implementation: `mfe.core.tests.unit_root_tests`
  - Key references: [2], [3]

### 3.2 Volatility Models

- **GARCH Family Models**: Univariate volatility models
  - Implementation: `mfe.models.univariate`
  - Variants: GARCH [5], EGARCH [9], GJR-GARCH [8], APARCH, FIGARCH
  - Key references: [5], [8], [9]

- **Multivariate GARCH Models**: Volatility models for multiple time series
  - Implementation: `mfe.models.multivariate`
  - Variants: BEKK [7], CCC, DCC
  - Key references: [7]

### 3.3 Bootstrap Methods

- **Block Bootstrap**: Resampling method for dependent data
  - Implementation: `mfe.core.bootstrap.block_bootstrap`
  - Key references: [11], [12]

- **Stationary Bootstrap**: Probabilistic block resampling
  - Implementation: `mfe.core.bootstrap.stationary_bootstrap`
  - Key references: [11]

### 3.4 High-Frequency Analysis

- **Realized Volatility Measures**: Non-parametric volatility estimation
  - Implementation: `mfe.models.realized.realized_volatility`
  - Key references: [13], [14], [15]

- **Microstructure Noise Filters**: Methods for handling market microstructure noise
  - Implementation: `mfe.models.realized.noise_filtering`
  - Key references: [15]

### 3.5 Statistical Distributions and Tests

- **Distribution Functions**: Advanced statistical distributions
  - Implementation: `mfe.core.distributions`
  - Variants: Generalized Error Distribution (GED), Hansen's Skewed T
  - Key references: [16]

- **Diagnostic Tests**: Tests for model adequacy
  - Implementation: `mfe.core.tests`
  - Variants: Jarque-Bera [16], White [18], Newey-West [17]
  - Key references: [16], [17], [18]

## 4. Implementation References

### 4.1 Core Python Libraries

- **NumPy** (Version 1.26.3)
  - Purpose: Fundamental package for scientific computing in Python
  - GitHub: [https://github.com/numpy/numpy](https://github.com/numpy/numpy)
  - Documentation: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)

- **SciPy** (Version 1.11.4)
  - Purpose: Scientific Python library for mathematics, science, and engineering
  - GitHub: [https://github.com/scipy/scipy](https://github.com/scipy/scipy)
  - Documentation: [https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/)

- **Pandas** (Version 2.1.4)
  - Purpose: Data analysis and manipulation library
  - GitHub: [https://github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas)
  - Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

- **Statsmodels** (Version 0.14.1)
  - Purpose: Statistical models, hypothesis tests, and data exploration
  - GitHub: [https://github.com/statsmodels/statsmodels](https://github.com/statsmodels/statsmodels)
  - Documentation: [https://www.statsmodels.org/stable/index.html](https://www.statsmodels.org/stable/index.html)

### 4.2 Performance Optimization

- **Numba** (Version 0.59.0)
  - Purpose: JIT compiler that translates Python and NumPy code to optimized machine code
  - GitHub: [https://github.com/numba/numba](https://github.com/numba/numba)
  - Documentation: [https://numba.pydata.org/numba-doc/latest/index.html](https://numba.pydata.org/numba-doc/latest/index.html)

### 4.3 User Interface

- **PyQt6** (Version 6.6.1)
  - Purpose: Python bindings for the Qt application framework
  - Documentation: [https://www.riverbankcomputing.com/static/Docs/PyQt6/](https://www.riverbankcomputing.com/static/Docs/PyQt6/)

### 4.4 Development Tools

- **pytest** (Version 7.4.3)
  - Purpose: Testing framework
  - GitHub: [https://github.com/pytest-dev/pytest](https://github.com/pytest-dev/pytest)
  - Documentation: [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/)

- **mypy** (Version 1.7.1)
  - Purpose: Static type checker for Python
  - GitHub: [https://github.com/python/mypy](https://github.com/python/mypy)
  - Documentation: [https://mypy.readthedocs.io/en/stable/](https://mypy.readthedocs.io/en/stable/)

- **Sphinx** (Version 7.2.6)
  - Purpose: Documentation generator
  - GitHub: [https://github.com/sphinx-doc/sphinx](https://github.com/sphinx-doc/sphinx)
  - Documentation: [https://www.sphinx-doc.org/en/master/](https://www.sphinx-doc.org/en/master/)

## 5. Software Design References

### 5.1 Python Best Practices

- **Python Enhancement Proposals (PEPs)**
  - PEP 8 - Style Guide for Python Code
  - PEP 484 - Type Hints
  - PEP 526 - Syntax for Variable Annotations
  - PEP 557 - Data Classes
  - PEP 585 - Type Hinting Generics In Standard Collections
  - PEP 604 - Complementary Syntax for Union[]
  - PEP 3119 - Introducing Abstract Base Classes

### 5.2 Asynchronous Programming

- **Python asyncio**
  - Documentation: [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)
  - Implementation in MFE: Used for non-blocking long-running computations in model estimation

### 5.3 Scientific Python Development

- **NumPy Enhancement Proposals (NEPs)**
  - NEP 29 - Recommend Python and NumPy version support as a community policy standard
  - NEP 18 - A dispatch mechanism for NumPy's high level array functions

- **Scientific Python Development Guide**
  - Documentation: [https://scientific-python.org/specs/](https://scientific-python.org/specs/)

## 6. Bibliography

[1] Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.

[2] Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

[3] Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag.

[4] Tsay, R. S. (2010). *Analysis of Financial Time Series*. John Wiley & Sons.

[5] Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

[6] Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.

[7] Engle, R. F., & Kroner, K. F. (1995). Multivariate simultaneous generalized ARCH. *Econometric Theory*, 11(1), 122-150.

[8] Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance*, 48(5), 1779-1801.

[9] Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347-370.

[10] Efron, B., & Tibshirani, R. (1994). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

[11] Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.

[12] Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241.

[13] Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579-625.

[14] Barndorff-Nielsen, O. E., & Shephard, N. (2004). Econometric analysis of realized covariation: High frequency-based covariance, regression, and correlation in financial economics. *Econometrica*, 72(3), 885-925.

[15] Hansen, P. R., & Lunde, A. (2006). Realized variance and market microstructure noise. *Journal of Business & Economic Statistics*, 24(2), 127-161.

[16] Jarque, C. M., & Bera, A. K. (1987). A test for normality of observations and regression residuals. *International Statistical Review*, 55(2), 163-172.

[17] Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

[18] White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.

---

*This document was last updated: 2024*