# APPENDICES

### GLOSSARY

#### Statistical Terms
- **AGARCH/NAGARCH**: Asymmetric/Nonlinear Asymmetric GARCH - Models for capturing leverage effects in volatility
- **APARCH**: Asymmetric Power ARCH - Model with flexible power transformation for volatility
- **ARMA**: AutoRegressive Moving Average - Time series model combining autoregressive and moving average components
- **ARMAX**: ARMA with eXogenous variables - Extension of ARMA including external regressors
- **BEKK**: Baba-Engle-Kraft-Kroner - Multivariate GARCH model ensuring positive definite covariance matrices
- **Block Bootstrap**: Resampling method for dependent data that preserves time series structure
- **CCC**: Constant Conditional Correlation - Multivariate volatility model with constant correlations
- **DCC**: Dynamic Conditional Correlation - Multivariate volatility model with time-varying correlations
- **EGARCH**: Exponential GARCH - Model for asymmetric volatility response
- **FIGARCH**: Fractionally Integrated GARCH - Long memory volatility model
- **HAR**: Heterogeneous AutoRegression - Model for capturing long-range dependence
- **IGARCH**: Integrated GARCH - Volatility model with unit root in variance
- **MCS**: Model Confidence Set - Statistical procedure for model comparison
- **SARIMA**: Seasonal ARIMA - Time series model with seasonal components
- **Stationary Bootstrap**: Probabilistic block resampling method for dependent data
- **TARCH**: Threshold ARCH - Model for asymmetric volatility effects
- **VAR**: Vector AutoRegression - Multivariate time series model

#### Technical Terms
- **Numba**: A Python library for just-in-time (JIT) compilation that accelerates performance-critical functions, replacing the legacy MATLAB MEX optimizations
- **GUI**: Graphical User Interface – now implemented using PyQt6 for a modern, cross-platform experience
- **LaTeX**: Document preparation system used for equation rendering
- **Newey-West HAC**: Heteroskedasticity and Autocorrelation Consistent estimation method
- **Async/Await**: Python language constructs enabling asynchronous programming for non-blocking operations, used to handle long-running computations
- **Dataclasses**: A Python module that provides a decorator and functions to automatically add special methods to classes, used for cleaner, class-based model parameter containers
- **Typing**: Python's standard library module to enforce strict type hints, enhancing code clarity, static analysis, and maintainability

### ACRONYMS

- **AIC**: Akaike Information Criterion
- **ARCH**: AutoRegressive Conditional Heteroskedasticity
- **BIC**: Bayesian Information Criterion
- **CDF**: Cumulative Distribution Function
- **GED**: Generalized Error Distribution
- **GARCH**: Generalized AutoRegressive Conditional Heteroskedasticity
- **MA**: Moving Average
- **MFE**: Financial Econometrics Toolbox re-implemented in Python 3.12
- **OLS**: Ordinary Least Squares
- **PACF**: Partial AutoCorrelation Function
- **PCA**: Principal Component Analysis
- **PDF**: Probability Density Function
- **QMLE**: Quasi-Maximum Likelihood Estimation
- **RARCH**: Rotated ARCH
- **RCC**: Rotated Conditional Correlation

### REFERENCES

#### Academic References
1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
3. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag.
4. Tsay, R. S. (2010). *Analysis of Financial Time Series*. John Wiley & Sons.
5. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
6. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
7. Engle, R. F., & Kroner, K. F. (1995). Multivariate simultaneous generalized ARCH. *Econometric Theory*, 11(1), 122-150.
8. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance*, 48(5), 1779-1801.
9. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347-370.
10. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.

#### Core Python Library References
- **NumPy** (Version 1.26.3): Fundamental package for scientific computing in Python
- **SciPy** (Version 1.11.4): Scientific Python library for mathematics, science, and engineering
- **Pandas** (Version 2.1.4): Data analysis and manipulation library
- **Statsmodels** (Version 0.14.1): Statistical models, hypothesis tests, and data exploration
- **Numba** (Version 0.59.0): JIT compiler that translates Python and NumPy code to optimized machine code
- **PyQt6** (Version 6.6.1): Python bindings for the Qt application framework

### CITATION GUIDELINES

When citing the MFE Toolbox in academic work, please use the following formats:

#### APA Style (7th Edition)
```
Sheppard, K., et al. (2024). MFE Toolbox: Python Financial Econometrics Library (Version 4.0) [Software]. University of Oxford. https://github.com/organization/mfe-toolbox
```

#### BibTeX Entry
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

### ACKNOWLEDGMENTS

The Python 3.12 reimplementation of the MFE Toolbox builds upon the original MATLAB version created by Kevin Sheppard. Special thanks to:

- The **NumPy**, **SciPy**, **Pandas**, and **Statsmodels** communities for their foundational work in scientific computing with Python
- The **Numba** team for enabling high-performance computing through just-in-time compilation
- The **PyQt6** developers for providing a robust cross-platform GUI framework
- All users who provided feedback, reported issues, and suggested improvements

### CONTRIBUTORS

#### Original MATLAB Version
- **Kevin Sheppard** - University of Oxford, Department of Economics (Original creator and main developer)

#### Python Reimplementation
- Python reimplementation team members