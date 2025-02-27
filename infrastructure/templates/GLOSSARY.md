# MFE Toolbox Glossary

## Overview

This document provides a comprehensive glossary for the MFE (MATLAB Financial Econometrics) Toolbox, which has been reimplemented in Python 3.12 while maintaining its legacy version 4.0 identity. The glossary includes statistical terms, technical terms, and acronyms related to financial econometrics and the Python implementation.

The MFE Toolbox leverages Python's scientific computing ecosystem, built upon foundational libraries including NumPy for matrix operations, SciPy for optimization and statistical functions, Pandas for time series handling, Statsmodels for econometric modeling, and Numba for performance optimization. This glossary serves as a reference for researchers, analysts, and practitioners using the toolbox.

## Statistical Terms

- **AGARCH/NAGARCH**: Asymmetric/Nonlinear Asymmetric GARCH - Models for capturing leverage effects in volatility that allow for asymmetric responses to positive and negative shocks.

- **APARCH**: Asymmetric Power ARCH - A versatile volatility model with flexible power transformation that can capture both asymmetric effects and long memory in volatility.

- **ARMA**: AutoRegressive Moving Average - Time series model combining autoregressive (AR) and moving average (MA) components to capture complex temporal dependencies.

- **ARMAX**: ARMA with eXogenous variables - Extension of ARMA including external regressors, allowing for explanatory variables to influence the conditional mean of a time series.

- **BEKK**: Baba-Engle-Kraft-Kroner - Multivariate GARCH model ensuring positive definite covariance matrices, suitable for modeling volatility dynamics across multiple assets simultaneously.

- **Block Bootstrap**: Resampling method for dependent data that preserves time series structure by resampling blocks of consecutive observations rather than individual points.

- **CCC**: Constant Conditional Correlation - Multivariate volatility model with constant correlations between assets, simplifying the estimation of multivariate volatility dynamics.

- **DCC**: Dynamic Conditional Correlation - Multivariate volatility model with time-varying correlations, capturing changing relationships between assets over time.

- **EGARCH**: Exponential GARCH - Model for asymmetric volatility response that uses a logarithmic specification to ensure positive variance without parameter constraints.

- **FIGARCH**: Fractionally Integrated GARCH - Long memory volatility model that captures persistence in volatility through fractional integration.

- **GARCH**: Generalized AutoRegressive Conditional Heteroskedasticity - Framework for modeling time-varying volatility in financial time series that captures volatility clustering.

- **GED**: Generalized Error Distribution - A symmetric probability distribution that generalizes the normal distribution with an additional shape parameter to capture fat tails.

- **HAR**: Heterogeneous AutoRegression - Model for capturing long-range dependence in realized volatility by incorporating multiple time scales.

- **IGARCH**: Integrated GARCH - Volatility model with unit root in variance, implying infinite persistence of volatility shocks.

- **MCS**: Model Confidence Set - Statistical procedure for model comparison that identifies a set of models that are statistically indistinguishable from the best model.

- **Newey-West HAC**: Heteroskedasticity and Autocorrelation Consistent estimation method for covariance matrices, robust to both heteroskedasticity and autocorrelation in error terms.

- **SARIMA**: Seasonal ARIMA - Time series model with seasonal components, extending ARIMA to capture regular patterns of fixed periods.

- **Stationary Bootstrap**: Probabilistic block resampling method for dependent data that uses random block lengths to improve the stationarity properties of the resampled series.

- **TARCH**: Threshold ARCH - Model for asymmetric volatility effects that uses a threshold to distinguish between positive and negative shocks.

- **VAR**: Vector AutoRegression - Multivariate time series model that captures linear interdependencies among multiple time series.

## Technical Terms

- **Async/Await**: Python language constructs enabling asynchronous programming for non-blocking operations, used in the MFE Toolbox to handle long-running computations while maintaining UI responsiveness. These patterns allow the execution of time-consuming tasks without freezing the application.

- **Dataclasses**: A Python module introduced in Python 3.7 that provides a decorator and functions to automatically add special methods to classes, used in the MFE Toolbox for cleaner, class-based model parameter containers. Dataclasses reduce boilerplate code while providing strong typing and improved readability.

- **GUI**: Graphical User Interface – implemented in the MFE Toolbox using PyQt6 for a modern, cross-platform experience. The GUI provides interactive access to models, visualizations, and diagnostics without requiring coding knowledge.

- **JIT Compilation**: Just-In-Time compilation, a technique used by Numba to compile Python functions to optimized machine code at runtime, significantly improving computational performance for numerical operations.

- **LaTeX**: Document preparation system used for equation rendering in the MFE Toolbox to display mathematical formulas with proper notation and formatting.

- **Numba**: A Python library for just-in-time (JIT) compilation that accelerates performance-critical functions in the MFE Toolbox, replacing the legacy MATLAB MEX optimizations. Numba translates Python functions to optimized machine code at runtime using the LLVM compiler infrastructure.

- **NumPy**: Core Python library for numerical computing that provides efficient array operations and serves as the foundation for the MFE Toolbox's matrix computations.

- **Pandas**: Python data analysis library used in the MFE Toolbox for time series data structures and manipulation, providing robust capabilities for working with time-indexed financial data.

- **SciPy**: Scientific computing library for Python used in the MFE Toolbox for optimization, statistical functions, and mathematical operations.

- **Statsmodels**: Python library for estimating and testing statistical models, used extensively in the MFE Toolbox for econometric modeling and statistical inference.

- **Type Hints**: Python's annotation system for specifying types, used throughout the MFE Toolbox to enhance code clarity, enable static analysis, and improve maintainability. Type hints specify the expected data types for function parameters and return values.

## Acronyms

- **AIC**: Akaike Information Criterion - Statistical measure used for model selection that balances goodness-of-fit and model complexity.

- **ARCH**: AutoRegressive Conditional Heteroskedasticity - Statistical model for time-varying volatility in time series.

- **BIC**: Bayesian Information Criterion - Statistical measure similar to AIC but with a stronger penalty for model complexity.

- **CDF**: Cumulative Distribution Function - Function describing the probability that a random variable takes a value less than or equal to a specific point.

- **GED**: Generalized Error Distribution - Probability distribution that generalizes the normal distribution to include heavier or lighter tails.

- **GARCH**: Generalized AutoRegressive Conditional Heteroskedasticity - Extension of ARCH model that includes lagged conditional variances.

- **MA**: Moving Average - Time series model component representing a weighted average of past random shocks.

- **MFE**: Financial Econometrics Toolbox re-implemented in Python 3.12 - A comprehensive suite of tools for financial time series analysis and econometric modeling.

- **OLS**: Ordinary Least Squares - Method for estimating parameters in linear regression models.

- **PACF**: Partial AutoCorrelation Function - Measure of the correlation between observations of a time series separated by k time units, controlling for the effects of intermediate observations.

- **PCA**: Principal Component Analysis - Statistical technique for dimensionality reduction and feature extraction.

- **PDF**: Probability Density Function - Function describing the relative likelihood of a continuous random variable taking a specific value.

- **QMLE**: Quasi-Maximum Likelihood Estimation - Parameter estimation method robust to certain forms of misspecification.

- **RARCH**: Rotated ARCH - Variation of ARCH models using rotated innovations for improved modeling of asymmetries.

- **RCC**: Rotated Conditional Correlation - Multivariate volatility model with rotated innovations for more flexible correlation structures.

## Using This Glossary

This glossary is designed to serve as a quick reference for users of the MFE Toolbox. Terms are organized into three categories:

1. **Statistical Terms**: Financial econometrics concepts and methods implemented in the toolbox
2. **Technical Terms**: Python-specific and implementation-related terminology
3. **Acronyms**: Common abbreviations used throughout the documentation and codebase

For more detailed information about specific implementations, please refer to the API documentation and examples available in the [MFE Toolbox Documentation](https://mfe-toolbox.readthedocs.io/).

---

*This document was last updated: 2024*

<!-- 
This file was generated as a template for the MFE Toolbox glossary.
It can be processed with Python functions for formatting glossary entries:

load_glossary_metadata(yaml_path): Loads glossary metadata from YAML
format_glossary_entry(entry_info): Formats entries consistently
generate_glossary_document(glossary_data): Generates complete glossary
-->