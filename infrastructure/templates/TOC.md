# MFE Toolbox Documentation

## Table of Contents

### 1. Introduction
- [Overview](#overview)
- [Features](#features)
- [Version History](#version-history)
- [Python Implementation](#python-implementation)
- [License](#license)

### 2. Getting Started
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Basic Workflow](#basic-workflow)
- [Python Environment Setup](#python-environment-setup)
- [Package Organization](#package-organization)

### 3. Core Statistical Modules (`mfe.core`)
- [Bootstrap Module](#bootstrap-module)
  - [Block Bootstrap](#block-bootstrap)
  - [Stationary Bootstrap](#stationary-bootstrap)
- [Cross-sectional Tools](#cross-sectional-tools)
  - [Regression Analysis](#regression-analysis)
  - [Principal Component Analysis](#principal-component-analysis)
- [Distribution Module](#distribution-module)
  - [Generalized Error Distribution (GED)](#generalized-error-distribution)
  - [Hansen's Skewed T](#hansens-skewed-t-distribution)
- [Statistical Testing](#statistical-testing)
  - [Unit Root Tests](#unit-root-tests)
  - [Normality Tests](#normality-tests)
  - [Specification Tests](#specification-tests)

### 4. Time Series & Volatility Modules (`mfe.models`)
- [Time Series Models](#time-series-models)
  - [ARMA/ARMAX Models](#arma-models)
  - [Parameter Estimation](#parameter-estimation)
  - [Forecasting with Async Support](#forecasting-with-async-support)
  - [Diagnostic Tools](#diagnostic-tools)
- [Univariate Volatility Models](#univariate-volatility-models)
  - [GARCH Models](#garch-models)
  - [EGARCH Models](#egarch-models)
  - [GJR-GARCH Models](#gjr-garch-models)
  - [APARCH Models](#aparch-models)
  - [FIGARCH Models](#figarch-models)
- [Multivariate Volatility Models](#multivariate-volatility-models)
  - [BEKK Models](#bekk-models)
  - [CCC Models](#ccc-models)
  - [DCC Models](#dcc-models)
- [High-Frequency Module](#high-frequency-module)
  - [Realized Volatility Measures](#realized-volatility-measures)
  - [Noise Filtering](#noise-filtering)
  - [Sampling Schemes](#sampling-schemes)

### 5. User Interface (`mfe.ui`)
- [GUI Components](#gui-components)
  - [Main Application Window](#main-application-window)
  - [Results Viewer](#results-viewer)
  - [Dialog Windows](#dialog-windows)
- [Asynchronous UI Updates](#asynchronous-ui-updates)
- [Interactive Plotting](#interactive-plotting)
- [LaTeX Rendering](#latex-rendering)

### 6. Utility Modules (`mfe.utils`)
- [Input Validation](#input-validation)
- [Data Transformations](#data-transformations)
- [Printing & Formatting](#printing-and-formatting)
- [Performance Utilities](#performance-utilities)
- [Type Hints & Annotations](#type-hints-and-annotations)

### 7. Numba Optimization
- [JIT Compilation Overview](#jit-compilation-overview)
- [Optimized Core Functions](#optimized-core-functions)
- [Performance Considerations](#performance-considerations)
- [Fallback Mechanisms](#fallback-mechanisms)
- [Writing Numba-compatible Code](#writing-numba-compatible-code)

### 8. Examples & Tutorials
- [Time Series Modeling](#time-series-modeling-tutorial)
- [Volatility Forecasting](#volatility-forecasting-tutorial)
- [Bootstrap Analysis](#bootstrap-analysis-tutorial)
- [High-Frequency Data Analysis](#high-frequency-data-tutorial)
- [Using the GUI Interface](#using-the-gui-tutorial)

### 9. API Reference
- [Module Index](#module-index)
- [Class Index](#class-index)
- [Function Index](#function-index)
- [Type Hints Reference](#type-hints-reference)

### 10. Development
- [Contributing Guidelines](#contributing-guidelines)
- [Build System](#build-system)
- [Testing Framework](#testing-framework)
- [Documentation Standards](#documentation-standards)
- [Version Control](#version-control)

### 11. References
- [Academic References](#academic-references)
- [Statistical Methods](#statistical-methods)
- [Implementation References](#implementation-references)
- [Bibliography](#bibliography)

### Appendices
- [A: Glossary](#glossary)
- [B: Acronyms](#acronyms)
- [C: Python Library Versions](#python-library-versions)
- [D: Migration from MATLAB](#migration-from-matlab)
- [E: Citation Guidelines](#citation-guidelines)

## Overview

The MFE (MATLAB Financial Econometrics) Toolbox is a comprehensive suite of Python modules designed for modeling financial time series and conducting advanced econometric analyses. While retaining its legacy version 4.0 identity, the toolbox has been completely re-implemented using Python 3.12, incorporating modern programming constructs such as async/await patterns and strict type hints.

The toolbox leverages Python's scientific computing ecosystem, built upon foundational libraries including NumPy for matrix operations, SciPy for optimization and statistical functions, Pandas for time series handling, Statsmodels for econometric modeling, and Numba for performance optimization.

### Key Features

- Financial time series modeling and forecasting
- Volatility and risk modeling using univariate and multivariate approaches
- High-frequency financial data analysis
- Cross-sectional econometric analysis
- Bootstrap-based statistical inference
- Advanced distribution modeling and simulation
- Numba-optimized performance-critical computations
- PyQt6-based graphical user interface
- Comprehensive API with strict type hints

### Python-Based Architecture

The system architecture follows a modern Python package structure organized into four main namespaces:

1. **Core Statistical Modules** (`mfe.core`):
   - Bootstrap: Robust resampling for dependent time series
   - Cross-section: Regression and principal component analysis
   - Distributions: Advanced statistical distributions
   - Tests: Comprehensive statistical testing suite

2. **Time Series & Volatility Modules** (`mfe.models`):
   - Timeseries: ARMA/ARMAX modeling and diagnostics
   - Univariate: Single-asset volatility models (AGARCH, APARCH, etc.)
   - Multivariate: Multi-asset volatility models (BEKK, CCC, DCC)
   - Realized: High-frequency financial econometrics

3. **User Interface** (`mfe.ui`):
   - GUI: Interactive modeling environment built with PyQt6
   - Visualization: Dynamic plotting and result display

4. **Utility Modules** (`mfe.utils`):
   - Validation: Input checking and parameter verification
   - Helpers: Common utility functions and tools
   - Performance: Numba-optimized computational kernels

### Performance Optimization

The toolbox employs Numba's just-in-time (JIT) compilation to achieve near-native performance:

- Performance-critical functions are decorated with `@jit` for automatic compilation
- NumPy arrays serve as the primary data structure for efficient numerical operations
- Asynchronous processing enables responsive UI during long-running computations
- Type specialization optimizes memory usage and execution speed

### Cross-Platform Support

The MFE Toolbox supports multiple platforms through its Python implementation:

- Windows (x86_64)
- Linux (x86_64)
- macOS (x86_64, arm64)

## Navigation and Additional Resources

- [Installation Guide](#installation): Set up the MFE Toolbox in your Python environment
- [Quick Start Guide](#quick-start-guide): Get started with basic examples
- [API Reference](#api-reference): Detailed documentation of all functions and classes
- [Examples](#examples-tutorials): Practical usage examples and tutorials
- [References](#references): Academic and implementation references
- [Glossary](#glossary): Definitions of statistical and technical terms

---

*This document serves as a template for the MFE Toolbox documentation table of contents.*
*Last Updated: 2024*