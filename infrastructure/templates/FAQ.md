# MFE Toolbox FAQ

This document provides answers to frequently asked questions about the MFE Toolbox (Version 4.0), a comprehensive suite of Python modules designed for modeling financial time series and conducting advanced econometric analyses.

## Table of Contents
- [Installation FAQ](#installation-faq)
- [Usage FAQ](#usage-faq)
- [Features FAQ](#features-faq)
- [Performance FAQ](#performance-faq)
- [Troubleshooting FAQ](#troubleshooting-faq)

## Installation FAQ

### What Python version do I need?
The MFE Toolbox requires Python 3.12 or higher. This specific version requirement is necessary to support the modern programming constructs used in the toolbox, including advanced async/await patterns and strict type hints.

### What packages does MFE Toolbox depend on?
The MFE Toolbox relies on several core scientific Python libraries:
- NumPy (1.26.3+) for matrix operations
- SciPy (1.11.4+) for optimization and statistical functions
- Pandas (2.1.4+) for time series handling
- Statsmodels (0.14.1+) for econometric modeling
- Numba (0.59.0+) for performance optimization
- PyQt6 (6.6.1+) for GUI components (if using the GUI)

### What operating systems are supported?
The MFE Toolbox supports:
- Windows (x86_64)
- Linux (x86_64)
- macOS (x86_64, arm64)

Since the toolbox is implemented in Python with platform-independent libraries, it works consistently across all supported platforms.

### How do I set up a virtual environment for MFE Toolbox?
Using a virtual environment is highly recommended to isolate dependencies. Here's how to set one up:

1. Create a virtual environment:
```bash
python -m venv mfe_env
```

2. Activate the environment:
   - On Windows: `mfe_env\Scripts\activate`
   - On macOS/Linux: `source mfe_env/bin/activate`

3. Install MFE Toolbox in the activated environment:
```bash
pip install mfe
```

### What's the recommended installation method?
The recommended installation method is via pip with a virtual environment:

```bash
# Create and activate virtual environment (see above)
# Then install the package
pip install mfe
```

For development installations, use:

```bash
pip install -e .
```

### How can I verify that the installation was successful?
You can verify the installation by importing the package and checking its version:

```python
import mfe
print(mfe.__version__)
```

This should display the installed version without any errors.

## Usage FAQ

### How do I get started with MFE Toolbox?
To get started with MFE Toolbox, import the package and use its modules:

```python
import mfe
from mfe.models import armax, garch
from mfe.core import bootstrap, optimization

# Example: Create and fit an ARMAX model
data = np.array([...])  # Your time series data
model = armax.ARMAX(p=1, q=1)
await model.async_fit(data)
```

For detailed examples, refer to the documentation and example scripts included with the package.

### How do I configure models?
Models in MFE Toolbox are configured through their constructor parameters. For example:

```python
# ARMAX model with AR(2), MA(1), and a constant term
model = armax.ARMAX(p=2, q=1, include_constant=True)

# GARCH model with p=1, q=1
garch_model = garch.GARCH(p=1, q=1)
```

Each model class has specific parameters for configuring model behavior. Refer to the API documentation for detailed parameter descriptions.

### How do I use the GUI interface?
The MFE Toolbox includes a PyQt6-based GUI for interactive modeling. To launch the GUI:

```python
from mfe.ui import widgets
from PyQt6.QtWidgets import QApplication

app = QApplication([])
main_window = widgets.ARMAXModelEstimation()
main_window.show()
app.exec()
```

The GUI provides interactive forms for model configuration, estimation, and result visualization.

### How do I import and export data?
MFE Toolbox works with NumPy arrays and Pandas DataFrames. You can import and export data using standard Pandas functions:

```python
import pandas as pd
import numpy as np

# Import data
data = pd.read_csv('your_data.csv')
time_series = data['your_column'].values

# Export results
results_df = pd.DataFrame({
    'Original': time_series,
    'Fitted': model._fitted,
    'Residuals': model.residuals
})
results_df.to_csv('results.csv')
```

For specialized financial data, you can use Pandas' financial data handling capabilities before passing the data to MFE Toolbox.

## Features FAQ

### What econometric models does MFE Toolbox support?
The MFE Toolbox supports a comprehensive range of models:

Time Series Models:
- ARMA/ARMAX (AutoRegressive Moving Average with eXogenous variables)
- SARIMA (Seasonal ARIMA)

Volatility Models:
- GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)
- EGARCH (Exponential GARCH)
- AGARCH/NAGARCH (Asymmetric/Nonlinear Asymmetric GARCH)
- APARCH (Asymmetric Power ARCH)
- FIGARCH (Fractionally Integrated GARCH)
- IGARCH (Integrated GARCH)
- TARCH (Threshold ARCH)

Multivariate Models:
- BEKK (Baba-Engle-Kraft-Kroner)
- CCC (Constant Conditional Correlation)
- DCC (Dynamic Conditional Correlation)

High-Frequency Models:
- Realized volatility measures
- Kernel-based estimators
- Noise-robust estimators

### What statistical tests are included?
The MFE Toolbox includes a comprehensive suite of statistical tests:

Diagnostic Tests:
- Ljung-Box test for autocorrelation
- Jarque-Bera test for normality
- ADF and KPSS tests for stationarity

Model Selection:
- Information criteria (AIC, BIC, HQIC)
- Likelihood ratio tests

Distributional Tests:
- Tests for distributional assumptions
- Goodness-of-fit tests

Bootstrapping:
- Block bootstrap for time series
- Stationary bootstrap methods

### What visualization options are available?
The MFE Toolbox offers rich visualization capabilities:

Time Series Visualization:
- Original and fitted time series plots
- Residual analysis plots

Diagnostic Plots:
- ACF/PACF plots
- Residual distribution plots
- QQ plots

Interactive Visualization:
- Real-time parameter estimation updates
- Interactive diagnostic tools
- Zoomable and exportable plots

All visualizations are powered by Matplotlib with PyQt6 integration for interactive features.

### What performance features does MFE Toolbox offer?
MFE Toolbox includes several performance-enhancing features:

Numba Optimization:
- JIT compilation for performance-critical functions
- Type specialization for optimal performance
- Hardware-specific optimizations

Asynchronous Operations:
- Async/await patterns for responsive UI during long computations
- Non-blocking estimation and forecasting

Efficient Algorithms:
- Optimized numerical methods for parameter estimation
- Memory-efficient implementations for large datasets
- Vectorized operations using NumPy

## Performance FAQ

### How does Numba optimization work in MFE Toolbox?
The MFE Toolbox uses Numba's Just-In-Time (JIT) compilation to optimize performance-critical functions:

- Functions are decorated with `@numba.jit(nopython=True)` for maximum performance
- The first call to a function compiles it to optimized machine code
- Subsequent calls use the compiled version, resulting in near-C performance
- Type specialization ensures optimal memory and CPU usage

For example, core estimation routines in modules like `optimization.py` and computation-intensive functions in model implementations are Numba-optimized.

### How can I manage memory usage with large datasets?
When working with large datasets in MFE Toolbox, consider these memory management practices:

1. Process data in chunks where possible:
```python
chunk_size = 10000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    process_chunk(chunk)
```

2. Release large arrays when no longer needed:
```python
import gc
large_array = None
gc.collect()
```

3. Use in-place operations where possible:
```python
# Use: result += values
# Instead of: result = result + values
```

4. Monitor memory usage during computation:
```python
from memory_profiler import profile

@profile
def memory_intensive_function(data):
    # Your function code
```

### How fast is MFE Toolbox compared to similar packages?
The MFE Toolbox is designed for high performance:

- Numba-optimized core functions provide near-native speed for critical computations
- Vectorized NumPy operations ensure efficient matrix calculations
- Performance benchmarks show competitive speed compared to similar packages:
  - Parameter estimation is typically 5-10x faster than pure Python implementations
  - Comparable speed to C/C++ implementations when using Numba
  - Faster than R implementations for many common tasks

### What are the resource requirements for optimal performance?
For optimal performance, we recommend:

Hardware:
- Multi-core CPU (Numba can utilize multiple cores)
- At least 4GB RAM for typical datasets
- 8GB+ RAM for large datasets or complex multivariate models

Software:
- Python 3.12 with up-to-date NumPy, SciPy, and Numba
- Operating system with proper LLVM support for Numba

Configuration:
- Set thread count appropriate for your CPU:
```python
import numba
numba.set_num_threads(8)  # Adjust based on your CPU
```
- Control memory usage with appropriate array data types:
```python
# Use float64 for precision, float32 for memory efficiency
data = np.asarray(data, dtype=np.float64)  # or np.float32
```

## Troubleshooting FAQ

### How do I resolve installation issues?

#### Python Version Compatibility
If you encounter errors about incompatible Python version:
- Verify your Python version with `python --version`
- Install Python 3.12 from [python.org](https://www.python.org/downloads/)
- Use virtual environments to manage multiple Python installations

#### Package Dependencies
If installation fails due to missing or incompatible dependencies:
- Install required dependencies with specific versions:
  ```bash
  pip install numpy==1.26.3 scipy==1.11.4 pandas==2.1.4 statsmodels==0.14.1 numba==0.59.0 PyQt6==6.6.1
  ```
- Consider creating a dedicated virtual environment for clean installation

#### Platform-Specific Issues
For Windows:
- Ensure Microsoft Visual C++ Build Tools are installed for Numba compilation
- For PyQt6 issues, ensure Visual C++ Redistributable is installed

For macOS:
- For Apple Silicon Macs, ensure you're using the arm64 version of Python
- Use Homebrew to install dependencies: `brew install python@3.12`

For Linux:
- Install system dependencies for PyQt6: `sudo apt-get install python3-pyqt6`
- For Numba issues, install LLVM: `sudo apt-get install llvm`

### How do I fix common runtime errors?

#### Import Errors
If you encounter `ModuleNotFoundError` or `ImportError`:
- Verify the package is installed: `pip list | grep mfe`
- Check import statements for typos
- Make sure you're in the correct virtual environment
- Use correct import patterns:
  ```python
  from mfe.core import optimization
  from mfe.models import armax
  ```

#### Type Errors
For `TypeError` during function calls:
- Check parameter types against function requirements
- Common issues include:
  - Passing lists instead of NumPy arrays
  - Using strings for numeric parameters
  - Providing arrays with incorrect dimensions
- Convert inputs to proper types:
  ```python
  data = np.asarray(data, dtype=np.float64)
  ```

#### Numerical Computation Errors
For numerical issues like NaN results or convergence problems:
- Try different initial values for optimization
- Add checks for non-finite values:
  ```python
  if not np.isfinite(data).all():
      data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
  ```
- Use controlled error handling:
  ```python
  with np.errstate(over='raise', divide='raise'):
      try:
          result = computation()
      except FloatingPointError:
          # Apply fallback method
  ```

### How do I troubleshoot GUI problems?

#### Display Problems
If PyQt6 windows don't appear or display incorrectly:
- Verify PyQt6 installation: `pip install --upgrade PyQt6==6.6.1`
- For scaling issues on high-DPI displays:
  ```python
  from PyQt6.QtCore import Qt
  from PyQt6.QtWidgets import QApplication
  app = QApplication.instance() or QApplication([])
  app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
  ```

#### Plot Rendering
If Matplotlib plots aren't showing in PyQt6 widgets:
- Verify Matplotlib backend configuration:
  ```python
  import matplotlib
  matplotlib.use('Qt6Agg')  # Set backend explicitly
  ```
- Force canvas updates:
  ```python
  self._figure.tight_layout()
  self._canvas.draw()
  self._canvas.flush_events()
  ```

### How do I resolve performance problems?

#### Numba Compilation Issues
If Numba functions aren't optimizing:
- Enable Numba debug output:
  ```python
  import numba
  numba.config.NUMBA_DUMP_OPTIMIZED = 1
  numba.config.NUMBA_DUMP_IR = 1
  ```
- Ensure functions are properly decorated:
  ```python
  @numba.jit(nopython=True)
  def optimize_garch(params, data, model_type_id):
      # Function code
  ```
- For first-time compilation delays, pre-compile functions:
  ```python
  # Call function once to trigger compilation
  dummy_data = np.random.randn(100)
  dummy_params = np.array([0.1, 0.1, 0.8])
  _ = optimize_garch(dummy_params, dummy_data, 0)
  ```

#### Slow Computation
If models are running slowly:
- Use Numba-optimized functions
- Enable parallel processing:
  ```python
  @numba.jit(nopython=True, parallel=True)
  def parallel_computation(data):
      # Parallel code
  ```
- Profile slow functions:
  ```python
  import cProfile
  cProfile.run('model.async_fit(data)', 'stats.prof')
  ```
- Use vectorized operations where possible:
  ```python
  # Fast vectorized operation
  result = data ** 2
  ```

For more detailed troubleshooting, refer to the full troubleshooting guide in the documentation.