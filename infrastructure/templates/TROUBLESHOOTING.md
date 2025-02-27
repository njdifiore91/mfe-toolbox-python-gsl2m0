# MFE Toolbox Troubleshooting Guide

This document provides solutions for common issues you might encounter when using the MFE Toolbox. It covers installation problems, runtime errors, numerical computation issues, GUI problems, and performance optimization.

## Installation Issues

### Python Version Compatibility

**Issue**: Error messages about incompatible Python version.

**Solution**:
- MFE Toolbox requires Python 3.12. Verify your Python version with `python --version`.
- If you have multiple Python installations, ensure you're using the correct one.
- Install Python 3.12 from [python.org](https://www.python.org/downloads/) if needed.

**Prevention**:
- Use virtual environments to manage Python versions.
- Consider `pyenv` or similar tools to manage multiple Python installations.

### Package Dependencies

**Issue**: Installation fails due to missing or incompatible dependencies.

**Solution**:
- Install required dependencies with specific versions:
  ```bash
  pip install numpy==1.26.3 scipy==1.11.4 pandas==2.1.4 statsmodels==0.14.1 numba==0.59.0 PyQt6==6.6.1
  ```
- For GPU acceleration issues, verify Numba has access to CUDA.
- If a dependency conflicts with existing packages, consider creating a dedicated virtual environment.

**Prevention**:
- Always check the `requirements.txt` file for the correct versions.
- Use virtual environments for isolated package installations.

### Virtual Environment Setup

**Issue**: Problems with package isolation or environment configuration.

**Solution**:
- Create a new virtual environment:
  ```bash
  python -m venv mfe_env
  ```
- Activate the environment:
  - Windows: `mfe_env\Scripts\activate`
  - macOS/Linux: `source mfe_env/bin/activate`
- Install the MFE Toolbox in the activated environment:
  ```bash
  pip install mfe
  ```

**Prevention**:
- Always use virtual environments for Python projects.
- Document your environment setup for reproducibility.

### Platform-Specific Issues

**Issue**: Installation or operation issues specific to Windows, macOS, or Linux.

**Solution**:

**For Windows**:
- Ensure Microsoft Visual C++ Build Tools are installed for Numba compilation.
- For PyQt6 issues, ensure Visual C++ Redistributable is installed.

**For macOS**:
- For Apple Silicon (M1/M2) Macs, ensure you're using the arm64 version of Python.
- Use Homebrew to install dependencies: `brew install python@3.12`

**For Linux**:
- Install system dependencies for PyQt6: 
  ```bash
  sudo apt-get install python3-pyqt6
  ```
- For Numba issues, install LLVM: 
  ```bash
  sudo apt-get install llvm
  ```

**Prevention**:
- Check platform-specific documentation before installation.
- Review system requirements in the documentation.

## Runtime Errors

### Import Errors

**Issue**: `ModuleNotFoundError` or `ImportError` when running code.

**Solution**:
- Verify the package is installed: `pip list | grep mfe`
- Check import statements for typos.
- Make sure you're in the correct virtual environment.
- For internal imports, check the package structure:
  ```python
  # Correct imports follow this pattern
  from mfe.core import optimization
  from mfe.models import armax
  from mfe.ui import armax_viewer
  ```

**Prevention**:
- Follow the import conventions in the documentation.
- Initialize the package properly as shown in examples.

### Type Errors

**Issue**: `TypeError` during function calls or parameter passing.

**Solution**:
- Check parameter types against function requirements.
- Common type errors in the MFE Toolbox:
  - Passing lists instead of NumPy arrays
  - Using strings for numeric parameters
  - Providing arrays with incorrect dimensions
- Convert inputs to proper types:
  ```python
  # Convert to proper numeric array
  data = np.asarray(data, dtype=np.float64)
  ```

**Prevention**:
- Review function signatures with type hints.
- Use the validation utilities:
  ```python
  from mfe.utils.validation import validate_array_input
  validate_array_input(data)
  ```

### Memory Errors

**Issue**: `MemoryError` or out-of-memory conditions with large datasets.

**Solution**:
- Reduce dataset size or segment processing.
- Close unused figures and clear variables:
  ```python
  import matplotlib.pyplot as plt
  plt.close('all')  # Close all matplotlib figures
  ```
- For PyQt6 memory issues, ensure widgets are properly disposed:
  ```python
  # In closeEvent or cleanup method
  if self._figure is not None:
      plt.close(self._figure)
  self._canvas = None
  self._figure = None
  ```

**Prevention**:
- Monitor memory usage during processing of large datasets.
- Release resources explicitly when finished.
- Use generators or iterators for large data processing.

### Path Configuration Issues

**Issue**: Unable to locate modules or package components.

**Solution**:
- Verify package installation is complete.
- Check Python's import path:
  ```python
  import sys
  print(sys.path)
  ```
- If using development installation, ensure the package directory is in PYTHONPATH.

**Prevention**:
- Use proper package installation methods (pip).
- Follow the initialization procedure in the documentation.

## Numerical Computation

### Convergence Problems

**Issue**: Optimization algorithms fail to converge or give warnings.

**Solution**:
- Try different initial values:
  ```python
  # Start with different initial parameters
  initial_params = np.array([0.05, 0.85, 0.1])  # For GARCH models
  ```
- Increase maximum iterations:
  ```python
  # Increase max iterations in optimizer
  result = optimize.minimize(
      neg_log_likelihood, 
      initial_params,
      method='SLSQP',
      options={'ftol': 1e-8, 'disp': True, 'maxiter': 2000}
  )
  ```
- Try alternative optimization methods:
  ```python
  # Try different optimization method
  result = optimize.minimize(
      neg_log_likelihood, 
      initial_params,
      method='L-BFGS-B',  # Alternative method
      bounds=bounds
  )
  ```

**Prevention**:
- Scale your data appropriately before modeling.
- Start with simpler models before increasing complexity.
- Check the stationarity of your time series data.

### Numerical Stability

**Issue**: NaN, infinite values, or numerical overflows during computation.

**Solution**:
- Add checks for non-finite values:
  ```python
  # Find and handle non-finite values
  if not np.isfinite(data).all():
      data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
  ```
- Use controlled error handling:
  ```python
  # Control numerical errors
  with np.errstate(over='raise', divide='raise'):
      try:
          # Your computation here
          result = computation()
      except FloatingPointError:
          logger.error("Numerical error encountered")
          # Apply fallback method
  ```
- Apply data transformations such as scaling or log transforms.

**Prevention**:
- Preprocess your data to handle outliers.
- Use the validation utilities before computations.
- Set numerical error handling preferences with numpy.

### Parameter Bounds

**Issue**: Parameter estimation results in invalid or unstable model parameters.

**Solution**:
- Add appropriate bounds to constrain parameters:
  ```python
  # For GARCH models, enforce stationarity
  bounds = [(1e-6, None), (0, 0.999), (0, 0.999)]  # omega, alpha, beta
  
  # Add constraint for alpha + beta < 1
  def constraint_func(params):
      return 0.999 - (params[1] + params[2])
  
  constraints = [{'type': 'ineq', 'fun': constraint_func}]
  ```
- Validate parameters after optimization:
  ```python
  from mfe.utils.validation import validate_parameters
  
  # Validate GARCH parameters
  validate_parameters(optimal_params, param_type='GARCH')
  ```

**Prevention**:
- Use the built-in parameter validation utilities.
- Set appropriate bounds during model estimation.
- Check model stability conditions in documentation.

### Precision Loss

**Issue**: Results with poor precision or unexpected small values.

**Solution**:
- Use double precision (float64) for calculations:
  ```python
  data = np.asarray(data, dtype=np.float64)
  ```
- Lower optimization tolerances:
  ```python
  result = optimize.minimize(
      neg_log_likelihood, 
      initial_params,
      tol=1e-10,  # Tighter tolerance
      options={'ftol': 1e-10, 'disp': False}
  )
  ```
- Apply a small epsilon to avoid division by zero:
  ```python
  # Add small constant to avoid division by zero
  denominator = np.maximum(value, 1e-10)
  result = numerator / denominator
  ```

**Prevention**:
- Always use double precision for financial calculations.
- Monitor precision in intermediate results.
- Validate input data ranges and scales.

## GUI Problems

### Display Problems

**Issue**: PyQt6 windows not appearing or displaying incorrectly.

**Solution**:
- Verify PyQt6 installation:
  ```bash
  pip install --upgrade PyQt6==6.6.1
  ```
- For layout issues, reset layouts:
  ```python
  # Clear and recreate layout
  while self._layout.count():
      item = self._layout.takeAt(0)
      if item.widget():
          item.widget().deleteLater()
  ```
- For scaling issues on high-DPI displays:
  ```python
  # Add to beginning of your application
  from PyQt6.QtCore import Qt
  from PyQt6.QtWidgets import QApplication
  app = QApplication.instance() or QApplication([])
  app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
  ```

**Prevention**:
- Test the UI on different resolutions and platforms.
- Use layout managers instead of fixed geometries.
- Add proper widget parenting.

### Plot Rendering

**Issue**: Matplotlib plots not showing in PyQt6 widgets.

**Solution**:
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
- Check for interactive mode:
  ```python
  import matplotlib.pyplot as plt
  plt.ion()  # Turn on interactive mode
  ```

**Prevention**:
- Use the provided `ResidualPlotWidget` which handles backend configuration.
- Always call `draw()` after updating plots.
- Set up proper signal connections for plot updates.

### Dialog Issues

**Issue**: Modal dialogs not functioning correctly or freezing the application.

**Solution**:
- Ensure proper dialog initialization:
  ```python
  dialog = ARMAXResultsViewer(self)
  dialog.display_results(model)
  result = dialog.exec()  # Use exec() for modal dialogs
  ```
- For frozen dialogs, use asynchronous operations:
  ```python
  # Use async methods with signals
  @pyqtSlot()
  async def show_results(self):
      try:
          # Long-running operation
          await self.process_data()
          # Update UI when done
          self.update_ui()
      except Exception as e:
          self.show_error_message(str(e))
  ```
- If dialog fails to close properly, implement explicit cleanup:
  ```python
  def close_dialog(self):
      try:
          # Clean up resources
          if self._residual_plots:
              self._residual_plots.clear_plots()
          
          # Close the dialog
          self.accept()
      except Exception as e:
          logger.error(f"Error closing dialog: {str(e)}")
          # Force close in case of error
          self.reject()
  ```

**Prevention**:
- Use async functions for long-running operations.
- Connect proper signals and slots for UI updates.
- Don't perform heavy computations in the UI thread.

### Event Handling

**Issue**: PyQt6 signals not connecting or events not firing.

**Solution**:
- Verify signal connections:
  ```python
  # Explicit connection with named slot
  self.button.clicked.connect(self.on_button_clicked)
  
  # Check if connection exists
  connections = self.button.receivers(self.button.clicked)
  print(f"Button has {connections} connections")
  ```
- Ensure event methods have correct signatures:
  ```python
  # Correct event method signature
  @pyqtSlot()  # Decorator for clarity
  def on_button_clicked(self):
      # Event handling code
      pass
  ```
- Debug event propagation with custom logging:
  ```python
  def eventFilter(self, watched, event):
      print(f"Event: {event.type()} on {watched}")
      return super().eventFilter(watched, event)
  ```

**Prevention**:
- Use proper PyQt6 signal/slot connections.
- Add logging to event handlers during development.
- Use `@pyqtSlot()` decorators to clarify slot methods.

## Performance Issues

### Numba Compilation

**Issue**: Numba functions not compiling or running slowly.

**Solution**:
- Enable Numba debug output:
  ```python
  import numba
  numba.config.NUMBA_DUMP_OPTIMIZED = 1
  numba.config.NUMBA_DUMP_IR = 1
  ```
- Ensure functions are properly decorated:
  ```python
  # Use nopython mode for best performance
  @numba.jit(nopython=True)
  def optimize_garch(params, data, model_type_id):
      # Function code
  ```
- For first-time compilation delays, pre-compile functions:
  ```python
  # Call function once with typical inputs to compile
  dummy_data = np.random.randn(100)
  dummy_params = np.array([0.1, 0.1, 0.8])
  _ = optimize_garch(dummy_params, dummy_data, 0)  # Triggers compilation
  ```
- If Numba fails to compile, check for Python objects in the function:
  ```python
  # Not Numba-friendly (uses Python objects):
  def bad_function(params, data, model_type):
      if model_type.upper() == 'GARCH':  # String operations not supported in nopython mode
          # ...
  
  # Numba-friendly (uses primitive types):
  def good_function(params, data, model_type_id):
      if model_type_id == 0:  # Integer comparison is supported
          # ...
  ```

**Prevention**:
- Use `nopython=True` for maximum performance.
- Keep Numba functions pure (avoid Python objects).
- Use contiguous arrays with `np.ascontiguousarray()`.

### Memory Usage

**Issue**: Excessive memory consumption during model estimation.

**Solution**:
- Profile memory usage:
  ```python
  from memory_profiler import profile
  
  @profile
  def memory_intensive_function(data):
      # Your function code
  ```
- Release memory explicitly:
  ```python
  # Clear large arrays when done
  import gc
  large_array = None
  gc.collect()
  ```
- Use memory-efficient operations:
  ```python
  # In-place operations to reduce memory usage
  result += values  # Instead of result = result + values
  ```
- For large datasets, process in chunks:
  ```python
  # Process data in chunks
  chunk_size = 10000
  for i in range(0, len(data), chunk_size):
      chunk = data[i:i+chunk_size]
      process_chunk(chunk)
  ```

**Prevention**:
- Process data in chunks where possible.
- Avoid unnecessary copies of large arrays.
- Use generators for large data processing.

### Computation Speed

**Issue**: Slow performance in model estimation or forecasting.

**Solution**:
- Use Numba-optimized functions:
  ```python
  # Ensure Numba is used for critical calculations
  from mfe.core.optimization import optimize_garch
  ```
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
  
  # Analyze profile
  import pstats
  p = pstats.Stats('stats.prof')
  p.sort_stats('cumulative').print_stats(20)
  ```
- Use vectorized operations where possible:
  ```python
  # Slow loop
  for i in range(len(data)):
      result[i] = data[i] ** 2
  
  # Fast vectorized operation
  result = data ** 2
  ```

**Prevention**:
- Use vectorized NumPy operations where possible.
- Leverage Numba for performance-critical functions.
- Minimize Python loops in performance-critical code.

### Resource Utilization

**Issue**: Poor CPU/GPU utilization for computational tasks.

**Solution**:
- For CPU utilization, enable thread parallelism:
  ```python
  # Set number of threads for NumPy/SciPy operations
  import os
  os.environ["OMP_NUM_THREADS"] = "4"  # Adjust to your CPU core count
  ```
- For GPU utilization with Numba:
  ```python
  @numba.cuda.jit
  def cuda_function(array):
      # CUDA GPU function
  ```
- Monitor resource usage during computation:
  ```python
  import psutil
  print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
  print(f"Memory usage: {psutil.virtual_memory().percent}%")
  ```
- Check if Numba is efficiently utilizing resources:
  ```python
  # Check Numba threading layer
  print(numba.get_num_threads())
  
  # Set thread count
  numba.set_num_threads(8)  # Adjust based on your CPU
  ```

**Prevention**:
- Structure code to enable parallelism.
- Use appropriate hardware acceleration when available.
- Monitor resource usage during development.

By following this troubleshooting guide, you can resolve common issues encountered when using the MFE Toolbox. If you continue to experience problems, please consult the detailed API documentation or contact support.