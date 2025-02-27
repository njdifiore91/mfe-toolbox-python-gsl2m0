# Known Issues and Limitations - MFE Toolbox v4.0

This document lists known issues, limitations, and workarounds for the MFE Toolbox Python implementation. It serves as a reference for developers and users encountering specific problems. For detailed troubleshooting assistance, please also refer to the [Troubleshooting Guide](TROUBLESHOOTING.md).

## Python Implementation

### Version Compatibility

**Issue**: MFE Toolbox requires Python 3.12 specifically, which may conflict with other Python projects or dependencies.

**Impact**: Installation failures or runtime errors may occur on systems with different Python versions.

**Workaround**: 
- Use virtual environments to isolate the MFE Toolbox installation
- If using conda, create a dedicated environment with `conda create -n mfe python=3.12`
- For development on systems where Python 3.12 cannot be installed, consider using Docker for containerization

### Package Dependencies

**Issue**: Specific version requirements for NumPy (1.26.3), SciPy (1.11.4), Pandas (2.1.4), Statsmodels (0.14.1), Numba (0.59.0), and PyQt6 (6.6.1).

**Impact**: Conflicts may arise when these requirements differ from other packages in the same environment.

**Workaround**:
- Use virtual environments for isolation
- Install dependencies with exact version specifications using:
  ```bash
  pip install numpy==1.26.3 scipy==1.11.4 pandas==2.1.4 statsmodels==0.14.1 numba==0.59.0 PyQt6==6.6.1
  ```
- For projects with conflicting dependencies, consider using separate environments

### Type Hint Limitations

**Issue**: Type hints may not fully capture the complexity of numerical operations, especially with NumPy arrays and their shapes.

**Impact**: Static type checkers might report false positives, and runtime type errors could still occur despite static checking passing.

**Workaround**:
- Use runtime validation with functions from `mfe.utils.validation` alongside static type checking
- Implement comprehensive input validation in performance-critical functions
- Be cautious of array shape, dtype, and dimensionality requirements even when type checking passes

### Async/Await Constraints

**Issue**: Async/await pattern integration with PyQt6 event loop can lead to unexpected behavior.

**Impact**: UI responsiveness issues, event handling delays, or deadlocks in certain scenarios, particularly during long-running computations.

**Workaround**:
- Ensure all asynchronous operations have proper error handling and timeouts
- Use `loop.run_in_executor()` for CPU-bound tasks as demonstrated in the optimization module
- Implement cancellation mechanisms for long-running asynchronous operations
- Keep UI thread responsive by offloading heavy computations to background tasks

## Numba Optimization

### Compilation Failures

**Issue**: Numba may fail to compile certain Python code patterns, especially with complex object interactions or dynamic types.

**Impact**: Runtime errors or fallback to non-optimized Python implementation, causing performance degradation.

**Workaround**:
- Implement graceful degradation to pure Python implementations when Numba compilation fails
- Use simple data types (numbers, NumPy arrays) in Numba-decorated functions
- Avoid Python objects, dictionaries, list comprehensions, and complex control flow in JIT-compiled functions
- Test Numba optimization on all target platforms during development

### Performance Limitations

**Issue**: Numba optimization may not achieve expected performance gains for all code patterns, particularly for operations that are already optimized in NumPy.

**Impact**: Some operations may remain slower than equivalent compiled language implementations.

**Workaround**:
- Focus Numba optimization on computation-heavy, loop-intensive code where gains are significant
- Use vectorized NumPy operations where possible before resorting to Numba
- Profile code to identify and optimize the most critical performance bottlenecks
- Consider pre-compilation of Numba functions during initialization to avoid JIT overhead:
  ```python
  # Pre-compile function with typical inputs
  dummy_data = np.random.randn(100)
  dummy_params = np.array([0.1, 0.1, 0.8])
  _ = optimize_garch(dummy_params, dummy_data, 0)  # Triggers compilation
  ```

### Platform Compatibility

**Issue**: Numba optimization effectiveness and compatibility may vary across operating systems and hardware platforms.

**Impact**: Performance inconsistencies or compilation failures between different deployment environments.

**Workaround**:
- Test on all target platforms during development
- Use continuous integration with multiple platform configurations
- Provide platform-specific optimizations with conditional code paths
- Implement fallback mechanisms for platforms where Numba optimizations fail:
  ```python
  try:
      result = numba_optimized_function(data)
  except numba.errors.NumbaError:
      logger.warning("Numba optimization failed, using fallback implementation")
      result = python_fallback_function(data)
  ```

### Type Specialization

**Issue**: Numba requires type stability for optimal performance, which can be challenging with dynamic Python code.

**Impact**: Sub-optimal performance or compilation failures when types are not stable or predictable.

**Workaround**:
- Use explicit type annotations in Numba-decorated functions
- Ensure consistent data types for function inputs using `np.asarray(data, dtype=np.float64)`
- Avoid mixing different numeric types in the same computation
- Use `np.ascontiguousarray()` to ensure array memory layout is optimized for Numba

## GUI Interface

### Display Problems

**Issue**: PyQt6 widget rendering and scaling issues across different platforms and display configurations.

**Impact**: Inconsistent appearance, cut-off elements, or improper scaling on high-DPI displays.

**Workaround**:
- Use layout managers instead of fixed positioning for all UI components
- Implement proper resize handling for components as shown in `residual_plot.py`
- For high-DPI display issues, use explicit scaling support:
  ```python
  from PyQt6.QtCore import Qt
  from PyQt6.QtWidgets import QApplication
  app = QApplication.instance() or QApplication([])
  app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
  ```
- Test on multiple display configurations and platforms

### Event Handling

**Issue**: Complex interaction between PyQt6's signal-slot mechanism and Python's async/await pattern.

**Impact**: Event processing delays, missed events, or UI freezing during long-running operations.

**Workaround**:
- Use `loop.run_in_executor()` for CPU-bound tasks to prevent UI freezing
- Implement proper cancellation mechanisms for async operations
- Connect signals properly with explicit slot methods:
  ```python
  # Explicit connection with named slot
  @pyqtSlot()
  def on_button_clicked(self):
      # Event handling code
      pass
      
  # In initialization
  self.button.clicked.connect(self.on_button_clicked)
  ```
- Use signals to communicate between async tasks and UI

### Plot Rendering

**Issue**: Integration challenges between Matplotlib and PyQt6, especially during resizing and UI updates.

**Impact**: Plot rendering artifacts, memory leaks, or performance degradation with frequent updates.

**Workaround**:
- Clean up Matplotlib resources properly in `closeEvent` handlers:
  ```python
  def closeEvent(self, event):
      # Clean up plot resources
      if self._figure is not None:
          plt.close(self._figure)
      self._canvas = None
      self._figure = None
      event.accept()
  ```
- Use the appropriate Matplotlib backend (Qt6Agg)
- Call `draw()` and `tight_layout()` after plot modifications
- For complex visualizations, implement double-buffering or offscreen rendering

### Dialog Management

**Issue**: Modal dialog handling complications with asynchronous operations.

**Impact**: Blocking UI interactions, dialog freezes, or incorrect UI state after dialog closure.

**Workaround**:
- Use non-modal dialogs for operations that require background processing
- Implement proper cleanup in `closeEvent` handlers to avoid resource leaks
- Manage dialog lifecycle carefully, especially with parent-child relationships
- Use PyQt6's `exec()` pattern correctly for modal dialogs:
  ```python
  dialog = ARMAXResultsViewer(self)
  dialog.display_results(model)
  result = dialog.exec()  # Use exec() for modal dialogs
  ```

## Numerical Computation

### Convergence Problems

**Issue**: Optimization algorithms may fail to converge for certain data patterns or initial conditions, particularly in GARCH and ARMAX models.

**Impact**: Model estimation failures, sub-optimal parameters, or excessive computation time.

**Workaround**:
- Implement multiple initialization strategies with different starting points:
  ```python
  # Try different initial values if optimization fails
  initial_values = [
      np.array([0.05, 0.85, 0.1]),
      np.array([0.01, 0.8, 0.15]),
      np.array([0.1, 0.7, 0.2])
  ]
  
  for init_values in initial_values:
      result = optimize.minimize(neg_log_likelihood, init_values, ...)
      if result.success:
          break
  ```
- Add constraints to parameter space to improve convergence
- Implement early stopping mechanisms with reasonable defaults
- Provide diagnostic information when convergence fails

### Precision Loss

**Issue**: Floating-point precision issues in numerical computations, especially for extreme values or long-running calculations.

**Impact**: Inaccurate results, overflow/underflow errors, or numerical instability in model estimation.

**Workaround**:
- Always use double precision (float64) for all critical calculations:
  ```python
  data = np.asarray(data, dtype=np.float64)
  ```
- Implement checks for non-finite values:
  ```python
  if not np.isfinite(result).all():
      logger.warning("Non-finite values detected in computation")
      result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
  ```
- Add small epsilon values to avoid division by zero:
  ```python
  # Add small constant to avoid division by zero
  denominator = np.maximum(value, 1e-10)
  result = numerator / denominator
  ```
- Scale data appropriately before processing to avoid extreme values

### Memory Usage

**Issue**: High memory consumption when processing large datasets, especially with multiple array copies in optimization routines.

**Impact**: Out-of-memory errors or excessive swapping, leading to performance degradation.

**Workaround**:
- Implement chunk-based processing for large datasets:
  ```python
  # Process data in chunks
  chunk_size = 10000
  for i in range(0, len(data), chunk_size):
      chunk = data[i:i+chunk_size]
      process_chunk(chunk)
  ```
- Use in-place operations where possible:
  ```python
  # In-place operations to reduce memory usage
  result += values  # Instead of result = result + values
  ```
- Explicitly delete large temporary arrays when no longer needed:
  ```python
  # Release memory
  import gc
  large_array = None
  gc.collect()
  ```
- Monitor memory usage during long-running operations

### Performance Bottlenecks

**Issue**: Computational bottlenecks in specific operations, especially for high model orders or multivariate models.

**Impact**: Excessive computation time or unresponsive UI during intensive calculations.

**Workaround**:
- Profile code to identify specific bottlenecks:
  ```python
  import cProfile
  cProfile.run('model.fit(data)', 'stats.prof')
  
  # Analyze profile
  import pstats
  p = pstats.Stats('stats.prof')
  p.sort_stats('cumulative').print_stats(20)
  ```
- Optimize the most critical paths with Numba
- Implement asynchronous processing with progress reporting
- Use vectorized operations wherever possible:
  ```python
  # Vectorized operation (fast)
  result = data ** 2
  
  # Instead of loop (slow)
  for i in range(len(data)):
      result[i] = data[i] ** 2
  ```

## Reporting New Issues

If you encounter an issue not listed in this document, please report it with the following information:
1. Detailed description of the issue
2. Steps to reproduce
3. System configuration (OS, Python version, package versions)
4. Any error messages or logs
5. Suggested workaround if discovered

## Change Log

This document was last updated for MFE Toolbox version 4.0.