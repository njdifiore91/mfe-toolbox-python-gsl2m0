# Monitoring Implementation Guide for MFE Toolbox

## Overview

This document provides comprehensive guidance for implementing monitoring capabilities in the MFE Toolbox. The monitoring system focuses on model diagnostics, performance tracking, and visual analysis through Python's scientific computing stack.

The monitoring framework leverages:
- Python's asynchronous programming with `async/await` patterns
- Numba-optimized performance tracking
- PyQt6-based interactive visualizations
- Integration with NumPy, Matplotlib, and Statsmodels
- Comprehensive logging through Python's native logging framework

## Basic Monitoring Implementation

### Model Diagnostic Monitoring

Model diagnostic monitoring implements specialized functions for tracking model quality, parameter significance, and statistical validity.

```python
import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

def monitor_model_diagnostics(model, data):
    """
    Monitor model diagnostic metrics during estimation.
    
    Parameters
    ----------
    model : ARMAX or other model instance
        Model being monitored
    data : np.ndarray
        Data used for model estimation
        
    Returns
    -------
    dict
        Dictionary of diagnostic metrics
    """
    try:
        # Initialize monitoring metrics
        monitoring_results = {}
        
        # Track log-likelihood
        monitoring_results['loglikelihood'] = model.loglikelihood
        
        # Track information criteria
        diagnostics = model.diagnostic_tests()
        monitoring_results['AIC'] = diagnostics['AIC']
        monitoring_results['BIC'] = diagnostics['BIC']
        
        # Track statistical tests
        monitoring_results['ljung_box'] = diagnostics['ljung_box']
        monitoring_results['jarque_bera'] = diagnostics['jarque_bera']
        
        # Track parameter significance
        monitoring_results['parameter_summary'] = diagnostics['parameter_summary']
        
        logger.info(f"Model diagnostics updated: Log-likelihood = {model.loglikelihood:.4f}")
        
        return monitoring_results
        
    except Exception as e:
        logger.error(f"Error in model diagnostic monitoring: {str(e)}")
        return None
```

#### Display Functions

Implement model-specific display functions for various GARCH models:

```python
def display_garch_diagnostics(model):
    """
    Display GARCH model diagnostic information.
    
    Parameters
    ----------
    model : GARCHModel
        Fitted GARCH model
    """
    # Get diagnostic tests
    diagnostics = model.diagnostic_tests()
    
    # Display model parameters
    print("Model Parameters:")
    print("================")
    for param in diagnostics['parameter_summary']:
        print(f"{param['name']}: {param['value']:.4f} (SE: {param['std_error']:.4f}, p-value: {param['p_value']:.4f})")
    
    # Display information criteria
    print("\nInformation Criteria:")
    print("====================")
    print(f"AIC: {diagnostics['AIC']:.4f}")
    print(f"BIC: {diagnostics['BIC']:.4f}")
    
    # Display statistical tests
    print("\nStatistical Tests:")
    print("================")
    print(f"Ljung-Box (lag={diagnostics['ljung_box']['lags']}): {diagnostics['ljung_box']['statistic']:.4f} (p-value: {diagnostics['ljung_box']['p_value']:.4f})")
    print(f"Jarque-Bera: {diagnostics['jarque_bera']['statistic']:.4f} (p-value: {diagnostics['jarque_bera']['p_value']:.4f})")
```

### Performance Metrics Tracking

The system tracks key model performance indicators using Python's scientific computing libraries:

```python
import time
import numpy as np

def track_optimization_performance(optimizer, optimization_result):
    """
    Track optimization performance metrics.
    
    Parameters
    ----------
    optimizer : Optimizer
        Optimizer instance
    optimization_result : OptimizeResult
        Result from SciPy optimization
        
    Returns
    -------
    dict
        Performance metrics
    """
    metrics = {}
    
    # Track convergence
    metrics['converged'] = optimizer.converged
    
    # Track number of iterations
    metrics['iterations'] = optimization_result.nit if hasattr(optimization_result, 'nit') else None
    
    # Track function evaluations
    metrics['function_evaluations'] = optimization_result.nfev if hasattr(optimization_result, 'nfev') else None
    
    # Track final function value (negative log-likelihood)
    metrics['final_value'] = optimization_result.fun if hasattr(optimization_result, 'fun') else None
    
    # Track computation time if available
    if hasattr(optimization_result, 'execution_time'):
        metrics['execution_time'] = optimization_result.execution_time
    
    return metrics
```

#### Numba Performance Tracking

For monitoring Numba-optimized functions:

```python
import time
import numba
import numpy as np

def benchmark_numba_function(func, *args, **kwargs):
    """
    Benchmark a Numba-optimized function.
    
    Parameters
    ----------
    func : function
        Numba-decorated function to benchmark
    *args, **kwargs
        Arguments to pass to the function
        
    Returns
    -------
    dict
        Benchmark results
    """
    results = {}
    
    # First call might include compilation time
    start_time = time.time()
    result = func(*args, **kwargs)
    first_call_time = time.time() - start_time
    
    # Second call should use compiled version
    start_time = time.time()
    result = func(*args, **kwargs)
    second_call_time = time.time() - start_time
    
    results['first_call_time'] = first_call_time
    results['second_call_time'] = second_call_time
    results['compilation_time'] = first_call_time - second_call_time
    results['speedup'] = first_call_time / second_call_time if second_call_time > 0 else float('inf')
    
    return results
```

### Visual Diagnostic Tools

Implement visual diagnostic tools using Matplotlib and Statsmodels:

```python
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.graphics.tsaplots as tsaplots
from scipy import stats

def create_residual_plots(residuals, figsize=(12, 10)):
    """
    Create comprehensive residual diagnostic plots.
    
    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : Figure
        Matplotlib figure with residual plots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Residual time series
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_title('Residuals')
    axes[0, 0].set_xlabel('Observation')
    axes[0, 0].set_ylabel('Residual')
    
    # Plot 2: Residual histogram
    axes[0, 1].hist(residuals, bins=20, density=True, alpha=0.7)
    # Add normal distribution overlay
    x = np.linspace(min(residuals), max(residuals), 100)
    mean, std = np.mean(residuals), np.std(residuals)
    axes[0, 1].plot(x, stats.norm.pdf(x, mean, std), 'r-')
    axes[0, 1].set_title('Residual Distribution')
    
    # Plot 3: ACF
    tsaplots.plot_acf(residuals, ax=axes[1, 0])
    axes[1, 0].set_title('Autocorrelation Function')
    
    # Plot 4: PACF
    tsaplots.plot_pacf(residuals, ax=axes[1, 1])
    axes[1, 1].set_title('Partial Autocorrelation Function')
    
    fig.tight_layout()
    return fig
```

### Interactive Monitoring

Implement interactive monitoring using PyQt6 and async/await patterns:

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
import asyncio

class ModelMonitoringWidget(QWidget):
    """
    Interactive widget for real-time model monitoring.
    
    This widget displays model estimation progress and updates
    diagnostic information in real-time.
    """
    
    update_signal = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up the UI
        self.setup_ui()
        
        # Connect signals
        self.update_signal.connect(self.update_display)
    
    def setup_ui(self):
        """Create UI components."""
        self.layout = QVBoxLayout(self)
        
        # Progress bar
        self.progress_label = QLabel("Estimation Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        
        # Likelihood value
        self.likelihood_label = QLabel("Log-likelihood: N/A")
        
        # Parameter values
        self.params_label = QLabel("Parameters:")
        self.params_value = QLabel("N/A")
        
        # Add components to layout
        self.layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.likelihood_label)
        self.layout.addWidget(self.params_label)
        self.layout.addWidget(self.params_value)
    
    @pyqtSlot(dict)
    def update_display(self, update_data):
        """
        Update the monitoring display with new data.
        
        Parameters
        ----------
        update_data : dict
            Dictionary with update information
        """
        # Update progress if available
        if 'progress' in update_data:
            self.progress_bar.setValue(int(update_data['progress'] * 100))
        
        # Update likelihood if available
        if 'likelihood' in update_data:
            self.likelihood_label.setText(f"Log-likelihood: {update_data['likelihood']:.4f}")
        
        # Update parameters if available
        if 'params' in update_data:
            params_text = "<br>".join([f"{name}: {value:.4f}" for name, value in update_data['params'].items()])
            self.params_value.setText(params_text)
    
    async def monitor_estimation(self, model, data):
        """
        Asynchronously monitor model estimation.
        
        Parameters
        ----------
        model : Model instance
            Model to monitor
        data : np.ndarray
            Data for estimation
        """
        try:
            # Start estimation
            estimation_task = asyncio.create_task(model.async_fit(data))
            
            # Monitor progress
            while not estimation_task.done():
                # Get current model state
                update_data = {}
                
                if hasattr(model, 'params') and model.params is not None:
                    update_data['params'] = model.params
                
                if hasattr(model, 'loglikelihood') and model.loglikelihood is not None:
                    update_data['likelihood'] = model.loglikelihood
                
                # Update display
                self.update_signal.emit(update_data)
                
                # Wait briefly
                await asyncio.sleep(0.1)
            
            # Get final result
            result = await estimation_task
            
            # Update with final values
            update_data = {
                'progress': 1.0,
                'likelihood': model.loglikelihood,
                'params': model.params
            }
            self.update_signal.emit(update_data)
            
            return result
            
        except Exception as e:
            # Handle errors
            print(f"Error monitoring estimation: {str(e)}")
            raise
```

## Performance Metrics Tracking

### Likelihood Monitoring

Track likelihood values during model estimation to monitor convergence:

```python
import matplotlib.pyplot as plt
import numpy as np

class LikelihoodTracker:
    """
    Track and visualize likelihood evolution during optimization.
    """
    
    def __init__(self):
        self.iteration_values = []
        self.likelihood_values = []
    
    def update(self, iteration, likelihood):
        """
        Update the tracker with new values.
        
        Parameters
        ----------
        iteration : int
            Current iteration
        likelihood : float
            Current likelihood value
        """
        self.iteration_values.append(iteration)
        self.likelihood_values.append(likelihood)
    
    def plot(self):
        """
        Plot likelihood evolution.
        
        Returns
        -------
        fig : Figure
            Matplotlib figure with likelihood plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.iteration_values, self.likelihood_values, 'b-')
        ax.set_title('Log-Likelihood Evolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-Likelihood')
        ax.grid(True)
        return fig
    
    def get_stats(self):
        """
        Get summary statistics.
        
        Returns
        -------
        dict
            Dictionary with summary statistics
        """
        if not self.likelihood_values:
            return {}
            
        return {
            'iterations': len(self.iteration_values),
            'initial_likelihood': self.likelihood_values[0],
            'final_likelihood': self.likelihood_values[-1],
            'improvement': self.likelihood_values[-1] - self.likelihood_values[0],
            'convergence_rate': (self.likelihood_values[-1] - self.likelihood_values[0]) / len(self.iteration_values)
        }
```

### Model Selection Criteria

Monitor information criteria for model selection:

```python
import numpy as np
import pandas as pd

def calculate_information_criteria(loglikelihood, n_params, n_obs):
    """
    Calculate information criteria for model selection.
    
    Parameters
    ----------
    loglikelihood : float
        Model log-likelihood
    n_params : int
        Number of model parameters
    n_obs : int
        Number of observations
        
    Returns
    -------
    dict
        Dictionary with information criteria
    """
    aic = -2 * loglikelihood + 2 * n_params
    bic = -2 * loglikelihood + n_params * np.log(n_obs)
    hqic = -2 * loglikelihood + 2 * n_params * np.log(np.log(n_obs))
    
    return {
        'AIC': aic,
        'BIC': bic,
        'HQIC': hqic,
        'loglikelihood': loglikelihood
    }

def compare_models(models, data):
    """
    Compare multiple models using information criteria.
    
    Parameters
    ----------
    models : list
        List of model instances
    data : np.ndarray
        Data used for estimation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with model comparison
    """
    results = []
    
    for i, model in enumerate(models):
        # Extract model information
        model_name = f"Model {i+1}"
        if hasattr(model, 'name'):
            model_name = model.name
        
        # Get diagnostics
        diagnostics = model.diagnostic_tests()
        
        # Create result entry
        result = {
            'Model': model_name,
            'Log-Likelihood': model.loglikelihood,
            'AIC': diagnostics['AIC'],
            'BIC': diagnostics['BIC'],
            'Parameters': len(model.params) - 3  # Subtract structural parameters
        }
        results.append(result)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Sort by AIC (lower is better)
    comparison_df = comparison_df.sort_values('AIC')
    
    return comparison_df
```

### Numba Performance Tracking

Comprehensive Numba performance monitor:

```python
import time
import numba
import numpy as np
import pandas as pd

class NumbaPerformanceMonitor:
    """
    Monitor and benchmark Numba-optimized functions.
    """
    
    def __init__(self):
        self.benchmarks = {}
    
    def benchmark_function(self, func_name, func, *args, **kwargs):
        """
        Benchmark a Numba-optimized function.
        
        Parameters
        ----------
        func_name : str
            Name of the function
        func : function
            Numba-decorated function
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        dict
            Benchmark results
        """
        # Check if function is Numba-decorated
        is_numba = hasattr(func, 'nopython') or hasattr(func, 'jit')
        
        # Create benchmark entry
        benchmark = {
            'function': func_name,
            'is_numba': is_numba,
            'measurements': []
        }
        
        # Run multiple measurements
        n_runs = 5
        for i in range(n_runs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            benchmark['measurements'].append(elapsed)
        
        # Calculate statistics
        measurements = np.array(benchmark['measurements'])
        benchmark['avg_time'] = float(np.mean(measurements))
        benchmark['min_time'] = float(np.min(measurements))
        benchmark['max_time'] = float(np.max(measurements))
        benchmark['std_time'] = float(np.std(measurements))
        
        # Store benchmark
        self.benchmarks[func_name] = benchmark
        
        return benchmark
    
    def compare_functions(self, numba_func, python_func, *args, **kwargs):
        """
        Compare Numba-optimized function with pure Python equivalent.
        
        Parameters
        ----------
        numba_func : function
            Numba-decorated function
        python_func : function
            Pure Python equivalent function
        *args, **kwargs
            Arguments to pass to both functions
            
        Returns
        -------
        dict
            Comparison results
        """
        # Benchmark Numba function
        numba_result = self.benchmark_function("numba_version", numba_func, *args, **kwargs)
        
        # Benchmark Python function
        python_result = self.benchmark_function("python_version", python_func, *args, **kwargs)
        
        # Calculate speedup
        speedup = python_result['avg_time'] / numba_result['avg_time']
        
        return {
            'numba_time': numba_result['avg_time'],
            'python_time': python_result['avg_time'],
            'speedup': speedup,
            'numba_result': numba_result,
            'python_result': python_result
        }
    
    def generate_report(self):
        """
        Generate a performance report.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with benchmark results
        """
        if not self.benchmarks:
            return pd.DataFrame()
        
        data = []
        for name, benchmark in self.benchmarks.items():
            data.append({
                'Function': name,
                'Numba Optimized': benchmark['is_numba'],
                'Avg Time (s)': benchmark['avg_time'],
                'Min Time (s)': benchmark['min_time'],
                'Max Time (s)': benchmark['max_time'],
                'Std Dev (s)': benchmark['std_time']
            })
        
        return pd.DataFrame(data)
```

## Visual Diagnostic Tools

### ResidualPlotWidget Implementation

Implement a PyQt6-based widget for displaying residual diagnostic plots:

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt6agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import statsmodels.graphics.tsaplots as tsaplots

class ResidualPlotWidget(QWidget):
    """
    PyQt6 widget for displaying and updating residual diagnostic plots.
    
    This widget integrates Matplotlib for high-quality plotting and provides
    interactive visualizations for model diagnostics.
    """
    
    plot_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create Matplotlib figure and canvas
        self._figure = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setParent(self)
        
        # Create Matplotlib toolbar
        self._toolbar = NavigationToolbar(self._canvas, self)
        
        # Create layout and add components
        self._layout = QVBoxLayout()
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        self.setLayout(self._layout)
        
        # Initialize plot axes
        self._axes = []
        nrows, ncols = (2, 2)
        for i in range(nrows * ncols):
            self._axes.append(self._figure.add_subplot(nrows, ncols, i + 1))
        
        # Set tight layout
        self._figure.tight_layout()
        
    def update_plots(self, residuals, diagnostic_results):
        """
        Update all residual diagnostic plots with new data.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        diagnostic_results : dict
            Dictionary with diagnostic test results
        """
        try:
            # Clear existing plots
            for ax in self._axes:
                ax.clear()
            
            # Get axes for specific plots
            residual_ax = self._axes[0]
            acf_ax = self._axes[1]
            pacf_ax = self._axes[2]
            hist_ax = self._axes[3]
            
            # Plot 1: Residual time series
            residual_ax.plot(residuals, 'b-', alpha=0.7)
            residual_ax.axhline(y=0, color='r', linestyle='-', linewidth=0.8)
            residual_ax.set_title('Residuals')
            residual_ax.set_xlabel('Observation')
            residual_ax.set_ylabel('Residual')
            
            # Plot 2: ACF of residuals
            tsaplots.plot_acf(residuals, ax=acf_ax, lags=20, alpha=0.05)
            acf_ax.set_title('Autocorrelation Function')
            
            # Plot 3: PACF of residuals
            tsaplots.plot_pacf(residuals, ax=pacf_ax, lags=20, alpha=0.05)
            pacf_ax.set_title('Partial Autocorrelation Function')
            
            # Plot 4: Residual histogram with normal distribution overlay
            hist_ax.hist(residuals, bins=20, density=True, alpha=0.7, color='b')
            
            # Add normal distribution overlay if residual stats are available
            if 'residual_stats' in diagnostic_results:
                # Extract mean and std dev
                mean = diagnostic_results['residual_stats']['mean']
                std = diagnostic_results['residual_stats']['std_dev']
                
                # Generate points for normal distribution curve
                x = np.linspace(min(residuals), max(residuals), 100)
                y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
                
                # Plot normal distribution
                hist_ax.plot(x, y, 'r-', linewidth=2)
                
                # Add Jarque-Bera test results if available
                if 'jarque_bera' in diagnostic_results:
                    jb_stat = diagnostic_results['jarque_bera']['statistic']
                    jb_pval = diagnostic_results['jarque_bera']['p_value']
                    hist_ax.text(0.05, 0.95, f"JB: {jb_stat:.2f}\np: {jb_pval:.4f}",
                                transform=hist_ax.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            hist_ax.set_title('Residual Distribution')
            hist_ax.set_xlabel('Residual')
            hist_ax.set_ylabel('Density')
            
            # Update layout
            self._figure.tight_layout()
            self._canvas.draw()
            
            # Emit signal that plots have been updated
            self.plot_updated.emit()
            
        except Exception as e:
            print(f"Error updating residual plots: {str(e)}")
            raise
```

### Interactive Plot Components

Create interactive diagnostic plots with PyQt6:

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton
from PyQt6.QtCore import pyqtSlot
from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
import scipy.stats as stats

class InteractiveDiagnosticPlot(QWidget):
    """
    Interactive plot with configurable plot types.
    
    This widget allows users to select different diagnostic
    plot types and update visualizations in real-time.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Plot types
        self.plot_types = [
            "Residuals",
            "ACF/PACF",
            "Distribution",
            "Q-Q Plot"
        ]
        
        # Set up UI
        self.setup_ui()
        
        # Data
        self.data = None
        self.model = None
    
    def setup_ui(self):
        """Set up UI components."""
        layout = QVBoxLayout(self)
        
        # Plot type selector
        self.plot_selector = QComboBox()
        self.plot_selector.addItems(self.plot_types)
        self.plot_selector.currentIndexChanged.connect(self.update_plot)
        
        # Update button
        self.update_button = QPushButton("Refresh Plot")
        self.update_button.clicked.connect(self.update_plot)
        
        # Matplotlib canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Add components to layout
        layout.addWidget(self.plot_selector)
        layout.addWidget(self.update_button)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
    
    def set_data(self, data, model=None):
        """
        Set data for plotting.
        
        Parameters
        ----------
        data : np.ndarray
            Data for plotting
        model : Model, optional
            Model instance if available
        """
        self.data = data
        self.model = model
        self.update_plot()
    
    @pyqtSlot()
    def update_plot(self):
        """Update the current plot."""
        if self.data is None:
            return
            
        # Clear current plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get selected plot type
        plot_type = self.plot_selector.currentText()
        
        # Create appropriate plot
        if plot_type == "Residuals":
            ax.plot(self.data, 'b-')
            ax.axhline(y=0, color='r', linestyle='-')
            ax.set_title('Residuals')
            ax.set_xlabel('Observation')
            ax.set_ylabel('Residual')
            
        elif plot_type == "ACF/PACF":
            import statsmodels.graphics.tsaplots as tsaplots
            
            # Clear for subplot creation
            self.figure.clear()
            
            # Create ACF subplot
            ax1 = self.figure.add_subplot(211)
            tsaplots.plot_acf(self.data, ax=ax1, lags=20)
            ax1.set_title('Autocorrelation Function')
            
            # Create PACF subplot
            ax2 = self.figure.add_subplot(212)
            tsaplots.plot_pacf(self.data, ax=ax2, lags=20)
            ax2.set_title('Partial Autocorrelation Function')
            
        elif plot_type == "Distribution":
            # Histogram
            ax.hist(self.data, bins=30, density=True, alpha=0.7)
            
            # Normal overlay
            x = np.linspace(min(self.data), max(self.data), 100)
            mu, sigma = np.mean(self.data), np.std(self.data)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
            
            # Distribution stats
            ax.set_title('Residual Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            
            # Add Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(self.data)
            ax.text(0.05, 0.95, f"Jarque-Bera: {jb_stat:.2f}\np-value: {jb_p:.4f}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        elif plot_type == "Q-Q Plot":
            stats.probplot(self.data, plot=ax)
            ax.set_title('Q-Q Plot')
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()
```

## Interactive Monitoring

### Async Monitor Implementation

Implement asynchronous monitoring for non-blocking model estimation:

```python
import asyncio
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

class AsyncMonitor(QObject):
    """
    Asynchronous monitoring handler for long-running estimation.
    
    This class bridges between async model estimation and the PyQt6
    event loop for real-time monitoring without blocking the UI.
    """
    
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(bool)
    error_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
    
    @pyqtSlot()
    async def monitor_estimation(self, model, data):
        """
        Asynchronously monitor model estimation.
        
        Parameters
        ----------
        model : Model instance
            Model to monitor
        data : np.ndarray
            Data for estimation
        """
        self._running = True
        
        try:
            # Start estimation in a separate task
            estimation_task = asyncio.create_task(model.async_fit(data))
            
            # Monitor progress while estimation is running
            while not estimation_task.done() and self._running:
                # Collect current model state
                update_data = {}
                
                # Monitor likelihood if available
                if hasattr(model, 'loglikelihood') and model.loglikelihood is not None:
                    update_data['likelihood'] = model.loglikelihood
                
                # Monitor parameters if available
                if hasattr(model, 'params') and model.params is not None:
                    update_data['params'] = model.params
                
                # Emit update signal
                self.update_signal.emit(update_data)
                
                # Sleep briefly to avoid overloading the event loop
                await asyncio.sleep(0.1)
            
            # Check if we were stopped
            if not self._running:
                return
                
            # Wait for estimation to complete
            result = await estimation_task
            
            # Emit final update
            final_data = {
                'complete': True,
                'likelihood': model.loglikelihood,
                'params': model.params,
                'converged': model._optimizer.converged if hasattr(model, '_optimizer') else None
            }
            self.update_signal.emit(final_data)
            
            # Emit finished signal
            self.finished_signal.emit(True)
            
        except Exception as e:
            # Emit error signal
            self.error_signal.emit(str(e))
            
        finally:
            self._running = False
    
    def stop_monitoring(self):
        """Stop the monitoring process."""
        self._running = False
```

### Parameter Tracking Table

Implement a PyQt6 table for tracking model parameters:

```python
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt

class ParameterMonitorTable(QTableWidget):
    """
    Table widget for displaying and tracking model parameters.
    
    This widget displays parameter values, standard errors, and
    significance tests in real-time during estimation.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up the table
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels([
            "Parameter", "Value", "Std. Error", "t-stat", "p-value"
        ])
        
        # Set column properties
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 5):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        # Set alternating row colors
        self.setAlternatingRowColors(True)
    
    def update_parameters(self, model):
        """
        Update parameter display with model information.
        
        Parameters
        ----------
        model : Model instance
            Model with parameters to display
        """
        if model is None or not hasattr(model, 'params') or model.params is None:
            return
            
        try:
            # Get diagnostic information
            diagnostics = model.diagnostic_tests()
            
            if 'parameter_summary' not in diagnostics:
                return
                
            param_summary = diagnostics['parameter_summary']
            
            # Set row count
            self.setRowCount(len(param_summary))
            
            # Fill parameter table
            for i, param in enumerate(param_summary):
                # Parameter name
                name_item = QTableWidgetItem(param['name'])
                self.setItem(i, 0, name_item)
                
                # Value
                value_item = QTableWidgetItem(f"{param['value']:.4f}")
                self.setItem(i, 1, value_item)
                
                # Standard error
                if param['std_error'] is not None:
                    std_err_item = QTableWidgetItem(f"{param['std_error']:.4f}")
                else:
                    std_err_item = QTableWidgetItem("N/A")
                self.setItem(i, 2, std_err_item)
                
                # t-statistic
                if param['t_statistic'] is not None:
                    t_stat_item = QTableWidgetItem(f"{param['t_statistic']:.4f}")
                else:
                    t_stat_item = QTableWidgetItem("N/A")
                self.setItem(i, 3, t_stat_item)
                
                # p-value
                if param['p_value'] is not None:
                    p_value_item = QTableWidgetItem(f"{param['p_value']:.4f}")
                    
                    # Highlight significant parameters
                    if param['p_value'] < 0.05:
                        p_value_item.setBackground(Qt.GlobalColor.green)
                    elif param['p_value'] < 0.1:
                        p_value_item.setBackground(Qt.GlobalColor.yellow)
                else:
                    p_value_item = QTableWidgetItem("N/A")
                self.setItem(i, 4, p_value_item)
                
        except Exception as e:
            print(f"Error updating parameter table: {str(e)}")
```

### Real-time Visualization

Implement real-time visualization with PyQt6 and Matplotlib:

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

class RealTimeMonitorPlot(QWidget):
    """
    Widget for real-time visualization of monitoring metrics.
    
    This widget provides real-time visualization of model estimation
    progress, likelihood evolution, and other monitoring metrics.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up data storage
        self.iterations = []
        self.likelihood_values = []
        self.param_values = {}
        
        # Create matplotlib figure and canvas
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Create axes
        self.likelihood_ax = self.fig.add_subplot(211)
        self.params_ax = self.fig.add_subplot(212)
        
        # Create layout and add components
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Set up update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(500)  # Update every 500ms
    
    def add_data_point(self, iteration, likelihood, params):
        """
        Add a new data point for visualization.
        
        Parameters
        ----------
        iteration : int
            Current iteration
        likelihood : float
            Current likelihood value
        params : dict
            Dictionary of parameter values
        """
        self.iterations.append(iteration)
        self.likelihood_values.append(likelihood)
        
        # Store parameter values
        for name, value in params.items():
            if name not in self.param_values:
                self.param_values[name] = []
            self.param_values[name].append(value)
    
    def update_plot(self):
        """Update the visualization with current data."""
        # Clear current plots
        self.likelihood_ax.clear()
        self.params_ax.clear()
        
        # Plot likelihood evolution if data is available
        if self.iterations and self.likelihood_values:
            self.likelihood_ax.plot(self.iterations, self.likelihood_values, 'b-')
            self.likelihood_ax.set_title('Log-Likelihood Evolution')
            self.likelihood_ax.set_xlabel('Iteration')
            self.likelihood_ax.set_ylabel('Log-Likelihood')
            self.likelihood_ax.grid(True)
        
        # Plot parameter evolution if data is available
        if self.iterations and self.param_values:
            for name, values in self.param_values.items():
                if len(values) == len(self.iterations):
                    self.params_ax.plot(self.iterations, values, label=name)
            
            self.params_ax.set_title('Parameter Evolution')
            self.params_ax.set_xlabel('Iteration')
            self.params_ax.set_ylabel('Parameter Value')
            self.params_ax.grid(True)
            self.params_ax.legend()
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def clear_data(self):
        """Clear all stored data."""
        self.iterations = []
        self.likelihood_values = []
        self.param_values = {}
        self.update_plot()
```

## Additional Resources

### Python Logging Configuration

```python
import logging
import sys
import os
from datetime import datetime

def configure_monitoring_logging(log_dir="logs", log_level=logging.INFO):
    """
    Configure logging for MFE Toolbox monitoring.
    
    Parameters
    ----------
    log_dir : str
        Directory for log files
    log_level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns
    -------
    logger : Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("mfe_monitor")
    logger.setLevel(log_level)
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mfe_monitor_{timestamp}.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```