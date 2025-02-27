# MFE Toolbox Usage Guide

This document provides comprehensive instructions for using the MFE (Financial Econometrics) Toolbox, a Python-based suite for financial time series modeling, volatility analysis, and econometric research.

## Table of Contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Model Examples](#model-examples)
4. [GUI Guide](#gui-guide)

## Installation

### Prerequisites

The MFE Toolbox requires Python 3.12 or later and depends on the following packages:

- NumPy (1.26.3 or later)
- SciPy (1.11.4 or later)
- Pandas (2.1.4 or later)
- Statsmodels (0.14.1 or later)
- Numba (0.59.0 or later)
- PyQt6 (6.6.1 or later) - for GUI components

### Standard Installation

The simplest way to install the MFE Toolbox is via pip:

```bash
pip install mfe
```

This will install the MFE Toolbox and all its dependencies.

### Development Installation

For development or to access the latest features:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/mfe-toolbox.git
   cd mfe-toolbox
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

### Virtual Environment (Recommended)

We recommend using a virtual environment to avoid conflicts with other packages:

```bash
# Create a virtual environment
python -m venv mfe-env

# Activate the environment
# On Windows:
mfe-env\Scripts\activate
# On macOS/Linux:
source mfe-env/bin/activate

# Install MFE Toolbox
pip install mfe
```

## Quickstart

### Basic Import and Setup

```python
# Import key modules
import numpy as np
import pandas as pd
from mfe.models import ARMAX
from mfe.ui import launch_gui

# Set random seed for reproducibility
np.random.seed(42)
```

### Simple Model Estimation

The following example demonstrates how to estimate a simple AR(1) model:

```python
# Generate sample data
n = 1000
ar_coef = 0.7
data = np.zeros(n)
for t in range(1, n):
    data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)

# Create and fit the model
model = ARMAX(p=1, q=0)  # AR(1) model

# Using async/await pattern for estimation
import asyncio

async def estimate():
    # Fit the model
    converged = await model.async_fit(data)
    if converged:
        print("Model converged successfully!")
        print(f"Estimated AR coefficient: {model._model_params['ar_params'][0]:.4f}")
        
        # Run diagnostic tests
        diagnostics = model.diagnostic_tests()
        print(f"Log-likelihood: {model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
    else:
        print("Model estimation did not converge")

# Run the estimation
asyncio.run(estimate())
```

### Forecasting Example

Once you have estimated a model, you can generate forecasts:

```python
# Generate 10-step ahead forecasts
forecasts = model.forecast(steps=10)
print("Forecasts:", forecasts)
```

### Launch the GUI

The MFE Toolbox includes a graphical user interface for interactive modeling:

```python
# Launch the GUI
from mfe.ui import launch_gui
launch_gui()
```

This will open the main application window where you can configure models, estimate parameters, and visualize results interactively.

## Model Examples

### ARMAX Model

#### Data Preparation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mfe.models import ARMAX

# Load data
# For example, using Pandas to load a CSV file
# df = pd.read_csv('your_data.csv')
# data = df['your_column'].values

# Or generate sample data
n = 1000
ar_params = [0.7, -0.2]  # AR(2) process
ma_params = [0.5]        # MA(1) process
data = np.zeros(n)
errors = np.random.normal(0, 1, n)

# Generate ARMA(2,1) process
for t in range(2, n):
    data[t] = ar_params[0] * data[t-1] + ar_params[1] * data[t-2] + errors[t] + ma_params[0] * errors[t-1]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Simulated ARMA(2,1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

#### Model Configuration and Estimation

```python
# Create the model
model = ARMAX(p=2, q=1, include_constant=True)

# Asynchronous estimation
import asyncio

async def estimate_model():
    print("Estimating ARMA(2,1) model...")
    
    # Fit the model
    converged = await model.async_fit(data)
    
    if converged:
        print("Model estimation converged successfully!")
        
        # Extract parameter estimates
        ar_params, ma_params, constant, _ = model._extract_params(model.params)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1): {ar_params[0]:.4f} (True: {ar_params[0]:.4f})")
        print(f"AR(2): {ar_params[1]:.4f} (True: {ar_params[1]:.4f})")
        print(f"MA(1): {ma_params[0]:.4f} (True: {ma_params[0]:.4f})")
        if constant is not None:
            print(f"Constant: {constant:.4f}")
            
        # Get diagnostics
        diagnostics = model.diagnostic_tests()
        
        # Print statistical metrics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Print residual diagnostics
        ljung_box = diagnostics['ljung_box']
        jarque_bera = diagnostics['jarque_bera']
        
        print(f"\nDiagnostic Tests:")
        print(f"Ljung-Box Q({ljung_box['lags']}): {ljung_box['statistic']:.4f}, p-value: {ljung_box['p_value']:.4f}")
        print(f"Jarque-Bera: {jarque_bera['statistic']:.4f}, p-value: {jarque_bera['p_value']:.4f}")
        
        # Plot residuals
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(model.residuals)
        plt.title('Model Residuals')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        
        plt.subplot(2, 1, 2)
        plt.hist(model.residuals, bins=30, density=True, alpha=0.7)
        plt.title('Residual Distribution')
        plt.xlabel('Residual')
        plt.ylabel('Density')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Model estimation did not converge")

# Run the estimation
asyncio.run(estimate_model())
```

#### Forecasting

```python
# Generate forecasts
forecast_steps = 20
forecasts = model.forecast(steps=forecast_steps)

# Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(data, label='Historical Data')
plt.plot(range(len(data), len(data) + forecast_steps), forecasts, 'r--', label='Forecast')
plt.title('ARMA(2,1) Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

#### ARMAX with Exogenous Variables

```python
# Generate sample data with exogenous variables
n = 1000
exog = np.random.normal(0, 1, (n, 2))  # Two exogenous variables
exog_params = [0.5, -0.3]              # True exogenous parameters

# Generate ARMAX process
data_exog = np.zeros(n)
errors = np.random.normal(0, 1, n)

for t in range(2, n):
    # AR component
    ar_component = ar_params[0] * data_exog[t-1] + ar_params[1] * data_exog[t-2]
    
    # MA component
    ma_component = errors[t] + ma_params[0] * errors[t-1]
    
    # Exogenous component
    exog_component = exog_params[0] * exog[t, 0] + exog_params[1] * exog[t, 1]
    
    # Combined
    data_exog[t] = ar_component + ma_component + exog_component

# Create and fit ARMAX model
armax_model = ARMAX(p=2, q=1, include_constant=True)

async def estimate_armax():
    print("Estimating ARMAX(2,1) model with exogenous variables...")
    
    # Fit the model
    converged = await armax_model.async_fit(data_exog, exog=exog)
    
    if converged:
        print("ARMAX estimation converged successfully!")
        
        # Extract parameter estimates
        ar_params_est, ma_params_est, constant, exog_params_est = armax_model._extract_params(armax_model.params)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1): {ar_params_est[0]:.4f} (True: {ar_params[0]:.4f})")
        print(f"AR(2): {ar_params_est[1]:.4f} (True: {ar_params[1]:.4f})")
        print(f"MA(1): {ma_params_est[0]:.4f} (True: {ma_params[0]:.4f})")
        
        if exog_params_est is not None:
            for i, param in enumerate(exog_params_est):
                print(f"Exog{i+1}: {param:.4f} (True: {exog_params[i]:.4f})")
        
        # Get diagnostics
        diagnostics = armax_model.diagnostic_tests()
        
        # Print statistical metrics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {armax_model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
    else:
        print("ARMAX estimation did not converge")

# Run the estimation
asyncio.run(estimate_armax())

# Generate forecasts with exogenous variables
forecast_steps = 10
exog_future = np.random.normal(0, 1, (forecast_steps, 2))  # Future exogenous values
forecasts_exog = armax_model.forecast(steps=forecast_steps, exog_future=exog_future)

print(f"ARMAX Forecasts: {forecasts_exog}")
```

### Volatility Models

For volatility modeling, the MFE Toolbox provides a range of GARCH-type models:

```python
from mfe.models import GARCH, EGARCH, GJR_GARCH

# Generate sample returns
n = 2000
returns = np.random.normal(0, 1, n)

# Add volatility clustering
volatility = np.ones(n)
for t in range(1, n):
    volatility[t] = 0.1 + 0.85 * volatility[t-1] + 0.1 * returns[t-1]**2
    returns[t] *= np.sqrt(volatility[t])

# Create and estimate GARCH(1,1) model
garch_model = GARCH(p=1, q=1)

async def estimate_garch():
    print("Estimating GARCH(1,1) model...")
    
    # Fit the model
    converged = await garch_model.async_fit(returns)
    
    if converged:
        print("GARCH estimation converged successfully!")
        print(f"\nEstimated Parameters:")
        print(f"omega: {garch_model._model_params['omega']:.4f}")
        print(f"alpha: {garch_model._model_params['alpha'][0]:.4f}")
        print(f"beta: {garch_model._model_params['beta'][0]:.4f}")
        
        # Get diagnostics
        diagnostics = garch_model.diagnostic_tests()
        
        # Print statistical metrics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {garch_model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Plot volatility
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(returns)
        plt.title('Returns')
        plt.xlabel('Time')
        plt.ylabel('Return')
        
        plt.subplot(2, 1, 2)
        plt.plot(garch_model.conditional_volatility)
        plt.title('Estimated Conditional Volatility')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        
        plt.tight_layout()
        plt.show()
    else:
        print("GARCH estimation did not converge")

# Run the estimation
asyncio.run(estimate_garch())

# Forecast volatility
volatility_forecast = garch_model.forecast_variance(steps=20)
print(f"Volatility Forecasts: {volatility_forecast}")
```

### High-Frequency Analysis

The MFE Toolbox provides tools for analyzing high-frequency financial data:

```python
from mfe.models import realized_variance, realized_kernel

# Sample high-frequency data
import pandas as pd
from datetime import datetime, timedelta

# Create sample intraday data
n_days = 5
n_intraday = 390  # e.g., 1-minute data for 6.5 hours
timestamps = []
prices = []
price = 100.0

for day in range(n_days):
    base_date = datetime(2023, 1, 2) + timedelta(days=day)
    daily_return = np.random.normal(0, 0.01)
    daily_vol = 0.02 + 0.5 * np.random.random()
    
    for minute in range(n_intraday):
        time = base_date + timedelta(minutes=minute)
        timestamps.append(time)
        
        # Add intraday pattern (U-shape volatility)
        intraday_factor = 1.0 + 0.5 * (
            np.exp(-((minute - 0) / 60)**2) + 
            np.exp(-((minute - (n_intraday-1)) / 60)**2)
        )
        
        # Generate price with realistic microstructure noise
        price_innovation = np.random.normal(daily_return/n_intraday, daily_vol/np.sqrt(n_intraday) * intraday_factor)
        price *= np.exp(price_innovation)
        
        # Add microstructure noise
        noisy_price = price * (1 + np.random.normal(0, 0.0001))
        prices.append(noisy_price)

# Create DataFrame
hf_data = pd.DataFrame({
    'timestamp': timestamps,
    'price': prices
})

# Convert to arrays for realized measures
times = np.array([t.timestamp() for t in hf_data['timestamp']])
price_array = np.array(hf_data['price'])

# Compute realized variance with different sampling schemes
rv, rv_ss = realized_variance(
    price_array, 
    times, 
    timeType='timestamp',
    samplingType='CalendarTime',
    samplingInterval=5  # 5-minute sampling
)

print(f"Realized Variance (5-min): {rv:.8f}")
print(f"Realized Variance (subsampled): {rv_ss:.8f}")

# Compute realized kernel (robust to noise)
rk = realized_kernel(
    price_array,
    times,
    timeType='timestamp',
    kernelType='Parzen'
)

print(f"Realized Kernel: {rk:.8f}")

# Plot high-frequency price and returns
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(hf_data['timestamp'], hf_data['price'])
plt.title('High-Frequency Price')
plt.ylabel('Price')

plt.subplot(2, 1, 2)
returns = np.diff(np.log(hf_data['price'])) * 100
plt.plot(hf_data['timestamp'][1:], returns)
plt.title('High-Frequency Returns (%)')
plt.ylabel('Return')
plt.xlabel('Time')

plt.tight_layout()
plt.show()
```

### Bootstrap Analysis

The MFE Toolbox provides robust bootstrap methods for dependent data:

```python
from mfe.core import block_bootstrap, stationary_bootstrap

# Example with AR(1) process
n = 500
ar_coef = 0.7
data = np.zeros(n)
for t in range(1, n):
    data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)

# Define statistic of interest (e.g., mean)
def compute_mean(x):
    return np.mean(x)

# Block bootstrap
n_bootstrap = 1000
block_size = 50  # Fixed block size
block_results = block_bootstrap(
    data, 
    compute_mean, 
    n_bootstrap=n_bootstrap, 
    block_size=block_size
)

# Stationary bootstrap
expected_block_size = 50  # Expected block size
stationary_results = stationary_bootstrap(
    data, 
    compute_mean, 
    n_bootstrap=n_bootstrap, 
    expected_block_size=expected_block_size
)

# Calculate bootstrap confidence intervals
alpha = 0.05
block_ci = np.percentile(block_results, [alpha/2*100, (1-alpha/2)*100])
stationary_ci = np.percentile(stationary_results, [alpha/2*100, (1-alpha/2)*100])

# Print results
print(f"Original Mean: {compute_mean(data):.4f}")
print(f"Block Bootstrap 95% CI: [{block_ci[0]:.4f}, {block_ci[1]:.4f}]")
print(f"Stationary Bootstrap 95% CI: [{stationary_ci[0]:.4f}, {stationary_ci[1]:.4f}]")

# Plot bootstrap distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(block_results, bins=30, alpha=0.7)
plt.axvline(compute_mean(data), color='r', linestyle='--', label='Sample Mean')
plt.axvline(block_ci[0], color='g', linestyle='-', label='95% CI')
plt.axvline(block_ci[1], color='g', linestyle='-')
plt.title('Block Bootstrap Distribution')
plt.xlabel('Mean')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(stationary_results, bins=30, alpha=0.7)
plt.axvline(compute_mean(data), color='r', linestyle='--', label='Sample Mean')
plt.axvline(stationary_ci[0], color='g', linestyle='-', label='95% CI')
plt.axvline(stationary_ci[1], color='g', linestyle='-')
plt.title('Stationary Bootstrap Distribution')
plt.xlabel('Mean')
plt.legend()

plt.tight_layout()
plt.show()
```

## GUI Guide

The MFE Toolbox provides a comprehensive graphical user interface (GUI) built with PyQt6 for interactive model estimation and analysis.

### Launching the GUI

```python
from mfe.ui import launch_gui

# Launch the main application
launch_gui()
```

### Main Application Window

The main application window is organized into several sections:

![Main Application Window](./images/main_window.png)

#### Model Configuration Panel

- **AR Order**: Specify the autoregressive order (p)
- **MA Order**: Specify the moving average order (q)
- **Include Constant**: Toggle to include a constant term in the model
- **Exogenous Variables**: Select and configure exogenous variables if available

#### Data Panel

- **Load Data**: Import time series data from CSV, Excel, or other formats
- **Sample Range**: Specify the sample period to use for estimation
- **Transformations**: Apply transformations like log, difference, etc.
- **View Data**: Plot the loaded data with basic statistics

#### Estimation Controls

- **Estimate Model**: Start the model estimation process
- **Reset**: Clear current results and reset parameters
- **Stop**: Cancel a running estimation
- **Progress Bar**: Shows estimation progress

#### Diagnostic Panel

This panel displays basic diagnostic information once the model is estimated:
- Residual plot
- Autocorrelation function (ACF)
- Partial autocorrelation function (PACF)
- Parameter estimates summary

### Results Viewer

After estimating a model, you can view detailed results in the Results Viewer dialog:

![Results Viewer](./images/results_viewer.png)

#### Model Equation

Displays the estimated model equation with parameter values.

#### Parameter Estimates

Shows a table with parameter estimates, standard errors, t-statistics, and p-values.

#### Statistical Metrics

Displays key model fit statistics:
- Log-likelihood
- Information criteria (AIC, BIC, HQIC)
- Other relevant metrics

#### Diagnostic Plots

The Results Viewer provides comprehensive diagnostic plots:
- Residual time series
- ACF and PACF of residuals
- Residual histogram with normal distribution overlay
- Jarque-Bera test results for normality

Navigation buttons at the bottom allow browsing through different plot pages.

### Interactive Plot Features

All plots support interactive features:
- Zoom in/out
- Pan
- Save as image
- Adjust axes
- Export data

To interact with plots:
1. Hover over the plot to see data values
2. Use the toolbar at the top of each plot for various functions
3. Right-click for additional context menu options

### Model Comparison

The GUI also supports comparing multiple models:

1. Estimate several models with different specifications
2. Click "Compare Models" in the main window
3. Select models to compare
4. View side-by-side comparisons of fit statistics and diagnostics

### Saving and Loading Results

You can save and load model results:

- **Save Results**: Saves the current model estimation to a file
- **Load Results**: Loads previously saved model estimations
- **Export Report**: Generates a comprehensive report in HTML or PDF format

### Tips for Effective Use

1. **Start Simple**: Begin with lower-order models and gradually increase complexity
2. **Check Diagnostics**: Always examine residual diagnostics for model adequacy
3. **Compare Models**: Estimate multiple models and compare using information criteria
4. **Use Tooltips**: Hover over UI elements to see helpful tooltips
5. **Keyboard Shortcuts**: Use keyboard shortcuts for common operations
   - Ctrl+E: Estimate model
   - Ctrl+R: Reset
   - Ctrl+S: Save results
   - F1: Open help