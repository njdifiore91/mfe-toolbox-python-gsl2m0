# MFE Toolbox Tutorials

A comprehensive guide to using the Python-based MFE (Financial Econometrics) Toolbox version 4.0.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Environment Setup Tutorial](#environment-setup-tutorial)
4. [ARMAX Modeling Tutorial](#armax-modeling-tutorial)
5. [GARCH Modeling Tutorial](#garch-modeling-tutorial)
6. [High-Frequency Analysis Tutorial](#high-frequency-analysis-tutorial)
7. [Bootstrap Analysis Tutorial](#bootstrap-analysis-tutorial)
8. [GUI Interface Tutorial](#gui-interface-tutorial)
9. [Advanced Topics](#advanced-topics)
10. [Performance Optimization with Numba](#performance-optimization-with-numba)

## Introduction

The MFE Toolbox is a comprehensive suite of Python modules designed for financial time series modeling and econometric analysis. Originally derived from a MATLAB implementation, this toolbox has been completely rewritten in Python 3.12, incorporating modern programming constructs such as async/await patterns and strict type hints.

### Key Features

- Time series modeling with ARMA/ARMAX
- Volatility modeling with various GARCH specifications
- High-frequency financial data analysis
- Bootstrap-based statistical inference
- Interactive GUI built with PyQt6
- Performance optimization with Numba

### Prerequisites

Before proceeding with these tutorials, you should have:
- Basic knowledge of Python programming
- Familiarity with financial time series concepts
- Working installation of Python 3.12 or higher

## Getting Started

To get started with the MFE Toolbox, first install the package following the instructions in the [Installation Guide](INSTALLATION.md).

Quick installation:

```bash
pip install mfe
```

For a more detailed setup, we recommend creating a virtual environment:

```bash
python -m venv mfe-env
source mfe-env/bin/activate  # On Windows: mfe-env\Scripts\activate
pip install mfe
```

Verify your installation:

```python
import mfe
print(mfe.__version__)  # Should print 4.0 or higher
```

## Environment Setup Tutorial

This tutorial guides you through setting up an optimal environment for financial econometrics analysis with the MFE Toolbox.

### Creating a Project Environment

Best practices suggest using a dedicated environment for financial analysis projects:

```bash
# Create a project directory
mkdir financial_analysis
cd financial_analysis

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the MFE Toolbox and dependencies
pip install mfe
```

### Setting Up a Jupyter Environment

For interactive analysis, you might want to use Jupyter notebooks:

```bash
# Install Jupyter in your virtual environment
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook
```

### Creating Your First Analysis Script

Create a new file called `initial_analysis.py`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mfe.models import ARMAX

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data (AR(1) process)
n = 1000
ar_coef = 0.7
data = np.zeros(n)
for t in range(1, n):
    data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Simulated AR(1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('ar1_process.png')
plt.show()

print("Environment setup complete!")
```

Run the script to verify your environment works correctly:

```bash
python initial_analysis.py
```

## ARMAX Modeling Tutorial

This tutorial demonstrates how to perform time series analysis using the ARMAX model from the MFE Toolbox.

### Understanding ARMAX Models

ARMAX (AutoRegressive Moving Average with eXogenous variables) models are extensions of ARMA models that include additional external regressors. The general form is:

y(t) = c + Σφᵢy(t-i) + Σθⱼε(t-j) + Σβₖx(t)ₖ + ε(t)

Where:
- y(t) is the dependent variable
- c is a constant
- φᵢ are the AR coefficients
- θⱼ are the MA coefficients
- βₖ are coefficients for exogenous variables x(t)ₖ
- ε(t) is white noise

### Basic ARMA Model Example

Let's start with a simple ARMA(2,1) model without exogenous variables:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from mfe.models import ARMAX

# Generate sample data: ARMA(2,1) process
n = 1000
ar_params = [0.7, -0.2]  # AR parameters
ma_params = [0.5]        # MA parameters
data = np.zeros(n)
errors = np.random.normal(0, 1, n)

for t in range(2, n):
    data[t] = ar_params[0] * data[t-1] + ar_params[1] * data[t-2] + errors[t] + ma_params[0] * errors[t-1]

# Create and estimate the model
model = ARMAX(p=2, q=1, include_constant=True)

async def estimate_model():
    print("Estimating ARMA(2,1) model...")
    
    # Fit the model using async/await pattern
    converged = await model.async_fit(data)
    
    if converged:
        print("Model estimation converged successfully!")
        
        # Extract parameter estimates
        ar_params_est, ma_params_est, constant, _ = model._extract_params(model.params)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1): {ar_params_est[0]:.4f} (True: {ar_params[0]:.4f})")
        print(f"AR(2): {ar_params_est[1]:.4f} (True: {ar_params[1]:.4f})")
        print(f"MA(1): {ma_params_est[0]:.4f} (True: {ma_params[0]:.4f})")
        if constant is not None:
            print(f"Constant: {constant:.4f}")
        
        # Get diagnostics
        diagnostics = model.diagnostic_tests()
        
        # Print key model fit statistics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Plot the results
        plot_results(model, data, diagnostics)
    else:
        print("Model estimation did not converge")

def plot_results(model, data, diagnostics):
    # Plot original data and residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot original data
    ax1.plot(data)
    ax1.set_title('Original Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # Plot residuals
    ax2.plot(model.residuals)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.set_title('Model Residuals')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residual')
    
    plt.tight_layout()
    plt.show()
    
    # Plot residual diagnostics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of residuals
    ax1.hist(model.residuals, bins=30, density=True, alpha=0.7)
    ax1.set_title('Residual Distribution')
    ax1.set_xlabel('Residual')
    ax1.set_ylabel('Density')
    
    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(model.residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

# Run the model estimation
asyncio.run(estimate_model())
```

### ARMAX Model with Exogenous Variables

Now, let's extend our model to include exogenous variables:

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

# Create and estimate ARMAX model
armax_model = ARMAX(p=2, q=1, include_constant=True)

async def estimate_armax():
    print("Estimating ARMAX(2,1) model with exogenous variables...")
    
    # Fit the model
    converged = await armax_model.async_fit(data_exog, exog=exog)
    
    if converged:
        print("ARMAX estimation converged successfully!")
        
        # Extract parameters
        ar_params_est, ma_params_est, constant, exog_params_est = armax_model._extract_params(armax_model.params)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1): {ar_params_est[0]:.4f} (True: {ar_params[0]:.4f})")
        print(f"AR(2): {ar_params_est[1]:.4f} (True: {ar_params[1]:.4f})")
        print(f"MA(1): {ma_params_est[0]:.4f} (True: {ma_params[0]:.4f})")
        
        if exog_params_est is not None:
            for i, param in enumerate(exog_params_est):
                print(f"Exog{i+1}: {param:.4f} (True: {exog_params[i]:.4f})")
        
        if constant is not None:
            print(f"Constant: {constant:.4f}")
        
        # Generate forecasts with future exogenous variables
        forecast_steps = 10
        exog_future = np.random.normal(0, 1, (forecast_steps, 2))
        forecasts = armax_model.forecast(steps=forecast_steps, exog_future=exog_future)
        
        print(f"\nForecasts (next {forecast_steps} steps):")
        for i, value in enumerate(forecasts):
            print(f"t+{i+1}: {value:.4f}")
    else:
        print("ARMAX estimation did not converge")

# Run the ARMAX estimation
asyncio.run(estimate_armax())
```

### Forecasting with ARMAX Models

Forecasting is a key application of ARMAX models. Here's how to generate forecasts:

```python
async def forecast_demo():
    # First fit the model
    converged = await model.async_fit(data)
    
    if not converged:
        print("Model estimation did not converge, cannot forecast")
        return
    
    # Generate forecasts for different horizons
    horizons = [1, 5, 10, 20]
    
    for h in horizons:
        forecast = model.forecast(steps=h)
        print(f"{h}-step ahead forecast: {forecast[-1]:.4f}")
    
    # Plot data with forecasts
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Historical Data')
    
    # Add forecasts for the longest horizon
    longest = max(horizons)
    forecast_long = model.forecast(steps=longest)
    forecast_indices = range(len(data), len(data) + longest)
    plt.plot(forecast_indices, forecast_long, 'r--', label=f'{longest}-step Forecast')
    
    # Add confidence intervals (simulated for demonstration)
    # In a full implementation, you would calculate proper confidence intervals
    upper_bound = forecast_long + 1.96 * np.std(model.residuals)
    lower_bound = forecast_long - 1.96 * np.std(model.residuals)
    
    plt.fill_between(forecast_indices, lower_bound, upper_bound, 
                     color='r', alpha=0.1, label='95% Confidence Interval')
    
    plt.title('ARMA Model Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Run the forecast demonstration
asyncio.run(forecast_demo())
```

### Model Selection and Diagnostics

It's important to select the best model using information criteria and diagnostic tests:

```python
async def model_selection():
    # Try different model orders
    p_values = [1, 2, 3]
    q_values = [0, 1, 2]
    
    results = []
    
    for p in p_values:
        for q in q_values:
            print(f"Estimating ARMA({p},{q})...")
            model = ARMAX(p=p, q=q, include_constant=True)
            
            try:
                converged = await model.async_fit(data)
                
                if converged:
                    # Get diagnostics
                    diagnostics = model.diagnostic_tests()
                    
                    # Store results
                    results.append({
                        'p': p,
                        'q': q,
                        'AIC': diagnostics['AIC'],
                        'BIC': diagnostics['BIC'],
                        'Log-likelihood': model.loglikelihood,
                        'Converged': converged
                    })
                else:
                    print(f"ARMA({p},{q}) did not converge")
            except Exception as e:
                print(f"Error estimating ARMA({p},{q}): {str(e)}")
    
    # Convert to DataFrame for easier analysis
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # Sort by AIC (lower is better)
    print("\nModel Selection Results (sorted by AIC):")
    print(results_df.sort_values('AIC').to_string(index=False))
    
    # Sort by BIC (lower is better)
    print("\nModel Selection Results (sorted by BIC):")
    print(results_df.sort_values('BIC').to_string(index=False))

# Run model selection
asyncio.run(model_selection())
```

### Real-World Example with Financial Data

Let's apply ARMAX modeling to real financial data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import yfinance as yahoo_finance  # For downloading financial data (pip install yfinance)
from mfe.models import ARMAX

async def analyze_financial_data():
    # Download S&P 500 data
    print("Downloading S&P 500 data...")
    sp500 = yahoo_finance.download('^GSPC', start='2018-01-01', end='2023-01-01')
    
    # Calculate daily returns
    sp500['Return'] = sp500['Adj Close'].pct_change() * 100
    returns = sp500['Return'].dropna().values
    
    # Plot returns
    plt.figure(figsize=(12, 6))
    plt.plot(returns)
    plt.title('S&P 500 Daily Returns (%)')
    plt.xlabel('Observation')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Create and fit ARMA model
    model = ARMAX(p=2, q=1, include_constant=True)
    
    print("Estimating ARMA(2,1) model on S&P 500 returns...")
    converged = await model.async_fit(returns)
    
    if converged:
        print("Model estimation converged successfully!")
        
        # Extract parameters
        ar_params, ma_params, constant, _ = model._extract_params(model.params)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1): {ar_params[0]:.4f}")
        print(f"AR(2): {ar_params[1]:.4f}")
        print(f"MA(1): {ma_params[0]:.4f}")
        if constant is not None:
            print(f"Constant: {constant:.4f}")
        
        # Get diagnostics
        diagnostics = model.diagnostic_tests()
        
        # Print key model fit statistics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Ljung-Box test for autocorrelation
        lb = diagnostics['ljung_box']
        print(f"Ljung-Box Q({lb['lags']}): {lb['statistic']:.4f}, p-value: {lb['p_value']:.4f}")
        
        # Generate forecasts
        forecast_steps = 5
        forecasts = model.forecast(steps=forecast_steps)
        
        print(f"\nForecasts (next {forecast_steps} days):")
        for i, value in enumerate(forecasts):
            print(f"Day {i+1}: {value:.4f}%")
    else:
        print("Model estimation did not converge")

# Run financial data analysis
asyncio.run(analyze_financial_data())
```

## GARCH Modeling Tutorial

This tutorial covers volatility modeling using the GARCH family of models in the MFE Toolbox.

### Understanding GARCH Models

GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models are used to analyze and forecast volatility in financial time series. The basic GARCH(1,1) model can be expressed as:

σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

Where:
- σ²(t) is the conditional variance at time t
- ω is a constant (omega)
- α is the ARCH parameter
- β is the GARCH parameter
- ε(t) is the innovation (residual) term

### Basic GARCH(1,1) Model Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from mfe.models import GARCH

# Generate sample returns with volatility clustering
np.random.seed(42)
n = 2000
returns = np.zeros(n)
volatility = np.zeros(n)
volatility[0] = 1.0

# True parameters
omega = 0.1
alpha = 0.1
beta = 0.8

# Generate GARCH(1,1) process
for t in range(1, n):
    volatility[t] = omega + alpha * returns[t-1]**2 + beta * volatility[t-1]
    returns[t] = np.random.normal(0, np.sqrt(volatility[t]))

# Plot simulated returns
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(returns)
plt.title('Simulated Returns with GARCH(1,1) Volatility')
plt.ylabel('Return')

plt.subplot(2, 1, 2)
plt.plot(np.sqrt(volatility))
plt.title('True Conditional Volatility')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.tight_layout()
plt.show()

# Create and fit GARCH(1,1) model
garch_model = GARCH(p=1, q=1)  # p=ARCH order, q=GARCH order

async def estimate_garch():
    print("Estimating GARCH(1,1) model...")
    
    # Fit the model using async/await pattern
    converged = await garch_model.async_fit(returns)
    
    if converged:
        print("GARCH estimation converged successfully!")
        
        # Extract parameters
        print(f"\nEstimated Parameters:")
        print(f"omega: {garch_model._model_params['omega']:.4f} (True: {omega:.4f})")
        print(f"alpha: {garch_model._model_params['alpha'][0]:.4f} (True: {alpha:.4f})")
        print(f"beta: {garch_model._model_params['beta'][0]:.4f} (True: {beta:.4f})")
        
        # Calculate persistence
        persistence = garch_model._model_params['alpha'][0] + garch_model._model_params['beta'][0]
        print(f"Persistence (α+β): {persistence:.4f}")
        
        # Get diagnostics
        diagnostics = garch_model.diagnostic_tests()
        
        # Print key model fit statistics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {garch_model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Plot estimated volatility vs true volatility
        estimated_vol = garch_model.conditional_volatility
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(returns)
        plt.title('Simulated Returns')
        plt.ylabel('Return')
        
        plt.subplot(3, 1, 2)
        plt.plot(np.sqrt(volatility), label='True')
        plt.plot(estimated_vol, 'r--', label='Estimated')
        plt.title('Conditional Volatility: True vs. Estimated')
        plt.ylabel('Volatility')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(np.sqrt(volatility) - estimated_vol)
        plt.title('Volatility Estimation Error')
        plt.xlabel('Time')
        plt.ylabel('Error')
        
        plt.tight_layout()
        plt.show()
        
        # Forecast volatility
        forecast_horizon = 20
        volatility_forecast = garch_model.forecast_variance(steps=forecast_horizon)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(estimated_vol)), estimated_vol, label='In-sample')
        plt.plot(range(len(estimated_vol), len(estimated_vol) + forecast_horizon), 
                 np.sqrt(volatility_forecast), 'r--', label='Forecast')
        plt.title('Volatility Forecast')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("GARCH estimation did not converge")

# Run GARCH estimation
asyncio.run(estimate_garch())
```

### Advanced GARCH Models: EGARCH and GJR-GARCH

The MFE Toolbox supports more advanced GARCH models that can capture asymmetric volatility responses:

```python
from mfe.models import EGARCH, GJR_GARCH

# Generate data with leverage effect (asymmetric volatility)
n = 2000
returns_asym = np.zeros(n)
volatility_asym = np.zeros(n)
volatility_asym[0] = 1.0

# True parameters (with asymmetric effect)
omega = 0.05
alpha = 0.05
beta = 0.85
gamma = 0.1  # Asymmetry parameter

# Simulate process with asymmetric volatility response
for t in range(1, n):
    # GJR-GARCH process: more volatility after negative returns
    leverage = 1.0 if returns_asym[t-1] < 0 else 0.0
    volatility_asym[t] = omega + alpha * returns_asym[t-1]**2 + gamma * leverage * returns_asym[t-1]**2 + beta * volatility_asym[t-1]
    returns_asym[t] = np.random.normal(0, np.sqrt(volatility_asym[t]))

# Plot simulated returns with leverage effect
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(returns_asym)
plt.title('Simulated Returns with Asymmetric Volatility')
plt.ylabel('Return')

plt.subplot(2, 1, 2)
plt.plot(np.sqrt(volatility_asym))
plt.title('True Conditional Volatility (with Leverage Effect)')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.tight_layout()
plt.show()

# Fit different GARCH models to see which captures asymmetry best
async def compare_garch_models():
    print("Comparing different GARCH models on data with leverage effect...")
    
    # Standard GARCH
    garch_model = GARCH(p=1, q=1)
    
    # GJR-GARCH (captures asymmetry through indicator function)
    gjr_model = GJR_GARCH(p=1, q=1)
    
    # EGARCH (captures asymmetry through log transformation)
    egarch_model = EGARCH(p=1, q=1)
    
    # Fit all models
    converged_garch = await garch_model.async_fit(returns_asym)
    converged_gjr = await gjr_model.async_fit(returns_asym)
    converged_egarch = await egarch_model.async_fit(returns_asym)
    
    # Print results
    print("\nModel Comparison Results:")
    results = []
    
    if converged_garch:
        diagnostics = garch_model.diagnostic_tests()
        results.append({
            'Model': 'GARCH(1,1)',
            'Log-likelihood': garch_model.loglikelihood,
            'AIC': diagnostics['AIC'],
            'BIC': diagnostics['BIC']
        })
    
    if converged_gjr:
        diagnostics = gjr_model.diagnostic_tests()
        results.append({
            'Model': 'GJR-GARCH(1,1)',
            'Log-likelihood': gjr_model.loglikelihood,
            'AIC': diagnostics['AIC'],
            'BIC': diagnostics['BIC']
        })
    
    if converged_egarch:
        diagnostics = egarch_model.diagnostic_tests()
        results.append({
            'Model': 'EGARCH(1,1)',
            'Log-likelihood': egarch_model.loglikelihood,
            'AIC': diagnostics['AIC'],
            'BIC': diagnostics['BIC']
        })
    
    # Convert to DataFrame and display
    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Plot volatility estimates from all models
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(returns_asym)
    plt.title('Returns with Asymmetric Volatility')
    plt.ylabel('Return')
    
    plt.subplot(4, 1, 2)
    plt.plot(np.sqrt(volatility_asym), 'k-', label='True')
    if converged_garch:
        plt.plot(garch_model.conditional_volatility, 'b--', label='GARCH')
    plt.legend()
    plt.title('True vs. GARCH(1,1)')
    plt.ylabel('Volatility')
    
    plt.subplot(4, 1, 3)
    plt.plot(np.sqrt(volatility_asym), 'k-', label='True')
    if converged_gjr:
        plt.plot(gjr_model.conditional_volatility, 'g--', label='GJR-GARCH')
    plt.legend()
    plt.title('True vs. GJR-GARCH(1,1)')
    plt.ylabel('Volatility')
    
    plt.subplot(4, 1, 4)
    plt.plot(np.sqrt(volatility_asym), 'k-', label='True')
    if converged_egarch:
        plt.plot(egarch_model.conditional_volatility, 'r--', label='EGARCH')
    plt.legend()
    plt.title('True vs. EGARCH(1,1)')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    
    plt.tight_layout()
    plt.show()

# Compare GARCH models
asyncio.run(compare_garch_models())
```

### Real-World Volatility Analysis with Financial Data

Let's apply GARCH modeling to real financial data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import yfinance as yahoo_finance
from mfe.models import GARCH, GJR_GARCH, EGARCH

async def analyze_market_volatility():
    # Download market data
    print("Downloading market data...")
    tickers = ['SPY', 'QQQ', 'GLD']  # S&P 500 ETF, NASDAQ ETF, Gold ETF
    
    market_data = {}
    for ticker in tickers:
        data = yahoo_finance.download(ticker, start='2018-01-01', end='2023-01-01')
        market_data[ticker] = data['Adj Close'].pct_change().dropna() * 100  # Convert to percentage
    
    # Combine into a DataFrame
    returns_df = pd.DataFrame({ticker: data for ticker, data in market_data.items()})
    
    # Plot returns
    plt.figure(figsize=(14, 8))
    for ticker in tickers:
        plt.plot(returns_df.index, returns_df[ticker], label=ticker)
    plt.title('Daily Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Apply GARCH models to each asset
    for ticker in tickers:
        returns = returns_df[ticker].values
        
        print(f"\nAnalyzing volatility for {ticker}...")
        
        # Create and fit GJR-GARCH model (captures leverage effect)
        model = GJR_GARCH(p=1, q=1)
        
        converged = await model.async_fit(returns)
        
        if converged:
            print(f"GJR-GARCH estimation for {ticker} converged successfully!")
            
            # Extract parameters
            param_dict = model._model_params
            print(f"\nEstimated Parameters:")
            print(f"omega: {param_dict.get('omega', 'N/A')}")
            print(f"alpha: {param_dict.get('alpha', ['N/A'])[0]}")
            print(f"beta: {param_dict.get('beta', ['N/A'])[0]}")
            print(f"gamma: {param_dict.get('gamma', ['N/A'])[0]}")  # Asymmetry parameter
            
            # Plot returns and estimated volatility
            plt.figure(figsize=(12, 8))
            
            # Returns
            plt.subplot(2, 1, 1)
            plt.plot(returns_df.index, returns_df[ticker])
            plt.title(f'{ticker} Daily Returns (%)')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3)
            
            # Volatility
            plt.subplot(2, 1, 2)
            plt.plot(returns_df.index, model.conditional_volatility)
            plt.title(f'{ticker} Estimated Conditional Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Forecast volatility
            forecast_days = 20
            volatility_forecast = model.forecast_variance(steps=forecast_days)
            
            # Display forecast
            print(f"\nVolatility Forecast for {ticker} (next {forecast_days} days):")
            for i, vol in enumerate(volatility_forecast):
                print(f"Day {i+1}: {np.sqrt(vol):.4f}")
        else:
            print(f"GJR-GARCH estimation for {ticker} did not converge")

# Run market volatility analysis
asyncio.run(analyze_market_volatility())
```

## High-Frequency Analysis Tutorial

This tutorial demonstrates how to use the MFE Toolbox for analyzing high-frequency financial data.

### Understanding Realized Volatility Measures

High-frequency financial data allows for more precise volatility estimation through realized measures. The basic realized variance is defined as:

RV = Σ(r²(i))

Where r(i) are intraday returns. The MFE Toolbox implements advanced realized measures that account for microstructure noise and other market frictions.

### Generating Synthetic High-Frequency Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mfe.models import realized_variance, realized_kernel

# Generate synthetic high-frequency data
np.random.seed(42)

# Simulation parameters
n_days = 5
n_intraday = 390  # Typical number of 1-minute observations in a 6.5-hour trading day
price_path = []
timestamps = []
true_daily_vol = []  # Store true daily volatility

# Initial price
price = 100.0

# Generate data for each day
for day in range(n_days):
    # Set date and daily volatility
    base_date = datetime(2023, 1, 2) + timedelta(days=day)
    daily_return = np.random.normal(0, 0.01)  # 1% average daily return
    daily_vol = 0.02 + 0.5 * np.random.random()  # Daily volatility between 2% and 52%
    true_daily_vol.append(daily_vol)
    
    # Simulate intraday price path
    for minute in range(n_intraday):
        # Create timestamp
        time = base_date + timedelta(minutes=minute)
        timestamps.append(time)
        
        # Add intraday pattern (U-shape volatility)
        intraday_factor = 1.0 + 0.5 * (
            np.exp(-((minute - 0) / 60)**2) + 
            np.exp(-((minute - (n_intraday-1)) / 60)**2)
        )
        
        # Generate price innovation with intraday pattern
        price_innovation = np.random.normal(
            daily_return/n_intraday,  # Expected return per minute
            daily_vol/np.sqrt(n_intraday) * intraday_factor  # Scaled volatility with pattern
        )
        
        # Update price (log-return process)
        price *= np.exp(price_innovation)
        
        # Add microstructure noise to observed price
        noise_level = 0.0001  # 1 basis point
        noisy_price = price * (1 + np.random.normal(0, noise_level))
        
        # Store noisy observed price
        price_path.append(noisy_price)

# Create DataFrame
hf_data = pd.DataFrame({
    'timestamp': timestamps,
    'price': price_path
})

# Plot full price path
plt.figure(figsize=(14, 7))
plt.plot(hf_data['timestamp'], hf_data['price'])
plt.title('Simulated High-Frequency Price Path')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)
plt.show()

# Look at data for a single day
day_1_data = hf_data[hf_data['timestamp'].dt.date == datetime(2023, 1, 2).date()]

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(day_1_data['timestamp'], day_1_data['price'])
plt.title('Day 1 Price Path')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)

# Calculate returns
day_1_data['return'] = np.log(day_1_data['price']).diff() * 100  # Percentage

plt.subplot(2, 1, 2)
plt.plot(day_1_data['timestamp'][1:], day_1_data['return'][1:])
plt.title('Day 1 Returns (%)')
plt.xlabel('Time')
plt.ylabel('Return (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Computing Realized Measures

```python
# Prepare data for realized measures computation
# We need to split by day and compute measures for each day

# Initialize arrays for realized measures
days = sorted(list(set([t.date() for t in hf_data['timestamp']])))
rv_results = []
rv_ss_results = []  # Subsampled
rk_results = []  # Realized kernel

for day in days:
    # Filter data for current day
    day_data = hf_data[hf_data['timestamp'].dt.date == day]
    
    # Extract arrays for computation
    times = np.array([t.timestamp() for t in day_data['timestamp']])
    prices = np.array(day_data['price'])
    
    # Compute realized variance with 5-minute sampling
    rv, rv_ss = realized_variance(
        prices,
        times,
        timeType='timestamp',
        samplingType='CalendarTime',
        samplingInterval=5  # 5-minute sampling
    )
    
    # Compute realized kernel (robust to noise)
    rk = realized_kernel(
        prices,
        times,
        timeType='timestamp',
        kernelType='Parzen'
    )
    
    # Store results
    rv_results.append(rv)
    rv_ss_results.append(rv_ss)
    rk_results.append(rk)

# Convert to volatility (standard deviation)
rv_vol = np.sqrt(rv_results)
rv_ss_vol = np.sqrt(rv_ss_results)
rk_vol = np.sqrt(rk_results)

# Create results table
results_df = pd.DataFrame({
    'Date': days,
    'True Volatility': true_daily_vol,
    'RV (5-min)': rv_vol,
    'RV (Subsampled)': rv_ss_vol,
    'Realized Kernel': rk_vol
})

print("Daily Volatility Estimates:")
print(results_df)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(results_df['Date'], results_df['True Volatility'], 'k-', label='True')
plt.plot(results_df['Date'], results_df['RV (5-min)'], 'b--', label='RV (5-min)')
plt.plot(results_df['Date'], results_df['RV (Subsampled)'], 'g--', label='RV (Subsampled)')
plt.title('Daily Volatility Estimates')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(results_df['Date'], results_df['True Volatility'], 'k-', label='True')
plt.plot(results_df['Date'], results_df['Realized Kernel'], 'r--', label='Realized Kernel')
plt.title('Realized Kernel Estimates')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Different Sampling Schemes

```python
# Experiment with different sampling schemes
sampling_types = [
    ('CalendarTime', 5),       # 5-minute calendar time sampling
    ('CalendarTime', 10),      # 10-minute calendar time sampling
    ('BusinessTime', 30),      # Business time sampling with 30 observations per day
    ('BusinessUniform', 30)    # Business time with jittering for uniform coverage
]

# Analyze a single day
test_day = days[0]
day_data = hf_data[hf_data['timestamp'].dt.date == test_day]
times = np.array([t.timestamp() for t in day_data['timestamp']])
prices = np.array(day_data['price'])

# Compute with different schemes
results = []

for scheme, interval in sampling_types:
    rv, rv_ss = realized_variance(
        prices,
        times,
        timeType='timestamp',
        samplingType=scheme,
        samplingInterval=interval
    )
    
    results.append({
        'Sampling Scheme': f"{scheme} ({interval})",
        'RV': np.sqrt(rv),
        'RV (Subsampled)': np.sqrt(rv_ss)
    })

# Display results
sampling_df = pd.DataFrame(results)
print("\nImpact of Different Sampling Schemes:")
print(sampling_df)

# Plot histogram of intraday returns for different sampling frequencies
plt.figure(figsize=(14, 10))

# Original 1-minute returns
day_data['return'] = np.log(day_data['price']).diff() * 100
minute_returns = day_data['return'].dropna()

plt.subplot(2, 2, 1)
plt.hist(minute_returns, bins=30, alpha=0.7)
plt.title('1-minute Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

# 5-minute returns
returns_5m = []
for i in range(0, len(prices) - 5, 5):
    ret = np.log(prices[i+5]/prices[i]) * 100
    returns_5m.append(ret)

plt.subplot(2, 2, 2)
plt.hist(returns_5m, bins=30, alpha=0.7)
plt.title('5-minute Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

# 10-minute returns
returns_10m = []
for i in range(0, len(prices) - 10, 10):
    ret = np.log(prices[i+10]/prices[i]) * 100
    returns_10m.append(ret)

plt.subplot(2, 2, 3)
plt.hist(returns_10m, bins=20, alpha=0.7)
plt.title('10-minute Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

# 30-minute returns
returns_30m = []
for i in range(0, len(prices) - 30, 30):
    ret = np.log(prices[i+30]/prices[i]) * 100
    returns_30m.append(ret)

plt.subplot(2, 2, 4)
plt.hist(returns_30m, bins=15, alpha=0.7)
plt.title('30-minute Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Real-World High-Frequency Analysis

```python
# Note: Acquiring real high-frequency data usually requires paid services
# This example demonstrates the workflow using simulated tick data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mfe.models import realized_variance, realized_kernel

def analyze_real_hf_data():
    # In practice, you would load your own high-frequency data
    # For example:
    # tick_data = pd.read_csv('tick_data.csv', parse_dates=['timestamp'])
    
    # For demonstration, we'll generate synthetic tick data
    # that mimics real market microstructure
    
    print("Simulating realistic high-frequency data...")
    
    # Parameters
    n_days = 3
    trading_hours = 6.5  # 6.5 hours typical trading day
    avg_ticks_per_second = 1  # Average 1 trade per second
    price = 150.0  # Starting price
    volatility = 0.20  # Annual volatility
    daily_vol = volatility / np.sqrt(252)  # Daily volatility
    tick_size = 0.01  # Minimum price increment
    
    # Generate data
    all_ticks = []
    
    for day in range(n_days):
        date = datetime(2023, 1, 2) + timedelta(days=day)
        open_time = datetime.combine(date.date(), datetime.min.time().replace(hour=9, minute=30))
        close_time = open_time + timedelta(hours=trading_hours)
        
        # Daily market return
        daily_return = np.random.normal(0, daily_vol)
        drift_per_second = daily_return / (trading_hours * 3600)
        
        # Simulate trading day
        current_time = open_time
        current_price = price * np.exp(np.random.normal(0, daily_vol))
        
        while current_time < close_time:
            # Generate random inter-arrival time (exponential distribution)
            seconds_to_next = np.random.exponential(1 / avg_ticks_per_second)
            current_time += timedelta(seconds=seconds_to_next)
            
            if current_time >= close_time:
                break
            
            # Compute price innovation
            time_fraction = seconds_to_next / (trading_hours * 3600)
            expected_return = drift_per_second * seconds_to_next
            vol_scaling = np.sqrt(time_fraction)
            
            # Add intraday U-shape volatility pattern
            tod_seconds = (current_time - open_time).total_seconds()
            rel_time = tod_seconds / (trading_hours * 3600)
            u_shape = 1.0 + 0.5 * (np.exp(-((rel_time - 0) / 0.1)**2) + np.exp(-((rel_time - 1) / 0.1)**2))
            
            # Price increment with microstructure noise
            price_increment = np.random.normal(expected_return, daily_vol * vol_scaling * u_shape)
            microstructure_noise = np.random.normal(0, tick_size * 0.5)
            
            # Update price
            current_price *= np.exp(price_increment)
            observed_price = current_price + microstructure_noise
            
            # Round to tick size
            observed_price = round(observed_price / tick_size) * tick_size
            
            # Add to tick data
            all_ticks.append({
                'timestamp': current_time,
                'price': observed_price,
                'volume': np.random.poisson(100)  # Random trade size
            })
    
    # Convert to DataFrame
    tick_data = pd.DataFrame(all_ticks)
    
    # Display sample
    print("\nSample of tick data:")
    print(tick_data.head())
    
    # Plot tick data for the first day
    day1_data = tick_data[tick_data['timestamp'].dt.date == datetime(2023, 1, 2).date()]
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(day1_data['timestamp'], day1_data['price'], 'b.')
    plt.title('Tick-by-Tick Price Data (Day 1)')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(day1_data['timestamp'], day1_data['volume'], 'g.')
    plt.title('Tick-by-Tick Volume')
    plt.ylabel('Volume')
    plt.grid(True, alpha=0.3)
    
    # Compute and plot tick-by-tick returns
    day1_data['return'] = np.log(day1_data['price']).diff() * 100
    
    plt.subplot(3, 1, 3)
    plt.plot(day1_data['timestamp'][1:], day1_data['return'][1:], 'r.')
    plt.title('Tick-by-Tick Returns (%)')
    plt.xlabel('Time')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compute daily realized measures
    days = sorted(list(set([t.date() for t in tick_data['timestamp']])))
    daily_results = []
    
    for day in days:
        day_data = tick_data[tick_data['timestamp'].dt.date == day]
        
        # Prepare data for realized measures
        times = np.array([t.timestamp() for t in day_data['timestamp']])
        prices = np.array(day_data['price'])
        
        # Compute measures
        rv_5min, rv_ss = realized_variance(
            prices, times, timeType='timestamp', 
            samplingType='CalendarTime', samplingInterval=5
        )
        
        rk = realized_kernel(
            prices, times, timeType='timestamp', kernelType='Parzen'
        )
        
        # Compute as volatility (annualized)
        daily_vol = np.sqrt(252) * np.sqrt(rv_5min)
        daily_vol_ss = np.sqrt(252) * np.sqrt(rv_ss)
        daily_vol_rk = np.sqrt(252) * np.sqrt(rk)
        
        daily_results.append({
            'Date': day,
            'Annualized Volatility (RV)': daily_vol,
            'Annualized Volatility (RV-SS)': daily_vol_ss,
            'Annualized Volatility (RK)': daily_vol_rk,
            'Number of Ticks': len(day_data)
        })
    
    # Display results
    daily_df = pd.DataFrame(daily_results)
    print("\nDaily Realized Measures:")
    print(daily_df)
    
    # Plot daily volatility
    plt.figure(figsize=(12, 6))
    plt.plot(daily_df['Date'], daily_df['Annualized Volatility (RV)'], 'b-o', label='RV (5-min)')
    plt.plot(daily_df['Date'], daily_df['Annualized Volatility (RV-SS)'], 'g-s', label='RV (Subsampled)')
    plt.plot(daily_df['Date'], daily_df['Annualized Volatility (RK)'], 'r-^', label='Realized Kernel')
    plt.title('Daily Realized Volatility Estimates (Annualized)')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Run high-frequency analysis
analyze_real_hf_data()
```

## Bootstrap Analysis Tutorial

This tutorial covers bootstrap methods for statistical inference in financial econometrics.

### Understanding Bootstrap Methods

Bootstrap methods are resampling techniques used to estimate the sampling distribution of a statistic without making strong distributional assumptions. For time series data, special bootstrap methods are needed to preserve temporal dependence:

1. Block Bootstrap: Resamples fixed-length contiguous blocks of data
2. Stationary Bootstrap: Resamples blocks of random length (geometrically distributed)

### Basic Block Bootstrap Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mfe.core import block_bootstrap, stationary_bootstrap

# Generate an AR(1) process for demonstration
np.random.seed(42)
n = 500
ar_coefficient = 0.7
data = np.zeros(n)

for t in range(1, n):
    data[t] = ar_coefficient * data[t-1] + np.random.normal(0, 1)

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Simulated AR(1) Process with φ = 0.7')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()

# Define statistics of interest
def compute_mean(x):
    return np.mean(x)

def compute_variance(x):
    return np.var(x, ddof=1)

def compute_ar1_coefficient(x):
    # Crude AR(1) coefficient estimation
    return np.sum(x[1:] * x[:-1]) / np.sum(x[:-1]**2)

# Perform block bootstrap
n_bootstrap = 1000
block_size = 50  # Fixed block size

# Run bootstrap for multiple statistics
mean_results = block_bootstrap(data, compute_mean, n_bootstrap=n_bootstrap, block_size=block_size)
var_results = block_bootstrap(data, compute_variance, n_bootstrap=n_bootstrap, block_size=block_size)
ar1_results = block_bootstrap(data, compute_ar1_coefficient, n_bootstrap=n_bootstrap, block_size=block_size)

# Compute true statistics from original data
true_mean = compute_mean(data)
true_var = compute_variance(data)
true_ar1 = compute_ar1_coefficient(data)

# Plot bootstrap distributions
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Mean bootstrap distribution
axes[0].hist(mean_results, bins=30, alpha=0.7)
axes[0].axvline(true_mean, color='r', linestyle='--', label=f'Sample mean: {true_mean:.4f}')
axes[0].set_title('Bootstrap Distribution of Mean')
axes[0].set_xlabel('Mean')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Variance bootstrap distribution
axes[1].hist(var_results, bins=30, alpha=0.7)
axes[1].axvline(true_var, color='r', linestyle='--', label=f'Sample variance: {true_var:.4f}')
axes[1].set_title('Bootstrap Distribution of Variance')
axes[1].set_xlabel('Variance')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# AR(1) coefficient bootstrap distribution
axes[2].hist(ar1_results, bins=30, alpha=0.7)
axes[2].axvline(true_ar1, color='r', linestyle='--', label=f'Sample AR(1): {true_ar1:.4f}')
axes[2].axvline(ar_coefficient, color='g', linestyle='-', label=f'True AR(1): {ar_coefficient:.4f}')
axes[2].set_title('Bootstrap Distribution of AR(1) Coefficient')
axes[2].set_xlabel('AR(1) Coefficient')
axes[2].set_ylabel('Frequency')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compute bootstrap confidence intervals
alpha = 0.05  # 5% significance level
mean_ci = np.percentile(mean_results, [alpha/2*100, (1-alpha/2)*100])
var_ci = np.percentile(var_results, [alpha/2*100, (1-alpha/2)*100])
ar1_ci = np.percentile(ar1_results, [alpha/2*100, (1-alpha/2)*100])

print("Bootstrap Confidence Intervals (95%)")
print(f"Mean: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")
print(f"Variance: [{var_ci[0]:.4f}, {var_ci[1]:.4f}]")
print(f"AR(1) Coefficient: [{ar1_ci[0]:.4f}, {ar1_ci[1]:.4f}]")
```

### Comparing Block and Stationary Bootstrap

```python
# Compare block bootstrap and stationary bootstrap
def compare_bootstrap_methods():
    # Define bootstrap parameters
    n_bootstrap = 1000
    block_size = 50  # Fixed block size for block bootstrap
    expected_block_size = 50  # Expected block size for stationary bootstrap
    
    # Run both bootstrap methods for AR(1) coefficient
    block_results = block_bootstrap(
        data, compute_ar1_coefficient, 
        n_bootstrap=n_bootstrap, block_size=block_size
    )
    
    stationary_results = stationary_bootstrap(
        data, compute_ar1_coefficient, 
        n_bootstrap=n_bootstrap, expected_block_size=expected_block_size
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(block_results, bins=30, alpha=0.7)
    plt.axvline(true_ar1, color='r', linestyle='--', label=f'Sample: {true_ar1:.4f}')
    plt.axvline(ar_coefficient, color='g', linestyle='-', label=f'True: {ar_coefficient:.4f}')
    plt.title('Block Bootstrap')
    plt.xlabel('AR(1) Coefficient')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(stationary_results, bins=30, alpha=0.7)
    plt.axvline(true_ar1, color='r', linestyle='--', label=f'Sample: {true_ar1:.4f}')
    plt.axvline(ar_coefficient, color='g', linestyle='-', label=f'True: {ar_coefficient:.4f}')
    plt.title('Stationary Bootstrap')
    plt.xlabel('AR(1) Coefficient')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compute confidence intervals
    block_ci = np.percentile(block_results, [alpha/2*100, (1-alpha/2)*100])
    stationary_ci = np.percentile(stationary_results, [alpha/2*100, (1-alpha/2)*100])
    
    print("\nAR(1) Coefficient Confidence Intervals (95%)")
    print(f"Block Bootstrap: [{block_ci[0]:.4f}, {block_ci[1]:.4f}]")
    print(f"Stationary Bootstrap: [{stationary_ci[0]:.4f}, {stationary_ci[1]:.4f}]")
    
    # Compute standard errors
    block_se = np.std(block_results, ddof=1)
    stationary_se = np.std(stationary_results, ddof=1)
    
    print("\nBootstrap Standard Errors")
    print(f"Block Bootstrap SE: {block_se:.4f}")
    print(f"Stationary Bootstrap SE: {stationary_se:.4f}")

# Run bootstrap comparison
compare_bootstrap_methods()
```

### Block Size Selection

```python
# Study the effect of block size on bootstrap performance
def study_block_size_effect():
    # Try different block sizes
    block_sizes = [10, 25, 50, 100, 200]
    n_bootstrap = 1000
    
    block_results = {}
    stationary_results = {}
    
    for size in block_sizes:
        # Block bootstrap
        block_results[size] = block_bootstrap(
            data, compute_ar1_coefficient, 
            n_bootstrap=n_bootstrap, block_size=size
        )
        
        # Stationary bootstrap
        stationary_results[size] = stationary_bootstrap(
            data, compute_ar1_coefficient, 
            n_bootstrap=n_bootstrap, expected_block_size=size
        )
    
    # Compute statistics
    results = []
    
    for size in block_sizes:
        # Block bootstrap stats
        block_mean = np.mean(block_results[size])
        block_se = np.std(block_results[size], ddof=1)
        block_ci = np.percentile(block_results[size], [alpha/2*100, (1-alpha/2)*100])
        block_ci_width = block_ci[1] - block_ci[0]
        
        # Stationary bootstrap stats
        stat_mean = np.mean(stationary_results[size])
        stat_se = np.std(stationary_results[size], ddof=1)
        stat_ci = np.percentile(stationary_results[size], [alpha/2*100, (1-alpha/2)*100])
        stat_ci_width = stat_ci[1] - stat_ci[0]
        
        # Bias from true value
        block_bias = block_mean - ar_coefficient
        stat_bias = stat_mean - ar_coefficient
        
        results.append({
            'Block Size': size,
            'Block Mean': block_mean,
            'Block SE': block_se,
            'Block CI Width': block_ci_width,
            'Block Bias': block_bias,
            'Stationary Mean': stat_mean,
            'Stationary SE': stat_se,
            'Stationary CI Width': stat_ci_width,
            'Stationary Bias': stat_bias
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print("\nEffect of Block Size on Bootstrap Performance")
    print(results_df.to_string(index=False))
    
    # Plot the results
    plt.figure(figsize=(14, 10))
    
    # Plot standard errors
    plt.subplot(2, 2, 1)
    plt.plot(results_df['Block Size'], results_df['Block SE'], 'bo-', label='Block Bootstrap')
    plt.plot(results_df['Block Size'], results_df['Stationary SE'], 'ro-', label='Stationary Bootstrap')
    plt.title('Standard Error vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('Standard Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot CI widths
    plt.subplot(2, 2, 2)
    plt.plot(results_df['Block Size'], results_df['Block CI Width'], 'bo-', label='Block Bootstrap')
    plt.plot(results_df['Block Size'], results_df['Stationary CI Width'], 'ro-', label='Stationary Bootstrap')
    plt.title('Confidence Interval Width vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('CI Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot bias
    plt.subplot(2, 2, 3)
    plt.plot(results_df['Block Size'], np.abs(results_df['Block Bias']), 'bo-', label='Block Bootstrap')
    plt.plot(results_df['Block Size'], np.abs(results_df['Stationary Bias']), 'ro-', label='Stationary Bootstrap')
    plt.title('Absolute Bias vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('|Bias|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot mean estimates
    plt.subplot(2, 2, 4)
    plt.plot(results_df['Block Size'], results_df['Block Mean'], 'bo-', label='Block Bootstrap')
    plt.plot(results_df['Block Size'], results_df['Stationary Mean'], 'ro-', label='Stationary Bootstrap')
    plt.axhline(ar_coefficient, color='g', linestyle='--', label=f'True Value: {ar_coefficient:.4f}')
    plt.title('Mean Estimate vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('Mean Estimate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Study block size effect
study_block_size_effect()
```

### Application: Bootstrap for Financial Risk Measures

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yahoo_finance
from mfe.core import block_bootstrap, stationary_bootstrap

def bootstrap_risk_measures():
    # Download market data
    print("Downloading market data...")
    tickers = ['SPY', 'AAPL', 'MSFT', 'AMZN']
    
    # Download daily data
    data = yahoo_finance.download(tickers, start='2020-01-01', end='2023-01-01')['Adj Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna() * 100  # Convert to percentage
    
    # Plot returns
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(returns.index, returns[ticker], label=ticker)
    plt.title('Daily Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Create a portfolio with equal weights
    weights = np.ones(len(tickers)) / len(tickers)
    portfolio_returns = returns.dot(weights)
    
    # Define risk measures
    def compute_var(x, alpha=0.05):
        # Value at Risk (VaR)
        return -np.percentile(x, alpha * 100)
    
    def compute_es(x, alpha=0.05):
        # Expected Shortfall (ES)
        var = compute_var(x, alpha)
        return -np.mean(x[x <= -var])
    
    def compute_sharpe(x):
        # Sharpe Ratio (assuming zero risk-free rate)
        return np.mean(x) / np.std(x, ddof=1) * np.sqrt(252)  # Annualized
    
    # Bootstrap parameters
    n_bootstrap = 2000
    block_size = 20  # ~1 month of trading days
    
    # Perform stationary bootstrap for each measure
    print("Running bootstrap for risk measures...")
    var_results = stationary_bootstrap(
        portfolio_returns.values, compute_var, 
        n_bootstrap=n_bootstrap, expected_block_size=block_size
    )
    
    es_results = stationary_bootstrap(
        portfolio_returns.values, compute_es, 
        n_bootstrap=n_bootstrap, expected_block_size=block_size
    )
    
    sharpe_results = stationary_bootstrap(
        portfolio_returns.values, compute_sharpe, 
        n_bootstrap=n_bootstrap, expected_block_size=block_size
    )
    
    # Compute point estimates and confidence intervals
    alpha = 0.05  # 5% significance level
    
    var_point = compute_var(portfolio_returns.values)
    var_ci = np.percentile(var_results, [alpha/2*100, (1-alpha/2)*100])
    
    es_point = compute_es(portfolio_returns.values)
    es_ci = np.percentile(es_results, [alpha/2*100, (1-alpha/2)*100])
    
    sharpe_point = compute_sharpe(portfolio_returns.values)
    sharpe_ci = np.percentile(sharpe_results, [alpha/2*100, (1-alpha/2)*100])
    
    # Print results
    print("\nRisk Measure Analysis")
    print(f"Value at Risk (5%): {var_point:.4f} [95% CI: {var_ci[0]:.4f}, {var_ci[1]:.4f}]")
    print(f"Expected Shortfall (5%): {es_point:.4f} [95% CI: {es_ci[0]:.4f}, {es_ci[1]:.4f}]")
    print(f"Sharpe Ratio (annualized): {sharpe_point:.4f} [95% CI: {sharpe_ci[0]:.4f}, {sharpe_ci[1]:.4f}]")
    
    # Plot the bootstrap distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(var_results, bins=30, alpha=0.7)
    plt.axvline(var_point, color='r', linestyle='--', label=f'Point Est.: {var_point:.4f}')
    plt.axvline(var_ci[0], color='g', linestyle='-', label='95% CI')
    plt.axvline(var_ci[1], color='g', linestyle='-')
    plt.title('VaR Bootstrap Distribution')
    plt.xlabel('Value at Risk (5%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(es_results, bins=30, alpha=0.7)
    plt.axvline(es_point, color='r', linestyle='--', label=f'Point Est.: {es_point:.4f}')
    plt.axvline(es_ci[0], color='g', linestyle='-', label='95% CI')
    plt.axvline(es_ci[1], color='g', linestyle='-')
    plt.title('ES Bootstrap Distribution')
    plt.xlabel('Expected Shortfall (5%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(sharpe_results, bins=30, alpha=0.7)
    plt.axvline(sharpe_point, color='r', linestyle='--', label=f'Point Est.: {sharpe_point:.4f}')
    plt.axvline(sharpe_ci[0], color='g', linestyle='-', label='95% CI')
    plt.axvline(sharpe_ci[1], color='g', linestyle='-')
    plt.title('Sharpe Ratio Bootstrap Distribution')
    plt.xlabel('Sharpe Ratio (annualized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run bootstrap for risk measures
bootstrap_risk_measures()
```

## GUI Interface Tutorial

This tutorial demonstrates how to use the MFE Toolbox's graphical user interface (GUI) for interactive financial econometric analysis.

### Launching the GUI

```python
from mfe.ui import launch_gui

# Launch the main application window
launch_gui()
```

### GUI Workflow Tutorial

The MFE Toolbox GUI provides an interactive interface for model estimation, diagnostic analysis, and results visualization. Here's a step-by-step guide to using the interface:

1. **Launching the GUI**
   - Import and call the `launch_gui()` function
   - The main application window will appear with model configuration options

2. **Loading Data**
   - Click the "Load Data" button to import your time series
   - The GUI supports CSV, Excel, and other standard formats
   - Once loaded, a preview of your data will be displayed

3. **Configuring the Model**
   - Set AR and MA orders using the input fields
   - Toggle "Include Constant" to include a constant term
   - Select exogenous variables if needed from the dropdown

4. **Estimating the Model**
   - Click "Estimate Model" to start the estimation process
   - A progress bar will display during estimation
   - The GUI will display basic results upon completion

5. **Viewing Results**
   - Click "View Results" to open the detailed results viewer
   - The results display includes:
     - Model equation with estimated parameters
     - Parameter estimates table with standard errors
     - Statistical metrics (log-likelihood, AIC, BIC)
     - Diagnostic plots for residual analysis

6. **Navigating Diagnostic Plots**
   - Use the "Previous" and "Next" buttons to navigate between plot pages
   - Different pages show different diagnostic information:
     - Residual time series and distribution
     - ACF and PACF plots
     - Jarque-Bera test results for normality
     - Additional statistical tests

7. **Model Comparison**
   - Estimate multiple models with different specifications
   - Use the comparison feature to select the best model

### GUI Components and Interactions

The GUI consists of these main components:

1. **Main Application Window**
   - Model configuration panel
   - Estimation controls
   - Basic diagnostic display
   - Menu for additional functions

2. **Results Viewer**
   - Parameter estimates
   - Statistical metrics
   - Interactive diagnostic plots
   - Navigation controls

3. **Dialog Windows**
   - About dialog showing version information
   - Confirmation dialogs for important actions
   - Error messages for troubleshooting

### Example: Model Estimation Workflow

1. Launch the GUI
```python
from mfe.ui import launch_gui
launch_gui()
```

2. Follow these steps in the GUI:
   - Set AR Order to 2
   - Set MA Order to 1
   - Check "Include Constant"
   - Click "Load Data" and select your time series file
   - Click "Estimate Model"
   - Once estimation completes, click "View Results"
   - Navigate through diagnostic plots using the "Next" and "Previous" buttons
   - Check parameter significance in the parameter table
   - Review model fit statistics

3. Try different model specifications and compare the results.

### Best Practices for GUI Usage

1. **Start Simple**: Begin with lower-order models before trying more complex specifications
2. **Check Diagnostics**: Always examine residual plots for model adequacy
3. **Compare Models**: Try multiple specifications and compare using information criteria
4. **Use Interactive Features**: The plot viewer supports zooming, panning, and data export
5. **Save Results**: Use the save functionality to preserve your analysis

## Performance Optimization with Numba

This tutorial demonstrates how the MFE Toolbox utilizes Numba for performance optimization of computationally intensive tasks.

### Understanding Numba Optimization

The MFE Toolbox leverages Numba's just-in-time (JIT) compilation to accelerate performance-critical numerical computations. Key components that benefit from Numba optimization include:

1. GARCH likelihood calculations
2. Realized volatility measures computation
3. Bootstrap resampling routines
4. Matrix operations in multivariate models

Here's a simplified example of how Numba is used in the toolbox:

```python
import numpy as np
import time
from numba import jit

# Define a computationally intensive function
def garch_likelihood_standard(returns, omega, alpha, beta):
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Log-likelihood computation
    llh = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * returns**2 / sigma2
    return -np.sum(llh[1:])  # Return negative log-likelihood for minimization

# Numba-optimized version
@jit(nopython=True)
def garch_likelihood_numba(returns, omega, alpha, beta):
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Log-likelihood computation
    llh = np.zeros(n)
    for t in range(n):
        llh[t] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2[t]) - 0.5 * returns[t]**2 / sigma2[t]
    
    return -np.sum(llh[1:])  # Return negative log-likelihood for minimization

# Generate sample data
np.random.seed(42)
n = 10000
returns = np.random.normal(0, 1, n)

# Parameters
omega = 0.1
alpha = 0.1
beta = 0.8

# Benchmark standard implementation
start_time = time.time()
standard_result = garch_likelihood_standard(returns, omega, alpha, beta)
standard_time = time.time() - start_time
print(f"Standard implementation: {standard_time:.6f} seconds")

# First run includes compilation time
start_time = time.time()
numba_result = garch_likelihood_numba(returns, omega, alpha, beta)
first_run_time = time.time() - start_time
print(f"Numba first run (includes compilation): {first_run_time:.6f} seconds")

# Second run shows true performance
start_time = time.time()
numba_result = garch_likelihood_numba(returns, omega, alpha, beta)
second_run_time = time.time() - start_time
print(f"Numba second run: {second_run_time:.6f} seconds")

# Verify results match
print(f"Results match: {np.isclose(standard_result, numba_result)}")
print(f"Speedup factor: {standard_time / second_run_time:.2f}x")
```

### Key Numba Optimization Strategies

The MFE Toolbox employs several strategies to optimize performance with Numba:

1. **Function Specialization**
   - Using `@jit(nopython=True)` for maximum performance
   - Optimizing array operations and memory access patterns
   - Proper type specialization for numerical functions

2. **Numba-Friendly Code Structure**
   - Avoiding Python-specific constructs in performance-critical sections
   - Using NumPy arrays with contiguous memory layouts
   - Structuring loops for efficient compilation

3. **Integration with Python's async/await**
   - Executing Numba-optimized functions in separate threads
   - Non-blocking UI updates during computation
   - Progress reporting from long-running operations

### Example: Optimized Bootstrap Implementation

Let's examine a simplified version of how the MFE Toolbox implements bootstrap with Numba optimization:

```python
import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt

# Standard block bootstrap implementation
def block_bootstrap_standard(data, statistic_func, n_bootstrap=1000, block_size=50):
    n = len(data)
    results = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Generate bootstrap sample
        bootstrap_sample = np.zeros(n)
        for i in range(0, n, block_size):
            # Choose random starting point for block
            start = np.random.randint(0, n - block_size + 1)
            # Copy block
            end = min(i + block_size, n)
            length = end - i
            bootstrap_sample[i:end] = data[start:start+length]
        
        # Compute statistic
        results[b] = statistic_func(bootstrap_sample)
    
    return results

# Numba-optimized block bootstrap
@jit(nopython=True)
def _bootstrap_sample_generator(data, n, block_size):
    bootstrap_sample = np.zeros(n)
    for i in range(0, n, block_size):
        # Choose random starting point for block
        start = np.random.randint(0, n - block_size + 1)
        # Copy block
        end = min(i + block_size, n)
        length = end - i
        bootstrap_sample[i:end] = data[start:start+length]
    return bootstrap_sample

# For simplicity, we'll define statistic functions that are Numba-friendly
@jit(nopython=True)
def compute_mean(x):
    return np.mean(x)

@jit(nopython=True)
def compute_variance(x):
    return np.var(x)

@jit(nopython=True)
def compute_ar1(x):
    return np.sum(x[1:] * x[:-1]) / np.sum(x[:-1]**2)

# Mixed Python/Numba block bootstrap implementation
def block_bootstrap_mixed(data, statistic_func, n_bootstrap=1000, block_size=50):
    n = len(data)
    results = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Generate bootstrap sample using Numba
        bootstrap_sample = _bootstrap_sample_generator(data, n, block_size)
        
        # Compute statistic
        results[b] = statistic_func(bootstrap_sample)
    
    return results

# Generate sample data (AR(1) process)
n = 1000
ar_coef = 0.7
data = np.zeros(n)
for t in range(1, n):
    data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)

# Benchmark parameters
n_bootstrap = 500
block_size = 50

# Benchmark standard implementation
start_time = time.time()
std_results = block_bootstrap_standard(data, compute_mean, n_bootstrap, block_size)
standard_time = time.time() - start_time
print(f"Standard implementation: {standard_time:.6f} seconds")

# Benchmark mixed implementation (includes compilation time)
start_time = time.time()
mixed_results = block_bootstrap_mixed(data, compute_mean, n_bootstrap, block_size)
mixed_time = time.time() - start_time
print(f"Mixed implementation (first run): {mixed_time:.6f} seconds")

# Second run of mixed implementation
start_time = time.time()
mixed_results2 = block_bootstrap_mixed(data, compute_mean, n_bootstrap, block_size)
mixed_time2 = time.time() - start_time
print(f"Mixed implementation (second run): {mixed_time2:.6f} seconds")

# Verify results are similar
print(f"Mean of standard results: {np.mean(std_results):.6f}")
print(f"Mean of mixed results: {np.mean(mixed_results):.6f}")
print(f"Speedup factor: {standard_time / mixed_time2:.2f}x")

# Plot distributions
plt.figure(figsize=(12, 6))
plt.hist(std_results, bins=30, alpha=0.5, label='Standard')
plt.hist(mixed_results, bins=30, alpha=0.5, label='Numba-optimized')
plt.axvline(np.mean(data), color='r', linestyle='--', label='Sample Mean')
plt.title('Bootstrap Distributions: Standard vs. Numba-optimized')
plt.xlabel('Mean')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Balancing Performance and Flexibility

The MFE Toolbox balances performance and flexibility by strategically applying Numba optimization:

1. **Core Computational Kernels**
   - Heavy use of Numba for performance-critical operations
   - Type specialized implementations of statistical calculations
   - Optimized matrix operations for large datasets

2. **High-Level Python Interface**
   - User-friendly Python API for model configuration
   - Flexible parameter handling and validation in Python
   - Comprehensive error checking outside of Numba code

3. **Asynchronous Operations**
   - Non-blocking UI updates during computation
   - Progress reporting from long-running operations
   - Responsive user interface even during heavy calculations

### Best Practices for Working with Numba-Optimized Code

When working with the MFE Toolbox, keep these Numba-related best practices in mind:

1. **Data Preparation**
   - Use NumPy arrays with contiguous memory layout
   - Pre-allocate arrays for better performance
   - Convert data types appropriately before passing to optimized functions

2. **Function Selection**
   - Use the asynchronous API for long-running computations
   - Leverage batch processing for multiple operations
   - Consider memory usage for very large datasets

3. **Error Handling**
   - Check input validity before calling optimized functions
   - Handle exceptions properly as they propagate from Numba code
   - Verify results for numerical stability

## Advanced Topics

### Working with Custom Models

The MFE Toolbox architecture allows for extending the existing models with custom implementations. Here's an example of creating a custom ARMA model with specialized features:

```python
import numpy as np
import asyncio
from typing import Optional, Dict, Any
from mfe.models import ARMAX
from mfe.core.optimization import Optimizer

class CustomARMA(ARMAX):
    """
    Custom ARMA model extending the base ARMAX model with additional features.
    """
    
    def __init__(self, p: int, q: int, include_constant: bool = True, 
                 custom_feature: Optional[str] = None):
        """
        Initialize the custom ARMA model.
        
        Parameters
        ----------
        p : int
            Autoregressive order
        q : int
            Moving average order
        include_constant : bool
            Whether to include a constant term
        custom_feature : Optional[str]
            Optional custom feature configuration
        """
        # Call parent initializer
        super().__init__(p, q, include_constant)
        
        # Add custom attributes
        self.custom_feature = custom_feature
        self._custom_results = {}
    
    async def custom_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform custom analysis on the data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
            
        Returns
        -------
        dict
            Custom analysis results
        """
        # First make sure model is fitted
        if self.params is None:
            # Fit the model if not already fitted
            await self.async_fit(data)
        
        # Now perform custom analysis
        results = {}
        
        # Example custom analysis: compute rolling statistics
        window = 20  # 20-day rolling window
        rolling_mean = np.zeros(len(data) - window + 1)
        rolling_std = np.zeros(len(data) - window + 1)
        
        for i in range(len(rolling_mean)):
            window_data = data[i:i+window]
            rolling_mean[i] = np.mean(window_data)
            rolling_std[i] = np.std(window_data, ddof=1)
        
        # Store results
        results['rolling_mean'] = rolling_mean
        results['rolling_std'] = rolling_std
        
        # If we have residuals, compute additional statistics
        if self.residuals is not None:
            residuals = self.residuals
            results['residual_acf'] = np.correlate(residuals, residuals, mode='full')
            results['residual_acf'] = results['residual_acf'][len(residuals)-1:] / np.var(residuals)
        
        # Save results for later access
        self._custom_results = results
        
        return results
    
    def plot_custom_analysis(self):
        """
        Plot the custom analysis results.
        """
        import matplotlib.pyplot as plt
        
        if not self._custom_results:
            print("No custom analysis results available. Run custom_analysis() first.")
            return
        
        # Plot rolling statistics
        if 'rolling_mean' in self._custom_results and 'rolling_std' in self._custom_results:
            rolling_mean = self._custom_results['rolling_mean']
            rolling_std = self._custom_results['rolling_std']
            
            plt.figure(figsize=(12, 6))
            plt.plot(rolling_mean, label='Rolling Mean')
            plt.plot(rolling_mean + 2*rolling_std, 'r--', label='Mean ± 2×Std')
            plt.plot(rolling_mean - 2*rolling_std, 'r--')
            plt.title('Rolling Statistics (20-day window)')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        # Plot residual ACF if available
        if 'residual_acf' in self._custom_results:
            residual_acf = self._custom_results['residual_acf']
            
            plt.figure(figsize=(12, 6))
            lags = min(50, len(residual_acf))  # Show up to 50 lags
            plt.stem(range(lags), residual_acf[:lags])
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=1.96/np.sqrt(len(self.residuals)), color='r', linestyle='--', alpha=0.7)
            plt.axhline(y=-1.96/np.sqrt(len(self.residuals)), color='r', linestyle='--', alpha=0.7)
            plt.title('Residual Autocorrelation Function')
            plt.xlabel('Lag')
            plt.ylabel('ACF')
            plt.grid(True, alpha=0.3)
            plt.show()

# Example usage
async def custom_model_demo():
    # Generate sample data
    n = 1000
    ar_params = [0.7, -0.2]
    ma_params = [0.5]
    data = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    
    for t in range(2, n):
        data[t] = ar_params[0] * data[t-1] + ar_params[1] * data[t-2] + errors[t] + ma_params[0] * errors[t-1]
    
    # Create and use custom model
    model = CustomARMA(p=2, q=1, include_constant=True, custom_feature='advanced')
    
    # Fit the model
    print("Fitting custom ARMA model...")
    converged = await model.async_fit(data)
    
    if converged:
        print("Model estimation converged successfully!")
        
        # Run standard diagnostics
        diagnostics = model.diagnostic_tests()
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Run custom analysis
        print("\nPerforming custom analysis...")
        custom_results = await model.custom_analysis(data)
        
        # Plot custom analysis results
        model.plot_custom_analysis()
    else:
        print("Model estimation did not converge")

# Run the custom model demo
# asyncio.run(custom_model_demo())
```

### Time Series Forecast Combination

This advanced example demonstrates how to implement forecast combination using multiple models:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from typing import List, Dict, Any
from mfe.models import ARMAX, GARCH

class ForecastCombiner:
    """
    Combines forecasts from multiple models using various weighting schemes.
    """
    
    def __init__(self):
        """Initialize the forecast combiner."""
        self.models = []
        self.model_names = []
        self.weights = None
        self.forecasts = {}
        self.combined_forecast = None
    
    def add_model(self, model, name: str):
        """
        Add a model to the combiner.
        
        Parameters
        ----------
        model : object
            Model instance with a forecast method
        name : str
            Name identifier for the model
        """
        self.models.append(model)
        self.model_names.append(name)
    
    async def fit_models(self, data: np.ndarray, exog: np.ndarray = None):
        """
        Fit all models asynchronously.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        exog : np.ndarray, optional
            Exogenous variables
        """
        for i, model in enumerate(self.models):
            name = self.model_names[i]
            print(f"Fitting model: {name}")
            
            # Check if model has async_fit method
            if hasattr(model, 'async_fit'):
                if exog is not None and hasattr(model, 'supports_exog') and model.supports_exog:
                    await model.async_fit(data, exog)
                else:
                    await model.async_fit(data)
            else:
                # Fallback for models without async_fit
                if exog is not None and hasattr(model, 'supports_exog') and model.supports_exog:
                    model.fit(data, exog)
                else:
                    model.fit(data)
    
    def generate_individual_forecasts(self, steps: int, exog_future: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Generate forecasts from each model.
        
        Parameters
        ----------
        steps : int
            Forecast horizon
        exog_future : np.ndarray, optional
            Future exogenous variables
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of forecasts from each model
        """
        self.forecasts = {}
        
        for i, model in enumerate(self.models):
            name = self.model_names[i]
            print(f"Generating forecast from model: {name}")
            
            # Check if model supports exogenous variables
            if exog_future is not None and hasattr(model, 'supports_exog') and model.supports_exog:
                forecast = model.forecast(steps=steps, exog_future=exog_future)
            else:
                forecast = model.forecast(steps=steps)
            
            self.forecasts[name] = forecast
        
        return self.forecasts
    
    def combine_forecasts(self, method: str = 'equal', **kwargs) -> np.ndarray:
        """
        Combine forecasts using specified method.
        
        Parameters
        ----------
        method : str
            Weighting method ('equal', 'performance', 'inverse_error', 'optimal')
        **kwargs
            Additional method-specific parameters
            
        Returns
        -------
        np.ndarray
            Combined forecast
        """
        if not self.forecasts:
            raise ValueError("No forecasts available. Run generate_individual_forecasts first.")
        
        # Get forecast arrays
        forecast_arrays = list(self.forecasts.values())
        
        # Check all forecasts have same length
        forecast_length = len(forecast_arrays[0])
        if not all(len(f) == forecast_length for f in forecast_arrays):
            raise ValueError("All forecasts must have the same length")
        
        # Compute weights based on method
        n_models = len(self.models)
        
        if method == 'equal':
            # Equal weights
            self.weights = np.ones(n_models) / n_models
            
        elif method == 'performance':
            # Weights based on in-sample performance (e.g., AIC)
            if 'criterion' not in kwargs:
                raise ValueError("Must specify 'criterion' for performance weighting")
                
            criterion = kwargs['criterion']
            criterion_values = []
            
            for model in self.models:
                if not hasattr(model, 'diagnostic_tests'):
                    raise ValueError("All models must have diagnostic_tests method")
                
                diagnostics = model.diagnostic_tests()
                if criterion not in diagnostics:
                    raise ValueError(f"Criterion '{criterion}' not found in model diagnostics")
                
                criterion_values.append(diagnostics[criterion])
            
            # For criteria like AIC/BIC, lower is better
            criterion_values = np.array(criterion_values)
            
            # Invert and normalize
            weights = 1.0 / criterion_values
            self.weights = weights / np.sum(weights)
            
        elif method == 'inverse_error':
            # Weights based on inverse in-sample error
            errors = []
            
            for model in self.models:
                if not hasattr(model, 'residuals'):
                    raise ValueError("All models must have residuals attribute")
                
                # Mean squared error
                mse = np.mean(model.residuals**2)
                errors.append(mse)
            
            # Convert to inverse (lower error = higher weight)
            errors = np.array(errors)
            weights = 1.0 / errors
            self.weights = weights / np.sum(weights)
            
        elif method == 'optimal':
            # Optimal weights based on forecast error covariance
            # This requires historical forecast errors
            if 'error_cov' not in kwargs:
                raise ValueError("Must provide 'error_cov' for optimal weighting")
                
            error_cov = kwargs['error_cov']
            ones = np.ones(n_models)
            
            # Optimal weights formula
            # w = (Σ^-1 * 1) / (1' * Σ^-1 * 1)
            inv_cov = np.linalg.inv(error_cov)
            self.weights = inv_cov.dot(ones) / (ones.dot(inv_cov).dot(ones))
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure weights sum to 1
        self.weights = self.weights / np.sum(self.weights)
        
        # Combine forecasts
        combined = np.zeros(forecast_length)
        for i, forecast in enumerate(forecast_arrays):
            combined += self.weights[i] * forecast
        
        self.combined_forecast = combined
        return combined
    
    def plot_forecasts(self, data: np.ndarray = None, dates=None):
        """
        Plot individual and combined forecasts.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Historical data to include in plot
        dates : array-like, optional
            Date labels for x-axis
        """
        if not self.forecasts or self.combined_forecast is None:
            raise ValueError("No forecasts available. Run combine_forecasts first.")
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical data if provided
        if data is not None:
            if dates is not None and len(dates) >= len(data):
                plt.plot(dates[:len(data)], data, 'k-', label='Historical')
            else:
                plt.plot(data, 'k-', label='Historical')
        
        # Plot forecasting period
        forecast_length = len(self.combined_forecast)
        
        # Create x-axis for forecasts
        if data is not None:
            forecast_start = len(data)
            forecast_end = forecast_start + forecast_length
            forecast_range = range(forecast_start, forecast_end)
            
            if dates is not None and len(dates) >= forecast_end:
                forecast_dates = dates[forecast_start:forecast_end]
            else:
                forecast_dates = forecast_range
        else:
            forecast_range = range(forecast_length)
            forecast_dates = forecast_range
        
        # Plot individual forecasts
        for name, forecast in self.forecasts.items():
            plt.plot(forecast_dates, forecast, '--', alpha=0.5, label=f'{name}')
        
        # Plot combined forecast
        plt.plot(forecast_dates, self.combined_forecast, 'r-', linewidth=2, label='Combined')
        
        # Add legend and labels
        plt.title('Forecast Comparison')
        plt.xlabel('Time' if dates is None else 'Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Display weight information
        weight_text = "Model Weights:\n"
        for i, name in enumerate(self.model_names):
            weight_text += f"{name}: {self.weights[i]:.4f}\n"
        
        plt.figtext(0.02, 0.02, weight_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

# Example usage
async def forecast_combination_demo():
    # Generate sample data
    n = 500
    np.random.seed(42)
    
    # AR(1) process with structural break
    data = np.zeros(n)
    for t in range(1, n):
        if t < n//2:
            data[t] = 0.8 * data[t-1] + np.random.normal(0, 1)
        else:
            data[t] = 0.4 * data[t-1] + np.random.normal(0, 1.5)
    
    # Split into training and test
    train_size = 450
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create models with different specifications
    model1 = ARMAX(p=1, q=0, include_constant=True)  # AR(1)
    model2 = ARMAX(p=2, q=0, include_constant=True)  # AR(2)
    model3 = ARMAX(p=1, q=1, include_constant=True)  # ARMA(1,1)
    
    # Create forecast combiner
    combiner = ForecastCombiner()
    combiner.add_model(model1, 'AR(1)')
    combiner.add_model(model2, 'AR(2)')
    combiner.add_model(model3, 'ARMA(1,1)')
    
    # Fit all models
    print("Fitting models...")
    await combiner.fit_models(train_data)
    
    # Generate individual forecasts
    forecast_horizon = len(test_data)
    print(f"\nGenerating {forecast_horizon}-step ahead forecasts...")
    individual_forecasts = combiner.generate_individual_forecasts(steps=forecast_horizon)
    
    # Try different combination methods
    print("\nCombining forecasts with different methods...")
    
    # Equal weights
    print("\n1. Equal weights:")
    equal_weights_forecast = combiner.combine_forecasts(method='equal')
    combiner.plot_forecasts(data=train_data)
    
    # Performance-based weights
    print("\n2. Performance-based weights (AIC):")
    perf_weights_forecast = combiner.combine_forecasts(method='performance', criterion='AIC')
    combiner.plot_forecasts(data=train_data)
    
    # Inverse error weights
    print("\n3. Inverse error weights:")
    inv_error_forecast = combiner.combine_forecasts(method='inverse_error')
    combiner.plot_forecasts(data=train_data)
    
    # Evaluate forecast accuracy
    def compute_rmse(forecast, actual):
        return np.sqrt(np.mean((forecast - actual)**2))
    
    print("\nForecast Evaluation (RMSE):")
    for name, forecast in individual_forecasts.items():
        rmse = compute_rmse(forecast, test_data)
        print(f"{name}: {rmse:.4f}")
    
    # Evaluate combined forecasts
    equal_rmse = compute_rmse(equal_weights_forecast, test_data)
    perf_rmse = compute_rmse(perf_weights_forecast, test_data)
    inv_rmse = compute_rmse(inv_error_forecast, test_data)
    
    print(f"Combined (Equal): {equal_rmse:.4f}")
    print(f"Combined (AIC): {perf_rmse:.4f}")
    print(f"Combined (Inverse Error): {inv_rmse:.4f}")
    
    # Plot final comparison with actual test data
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(range(len(train_data)), train_data, 'k-', label='Training Data')
    
    # Plot test data
    plt.plot(range(len(train_data), len(data)), test_data, 'k-', label='Test Data')
    
    # Plot combined forecasts
    forecast_range = range(len(train_data), len(data))
    plt.plot(forecast_range, equal_weights_forecast, 'b--', label=f'Equal Weights (RMSE: {equal_rmse:.4f})')
    plt.plot(forecast_range, perf_weights_forecast, 'g--', label=f'AIC Weights (RMSE: {perf_rmse:.4f})')
    plt.plot(forecast_range, inv_error_forecast, 'r--', label=f'Inverse Error Weights (RMSE: {inv_rmse:.4f})')
    
    plt.title('Forecast Combination Evaluation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Run the forecast combination demo
# asyncio.run(forecast_combination_demo())
```

## Additional Resources

### Official Documentation

For more detailed information, refer to the complete MFE Toolbox documentation:

- Online Documentation: [MFE Toolbox Documentation](https://mfe-toolbox.readthedocs.io/)
- API Reference: [API Documentation](https://mfe-toolbox.readthedocs.io/en/latest/api.html)
- Example Gallery: [Example Gallery](https://mfe-toolbox.readthedocs.io/en/latest/examples/index.html)

### Academic References

The MFE Toolbox implements methods from several key academic papers:

1. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
2. Engle, R. F., & Sheppard, K. (2001). Theoretical and empirical properties of dynamic conditional correlation multivariate GARCH (No. w8554). National Bureau of Economic Research.
3. Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.
4. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association, 89(428), 1303-1313.
5. Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. Econometrica, 71(2), 579-625.

### Useful Links

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [Numba Documentation](https://numba.pydata.org/numba-doc/latest/index.html)
- [PyQt6 Documentation](https://doc.qt.io/qtforpython-6/)
```

# infrastructure/templates/TUTORIALS.md
``` markdown
# MFE Toolbox Tutorials

A comprehensive guide to using the Python-based MFE (Financial Econometrics) Toolbox version 4.0.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Environment Setup Tutorial](#environment-setup-tutorial)
4. [ARMAX Modeling Tutorial](#armax-modeling-tutorial)
5. [GARCH Modeling Tutorial](#garch-modeling-tutorial)
6. [High-Frequency Analysis Tutorial](#high-frequency-analysis-tutorial)
7. [Bootstrap Analysis Tutorial](#bootstrap-analysis-tutorial)
8. [GUI Interface Tutorial](#gui-interface-tutorial)
9. [Advanced Topics](#advanced-topics)
10. [Performance Optimization with Numba](#performance-optimization-with-numba)

## Introduction

The MFE Toolbox is a comprehensive suite of Python modules designed for financial time series modeling and econometric analysis. Originally derived from a MATLAB implementation, this toolbox has been completely rewritten in Python 3.12, incorporating modern programming constructs such as async/await patterns and strict type hints.

### Key Features

- Time series modeling with ARMA/ARMAX
- Volatility modeling with various GARCH specifications
- High-frequency financial data analysis
- Bootstrap-based statistical inference
- Interactive GUI built with PyQt6
- Performance optimization with Numba

### Prerequisites

Before proceeding with these tutorials, you should have:
- Basic knowledge of Python programming
- Familiarity with financial time series concepts
- Working installation of Python 3.12 or higher

## Getting Started

To get started with the MFE Toolbox, first install the package following the instructions in the [Installation Guide](INSTALLATION.md).

Quick installation:

```bash
pip install mfe
```

For a more detailed setup, we recommend creating a virtual environment:

```bash
python -m venv mfe-env
source mfe-env/bin/activate  # On Windows: mfe-env\Scripts\activate
pip install mfe
```

Verify your installation:

```python
import mfe
print(mfe.__version__)  # Should print 4.0 or higher
```

## Environment Setup Tutorial

This tutorial guides you through setting up an optimal environment for financial econometrics analysis with the MFE Toolbox.

### Creating a Project Environment

Best practices suggest using a dedicated environment for financial analysis projects:

```bash
# Create a project directory
mkdir financial_analysis
cd financial_analysis

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the MFE Toolbox and dependencies
pip install mfe
```

### Setting Up a Jupyter Environment

For interactive analysis, you might want to use Jupyter notebooks:

```bash
# Install Jupyter in your virtual environment
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook
```

### Creating Your First Analysis Script

Create a new file called `initial_analysis.py`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mfe.models import ARMAX

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data (AR(1) process)
n = 1000
ar_coef = 0.7
data = np.zeros(n)
for t in range(1, n):
    data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Simulated AR(1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('ar1_process.png')
plt.show()

print("Environment setup complete!")
```

Run the script to verify your environment works correctly:

```bash
python initial_analysis.py
```

## ARMAX Modeling Tutorial

This tutorial demonstrates how to perform time series analysis using the ARMAX model from the MFE Toolbox.

### Understanding ARMAX Models

ARMAX (AutoRegressive Moving Average with eXogenous variables) models are extensions of ARMA models that include additional external regressors. The general form is:

y(t) = c + Σφᵢy(t-i) + Σθⱼε(t-j) + Σβₖx(t)ₖ + ε(t)

Where:
- y(t) is the dependent variable
- c is a constant
- φᵢ are the AR coefficients
- θⱼ are the MA coefficients
- βₖ are coefficients for exogenous variables x(t)ₖ
- ε(t) is white noise

### Basic ARMA Model Example

Let's start with a simple ARMA(2,1) model without exogenous variables:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from mfe.models import ARMAX

# Generate sample data: ARMA(2,1) process
n = 1000
ar_params = [0.7, -0.2]  # AR parameters
ma_params = [0.5]        # MA parameters
data = np.zeros(n)
errors = np.random.normal(0, 1, n)

for t in range(2, n):
    data[t] = ar_params[0] * data[t-1] + ar_params[1] * data[t-2] + errors[t] + ma_params[0] * errors[t-1]

# Create and estimate the model
model = ARMAX(p=2, q=1, include_constant=True)

async def estimate_model():
    print("Estimating ARMA(2,1) model...")
    
    # Fit the model using async/await pattern
    converged = await model.async_fit(data)
    
    if converged:
        print("Model estimation converged successfully!")
        
        # Extract parameter estimates
        ar_params_est, ma_params_est, constant, _ = model._extract_params(model.params)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1): {ar_params_est[0]:.4f} (True: {ar_params[0]:.4f})")
        print(f"AR(2): {ar_params_est[1]:.4f} (True: {ar_params[1]:.4f})")
        print(f"MA(1): {ma_params_est[0]:.4f} (True: {ma_params[0]:.4f})")
        if constant is not None:
            print(f"Constant: {constant:.4f}")
        
        # Get diagnostics
        diagnostics = model.diagnostic_tests()
        
        # Print key model fit statistics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Plot the results
        plot_results(model, data, diagnostics)
    else:
        print("Model estimation did not converge")

def plot_results(model, data, diagnostics):
    # Plot original data and residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot original data
    ax1.plot(data)
    ax1.set_title('Original Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # Plot residuals
    ax2.plot(model.residuals)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.set_title('Model Residuals')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residual')
    
    plt.tight_layout()
    plt.show()
    
    # Plot residual diagnostics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of residuals
    ax1.hist(model.residuals, bins=30, density=True, alpha=0.7)
    ax1.set_title('Residual Distribution')
    ax1.set_xlabel('Residual')
    ax1.set_ylabel('Density')
    
    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(model.residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

# Run the model estimation
asyncio.run(estimate_model())
```

### ARMAX Model with Exogenous Variables

Now, let's extend our model to include exogenous variables:

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

# Create and estimate ARMAX model
armax_model = ARMAX(p=2, q=1, include_constant=True)

async def estimate_armax():
    print("Estimating ARMAX(2,1) model with exogenous variables...")
    
    # Fit the model
    converged = await armax_model.async_fit(data_exog, exog=exog)
    
    if converged:
        print("ARMAX estimation converged successfully!")
        
        # Extract parameters
        ar_params_est, ma_params_est, constant, exog_params_est = armax_model._extract_params(armax_model.params)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1): {ar_params_est[0]:.4f} (True: {ar_params[0]:.4f})")
        print(f"AR(2): {ar_params_est[1]:.4f} (True: {ar_params[1]:.4f})")
        print(f"MA(1): {ma_params_est[0]:.4f} (True: {ma_params[0]:.4f})")
        
        if exog_params_est is not None:
            for i, param in enumerate(exog_params_est):
                print(f"Exog{i+1}: {param:.4f} (True: {exog_params[i]:.4f})")
        
        if constant is not None:
            print(f"Constant: {constant:.4f}")
        
        # Generate forecasts with future exogenous variables
        forecast_steps = 10
        exog_future = np.random.normal(0, 1, (forecast_steps, 2))
        forecasts = armax_model.forecast(steps=forecast_steps, exog_future=exog_future)
        
        print(f"\nForecasts (next {forecast_steps} steps):")
        for i, value in enumerate(forecasts):
            print(f"t+{i+1}: {value:.4f}")
    else:
        print("ARMAX estimation did not converge")

# Run the ARMAX estimation
asyncio.run(estimate_armax())
```

### Forecasting with ARMAX Models

Forecasting is a key application of ARMAX models. Here's how to generate forecasts:

```python
async def forecast_demo():
    # First fit the model
    converged = await model.async_fit(data)
    
    if not converged:
        print("Model estimation did not converge, cannot forecast")
        return
    
    # Generate forecasts for different horizons
    horizons = [1, 5, 10, 20]
    
    for h in horizons:
        forecast = model.forecast(steps=h)
        print(f"{h}-step ahead forecast: {forecast[-1]:.4f}")
    
    # Plot data with forecasts
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Historical Data')
    
    # Add forecasts for the longest horizon
    longest = max(horizons)
    forecast_long = model.forecast(steps=longest)
    forecast_indices = range(len(data), len(data) + longest)
    plt.plot(forecast_indices, forecast_long, 'r--', label=f'{longest}-step Forecast')
    
    # Add confidence intervals (simulated for demonstration)
    # In a full implementation, you would calculate proper confidence intervals
    upper_bound = forecast_long + 1.96 * np.std(model.residuals)
    lower_bound = forecast_long - 1.96 * np.std(model.residuals)
    
    plt.fill_between(forecast_indices, lower_bound, upper_bound, 
                     color='r', alpha=0.1, label='95% Confidence Interval')
    
    plt.title('ARMA Model Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Run the forecast demonstration
asyncio.run(forecast_demo())
```

### Model Selection and Diagnostics

It's important to select the best model using information criteria and diagnostic tests:

```python
async def model_selection():
    # Try different model orders
    p_values = [1, 2, 3]
    q_values = [0, 1, 2]
    
    results = []
    
    for p in p_values:
        for q in q_values:
            print(f"Estimating ARMA({p},{q})...")
            model = ARMAX(p=p, q=q, include_constant=True)
            
            try:
                converged = await model.async_fit(data)
                
                if converged:
                    # Get diagnostics
                    diagnostics = model.diagnostic_tests()
                    
                    # Store results
                    results.append({
                        'p': p,
                        'q': q,
                        'AIC': diagnostics['AIC'],
                        'BIC': diagnostics['BIC'],
                        'Log-likelihood': model.loglikelihood,
                        'Converged': converged
                    })
                else:
                    print(f"ARMA({p},{q}) did not converge")
            except Exception as e:
                print(f"Error estimating ARMA({p},{q}): {str(e)}")
    
    # Convert to DataFrame for easier analysis
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # Sort by AIC (lower is better)
    print("\nModel Selection Results (sorted by AIC):")
    print(results_df.sort_values('AIC').to_string(index=False))
    
    # Sort by BIC (lower is better)
    print("\nModel Selection Results (sorted by BIC):")
    print(results_df.sort_values('BIC').to_string(index=False))

# Run model selection
asyncio.run(model_selection())
```

### Real-World Example with Financial Data

Let's apply ARMAX modeling to real financial data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import yfinance as yahoo_finance  # For downloading financial data (pip install yfinance)
from mfe.models import ARMAX

async def analyze_financial_data():
    # Download S&P 500 data
    print("Downloading S&P 500 data...")
    sp500 = yahoo_finance.download('^GSPC', start='2018-01-01', end='2023-01-01')
    
    # Calculate daily returns
    sp500['Return'] = sp500['Adj Close'].pct_change() * 100
    returns = sp500['Return'].dropna().values
    
    # Plot returns
    plt.figure(figsize=(12, 6))
    plt.plot(returns)
    plt.title('S&P 500 Daily Returns (%)')
    plt.xlabel('Observation')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Create and fit ARMA model
    model = ARMAX(p=2, q=1, include_constant=True)
    
    print("Estimating ARMA(2,1) model on S&P 500 returns...")
    converged = await model.async_fit(returns)
    
    if converged:
        print("Model estimation converged successfully!")
        
        # Extract parameters
        ar_params, ma_params, constant, _ = model._extract_params(model.params)
        
        print(f"\nEstimated Parameters:")
        print(f"AR(1): {ar_params[0]:.4f}")
        print(f"AR(2): {ar_params[1]:.4f}")
        print(f"MA(1): {ma_params[0]:.4f}")
        if constant is not None:
            print(f"Constant: {constant:.4f}")
        
        # Get diagnostics
        diagnostics = model.diagnostic_tests()
        
        # Print key model fit statistics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Ljung-Box test for autocorrelation
        lb = diagnostics['ljung_box']
        print(f"Ljung-Box Q({lb['lags']}): {lb['statistic']:.4f}, p-value: {lb['p_value']:.4f}")
        
        # Generate forecasts
        forecast_steps = 5
        forecasts = model.forecast(steps=forecast_steps)
        
        print(f"\nForecasts (next {forecast_steps} days):")
        for i, value in enumerate(forecasts):
            print(f"Day {i+1}: {value:.4f}%")
    else:
        print("Model estimation did not converge")

# Run financial data analysis
asyncio.run(analyze_financial_data())
```

## GARCH Modeling Tutorial

This tutorial covers volatility modeling using the GARCH family of models in the MFE Toolbox.

### Understanding GARCH Models

GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models are used to analyze and forecast volatility in financial time series. The basic GARCH(1,1) model can be expressed as:

σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

Where:
- σ²(t) is the conditional variance at time t
- ω is a constant (omega)
- α is the ARCH parameter
- β is the GARCH parameter
- ε(t) is the innovation (residual) term

### Basic GARCH(1,1) Model Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from mfe.models import GARCH

# Generate sample returns with volatility clustering
np.random.seed(42)
n = 2000
returns = np.zeros(n)
volatility = np.zeros(n)
volatility[0] = 1.0

# True parameters
omega = 0.1
alpha = 0.1
beta = 0.8

# Generate GARCH(1,1) process
for t in range(1, n):
    volatility[t] = omega + alpha * returns[t-1]**2 + beta * volatility[t-1]
    returns[t] = np.random.normal(0, np.sqrt(volatility[t]))

# Plot simulated returns
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(returns)
plt.title('Simulated Returns with GARCH(1,1) Volatility')
plt.ylabel('Return')

plt.subplot(2, 1, 2)
plt.plot(np.sqrt(volatility))
plt.title('True Conditional Volatility')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.tight_layout()
plt.show()

# Create and fit GARCH(1,1) model
garch_model = GARCH(p=1, q=1)  # p=ARCH order, q=GARCH order

async def estimate_garch():
    print("Estimating GARCH(1,1) model...")
    
    # Fit the model using async/await pattern
    converged = await garch_model.async_fit(returns)
    
    if converged:
        print("GARCH estimation converged successfully!")
        
        # Extract parameters
        print(f"\nEstimated Parameters:")
        print(f"omega: {garch_model._model_params['omega']:.4f} (True: {omega:.4f})")
        print(f"alpha: {garch_model._model_params['alpha'][0]:.4f} (True: {alpha:.4f})")
        print(f"beta: {garch_model._model_params['beta'][0]:.4f} (True: {beta:.4f})")
        
        # Calculate persistence
        persistence = garch_model._model_params['alpha'][0] + garch_model._model_params['beta'][0]
        print(f"Persistence (α+β): {persistence:.4f}")
        
        # Get diagnostics
        diagnostics = garch_model.diagnostic_tests()
        
        # Print key model fit statistics
        print(f"\nModel Fit Statistics:")
        print(f"Log-likelihood: {garch_model.loglikelihood:.4f}")
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Plot estimated volatility vs true volatility
        estimated_vol = garch_model.conditional_volatility
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(returns)
        plt.title('Simulated Returns')
        plt.ylabel('Return')
        
        plt.subplot(3, 1, 2)
        plt.plot(np.sqrt(volatility), label='True')
        plt.plot(estimated_vol, 'r--', label='Estimated')
        plt.title('Conditional Volatility: True vs. Estimated')
        plt.ylabel('Volatility')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(np.sqrt(volatility) - estimated_vol)
        plt.title('Volatility Estimation Error')
        plt.xlabel('Time')
        plt.ylabel('Error')
        
        plt.tight_layout()
        plt.show()
        
        # Forecast volatility
        forecast_horizon = 20
        volatility_forecast = garch_model.forecast_variance(steps=forecast_horizon)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(estimated_vol)), estimated_vol, label='In-sample')
        plt.plot(range(len(estimated_vol), len(estimated_vol) + forecast_horizon), 
                 np.sqrt(volatility_forecast), 'r--', label='Forecast')
        plt.title('Volatility Forecast')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("GARCH estimation did not converge")

# Run GARCH estimation
asyncio.run(estimate_garch())
```

### Advanced GARCH Models: EGARCH and GJR-GARCH

The MFE Toolbox supports more advanced GARCH models that can capture asymmetric volatility responses:

```python
from mfe.models import EGARCH, GJR_GARCH

# Generate data with leverage effect (asymmetric volatility)
n = 2000
returns_asym = np.zeros(n)
volatility_asym = np.zeros(n)
volatility_asym[0] = 1.0

# True parameters (with asymmetric effect)
omega = 0.05
alpha = 0.05
beta = 0.85
gamma = 0.1  # Asymmetry parameter

# Simulate process with asymmetric volatility response
for t in range(1, n):
    # GJR-GARCH process: more volatility after negative returns
    leverage = 1.0 if returns_asym[t-1] < 0 else 0.0
    volatility_asym[t] = omega + alpha * returns_asym[t-1]**2 + gamma * leverage * returns_asym[t-1]**2 + beta * volatility_asym[t-1]
    returns_asym[t] = np.random.normal(0, np.sqrt(volatility_asym[t]))

# Plot simulated returns with leverage effect
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(returns_asym)
plt.title('Simulated Returns with Asymmetric Volatility')
plt.ylabel('Return')

plt.subplot(2, 1, 2)
plt.plot(np.sqrt(volatility_asym))
plt.title('True Conditional Volatility (with Leverage Effect)')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.tight_layout()
plt.show()

# Fit different GARCH models to see which captures asymmetry best
async def compare_garch_models():
    print("Comparing different GARCH models on data with leverage effect...")
    
    # Standard GARCH
    garch_model = GARCH(p=1, q=1)
    
    # GJR-GARCH (captures asymmetry through indicator function)
    gjr_model = GJR_GARCH(p=1, q=1)
    
    # EGARCH (captures asymmetry through log transformation)
    egarch_model = EGARCH(p=1, q=1)
    
    # Fit all models
    converged_garch = await garch_model.async_fit(returns_asym)
    converged_gjr = await gjr_model.async_fit(returns_asym)
    converged_egarch = await egarch_model.async_fit(returns_asym)
    
    # Print results
    print("\nModel Comparison Results:")
    results = []
    
    if converged_garch:
        diagnostics = garch_model.diagnostic_tests()
        results.append({
            'Model': 'GARCH(1,1)',
            'Log-likelihood': garch_model.loglikelihood,
            'AIC': diagnostics['AIC'],
            'BIC': diagnostics['BIC']
        })
    
    if converged_gjr:
        diagnostics = gjr_model.diagnostic_tests()
        results.append({
            'Model': 'GJR-GARCH(1,1)',
            'Log-likelihood': gjr_model.loglikelihood,
            'AIC': diagnostics['AIC'],
            'BIC': diagnostics['BIC']
        })
    
    if converged_egarch:
        diagnostics = egarch_model.diagnostic_tests()
        results.append({
            'Model': 'EGARCH(1,1)',
            'Log-likelihood': egarch_model.loglikelihood,
            'AIC': diagnostics['AIC'],
            'BIC': diagnostics['BIC']
        })
    
    # Convert to DataFrame and display
    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Plot volatility estimates from all models
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(returns_asym)
    plt.title('Returns with Asymmetric Volatility')
    plt.ylabel('Return')
    
    plt.subplot(4, 1, 2)
    plt.plot(np.sqrt(volatility_asym), 'k-', label='True')
    if converged_garch:
        plt.plot(garch_model.conditional_volatility, 'b--', label='GARCH')
    plt.legend()
    plt.title('True vs. GARCH(1,1)')
    plt.ylabel('Volatility')
    
    plt.subplot(4, 1, 3)
    plt.plot(np.sqrt(volatility_asym), 'k-', label='True')
    if converged_gjr:
        plt.plot(gjr_model.conditional_volatility, 'g--', label='GJR-GARCH')
    plt.legend()
    plt.title('True vs. GJR-GARCH(1,1)')
    plt.ylabel('Volatility')
    
    plt.subplot(4, 1, 4)
    plt.plot(np.sqrt(volatility_asym), 'k-', label='True')
    if converged_egarch:
        plt.plot(egarch_model.conditional_volatility, 'r--', label='EGARCH')
    plt.legend()
    plt.title('True vs. EGARCH(1,1)')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    
    plt.tight_layout()
    plt.show()

# Compare GARCH models
asyncio.run(compare_garch_models())
```

### Real-World Volatility Analysis with Financial Data

Let's apply GARCH modeling to real financial data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import yfinance as yahoo_finance
from mfe.models import GARCH, GJR_GARCH, EGARCH

async def analyze_market_volatility():
    # Download market data
    print("Downloading market data...")
    tickers = ['SPY', 'QQQ', 'GLD']  # S&P 500 ETF, NASDAQ ETF, Gold ETF
    
    market_data = {}
    for ticker in tickers:
        data = yahoo_finance.download(ticker, start='2018-01-01', end='2023-01-01')
        market_data[ticker] = data['Adj Close'].pct_change().dropna() * 100  # Convert to percentage
    
    # Combine into a DataFrame
    returns_df = pd.DataFrame({ticker: data for ticker, data in market_data.items()})
    
    # Plot returns
    plt.figure(figsize=(14, 8))
    for ticker in tickers:
        plt.plot(returns_df.index, returns_df[ticker], label=ticker)
    plt.title('Daily Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Apply GARCH models to each asset
    for ticker in tickers:
        returns = returns_df[ticker].values
        
        print(f"\nAnalyzing volatility for {ticker}...")
        
        # Create and fit GJR-GARCH model (captures leverage effect)
        model = GJR_GARCH(p=1, q=1)
        
        converged = await model.async_fit(returns)
        
        if converged:
            print(f"GJR-GARCH estimation for {ticker} converged successfully!")
            
            # Extract parameters
            param_dict = model._model_params
            print(f"\nEstimated Parameters:")
            print(f"omega: {param_dict.get('omega', 'N/A')}")
            print(f"alpha: {param_dict.get('alpha', ['N/A'])[0]}")
            print(f"beta: {param_dict.get('beta', ['N/A'])[0]}")
            print(f"gamma: {param_dict.get('gamma', ['N/A'])[0]}")  # Asymmetry parameter
            
            # Plot returns and estimated volatility
            plt.figure(figsize=(12, 8))
            
            # Returns
            plt.subplot(2, 1, 1)
            plt.plot(returns_df.index, returns_df[ticker])
            plt.title(f'{ticker} Daily Returns (%)')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3)
            
            # Volatility
            plt.subplot(2, 1, 2)
            plt.plot(returns_df.index, model.conditional_volatility)
            plt.title(f'{ticker} Estimated Conditional Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Forecast volatility
            forecast_days = 20
            volatility_forecast = model.forecast_variance(steps=forecast_days)
            
            # Display forecast
            print(f"\nVolatility Forecast for {ticker} (next {forecast_days} days):")
            for i, vol in enumerate(volatility_forecast):
                print(f"Day {i+1}: {np.sqrt(vol):.4f}")
        else:
            print(f"GJR-GARCH estimation for {ticker} did not converge")

# Run market volatility analysis
asyncio.run(analyze_market_volatility())
```

## High-Frequency Analysis Tutorial

This tutorial demonstrates how to use the MFE Toolbox for analyzing high-frequency financial data.

### Understanding Realized Volatility Measures

High-frequency financial data allows for more precise volatility estimation through realized measures. The basic realized variance is defined as:

RV = Σ(r²(i))

Where r(i) are intraday returns. The MFE Toolbox implements advanced realized measures that account for microstructure noise and other market frictions.

### Generating Synthetic High-Frequency Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mfe.models import realized_variance, realized_kernel

# Generate synthetic high-frequency data
np.random.seed(42)

# Simulation parameters
n_days = 5
n_intraday = 390  # Typical number of 1-minute observations in a 6.5-hour trading day
price_path = []
timestamps = []
true_daily_vol = []  # Store true daily volatility

# Initial price
price = 100.0

# Generate data for each day
for day in range(n_days):
    # Set date and daily volatility
    base_date = datetime(2023, 1, 2) + timedelta(days=day)
    daily_return = np.random.normal(0, 0.01)  # 1% average daily return
    daily_vol = 0.02 + 0.5 * np.random.random()  # Daily volatility between 2% and 52%
    true_daily_vol.append(daily_vol)
    
    # Simulate intraday price path
    for minute in range(n_intraday):
        # Create timestamp
        time = base_date + timedelta(minutes=minute)
        timestamps.append(time)
        
        # Add intraday pattern (U-shape volatility)
        intraday_factor = 1.0 + 0.5 * (
            np.exp(-((minute - 0) / 60)**2) + 
            np.exp(-((minute - (n_intraday-1)) / 60)**2)
        )
        
        # Generate price innovation with intraday pattern
        price_innovation = np.random.normal(
            daily_return/n_intraday,  # Expected return per minute
            daily_vol/np.sqrt(n_intraday) * intraday_factor  # Scaled volatility with pattern
        )
        
        # Update price (log-return process)
        price *= np.exp(price_innovation)
        
        # Add microstructure noise to observed price
        noise_level = 0.0001  # 1 basis point
        noisy_price = price * (1 + np.random.normal(0, noise_level))
        
        # Store noisy observed price
        price_path.append(noisy_price)

# Create DataFrame
hf_data = pd.DataFrame({
    'timestamp': timestamps,
    'price': price_path
})

# Plot full price path
plt.figure(figsize=(14, 7))
plt.plot(hf_data['timestamp'], hf_data['price'])
plt.title('Simulated High-Frequency Price Path')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)
plt.show()

# Look at data for a single day
day_1_data = hf_data[hf_data['timestamp'].dt.date == datetime(2023, 1, 2).date()]

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(day_1_data['timestamp'], day_1_data['price'])
plt.title('Day 1 Price Path')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)

# Calculate returns
day_1_data['return'] = np.log(day_1_data['price']).diff() * 100  # Percentage

plt.subplot(2, 1, 2)
plt.plot(day_1_data['timestamp'][1:], day_1_data['return'][1:])
plt.title('Day 1 Returns (%)')
plt.xlabel('Time')
plt.ylabel('Return (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Computing Realized Measures

```python
# Prepare data for realized measures computation
# We need to split by day and compute measures for each day

# Initialize arrays for realized measures
days = sorted(list(set([t.date() for t in hf_data['timestamp']])))
rv_results = []
rv_ss_results = []  # Subsampled
rk_results = []  # Realized kernel

for day in days:
    # Filter data for current day
    day_data = hf_data[hf_data['timestamp'].dt.date == day]
    
    # Extract arrays for computation
    times = np.array([t.timestamp() for t in day_data['timestamp']])
    prices = np.array(day_data['price'])
    
    # Compute realized variance with 5-minute sampling
    rv, rv_ss = realized_variance(
        prices,
        times,
        timeType='timestamp',
        samplingType='CalendarTime',
        samplingInterval=5  # 5-minute sampling
    )
    
    # Compute realized kernel (robust to noise)
    rk = realized_kernel(
        prices,
        times,
        timeType='timestamp',
        kernelType='Parzen'
    )
    
    # Store results
    rv_results.append(rv)
    rv_ss_results.append(rv_ss)
    rk_results.append(rk)

# Convert to volatility (standard deviation)
rv_vol = np.sqrt(rv_results)
rv_ss_vol = np.sqrt(rv_ss_results)
rk_vol = np.sqrt(rk_results)

# Create results table
results_df = pd.DataFrame({
    'Date': days,
    'True Volatility': true_daily_vol,
    'RV (5-min)': rv_vol,
    'RV (Subsampled)': rv_ss_vol,
    'Realized Kernel': rk_vol
})

print("Daily Volatility Estimates:")
print(results_df)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(results_df['Date'], results_df['True Volatility'], 'k-', label='True')
plt.plot(results_df['Date'], results_df['RV (5-min)'], 'b--', label='RV (5-min)')
plt.plot(results_df['Date'], results_df['RV (Subsampled)'], 'g--', label='RV (Subsampled)')
plt.title('Daily Volatility Estimates')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(results_df['Date'], results_df['True Volatility'], 'k-', label='True')
plt.plot(results_df['Date'], results_df['Realized Kernel'], 'r--', label='Realized Kernel')
plt.title('Realized Kernel Estimates')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Different Sampling Schemes

```python
# Experiment with different sampling schemes
sampling_types = [
    ('CalendarTime', 5),       # 5-minute calendar time sampling
    ('CalendarTime', 10),      # 10-minute calendar time sampling
    ('BusinessTime', 30),      # Business time sampling with 30 observations per day
    ('BusinessUniform', 30)    # Business time with jittering for uniform coverage
]

# Analyze a single day
test_day = days[0]
day_data = hf_data[hf_data['timestamp'].dt.date == test_day]
times = np.array([t.timestamp() for t in day_data['timestamp']])
prices = np.array(day_data['price'])

# Compute with different schemes
results = []

for scheme, interval in sampling_types:
    rv, rv_ss = realized_variance(
        prices,
        times,
        timeType='timestamp',
        samplingType=scheme,
        samplingInterval=interval
    )
    
    results.append({
        'Sampling Scheme': f"{scheme} ({interval})",
        'RV': np.sqrt(rv),
        'RV (Subsampled)': np.sqrt(rv_ss)
    })

# Display results
sampling_df = pd.DataFrame(results)
print("\nImpact of Different Sampling Schemes:")
print(sampling_df)

# Plot histogram of intraday returns for different sampling frequencies
plt.figure(figsize=(14, 10))

# Original 1-minute returns
day_data['return'] = np.log(day_data['price']).diff() * 100
minute_returns = day_data['return'].dropna()

plt.subplot(2, 2, 1)
plt.hist(minute_returns, bins=30, alpha=0.7)
plt.title('1-minute Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

# 5-minute returns
returns_5m = []
for i in range(0, len(prices) - 5, 5):
    ret = np.log(prices[i+5]/prices[i]) * 100
    returns_5m.append(ret)

plt.subplot(2, 2, 2)
plt.hist(returns_5m, bins=30, alpha=0.7)
plt.title('5-minute Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

# 10-minute returns
returns_10m = []
for i in range(0, len(prices) - 10, 10):
    ret = np.log(prices[i+10]/prices[i]) * 100
    returns_10m.append(ret)

plt.subplot(2, 2, 3)
plt.hist(returns_10m, bins=20, alpha=0.7)
plt.title('10-minute Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

# 30-minute returns
returns_30m = []
for i in range(0, len(prices) - 30, 30):
    ret = np.log(prices[i+30]/prices[i]) * 100
    returns_30m.append(ret)

plt.subplot(2, 2, 4)
plt.hist(returns_30m, bins=15, alpha=0.7)
plt.title('30-minute Returns Distribution')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Real-World High-Frequency Analysis

```python
# Note: Acquiring real high-frequency data usually requires paid services
# This example demonstrates the workflow using simulated tick data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mfe.models import realized_variance, realized_kernel

def analyze_real_hf_data():
    # In practice, you would load your own high-frequency data
    # For example:
    # tick_data = pd.read_csv('tick_data.csv', parse_dates=['timestamp'])
    
    # For demonstration, we'll generate synthetic tick data
    # that mimics real market microstructure
    
    print("Simulating realistic high-frequency data...")
    
    # Parameters
    n_days = 3
    trading_hours = 6.5  # 6.5 hours typical trading day
    avg_ticks_per_second = 1  # Average 1 trade per second
    price = 150.0  # Starting price
    volatility = 0.20  # Annual volatility
    daily_vol = volatility / np.sqrt(252)  # Daily volatility
    tick_size = 0.01  # Minimum price increment
    
    # Generate data
    all_ticks = []
    
    for day in range(n_days):
        date = datetime(2023, 1, 2) + timedelta(days=day)
        open_time = datetime.combine(date.date(), datetime.min.time().replace(hour=9, minute=30))
        close_time = open_time + timedelta(hours=trading_hours)
        
        # Daily market return
        daily_return = np.random.normal(0, daily_vol)
        drift_per_second = daily_return / (trading_hours * 3600)
        
        # Simulate trading day
        current_time = open_time
        current_price = price * np.exp(np.random.normal(0, daily_vol))
        
        while current_time < close_time:
            # Generate random inter-arrival time (exponential distribution)
            seconds_to_next = np.random.exponential(1 / avg_ticks_per_second)
            current_time += timedelta(seconds=seconds_to_next)
            
            if current_time >= close_time:
                break
            
            # Compute price innovation
            time_fraction = seconds_to_next / (trading_hours * 3600)
            expected_return = drift_per_second * seconds_to_next
            vol_scaling = np.sqrt(time_fraction)
            
            # Add intraday U-shape volatility pattern
            tod_seconds = (current_time - open_time).total_seconds()
            rel_time = tod_seconds / (trading_hours * 3600)
            u_shape = 1.0 + 0.5 * (np.exp(-((rel_time - 0) / 0.1)**2) + np.exp(-((rel_time - 1) / 0.1)**2))
            
            # Price increment with microstructure noise
            price_increment = np.random.normal(expected_return, daily_vol * vol_scaling * u_shape)
            microstructure_noise = np.random.normal(0, tick_size * 0.5)
            
            # Update price
            current_price *= np.exp(price_increment)
            observed_price = current_price + microstructure_noise
            
            # Round to tick size
            observed_price = round(observed_price / tick_size) * tick_size
            
            # Add to tick data
            all_ticks.append({
                'timestamp': current_time,
                'price': observed_price,
                'volume': np.random.poisson(100)  # Random trade size
            })
    
    # Convert to DataFrame
    tick_data = pd.DataFrame(all_ticks)
    
    # Display sample
    print("\nSample of tick data:")
    print(tick_data.head())
    
    # Plot tick data for the first day
    day1_data = tick_data[tick_data['timestamp'].dt.date == datetime(2023, 1, 2).date()]
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(day1_data['timestamp'], day1_data['price'], 'b.')
    plt.title('Tick-by-Tick Price Data (Day 1)')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(day1_data['timestamp'], day1_data['volume'], 'g.')
    plt.title('Tick-by-Tick Volume')
    plt.ylabel('Volume')
    plt.grid(True, alpha=0.3)
    
    # Compute and plot tick-by-tick returns
    day1_data['return'] = np.log(day1_data['price']).diff() * 100
    
    plt.subplot(3, 1, 3)
    plt.plot(day1_data['timestamp'][1:], day1_data['return'][1:], 'r.')
    plt.title('Tick-by-Tick Returns (%)')
    plt.xlabel('Time')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compute daily realized measures
    days = sorted(list(set([t.date() for t in tick_data['timestamp']])))
    daily_results = []
    
    for day in days:
        day_data = tick_data[tick_data['timestamp'].dt.date == day]
        
        # Prepare data for realized measures
        times = np.array([t.timestamp() for t in day_data['timestamp']])
        prices = np.array(day_data['price'])
        
        # Compute measures
        rv_5min, rv_ss = realized_variance(
            prices, times, timeType='timestamp', 
            samplingType='CalendarTime', samplingInterval=5
        )
        
        rk = realized_kernel(
            prices, times, timeType='timestamp', kernelType='Parzen'
        )
        
        # Compute as volatility (annualized)
        daily_vol = np.sqrt(252) * np.sqrt(rv_5min)
        daily_vol_ss = np.sqrt(252) * np.sqrt(rv_ss)
        daily_vol_rk = np.sqrt(252) * np.sqrt(rk)
        
        daily_results.append({
            'Date': day,
            'Annualized Volatility (RV)': daily_vol,
            'Annualized Volatility (RV-SS)': daily_vol_ss,
            'Annualized Volatility (RK)': daily_vol_rk,
            'Number of Ticks': len(day_data)
        })
    
    # Display results
    daily_df = pd.DataFrame(daily_results)
    print("\nDaily Realized Measures:")
    print(daily_df)
    
    # Plot daily volatility
    plt.figure(figsize=(12, 6))
    plt.plot(daily_df['Date'], daily_df['Annualized Volatility (RV)'], 'b-o', label='RV (5-min)')
    plt.plot(daily_df['Date'], daily_df['Annualized Volatility (RV-SS)'], 'g-s', label='RV (Subsampled)')
    plt.plot(daily_df['Date'], daily_df['Annualized Volatility (RK)'], 'r-^', label='Realized Kernel')
    plt.title('Daily Realized Volatility Estimates (Annualized)')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Run high-frequency analysis
analyze_real_hf_data()
```

## Bootstrap Analysis Tutorial

This tutorial covers bootstrap methods for statistical inference in financial econometrics.

### Understanding Bootstrap Methods

Bootstrap methods are resampling techniques used to estimate the sampling distribution of a statistic without making strong distributional assumptions. For time series data, special bootstrap methods are needed to preserve temporal dependence:

1. Block Bootstrap: Resamples fixed-length contiguous blocks of data
2. Stationary Bootstrap: Resamples blocks of random length (geometrically distributed)

### Basic Block Bootstrap Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mfe.core import block_bootstrap, stationary_bootstrap

# Generate an AR(1) process for demonstration
np.random.seed(42)
n = 500
ar_coefficient = 0.7
data = np.zeros(n)

for t in range(1, n):
    data[t] = ar_coefficient * data[t-1] + np.random.normal(0, 1)

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Simulated AR(1) Process with φ = 0.7')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()

# Define statistics of interest
def compute_mean(x):
    return np.mean(x)

def compute_variance(x):
    return np.var(x, ddof=1)

def compute_ar1_coefficient(x):
    # Crude AR(1) coefficient estimation
    return np.sum(x[1:] * x[:-1]) / np.sum(x[:-1]**2)

# Perform block bootstrap
n_bootstrap = 1000
block_size = 50  # Fixed block size

# Run bootstrap for multiple statistics
mean_results = block_bootstrap(data, compute_mean, n_bootstrap=n_bootstrap, block_size=block_size)
var_results = block_bootstrap(data, compute_variance, n_bootstrap=n_bootstrap, block_size=block_size)
ar1_results = block_bootstrap(data, compute_ar1_coefficient, n_bootstrap=n_bootstrap, block_size=block_size)

# Compute true statistics from original data
true_mean = compute_mean(data)
true_var = compute_variance(data)
true_ar1 = compute_ar1_coefficient(data)

# Plot bootstrap distributions
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Mean bootstrap distribution
axes[0].hist(mean_results, bins=30, alpha=0.7)
axes[0].axvline(true_mean, color='r', linestyle='--', label=f'Sample mean: {true_mean:.4f}')
axes[0].set_title('Bootstrap Distribution of Mean')
axes[0].set_xlabel('Mean')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Variance bootstrap distribution
axes[1].hist(var_results, bins=30, alpha=0.7)
axes[1].axvline(true_var, color='r', linestyle='--', label=f'Sample variance: {true_var:.4f}')
axes[1].set_title('Bootstrap Distribution of Variance')
axes[1].set_xlabel('Variance')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# AR(1) coefficient bootstrap distribution
axes[2].hist(ar1_results, bins=30, alpha=0.7)
axes[2].axvline(true_ar1, color='r', linestyle='--', label=f'Sample AR(1): {true_ar1:.4f}')
axes[2].axvline(ar_coefficient, color='g', linestyle='-', label=f'True AR(1): {ar_coefficient:.4f}')
axes[2].set_title('Bootstrap Distribution of AR(1) Coefficient')
axes[2].set_xlabel('AR(1) Coefficient')
axes[2].set_ylabel('Frequency')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compute bootstrap confidence intervals
alpha = 0.05  # 5% significance level
mean_ci = np.percentile(mean_results, [alpha/2*100, (1-alpha/2)*100])
var_ci = np.percentile(var_results, [alpha/2*100, (1-alpha/2)*100])
ar1_ci = np.percentile(ar1_results, [alpha/2*100, (1-alpha/2)*100])

print("Bootstrap Confidence Intervals (95%)")
print(f"Mean: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")
print(f"Variance: [{var_ci[0]:.4f}, {var_ci[1]:.4f}]")
print(f"AR(1) Coefficient: [{ar1_ci[0]:.4f}, {ar1_ci[1]:.4f}]")
```

### Comparing Block and Stationary Bootstrap

```python
# Compare block bootstrap and stationary bootstrap
def compare_bootstrap_methods():
    # Define bootstrap parameters
    n_bootstrap = 1000
    block_size = 50  # Fixed block size for block bootstrap
    expected_block_size = 50  # Expected block size for stationary bootstrap
    
    # Run both bootstrap methods for AR(1) coefficient
    block_results = block_bootstrap(
        data, compute_ar1_coefficient, 
        n_bootstrap=n_bootstrap, block_size=block_size
    )
    
    stationary_results = stationary_bootstrap(
        data, compute_ar1_coefficient, 
        n_bootstrap=n_bootstrap, expected_block_size=expected_block_size
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(block_results, bins=30, alpha=0.7)
    plt.axvline(true_ar1, color='r', linestyle='--', label=f'Sample: {true_ar1:.4f}')
    plt.axvline(ar_coefficient, color='g', linestyle='-', label=f'True: {ar_coefficient:.4f}')
    plt.title('Block Bootstrap')
    plt.xlabel('AR(1) Coefficient')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(stationary_results, bins=30, alpha=0.7)
    plt.axvline(true_ar1, color='r', linestyle='--', label=f'Sample: {true_ar1:.4f}')
    plt.axvline(ar_coefficient, color='g', linestyle='-', label=f'True: {ar_coefficient:.4f}')
    plt.title('Stationary Bootstrap')
    plt.xlabel('AR(1) Coefficient')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compute confidence intervals
    block_ci = np.percentile(block_results, [alpha/2*100, (1-alpha/2)*100])
    stationary_ci = np.percentile(stationary_results, [alpha/2*100, (1-alpha/2)*100])
    
    print("\nAR(1) Coefficient Confidence Intervals (95%)")
    print(f"Block Bootstrap: [{block_ci[0]:.4f}, {block_ci[1]:.4f}]")
    print(f"Stationary Bootstrap: [{stationary_ci[0]:.4f}, {stationary_ci[1]:.4f}]")
    
    # Compute standard errors
    block_se = np.std(block_results, ddof=1)
    stationary_se = np.std(stationary_results, ddof=1)
    
    print("\nBootstrap Standard Errors")
    print(f"Block Bootstrap SE: {block_se:.4f}")
    print(f"Stationary Bootstrap SE: {stationary_se:.4f}")

# Run bootstrap comparison
compare_bootstrap_methods()
```

### Block Size Selection

```python
# Study the effect of block size on bootstrap performance
def study_block_size_effect():
    # Try different block sizes
    block_sizes = [10, 25, 50, 100, 200]
    n_bootstrap = 1000
    
    block_results = {}
    stationary_results = {}
    
    for size in block_sizes:
        # Block bootstrap
        block_results[size] = block_bootstrap(
            data, compute_ar1_coefficient, 
            n_bootstrap=n_bootstrap, block_size=size
        )
        
        # Stationary bootstrap
        stationary_results[size] = stationary_bootstrap(
            data, compute_ar1_coefficient, 
            n_bootstrap=n_bootstrap, expected_block_size=size
        )
    
    # Compute statistics
    results = []
    
    for size in block_sizes:
        # Block bootstrap stats
        block_mean = np.mean(block_results[size])
        block_se = np.std(block_results[size], ddof=1)
        block_ci = np.percentile(block_results[size], [alpha/2*100, (1-alpha/2)*100])
        block_ci_width = block_ci[1] - block_ci[0]
        
        # Stationary bootstrap stats
        stat_mean = np.mean(stationary_results[size])
        stat_se = np.std(stationary_results[size], ddof=1)
        stat_ci = np.percentile(stationary_results[size], [alpha/2*100, (1-alpha/2)*100])
        stat_ci_width = stat_ci[1] - stat_ci[0]
        
        # Bias from true value
        block_bias = block_mean - ar_coefficient
        stat_bias = stat_mean - ar_coefficient
        
        results.append({
            'Block Size': size,
            'Block Mean': block_mean,
            'Block SE': block_se,
            'Block CI Width': block_ci_width,
            'Block Bias': block_bias,
            'Stationary Mean': stat_mean,
            'Stationary SE': stat_se,
            'Stationary CI Width': stat_ci_width,
            'Stationary Bias': stat_bias
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print("\nEffect of Block Size on Bootstrap Performance")
    print(results_df.to_string(index=False))
    
    # Plot the results
    plt.figure(figsize=(14, 10))
    
    # Plot standard errors
    plt.subplot(2, 2, 1)
    plt.plot(results_df['Block Size'], results_df['Block SE'], 'bo-', label='Block Bootstrap')
    plt.plot(results_df['Block Size'], results_df['Stationary SE'], 'ro-', label='Stationary Bootstrap')
    plt.title('Standard Error vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('Standard Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot CI widths
    plt.subplot(2, 2, 2)
    plt.plot(results_df['Block Size'], results_df['Block CI Width'], 'bo-', label='Block Bootstrap')
    plt.plot(results_df['Block Size'], results_df['Stationary CI Width'], 'ro-', label='Stationary Bootstrap')
    plt.title('Confidence Interval Width vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('CI Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot bias
    plt.subplot(2, 2, 3)
    plt.plot(results_df['Block Size'], np.abs(results_df['Block Bias']), 'bo-', label='Block Bootstrap')
    plt.plot(results_df['Block Size'], np.abs(results_df['Stationary Bias']), 'ro-', label='Stationary Bootstrap')
    plt.title('Absolute Bias vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('|Bias|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot mean estimates
    plt.subplot(2, 2, 4)
    plt.plot(results_df['Block Size'], results_df['Block Mean'], 'bo-', label='Block Bootstrap')
    plt.plot(results_df['Block Size'], results_df['Stationary Mean'], 'ro-', label='Stationary Bootstrap')
    plt.axhline(ar_coefficient, color='g', linestyle='--', label=f'True Value: {ar_coefficient:.4f}')
    plt.title('Mean Estimate vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('Mean Estimate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Study block size effect
study_block_size_effect()
```

### Application: Bootstrap for Financial Risk Measures

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yahoo_finance
from mfe.core import block_bootstrap, stationary_bootstrap

def bootstrap_risk_measures():
    # Download market data
    print("Downloading market data...")
    tickers = ['SPY', 'AAPL', 'MSFT', 'AMZN']
    
    # Download daily data
    data = yahoo_finance.download(tickers, start='2020-01-01', end='2023-01-01')['Adj Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna() * 100  # Convert to percentage
    
    # Plot returns
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(returns.index, returns[ticker], label=ticker)
    plt.title('Daily Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Create a portfolio with equal weights
    weights = np.ones(len(tickers)) / len(tickers)
    portfolio_returns = returns.dot(weights)
    
    # Define risk measures
    def compute_var(x, alpha=0.05):
        # Value at Risk (VaR)
        return -np.percentile(x, alpha * 100)
    
    def compute_es(x, alpha=0.05):
        # Expected Shortfall (ES)
        var = compute_var(x, alpha)
        return -np.mean(x[x <= -var])
    
    def compute_sharpe(x):
        # Sharpe Ratio (assuming zero risk-free rate)
        return np.mean(x) / np.std(x, ddof=1) * np.sqrt(252)  # Annualized
    
    # Bootstrap parameters
    n_bootstrap = 2000
    block_size = 20  # ~1 month of trading days
    
    # Perform stationary bootstrap for each measure
    print("Running bootstrap for risk measures...")
    var_results = stationary_bootstrap(
        portfolio_returns.values, compute_var, 
        n_bootstrap=n_bootstrap, expected_block_size=block_size
    )
    
    es_results = stationary_bootstrap(
        portfolio_returns.values, compute_es, 
        n_bootstrap=n_bootstrap, expected_block_size=block_size
    )
    
    sharpe_results = stationary_bootstrap(
        portfolio_returns.values, compute_sharpe, 
        n_bootstrap=n_bootstrap, expected_block_size=block_size
    )
    
    # Compute point estimates and confidence intervals
    alpha = 0.05  # 5% significance level
    
    var_point = compute_var(portfolio_returns.values)
    var_ci = np.percentile(var_results, [alpha/2*100, (1-alpha/2)*100])
    
    es_point = compute_es(portfolio_returns.values)
    es_ci = np.percentile(es_results, [alpha/2*100, (1-alpha/2)*100])
    
    sharpe_point = compute_sharpe(portfolio_returns.values)
    sharpe_ci = np.percentile(sharpe_results, [alpha/2*100, (1-alpha/2)*100])
    
    # Print results
    print("\nRisk Measure Analysis")
    print(f"Value at Risk (5%): {var_point:.4f} [95% CI: {var_ci[0]:.4f}, {var_ci[1]:.4f}]")
    print(f"Expected Shortfall (5%): {es_point:.4f} [95% CI: {es_ci[0]:.4f}, {es_ci[1]:.4f}]")
    print(f"Sharpe Ratio (annualized): {sharpe_point:.4f} [95% CI: {sharpe_ci[0]:.4f}, {sharpe_ci[1]:.4f}]")
    
    # Plot the bootstrap distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(var_results, bins=30, alpha=0.7)
    plt.axvline(var_point, color='r', linestyle='--', label=f'Point Est.: {var_point:.4f}')
    plt.axvline(var_ci[0], color='g', linestyle='-', label='95% CI')
    plt.axvline(var_ci[1], color='g', linestyle='-')
    plt.title('VaR Bootstrap Distribution')
    plt.xlabel('Value at Risk (5%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(es_results, bins=30, alpha=0.7)
    plt.axvline(es_point, color='r', linestyle='--', label=f'Point Est.: {es_point:.4f}')
    plt.axvline(es_ci[0], color='g', linestyle='-', label='95% CI')
    plt.axvline(es_ci[1], color='g', linestyle='-')
    plt.title('ES Bootstrap Distribution')
    plt.xlabel('Expected Shortfall (5%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(sharpe_results, bins=30, alpha=0.7)
    plt.axvline(sharpe_point, color='r', linestyle='--', label=f'Point Est.: {sharpe_point:.4f}')
    plt.axvline(sharpe_ci[0], color='g', linestyle='-', label='95% CI')
    plt.axvline(sharpe_ci[1], color='g', linestyle='-')
    plt.title('Sharpe Ratio Bootstrap Distribution')
    plt.xlabel('Sharpe Ratio (annualized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run bootstrap for risk measures
bootstrap_risk_measures()
```

## GUI Interface Tutorial

This tutorial demonstrates how to use the MFE Toolbox's graphical user interface (GUI) for interactive financial econometric analysis.

### Launching the GUI

```python
from mfe.ui import launch_gui

# Launch the main application window
launch_gui()
```

### GUI Workflow Tutorial

The MFE Toolbox GUI provides an interactive interface for model estimation, diagnostic analysis, and results visualization. Here's a step-by-step guide to using the interface:

1. **Launching the GUI**
   - Import and call the `launch_gui()` function
   - The main application window will appear with model configuration options

2. **Loading Data**
   - Click the "Load Data" button to import your time series
   - The GUI supports CSV, Excel, and other standard formats
   - Once loaded, a preview of your data will be displayed

3. **Configuring the Model**
   - Set AR and MA orders using the input fields
   - Toggle "Include Constant" to include a constant term
   - Select exogenous variables if needed from the dropdown

4. **Estimating the Model**
   - Click "Estimate Model" to start the estimation process
   - A progress bar will display during estimation
   - The GUI will display basic results upon completion

5. **Viewing Results**
   - Click "View Results" to open the detailed results viewer
   - The results display includes:
     - Model equation with estimated parameters
     - Parameter estimates table with standard errors
     - Statistical metrics (log-likelihood, AIC, BIC)
     - Diagnostic plots for residual analysis

6. **Navigating Diagnostic Plots**
   - Use the "Previous" and "Next" buttons to navigate between plot pages
   - Different pages show different diagnostic information:
     - Residual time series and distribution
     - ACF and PACF plots
     - Jarque-Bera test results for normality
     - Additional statistical tests

7. **Model Comparison**
   - Estimate multiple models with different specifications
   - Use the comparison feature to select the best model

### GUI Components and Interactions

The GUI consists of these main components:

1. **Main Application Window**
   - Model configuration panel
   - Estimation controls
   - Basic diagnostic display
   - Menu for additional functions

2. **Results Viewer**
   - Parameter estimates
   - Statistical metrics
   - Interactive diagnostic plots
   - Navigation controls

3. **Dialog Windows**
   - About dialog showing version information
   - Confirmation dialogs for important actions
   - Error messages for troubleshooting

### Example: Model Estimation Workflow

1. Launch the GUI
```python
from mfe.ui import launch_gui
launch_gui()
```

2. Follow these steps in the GUI:
   - Set AR Order to 2
   - Set MA Order to 1
   - Check "Include Constant"
   - Click "Load Data" and select your time series file
   - Click "Estimate Model"
   - Once estimation completes, click "View Results"
   - Navigate through diagnostic plots using the "Next" and "Previous" buttons
   - Check parameter significance in the parameter table
   - Review model fit statistics

3. Try different model specifications and compare the results.

### Best Practices for GUI Usage

1. **Start Simple**: Begin with lower-order models before trying more complex specifications
2. **Check Diagnostics**: Always examine residual plots for model adequacy
3. **Compare Models**: Try multiple specifications and compare using information criteria
4. **Use Interactive Features**: The plot viewer supports zooming, panning, and data export
5. **Save Results**: Use the save functionality to preserve your analysis

## Performance Optimization with Numba

This tutorial demonstrates how the MFE Toolbox utilizes Numba for performance optimization of computationally intensive tasks.

### Understanding Numba Optimization

The MFE Toolbox leverages Numba's just-in-time (JIT) compilation to accelerate performance-critical numerical computations. Key components that benefit from Numba optimization include:

1. GARCH likelihood calculations
2. Realized volatility measures computation
3. Bootstrap resampling routines
4. Matrix operations in multivariate models

Here's a simplified example of how Numba is used in the toolbox:

```python
import numpy as np
import time
from numba import jit

# Define a computationally intensive function
def garch_likelihood_standard(returns, omega, alpha, beta):
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Log-likelihood computation
    llh = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * returns**2 / sigma2
    return -np.sum(llh[1:])  # Return negative log-likelihood for minimization

# Numba-optimized version
@jit(nopython=True)
def garch_likelihood_numba(returns, omega, alpha, beta):
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Log-likelihood computation
    llh = np.zeros(n)
    for t in range(n):
        llh[t] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2[t]) - 0.5 * returns[t]**2 / sigma2[t]
    
    return -np.sum(llh[1:])  # Return negative log-likelihood for minimization

# Generate sample data
np.random.seed(42)
n = 10000
returns = np.random.normal(0, 1, n)

# Parameters
omega = 0.1
alpha = 0.1
beta = 0.8

# Benchmark standard implementation
start_time = time.time()
standard_result = garch_likelihood_standard(returns, omega, alpha, beta)
standard_time = time.time() - start_time
print(f"Standard implementation: {standard_time:.6f} seconds")

# First run includes compilation time
start_time = time.time()
numba_result = garch_likelihood_numba(returns, omega, alpha, beta)
first_run_time = time.time() - start_time
print(f"Numba first run (includes compilation): {first_run_time:.6f} seconds")

# Second run shows true performance
start_time = time.time()
numba_result = garch_likelihood_numba(returns, omega, alpha, beta)
second_run_time = time.time() - start_time
print(f"Numba second run: {second_run_time:.6f} seconds")

# Verify results match
print(f"Results match: {np.isclose(standard_result, numba_result)}")
print(f"Speedup factor: {standard_time / second_run_time:.2f}x")
```

### Key Numba Optimization Strategies

The MFE Toolbox employs several strategies to optimize performance with Numba:

1. **Function Specialization**
   - Using `@jit(nopython=True)` for maximum performance
   - Optimizing array operations and memory access patterns
   - Proper type specialization for numerical functions

2. **Numba-Friendly Code Structure**
   - Avoiding Python-specific constructs in performance-critical sections
   - Using NumPy arrays with contiguous memory layouts
   - Structuring loops for efficient compilation

3. **Integration with Python's async/await**
   - Executing Numba-optimized functions in separate threads
   - Non-blocking UI updates during computation
   - Progress reporting from long-running operations

### Example: Optimized Bootstrap Implementation

Let's examine a simplified version of how the MFE Toolbox implements bootstrap with Numba optimization:

```python
import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt

# Standard block bootstrap implementation
def block_bootstrap_standard(data, statistic_func, n_bootstrap=1000, block_size=50):
    n = len(data)
    results = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Generate bootstrap sample
        bootstrap_sample = np.zeros(n)
        for i in range(0, n, block_size):
            # Choose random starting point for block
            start = np.random.randint(0, n - block_size + 1)
            # Copy block
            end = min(i + block_size, n)
            length = end - i
            bootstrap_sample[i:end] = data[start:start+length]
        
        # Compute statistic
        results[b] = statistic_func(bootstrap_sample)
    
    return results

# Numba-optimized block bootstrap
@jit(nopython=True)
def _bootstrap_sample_generator(data, n, block_size):
    bootstrap_sample = np.zeros(n)
    for i in range(0, n, block_size):
        # Choose random starting point for block
        start = np.random.randint(0, n - block_size + 1)
        # Copy block
        end = min(i + block_size, n)
        length = end - i
        bootstrap_sample[i:end] = data[start:start+length]
    return bootstrap_sample

# For simplicity, we'll define statistic functions that are Numba-friendly
@jit(nopython=True)
def compute_mean(x):
    return np.mean(x)

@jit(nopython=True)
def compute_variance(x):
    return np.var(x)

@jit(nopython=True)
def compute_ar1(x):
    return np.sum(x[1:] * x[:-1]) / np.sum(x[:-1]**2)

# Mixed Python/Numba block bootstrap implementation
def block_bootstrap_mixed(data, statistic_func, n_bootstrap=1000, block_size=50):
    n = len(data)
    results = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Generate bootstrap sample using Numba
        bootstrap_sample = _bootstrap_sample_generator(data, n, block_size)
        
        # Compute statistic
        results[b] = statistic_func(bootstrap_sample)
    
    return results

# Generate sample data (AR(1) process)
n = 1000
ar_coef = 0.7
data = np.zeros(n)
for t in range(1, n):
    data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)

# Benchmark parameters
n_bootstrap = 500
block_size = 50

# Benchmark standard implementation
start_time = time.time()
std_results = block_bootstrap_standard(data, compute_mean, n_bootstrap, block_size)
standard_time = time.time() - start_time
print(f"Standard implementation: {standard_time:.6f} seconds")

# Benchmark mixed implementation (includes compilation time)
start_time = time.time()
mixed_results = block_bootstrap_mixed(data, compute_mean, n_bootstrap, block_size)
mixed_time = time.time() - start_time
print(f"Mixed implementation (first run): {mixed_time:.6f} seconds")

# Second run of mixed implementation
start_time = time.time()
mixed_results2 = block_bootstrap_mixed(data, compute_mean, n_bootstrap, block_size)
mixed_time2 = time.time() - start_time
print(f"Mixed implementation (second run): {mixed_time2:.6f} seconds")

# Verify results are similar
print(f"Mean of standard results: {np.mean(std_results):.6f}")
print(f"Mean of mixed results: {np.mean(mixed_results):.6f}")
print(f"Speedup factor: {standard_time / mixed_time2:.2f}x")

# Plot distributions
plt.figure(figsize=(12, 6))
plt.hist(std_results, bins=30, alpha=0.5, label='Standard')
plt.hist(mixed_results, bins=30, alpha=0.5, label='Numba-optimized')
plt.axvline(np.mean(data), color='r', linestyle='--', label='Sample Mean')
plt.title('Bootstrap Distributions: Standard vs. Numba-optimized')
plt.xlabel('Mean')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Balancing Performance and Flexibility

The MFE Toolbox balances performance and flexibility by strategically applying Numba optimization:

1. **Core Computational Kernels**
   - Heavy use of Numba for performance-critical operations
   - Type specialized implementations of statistical calculations
   - Optimized matrix operations for large datasets

2. **High-Level Python Interface**
   - User-friendly Python API for model configuration
   - Flexible parameter handling and validation in Python
   - Comprehensive error checking outside of Numba code

3. **Asynchronous Operations**
   - Non-blocking UI updates during computation
   - Progress reporting from long-running operations
   - Responsive user interface even during heavy calculations

### Best Practices for Working with Numba-Optimized Code

When working with the MFE Toolbox, keep these Numba-related best practices in mind:

1. **Data Preparation**
   - Use NumPy arrays with contiguous memory layout
   - Pre-allocate arrays for better performance
   - Convert data types appropriately before passing to optimized functions

2. **Function Selection**
   - Use the asynchronous API for long-running computations
   - Leverage batch processing for multiple operations
   - Consider memory usage for very large datasets

3. **Error Handling**
   - Check input validity before calling optimized functions
   - Handle exceptions properly as they propagate from Numba code
   - Verify results for numerical stability

## Advanced Topics

### Working with Custom Models

The MFE Toolbox architecture allows for extending the existing models with custom implementations. Here's an example of creating a custom ARMA model with specialized features:

```python
import numpy as np
import asyncio
from typing import Optional, Dict, Any
from mfe.models import ARMAX
from mfe.core.optimization import Optimizer

class CustomARMA(ARMAX):
    """
    Custom ARMA model extending the base ARMAX model with additional features.
    """
    
    def __init__(self, p: int, q: int, include_constant: bool = True, 
                 custom_feature: Optional[str] = None):
        """
        Initialize the custom ARMA model.
        
        Parameters
        ----------
        p : int
            Autoregressive order
        q : int
            Moving average order
        include_constant : bool
            Whether to include a constant term
        custom_feature : Optional[str]
            Optional custom feature configuration
        """
        # Call parent initializer
        super().__init__(p, q, include_constant)
        
        # Add custom attributes
        self.custom_feature = custom_feature
        self._custom_results = {}
    
    async def custom_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform custom analysis on the data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
            
        Returns
        -------
        dict
            Custom analysis results
        """
        # First make sure model is fitted
        if self.params is None:
            # Fit the model if not already fitted
            await self.async_fit(data)
        
        # Now perform custom analysis
        results = {}
        
        # Example custom analysis: compute rolling statistics
        window = 20  # 20-day rolling window
        rolling_mean = np.zeros(len(data) - window + 1)
        rolling_std = np.zeros(len(data) - window + 1)
        
        for i in range(len(rolling_mean)):
            window_data = data[i:i+window]
            rolling_mean[i] = np.mean(window_data)
            rolling_std[i] = np.std(window_data, ddof=1)
        
        # Store results
        results['rolling_mean'] = rolling_mean
        results['rolling_std'] = rolling_std
        
        # If we have residuals, compute additional statistics
        if self.residuals is not None:
            residuals = self.residuals
            results['residual_acf'] = np.correlate(residuals, residuals, mode='full')
            results['residual_acf'] = results['residual_acf'][len(residuals)-1:] / np.var(residuals)
        
        # Save results for later access
        self._custom_results = results
        
        return results
    
    def plot_custom_analysis(self):
        """
        Plot the custom analysis results.
        """
        import matplotlib.pyplot as plt
        
        if not self._custom_results:
            print("No custom analysis results available. Run custom_analysis() first.")
            return
        
        # Plot rolling statistics
        if 'rolling_mean' in self._custom_results and 'rolling_std' in self._custom_results:
            rolling_mean = self._custom_results['rolling_mean']
            rolling_std = self._custom_results['rolling_std']
            
            plt.figure(figsize=(12, 6))
            plt.plot(rolling_mean, label='Rolling Mean')
            plt.plot(rolling_mean + 2*rolling_std, 'r--', label='Mean ± 2×Std')
            plt.plot(rolling_mean - 2*rolling_std, 'r--')
            plt.title('Rolling Statistics (20-day window)')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        # Plot residual ACF if available
        if 'residual_acf' in self._custom_results:
            residual_acf = self._custom_results['residual_acf']
            
            plt.figure(figsize=(12, 6))
            lags = min(50, len(residual_acf))  # Show up to 50 lags
            plt.stem(range(lags), residual_acf[:lags])
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=1.96/np.sqrt(len(self.residuals)), color='r', linestyle='--', alpha=0.7)
            plt.axhline(y=-1.96/np.sqrt(len(self.residuals)), color='r', linestyle='--', alpha=0.7)
            plt.title('Residual Autocorrelation Function')
            plt.xlabel('Lag')
            plt.ylabel('ACF')
            plt.grid(True, alpha=0.3)
            plt.show()

# Example usage
async def custom_model_demo():
    # Generate sample data
    n = 1000
    ar_params = [0.7, -0.2]
    ma_params = [0.5]
    data = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    
    for t in range(2, n):
        data[t] = ar_params[0] * data[t-1] + ar_params[1] * data[t-2] + errors[t] + ma_params[0] * errors[t-1]
    
    # Create and use custom model
    model = CustomARMA(p=2, q=1, include_constant=True, custom_feature='advanced')
    
    # Fit the model
    print("Fitting custom ARMA model...")
    converged = await model.async_fit(data)
    
    if converged:
        print("Model estimation converged successfully!")
        
        # Run standard diagnostics
        diagnostics = model.diagnostic_tests()
        print(f"AIC: {diagnostics['AIC']:.4f}")
        print(f"BIC: {diagnostics['BIC']:.4f}")
        
        # Run custom analysis
        print("\nPerforming custom analysis...")
        custom_results = await model.custom_analysis(data)
        
        # Plot custom analysis results
        model.plot_custom_analysis()
    else:
        print("Model estimation did not converge")

# Run the custom model demo
# asyncio.run(custom_model_demo())
```

### Time Series Forecast Combination

This advanced example demonstrates how to implement forecast combination using multiple models:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from typing import List, Dict, Any
from mfe.models import ARMAX, GARCH

class ForecastCombiner:
    """
    Combines forecasts from multiple models using various weighting schemes.
    """
    
    def __init__(self):
        """Initialize the forecast combiner."""
        self.models = []
        self.model_names = []
        self.weights = None
        self.forecasts = {}
        self.combined_forecast = None
    
    def add_model(self, model, name: str):
        """
        Add a model to the combiner.
        
        Parameters
        ----------
        model : object
            Model instance with a forecast method
        name : str
            Name identifier for the model
        """
        self.models.append(model)
        self.model_names.append(name)
    
    async def fit_models(self, data: np.ndarray, exog: np.ndarray = None):
        """
        Fit all models asynchronously.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        exog : np.ndarray, optional
            Exogenous variables
        """
        for i, model in enumerate(self.models):
            name = self.model_names[i]
            print(f"Fitting model: {name}")
            
            # Check if model has async_fit method
            if hasattr(model, 'async_fit'):
                if exog is not None and hasattr(model, 'supports_exog') and model.supports_exog:
                    await model.async_fit(data, exog)
                else:
                    await model.async_fit(data)
            else:
                # Fallback for models without async_fit
                if exog is not None and hasattr(model, 'supports_exog') and model.supports_exog:
                    model.fit(data, exog)
                else:
                    model.fit(data)
    
    def generate_individual_forecasts(self, steps: int, exog_future: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Generate forecasts from each model.
        
        Parameters
        ----------
        steps : int
            Forecast horizon
        exog_future : np.ndarray, optional
            Future exogenous variables
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of forecasts from each model
        """
        self.forecasts = {}
        
        for i, model in enumerate(self.models):
            name = self.model_names[i]
            print(f"Generating forecast from model: {name}")
            
            # Check if model supports exogenous variables
            if exog_future is not None and hasattr(model, 'supports_exog') and model.supports_exog:
                forecast = model.forecast(steps=steps, exog_future=exog_future)
            else:
                forecast = model.forecast(steps=steps)
            
            self.forecasts[name] = forecast
        
        return self.forecasts
    
    def combine_forecasts(self, method: str = 'equal', **kwargs) -> np.ndarray:
        """
        Combine forecasts using specified method.
        
        Parameters
        ----------
        method : str
            Weighting method ('equal', 'performance', 'inverse_error', 'optimal')
        **kwargs
            Additional method-specific parameters
            
        Returns
        -------
        np.ndarray
            Combined forecast
        """
        if not self.forecasts:
            raise ValueError("No forecasts available. Run generate_individual_forecasts first.")
        
        # Get forecast arrays
        forecast_arrays = list(self.forecasts.values())
        
        # Check all forecasts have same length
        forecast_length = len(forecast_arrays[0])
        if not all(len(f) == forecast_length for f in forecast_arrays):
            raise ValueError("All forecasts must have the same length")
        
        # Compute weights based on method
        n_models = len(self.models)
        
        if method == 'equal':
            # Equal weights
            self.weights = np.ones(n_models) / n_models
            
        elif method == 'performance':
            # Weights based on in-sample performance (e.g., AIC)
            if 'criterion' not in kwargs:
                raise ValueError("Must specify 'criterion' for performance weighting")
                
            criterion = kwargs['criterion']
            criterion_values = []
            
            for model in self.models:
                if not hasattr(model, 'diagnostic_tests'):
                    raise ValueError("All models must have diagnostic_tests method")
                
                diagnostics = model.diagnostic_tests()
                if criterion not in diagnostics:
                    raise ValueError(f"Criterion '{criterion}' not found in model diagnostics")
                
                criterion_values.append(diagnostics[criterion])
            
            # For criteria like AIC/BIC, lower is better
            criterion_values = np.array(criterion_values)
            
            # Invert and normalize
            weights = 1.0 / criterion_values
            self.weights = weights / np.sum(weights)
            
        elif method == 'inverse_error':
            # Weights based on inverse in-sample error
            errors = []
            
            for model in self.models:
                if not hasattr(model, 'residuals'):
                    raise ValueError("All models must have residuals attribute")
                
                # Mean squared error
                mse = np.mean(model.residuals**2)
                errors.append(mse)
            
            # Convert to inverse (lower error = higher weight)
            errors = np.array(errors)
            weights = 1.0 / errors
            self.weights = weights / np.sum(weights)
            
        elif method == 'optimal':
            # Optimal weights based on forecast error covariance
            # This requires historical forecast errors
            if 'error_cov' not in kwargs:
                raise ValueError("Must provide 'error_cov' for optimal weighting")
                
            error_cov = kwargs['error_cov']
            ones = np.ones(n_models)
            
            # Optimal weights formula
            # w = (Σ^-1 * 1) / (1' * Σ^-1 * 1)
            inv_cov = np.linalg.inv(error_cov)
            self.weights = inv_cov.dot(ones) / (ones.dot(inv_cov).dot(ones))
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure weights sum to 1
        self.weights = self.weights / np.sum(self.weights)
        
        # Combine forecasts
        combined = np.zeros(forecast_length)
        for i, forecast in enumerate(forecast_arrays):
            combined += self.weights[i] * forecast
        
        self.combined_forecast = combined
        return combined
    
    def plot_forecasts(self, data: np.ndarray = None, dates=None):
        """
        Plot individual and combined forecasts.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Historical data to include in plot
        dates : array-like, optional
            Date labels for x-axis
        """
        if not self.forecasts or self.combined_forecast is None:
            raise ValueError("No forecasts available. Run combine_forecasts first.")
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical data if provided
        if data is not None:
            if dates is not None and len(dates) >= len(data):
                plt.plot(dates[:len(data)], data, 'k-', label='Historical')
            else:
                plt.plot(data, 'k-', label='Historical')
        
        # Plot forecasting period
        forecast_length = len(self.combined_forecast)
        
        # Create x-axis for forecasts
        if data is not None:
            forecast_start = len(data)
            forecast_end = forecast_start + forecast_length
            forecast_range = range(forecast_start, forecast_end)
            
            if dates is not None and len(dates) >= forecast_end:
                forecast_dates = dates[forecast_start:forecast_end]
            else:
                forecast_dates = forecast_range
        else:
            forecast_range = range(forecast_length)
            forecast_dates = forecast_range
        
        # Plot individual forecasts
        for name, forecast in self.forecasts.items():
            plt.plot(forecast_dates, forecast, '--', alpha=0.5, label=f'{name}')
        
        # Plot combined forecast
        plt.plot(forecast_dates, self.combined_forecast, 'r-', linewidth=2, label='Combined')
        
        # Add legend and labels
        plt.title('Forecast Comparison')
        plt.xlabel('Time' if dates is None else 'Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Display weight information
        weight_text = "Model Weights:\n"
        for i, name in enumerate(self.model_names):
            weight_text += f"{name}: {self.weights[i]:.4f}\n"
        
        plt.figtext(0.02, 0.02, weight_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

# Example usage
async def forecast_combination_demo():
    # Generate sample data
    n = 500
    np.random.seed(42)
    
    # AR(1) process with structural break
    data = np.zeros(n)
    for t in range(1, n):
        if t < n//2:
            data[t] = 0.8 * data[t-1] + np.random.normal(0, 1)
        else:
            data[t] = 0.4 * data[t-1] + np.random.normal(0, 1.5)
    
    # Split into training and test
    train_size = 450
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create models with different specifications
    model1 = ARMAX(p=1, q=0, include_constant=True)  # AR(1)
    model2 = ARMAX(p=2, q=0, include_constant=True)  # AR(2)
    model3 = ARMAX(p=1, q=1, include_constant=True)  # ARMA(1,1)
    
    # Create forecast combiner
    combiner = ForecastCombiner()
    combiner.add_model(model1, 'AR(1)')
    combiner.add_model(model2, 'AR(2)')
    combiner.add_model(model3, 'ARMA(1,1)')
    
    # Fit all models
    print("Fitting models...")
    await combiner.fit_models(train_data)
    
    # Generate individual forecasts
    forecast_horizon = len(test_data)
    print(f"\nGenerating {forecast_horizon}-step ahead forecasts...")
    individual_forecasts = combiner.generate_individual_forecasts(steps=forecast_horizon)
    
    # Try different combination methods
    print("\nCombining forecasts with different methods...")
    
    # Equal weights
    print("\n1. Equal weights:")
    equal_weights_forecast = combiner.combine_forecasts(method='equal')
    combiner.plot_forecasts(data=train_data)
    
    # Performance-based weights
    print("\n2. Performance-based weights (AIC):")
    perf_weights_forecast = combiner.combine_forecasts(method='performance', criterion='AIC')
    combiner.plot_forecasts(data=train_data)
    
    # Inverse error weights
    print("\n3. Inverse error weights:")
    inv_error_forecast = combiner.combine_forecasts(method='inverse_error')
    combiner.plot_forecasts(data=train_data)
    
    # Evaluate forecast accuracy
    def compute_rmse(forecast, actual):
        return np.sqrt(np.mean((forecast - actual)**2))
    
    print("\nForecast Evaluation (RMSE):")
    for name, forecast in individual_forecasts.items():
        rmse = compute_rmse(forecast, test_data)
        print(f"{name}: {rmse:.4f}")
    
    # Evaluate combined forecasts
    equal_rmse = compute_rmse(equal_weights_forecast, test_data)
    perf_rmse = compute_rmse(perf_weights_forecast, test_data)
    inv_rmse = compute_rmse(inv_error_forecast, test_data)
    
    print(f"Combined (Equal): {equal_rmse:.4f}")
    print(f"Combined (AIC): {perf_rmse:.4f}")
    print(f"Combined (Inverse Error): {inv_rmse:.4f}")
    
    # Plot final comparison with actual test data
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(range(len(train_data)), train_data, 'k-', label='Training Data')
    
    # Plot test data
    plt.plot(range(len(train_data), len(data)), test_data, 'k-', label='Test Data')
    
    # Plot combined forecasts
    forecast_range = range(len(train_data), len(data))
    plt.plot(forecast_range, equal_weights_forecast, 'b--', label=f'Equal Weights (RMSE: {equal_rmse:.4f})')
    plt.plot(forecast_range, perf_weights_forecast, 'g--', label=f'AIC Weights (RMSE: {perf_rmse:.4f})')
    plt.plot(forecast_range, inv_error_forecast, 'r--', label=f'Inverse Error Weights (RMSE: {inv_rmse:.4f})')
    
    plt.title('Forecast Combination Evaluation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Run the forecast combination demo
# asyncio.run(forecast_combination_demo())
```

## Additional Resources

### Official Documentation

For more detailed information, refer to the complete MFE Toolbox documentation:

- Online Documentation: [MFE Toolbox Documentation](https://mfe-toolbox.readthedocs.io/)
- API Reference: [API Documentation](https://mfe-toolbox.readthedocs.io/en/latest/api.html)
- Example Gallery: [Example Gallery](https://mfe-toolbox.readthedocs.io/en/latest/examples/index.html)

### Academic References

The MFE Toolbox implements methods from several key academic papers:

1. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
2. Engle, R. F., & Sheppard, K. (2001). Theoretical and empirical properties of dynamic conditional correlation multivariate GARCH (No. w8554). National Bureau of Economic Research.
3. Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.
4. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association, 89(428), 1303-1313.
5. Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. Econometrica, 71(2), 579-625.

### Useful Links

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [Numba Documentation](https://numba.pydata.org/numba-doc/latest/index.html)
- [PyQt6 Documentation](https://doc.qt.io/qtforpython-6/)