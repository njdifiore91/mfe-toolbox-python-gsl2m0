"""
MFE Toolbox - Usage Examples
===========================

This file provides comprehensive usage examples for the MFE (Financial Econometrics) Toolbox,
a Python-based suite of tools for financial time series modeling and econometric analysis.

The MFE Toolbox leverages modern Python 3.12 features including async/await patterns and
strict type hints, built upon NumPy, SciPy, Pandas, Statsmodels, and Numba for performance
optimization.

Sections:
- Getting Started (Installation and Data Preparation)
- Time Series Modeling (ARMAX Examples) 
- Volatility Analysis (GARCH Examples)
- Advanced Features (Performance Optimization and Error Handling)
"""

#----------------------------------------
# Getting Started
#----------------------------------------

"""
## Installation

Install the MFE Toolbox using pip:
```
pip install mfe
```

Required dependencies:
- Python 3.12+
- NumPy 1.26.3+
- SciPy 1.11.4+
- Pandas 2.1.4+
- Statsmodels 0.14.1+
- Numba 0.59.0+
- PyQt6 6.6.1+ (optional, for GUI)
"""

# Basic imports
import numpy as np
import pandas as pd

# MFE Toolbox imports
from mfe.models import ARMAX, GARCHModel
import mfe.core.bootstrap as bootstrap
import mfe.core.distributions as distributions
import mfe.utils.validation as validation

# For performance-critical functions
import numba

"""
## Data Preparation

Before using the MFE toolbox, you need to prepare your financial time series data.
The toolbox is designed to work with NumPy arrays and Pandas Series/DataFrames.
"""

# Example: Generate synthetic return data
def prepare_sample_data() -> pd.Series:
    """Generate sample financial return data for examples."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000)
    returns = pd.Series(np.random.normal(0, 1, 1000), index=dates)
    return returns

# Example: Data validation function
def validate_returns(returns: pd.Series) -> pd.Series:
    """
    Validate financial return data before using with MFE Toolbox models.
    
    Parameters:
        returns: A pandas Series containing financial returns
        
    Returns:
        Validated returns series
        
    Raises:
        ValueError: If the data does not meet validation criteria
    """
    # Check for NaN values
    if returns.isna().any():
        raise ValueError("Return series contains NaN values")
    
    # Check for adequate length
    if len(returns) < 10:
        raise ValueError("Return series too short for meaningful analysis")
        
    # Additional validation can be performed here
    
    return returns

# Using the validation function
sample_returns = prepare_sample_data()
validated_returns = validate_returns(sample_returns)
print(f"Prepared {len(validated_returns)} valid return observations")

#----------------------------------------
# Time Series Modeling (ARMAX Examples)
#----------------------------------------

"""
## ARMAX Modeling

The ARMAX (AutoRegressive Moving Average with eXogenous variables) model 
combines AR and MA components with optional exogenous regressors.
"""

async def armax_example():
    """Example demonstrating ARMAX model fitting and forecasting."""
    # Prepare data
    returns = prepare_sample_data()
    
    # Initialize ARMAX model
    model = ARMAX(
        ar_order=1,       # AR order (p)
        ma_order=1,       # MA order (q)
        include_constant=True  # Include a constant term
    )
    
    # Set estimation options
    model.set_options(
        optimizer='BFGS',
        max_iterations=1000,
        tolerance=1e-08
    )
    
    # Asynchronous parameter estimation
    print("Estimating ARMAX model parameters...")
    results = await model.estimate_async(returns)
    
    # Display results
    print(f"ARMAX(1,1) Model Results:")
    print(f"Log-likelihood: {results.log_likelihood:.4f}")
    print(f"AIC: {results.aic:.4f}")
    print(f"BIC: {results.bic:.4f}")
    
    # Parameter estimates
    print("\nParameter Estimates:")
    for name, value, std_err, t_stat, p_val in zip(
        results.param_names,
        results.params,
        results.std_errors,
        results.t_stats,
        results.p_values
    ):
        print(f"{name}: {value:.4f} (SE: {std_err:.4f}, t: {t_stat:.4f}, p: {p_val:.4f})")
    
    # Forecasting
    print("\nForecasting next 5 periods:")
    forecasts = await model.forecast_async(returns, h=5)
    for i, forecast in enumerate(forecasts):
        print(f"t+{i+1}: {forecast:.4f}")
    
    # Diagnostic testing
    print("\nDiagnostic Tests:")
    diagnostics = model.diagnostic_tests(returns)
    print(f"Ljung-Box Q(10): {diagnostics['ljung_box_q'][0]:.4f} (p-value: {diagnostics['ljung_box_q'][1]:.4f})")
    print(f"Jarque-Bera: {diagnostics['jarque_bera'][0]:.4f} (p-value: {diagnostics['jarque_bera'][1]:.4f})")
    
    return results

# To run the ARMAX example, use:
# import asyncio
# asyncio.run(armax_example())

#----------------------------------------
# Volatility Analysis (GARCH Examples)
#----------------------------------------

"""
## GARCH Modeling

The GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) model
is used for modeling and forecasting volatility in financial time series.
"""

async def garch_example():
    """Example demonstrating GARCH model fitting and volatility forecasting."""
    # Prepare data
    returns = prepare_sample_data()
    
    # Initialize GARCH model with Numba optimization
    model = GARCHModel(
        p=1,  # GARCH order
        q=1,  # ARCH order
        distribution='student-t'  # Error distribution
    )
    
    # Configure Numba optimization for critical functions
    model.set_numba_config(
        nopython=True,  # Use nopython mode for maximum performance
        parallel=True,  # Enable parallel execution
        cache=True      # Cache the compiled function
    )
    
    # Asynchronous parameter estimation
    print("Estimating GARCH model parameters...")
    results = await model.estimate_async(returns)
    
    # Display results
    print(f"GARCH(1,1) Model Results:")
    print(f"Log-likelihood: {results.log_likelihood:.4f}")
    print(f"AIC: {results.aic:.4f}")
    print(f"BIC: {results.bic:.4f}")
    
    # Parameter estimates
    print("\nParameter Estimates:")
    for name, value, std_err, t_stat, p_val in zip(
        results.param_names,
        results.params,
        results.std_errors,
        results.t_stats,
        results.p_values
    ):
        print(f"{name}: {value:.4f} (SE: {std_err:.4f}, t: {t_stat:.4f}, p: {p_val:.4f})")
    
    # Volatility forecasting
    print("\nForecasting volatility for next 5 periods:")
    vol_forecasts = await model.forecast_volatility_async(returns, h=5)
    for i, forecast in enumerate(vol_forecasts):
        print(f"t+{i+1}: {forecast:.6f}")
    
    # Monte Carlo simulation
    print("\nPerforming Monte Carlo simulation (1000 paths, 5 steps ahead):")
    simulations = await model.simulate_async(returns, n_paths=1000, horizon=5)
    
    # Calculate VaR from simulations
    alpha = 0.05  # 95% confidence level
    var_estimates = np.percentile(simulations, alpha * 100, axis=0)
    print(f"Value-at-Risk (95%) estimates for next 5 periods:")
    for i, var in enumerate(var_estimates):
        print(f"t+{i+1}: {var:.6f}")
    
    return results

# To run the GARCH example, use:
# import asyncio
# asyncio.run(garch_example())

#----------------------------------------
# Advanced Features
#----------------------------------------

"""
## Performance Optimization

The MFE Toolbox uses Numba's just-in-time (JIT) compilation for performance-critical
functions, providing near-C performance while maintaining Python syntax.
"""

# Example: Numba-optimized function for rolling volatility calculation
@numba.jit(nopython=True, parallel=True, cache=True)
def calculate_rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation) of returns.
    
    Uses Numba's JIT compilation for efficient computation.
    
    Parameters:
        returns: Array of return data
        window: Rolling window size
        
    Returns:
        Array of rolling volatility estimates
    """
    n = len(returns)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        window_slice = returns[i-window+1:i+1]
        result[i] = np.std(window_slice)
    
    return result

# Example: Using the Numba-optimized function
def rolling_volatility_example():
    """Demonstrate the Numba-optimized rolling volatility function."""
    returns = prepare_sample_data().values
    window = 20
    
    # Calculate rolling volatility
    vol = calculate_rolling_volatility(returns, window)
    
    print(f"Calculated {len(vol) - window + 1} rolling volatility estimates")
    print(f"Last 5 volatility estimates:")
    for i in range(-5, 0):
        print(f"Period {len(vol) + i}: {vol[i]:.6f}")

# Asynchronous pattern for processing multiple models concurrently
async def process_multiple_models(returns: pd.Series):
    """
    Process multiple models concurrently using Python's async/await pattern.
    
    This demonstrates how to efficiently run multiple time-consuming model
    estimations concurrently.
    
    Parameters:
        returns: Return series to analyze
    """
    import asyncio
    
    # Define model estimation coroutines
    async def estimate_armax():
        model = ARMAX(ar_order=1, ma_order=1, include_constant=True)
        return await model.estimate_async(returns)
    
    async def estimate_garch():
        model = GARCHModel(p=1, q=1, distribution='student-t')
        return await model.estimate_async(returns)
    
    # Run model estimations concurrently
    print("Estimating multiple models concurrently...")
    armax_task = asyncio.create_task(estimate_armax())
    garch_task = asyncio.create_task(estimate_garch())
    
    # Gather results
    armax_results, garch_results = await asyncio.gather(armax_task, garch_task)
    
    # Compare model information criteria
    print("\nModel Comparison (Information Criteria):")
    print(f"ARMAX(1,1): AIC = {armax_results.aic:.4f}, BIC = {armax_results.bic:.4f}")
    print(f"GARCH(1,1): AIC = {garch_results.aic:.4f}, BIC = {garch_results.bic:.4f}")
    
    return armax_results, garch_results

"""
## Error Handling

The MFE Toolbox implements comprehensive error handling and input validation
to ensure robust operation and clear error messages.
"""

# Example: Input validation for model parameters
def validate_model_inputs(data: np.ndarray, order: tuple) -> bool:
    """
    Validate inputs for time series models.
    
    Parameters:
        data: Array of time series data
        order: Model order specification (p, q)
        
    Returns:
        True if inputs are valid
        
    Raises:
        TypeError: If input types are incorrect
        ValueError: If input values are invalid
    """
    # Data type validation
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("Data must be a NumPy array or Pandas Series")
        
    # Convert pandas Series to numpy array if necessary
    if isinstance(data, pd.Series):
        data = data.values
    
    # Order validation
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError("Order must be a tuple of (p, q)")
    
    p, q = order
    if not isinstance(p, int) or not isinstance(q, int):
        raise TypeError("Order components p and q must be integers")
        
    if p < 0 or q < 0:
        raise ValueError("Order components p and q must be non-negative")
    
    # Data length validation
    min_length = max(p, q) + 10
    if len(data) < min_length:
        raise ValueError(f"Data length must be at least {min_length} for order {order}")
    
    # Check for NaN and inf values
    if not np.isfinite(data).all():
        raise ValueError("Data contains NaN or infinite values")
    
    return True

# Example: Function to check numerical stability
def check_numerical_stability(tolerance: float = 1e-8) -> None:
    """
    Check numerical stability of computations.
    
    Parameters:
        tolerance: Tolerance for checking near-zero values
        
    Raises:
        ValueError: If numerical instability is detected
    """
    # Generate some data that might cause numerical issues
    np.random.seed(42)
    values = np.random.normal(0, 1e-9, 1000)
    
    # Check for NaN values
    if np.isnan(values).any():
        raise ValueError("NaN values detected, indicating numerical instability")
    
    # Check for infinite values
    if np.isinf(values).any():
        raise ValueError("Infinite values detected, indicating numerical instability")
    
    # Check for values that are too close to zero for numerical stability
    near_zero_mask = np.abs(values) < tolerance
    if near_zero_mask.any() and np.abs(values[near_zero_mask]).max() > 0:
        print(f"Warning: Near-zero values detected (< {tolerance}), potential numerical instability")

# Example of using error handling functions
def error_handling_example():
    """Demonstrate error handling and validation functions."""
    try:
        # Generate valid data
        returns = prepare_sample_data().values
        
        # Validate model inputs
        validate_model_inputs(returns, (1, 1))
        print("Model inputs are valid")
        
        # Check numerical stability
        check_numerical_stability()
        print("Numerical stability check completed")
        
        # Intentionally create invalid data
        invalid_data = np.array([1.0, 2.0, np.nan, 4.0])
        
        # This should raise an error
        validate_model_inputs(invalid_data, (1, 1))
        
    except (TypeError, ValueError) as e:
        print(f"Error caught: {str(e)}")
        
    # Demonstrate proper error handling in model estimation
    try:
        # Create invalid order
        invalid_order = (-1, 1)
        
        # This should raise an error
        validate_model_inputs(returns, invalid_order)
        
    except ValueError as e:
        print(f"Order validation error caught: {str(e)}")

if __name__ == "__main__":
    print("MFE Toolbox Usage Examples")
    print("=========================")
    print("\nRun the individual examples to see the MFE Toolbox in action.")
    print("For example:")
    print("  import asyncio")
    print("  asyncio.run(armax_example())")
    
    # Uncomment to run the data preparation example
    # sample_returns = prepare_sample_data()
    # validated_returns = validate_returns(sample_returns)
    # print(f"Prepared {len(validated_returns)} valid return observations")
    
    # Uncomment to run the rolling volatility example
    # rolling_volatility_example()
    
    # Uncomment to run the error handling example
    # error_handling_example()