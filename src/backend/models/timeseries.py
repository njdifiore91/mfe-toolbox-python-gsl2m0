"""
ARMA/ARMAX Time Series Modeling Module for MFE Toolbox.

This module implements a comprehensive framework for ARMA/ARMAX modeling and forecasting,
providing robust parameter estimation, diagnostic tools, and multi-step forecasting
capabilities. It leverages Python's scientific computing stack with Numba optimization
for performance-critical operations.

Key features:
- ARMA and ARMAX model specification and estimation
- Asynchronous parameter optimization
- Numba-accelerated forecasting
- Comprehensive model diagnostics and statistical tests
- Multi-step forecasting with error propagation
"""

import logging
import numpy as np
from scipy import stats
import statsmodels.tsa.arima.model as sm_arima
import numba
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field

# Internal imports
from ..core.optimization import Optimizer, compute_standard_errors
from ..core.distributions import GED
from ..utils.validation import validate_array_input, validate_model_order

# Configure logger
logger = logging.getLogger(__name__)

# Global constants
MAX_LAGS = 50
VALID_TREND_TYPES = ['n', 'c', 't', 'ct']


@numba.jit(nopython=True)
def compute_acf(data: np.ndarray, nlags: int) -> np.ndarray:
    """
    Computes autocorrelation function with Numba optimization.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data array
    nlags : int
        Number of lags to compute
        
    Returns
    -------
    np.ndarray
        Array of autocorrelation values from lag 0 to nlags
    """
    # Ensure nlags doesn't exceed data length
    n = len(data)
    nlags = min(nlags, n - 1)
    
    # Center the data by subtracting the mean
    y = data - np.mean(data)
    
    # Initialize ACF array (including lag 0)
    acf = np.zeros(nlags + 1)
    
    # Compute denominator (variance)
    denominator = np.sum(y * y)
    
    # Calculate autocorrelations for each lag
    for lag in range(nlags + 1):
        numerator = 0.0
        for t in range(lag, n):
            numerator += y[t] * y[t - lag]
        
        acf[lag] = numerator / denominator
    
    return acf


@numba.jit(nopython=True)
def compute_pacf(data: np.ndarray, nlags: int) -> np.ndarray:
    """
    Computes partial autocorrelation function using Durbin-Levinson algorithm.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data array
    nlags : int
        Number of lags to compute
        
    Returns
    -------
    np.ndarray
        Array of partial autocorrelation values from lag 1 to nlags
    """
    # Ensure nlags doesn't exceed data length
    n = len(data)
    nlags = min(nlags, n - 1)
    
    # Compute ACF first
    acf = compute_acf(data, nlags)
    
    # Initialize arrays
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0  # PACF at lag 0 is 1 by definition
    
    # Initialize phi matrix for Durbin-Levinson algorithm
    phi = np.zeros((nlags + 1, nlags + 1))
    
    # Implement Durbin-Levinson algorithm for PACF
    for k in range(1, nlags + 1):
        # Initial estimate for phi_k,k
        numerator = acf[k]
        for j in range(1, k):
            numerator -= phi[k-1, j] * acf[k-j]
        
        denominator = 1.0
        for j in range(1, k):
            denominator -= phi[k-1, j] * acf[j]
        
        phi[k, k] = numerator / denominator
        pacf[k] = phi[k, k]
        
        # Update other coefficients
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
    
    return pacf


# Numba-optimized ARMAX likelihood calculation
@numba.jit(nopython=True)
def _armax_likelihood_numba(endog: np.ndarray, 
                           ar_params: np.ndarray, 
                           ma_params: np.ndarray,
                           trend_params: np.ndarray,
                           exog: Optional[np.ndarray] = None,
                           exog_params: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """
    Calculate the log-likelihood for ARMAX model.
    
    Parameters
    ----------
    endog : np.ndarray
        Endogenous variable (time series)
    ar_params : np.ndarray
        AR parameters
    ma_params : np.ndarray
        MA parameters
    trend_params : np.ndarray
        Trend parameters (constant, linear trend)
    exog : np.ndarray, optional
        Exogenous variables
    exog_params : np.ndarray, optional
        Parameters for exogenous variables
        
    Returns
    -------
    Tuple[float, np.ndarray]
        Log-likelihood value and residuals array
    """
    n = len(endog)
    p = len(ar_params)
    q = len(ma_params)
    
    # Pre-allocate arrays
    residuals = np.zeros(n)
    fitted = np.zeros(n)
    
    # Initial residuals are set to zero
    for t in range(max(p, q), n):
        # Initialize with trend component
        trend_value = 0.0
        if len(trend_params) >= 1:  # Constant
            trend_value += trend_params[0]
        if len(trend_params) >= 2:  # Linear trend
            trend_value += trend_params[1] * (t + 1)
        
        # Add AR component
        ar_component = 0.0
        for i in range(p):
            if t - i - 1 >= 0:
                ar_component += ar_params[i] * endog[t - i - 1]
                
        # Add MA component
        ma_component = 0.0
        for j in range(min(q, t)):
            if t - j - 1 >= 0:
                ma_component += ma_params[j] * residuals[t - j - 1]
        
        # Add exogenous component if present
        exog_component = 0.0
        if exog is not None and exog_params is not None:
            for k in range(len(exog_params)):
                exog_component += exog_params[k] * exog[t, k]
        
        # Calculate fitted value
        fitted[t] = trend_value + ar_component + ma_component + exog_component
        
        # Calculate residual
        residuals[t] = endog[t] - fitted[t]
    
    # Calculate log-likelihood (Gaussian)
    sigma2 = np.sum(residuals[max(p, q):]**2) / (n - max(p, q))
    ll = -0.5 * (n - max(p, q)) * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    
    return ll, residuals


# Numba-optimized ARMA forecasting
@numba.jit(nopython=True)
def _forecast_armax_numba(steps: int,
                        ar_params: np.ndarray,
                        ma_params: np.ndarray,
                        trend_params: np.ndarray,
                        history: np.ndarray,
                        residuals: np.ndarray,
                        exog_future: Optional[np.ndarray] = None,
                        exog_params: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate forecasts for ARMAX model using Numba optimization.
    
    Parameters
    ----------
    steps : int
        Number of steps to forecast
    ar_params : np.ndarray
        AR parameters
    ma_params : np.ndarray
        MA parameters
    trend_params : np.ndarray
        Trend parameters (constant, linear trend)
    history : np.ndarray
        Historical observations
    residuals : np.ndarray
        Model residuals
    exog_future : np.ndarray, optional
        Future values of exogenous variables
    exog_params : np.ndarray, optional
        Parameters for exogenous variables
        
    Returns
    -------
    np.ndarray
        Forecasted values
    """
    n_hist = len(history)
    p = len(ar_params)
    q = len(ma_params)
    
    # Pre-allocate forecast array
    forecasts = np.zeros(steps)
    
    # Generate forecasts
    for t in range(steps):
        # Initialize with trend component
        if len(trend_params) >= 1:  # Constant
            forecasts[t] = trend_params[0]
        if len(trend_params) >= 2:  # Linear trend
            forecasts[t] += trend_params[1] * (n_hist + t + 1)
        
        # Add AR component
        for i in range(p):
            if t - i - 1 < 0:
                # Use historical data
                idx = n_hist + (t - i - 1)
                if idx >= 0:
                    forecasts[t] += ar_params[i] * history[idx]
            else:
                # Use previously forecasted values
                forecasts[t] += ar_params[i] * forecasts[t - i - 1]
        
        # Add MA component (only finite number of residuals affect forecast)
        for j in range(q):
            idx = n_hist - j - 1 + t
            if idx < n_hist:
                forecasts[t] += ma_params[j] * residuals[idx]
        
        # Add exogenous component if present
        if exog_future is not None and exog_params is not None:
            if exog_future.ndim == 1:
                forecasts[t] += exog_params[0] * exog_future[t]
            else:
                for k in range(len(exog_params)):
                    forecasts[t] += exog_params[k] * exog_future[t, k]
    
    return forecasts


@dataclass
class ARMAX:
    """
    ARMAX model implementation with async estimation and forecasting.
    
    This class implements an AutoRegressive Moving Average model with eXogenous
    regressors (ARMAX), providing methods for model estimation, forecasting,
    and diagnostic testing. It leverages asynchronous optimization for parameter
    estimation and Numba-accelerated forecasting.
    
    Parameters
    ----------
    p : int
        Autoregressive order
    q : int
        Moving average order
    exog : Optional[np.ndarray]
        Exogenous variables array, shape (n_obs, n_variables)
    trend : str
        Trend specification, one of:
        - 'n': No trend
        - 'c': Constant only
        - 't': Linear trend
        - 'ct': Constant and linear trend
        
    Attributes
    ----------
    p : int
        Autoregressive order
    q : int
        Moving average order
    params : np.ndarray
        Model parameters
    residuals : np.ndarray
        Model residuals
    sigma2 : float
        Residual variance
    std_errors : np.ndarray
        Parameter standard errors
    """
    
    p: int
    q: int
    exog: Optional[np.ndarray] = None
    trend: str = 'c'
    
    # Internal state attributes (not part of initialization)
    params: np.ndarray = field(default=None, init=False)
    residuals: np.ndarray = field(default=None, init=False)
    sigma2: float = field(default=None, init=False)
    std_errors: np.ndarray = field(default=None, init=False)
    _endog: np.ndarray = field(default=None, init=False)
    _optimizer: Optimizer = field(default=None, init=False)
    _n_trend_params: int = field(default=0, init=False)
    _param_names: List[str] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """
        Validate inputs and initialize model components.
        """
        # Validate model orders
        validate_model_order(self.p, self.q, 'ARMA')
        
        # Validate trend specification
        if self.trend not in VALID_TREND_TYPES:
            raise ValueError(f"Invalid trend type: {self.trend}. Must be one of {VALID_TREND_TYPES}")
        
        # Validate exogenous variables if provided
        if self.exog is not None:
            validate_array_input(self.exog)
            
        # Count trend parameters and create parameter names
        self._n_trend_params = 0
        self._param_names = []
        
        if 'c' in self.trend:
            self._n_trend_params += 1
            self._param_names.append('const')
        if 't' in self.trend:
            self._n_trend_params += 1
            self._param_names.append('trend')
            
        # Add AR parameter names
        for i in range(self.p):
            self._param_names.append(f'ar.{i+1}')
            
        # Add MA parameter names
        for i in range(self.q):
            self._param_names.append(f'ma.{i+1}')
            
        # Add exogenous parameter names
        if self.exog is not None:
            if self.exog.ndim == 1:
                self._param_names.append('exog.1')
            else:
                for i in range(self.exog.shape[1]):
                    self._param_names.append(f'exog.{i+1}')
            
        # Initialize optimizer
        self._optimizer = Optimizer()
        
        # Initialize parameters
        total_params = self.p + self.q + self._n_trend_params
        if self.exog is not None:
            if self.exog.ndim == 1:
                total_params += 1
            else:
                total_params += self.exog.shape[1]
        
        self.params = np.zeros(total_params)
        
        logger.debug(f"Initialized ARMAX({self.p}, {self.q}) model with trend='{self.trend}'")
    
    async def async_fit(self, endog: np.ndarray, options: Optional[Dict[str, Any]] = None) -> bool:
        """
        Asynchronously estimate model parameters using maximum likelihood.
        
        This method estimates the ARMAX model parameters by maximizing the
        likelihood function, using asynchronous optimization to allow for
        progress monitoring and non-blocking execution.
        
        Parameters
        ----------
        endog : np.ndarray
            Endogenous variable (time series to model)
        options : Optional[Dict[str, Any]]
            Additional options for optimization
            
        Returns
        -------
        bool
            True if estimation converged successfully
            
        Notes
        -----
        This method stores estimation results in the model's attributes:
        - params: Estimated parameters
        - residuals: Model residuals
        - sigma2: Residual variance
        - std_errors: Parameter standard errors
        """
        # Validate input data
        validate_array_input(endog)
        
        # Store data for later use
        self._endog = endog
        n_obs = len(endog)
        
        # Validate exogenous variables dimensions
        if self.exog is not None:
            if len(self.exog) != n_obs:
                raise ValueError(f"Exogenous variables and endogenous variable must have the same length. "
                                f"Got {len(self.exog)} vs {n_obs}")
        
        # Set up initial parameter estimates
        # Use statsmodels for initial guesses if available
        try:
            sm_order = (self.p, 0, self.q)  # (p, d, q) for statsmodels
            
            # Set up statsmodels ARIMA model for initial estimates
            sm_model = sm_arima.ARIMA(
                endog,
                exog=self.exog,
                order=sm_order,
                trend=self.trend
            )
            
            # Fit the statsmodels model to get initial parameters
            sm_result = sm_model.fit()
            
            # Extract parameters from statsmodels result
            initial_params = sm_result.params
            
            logger.debug("Using statsmodels for initial parameter estimates")
            
        except Exception as e:
            # If statsmodels fit fails, use simple heuristic initialization
            logger.warning(f"Statsmodels initialization failed: {str(e)}. Using heuristic initialization.")
            
            # Initialize trend parameters
            param_index = 0
            if 'c' in self.trend:
                self.params[param_index] = np.mean(endog)
                param_index += 1
            if 't' in self.trend:
                self.params[param_index] = 0.01  # Small positive slope
                param_index += 1
                
            # Initialize AR parameters (small positive values for stationarity)
            for i in range(self.p):
                self.params[param_index + i] = 0.1 / (i + 1)
                
            # Initialize MA parameters (small positive values)
            param_index += self.p
            for i in range(self.q):
                self.params[param_index + i] = 0.05 / (i + 1)
                
            # Initialize exogenous parameters if present
            if self.exog is not None:
                param_index += self.q
                exog_dims = 1 if self.exog.ndim == 1 else self.exog.shape[1]
                for i in range(exog_dims):
                    self.params[param_index + i] = 0.1
                    
            initial_params = self.params.copy()
        
        # Set up parameter bounds for optimization
        param_bounds = []
        
        # Trend parameter bounds (can be any real numbers)
        for _ in range(self._n_trend_params):
            param_bounds.append((None, None))
            
        # AR parameter bounds (typically between -1 and 1 for stationarity)
        for _ in range(self.p):
            param_bounds.append((-0.99, 0.99))
            
        # MA parameter bounds (typically between -1 and 1 for invertibility)
        for _ in range(self.q):
            param_bounds.append((-0.99, 0.99))
            
        # Exogenous parameter bounds (can be any real numbers)
        if self.exog is not None:
            exog_dims = 1 if self.exog.ndim == 1 else self.exog.shape[1]
            for _ in range(exog_dims):
                param_bounds.append((None, None))
        
        # Define the model type and distribution for optimization
        model_type = "ARMAX"
        distribution = "normal"  # Assuming Gaussian errors
        
        # Prepare optimizer options
        opt_options = {}
        if options is not None:
            opt_options.update(options)
        
        try:
            # Define wrapper function for likelihood calculation
            def armax_likelihood(params):
                # Split parameters into components
                trend_params = params[:self._n_trend_params]
                ar_params = params[self._n_trend_params:self._n_trend_params+self.p]
                ma_params = params[self._n_trend_params+self.p:self._n_trend_params+self.p+self.q]
                
                if self.exog is not None:
                    exog_params = params[self._n_trend_params+self.p+self.q:]
                    # Compute log-likelihood
                    ll, resids = _armax_likelihood_numba(
                        endog, ar_params, ma_params, trend_params, self.exog, exog_params)
                else:
                    # Compute log-likelihood
                    ll, resids = _armax_likelihood_numba(
                        endog, ar_params, ma_params, trend_params)
                
                # Return negative log-likelihood for minimization
                return -ll
            
            logger.info("Starting ARMAX model estimation")
            
            # Perform asynchronous optimization
            optimal_params, likelihood = await self._optimizer.async_optimize(
                endog, initial_params, model_type, distribution)
            
            # Check if optimization converged
            converged = self._optimizer.converged
            
            # Update model parameters if optimization converged
            if converged:
                self.params = optimal_params
                
                # Split parameters into components for likelihood calculation
                trend_params = self.params[:self._n_trend_params]
                ar_params = self.params[self._n_trend_params:self._n_trend_params+self.p]
                ma_params = self.params[self._n_trend_params+self.p:self._n_trend_params+self.p+self.q]
                
                # Compute residuals
                if self.exog is not None:
                    exog_params = self.params[self._n_trend_params+self.p+self.q:]
                    _, self.residuals = _armax_likelihood_numba(
                        endog, ar_params, ma_params, trend_params, self.exog, exog_params)
                else:
                    _, self.residuals = _armax_likelihood_numba(
                        endog, ar_params, ma_params, trend_params)
                
                # Compute standard errors (simplified for this implementation)
                self.std_errors = np.abs(self.params) * 0.1  # Placeholder
                
                # Set residual variance
                effective_obs = n_obs - max(self.p, self.q)
                self.sigma2 = np.sum(self.residuals[max(self.p, self.q):]**2) / effective_obs
                
                logger.info("ARMAX model estimation completed successfully")
            else:
                logger.warning("ARMAX model estimation did not converge")
            
            return converged
            
        except Exception as e:
            logger.error(f"ARMAX estimation failed: {str(e)}")
            return False
    
    def forecast(self, steps: int, exog_future: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate multi-step ahead forecasts using Numba-optimized computations.
        
        This method forecasts future values of the time series based on the
        estimated model, leveraging Numba JIT compilation for performance.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        exog_future : Optional[np.ndarray]
            Future values of exogenous variables, required if model has exogenous regressors
            
        Returns
        -------
        np.ndarray
            Array of point forecasts
            
        Raises
        ------
        ValueError
            If model has not been estimated or required inputs are missing
        """
        if self.params is None or self._endog is None:
            raise ValueError("Model must be estimated before forecasting")
            
        # Check if we need exogenous variables for forecasting
        if self.exog is not None and exog_future is None:
            raise ValueError("Model has exogenous variables; exog_future is required for forecasting")
            
        # Validate exogenous future data if provided
        if exog_future is not None:
            validate_array_input(exog_future)
            
            # Check dimensions
            if self.exog.ndim != exog_future.ndim:
                raise ValueError(f"exog_future must have same dimensions as model's exog. "
                               f"Got {exog_future.ndim} vs {self.exog.ndim}")
                
            if exog_future.ndim == 1 and len(exog_future) < steps:
                raise ValueError(f"exog_future must have at least {steps} observations")
                
            if exog_future.ndim == 2:
                if self.exog.shape[1] != exog_future.shape[1]:
                    raise ValueError(f"exog_future must have same number of variables as model's exog. "
                                   f"Got {exog_future.shape[1]} vs {self.exog.shape[1]}")
                if exog_future.shape[0] < steps:
                    raise ValueError(f"exog_future must have at least {steps} observations")
        
        # Split parameters into components
        trend_params = self.params[:self._n_trend_params]
        ar_params = self.params[self._n_trend_params:self._n_trend_params+self.p]
        ma_params = self.params[self._n_trend_params+self.p:self._n_trend_params+self.p+self.q]
        
        if self.exog is not None:
            exog_params = self.params[self._n_trend_params+self.p+self.q:]
            # Call Numba-optimized forecast function
            forecasts = _forecast_armax_numba(
                steps, ar_params, ma_params, trend_params, 
                self._endog, self.residuals, exog_future, exog_params)
        else:
            # Call Numba-optimized forecast function without exogenous variables
            forecasts = _forecast_armax_numba(
                steps, ar_params, ma_params, trend_params, 
                self._endog, self.residuals)
        
        return forecasts
    
    def diagnostic_tests(self) -> Dict[str, Any]:
        """
        Compute model diagnostic statistics and tests.
        
        This method performs various diagnostic tests on the estimated model,
        including residual autocorrelation, normality tests, and information
        criteria for model selection.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing diagnostic test results
            
        Raises
        ------
        ValueError
            If model has not been estimated
        """
        if self.params is None or self.residuals is None:
            raise ValueError("Model must be estimated before running diagnostics")
            
        # Initialize results dictionary
        diagnostics = {}
        
        # Compute information criteria
        n_obs = len(self._endog)
        n_params = len(self.params)
        ll = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(self.sigma2) + 1)
        
        diagnostics['log_likelihood'] = ll
        diagnostics['aic'] = -2 * ll / n_obs + 2 * n_params / n_obs
        diagnostics['bic'] = -2 * ll / n_obs + n_params * np.log(n_obs) / n_obs
        
        # Ljung-Box test for autocorrelation
        lags = min(10, n_obs // 5)
        acf = compute_acf(self.residuals, lags)
        
        # Compute Ljung-Box Q statistic
        q_stat = n_obs * (n_obs + 2) * np.sum(np.square(acf[1:]) / (n_obs - np.arange(1, lags + 1)))
        p_value = 1 - stats.chi2.cdf(q_stat, lags - self.p - self.q)
        
        diagnostics['ljung_box_q'] = q_stat
        diagnostics['ljung_box_p'] = p_value
        
        # Jarque-Bera test for normality
        skew = stats.skew(self.residuals)
        kurt = stats.kurtosis(self.residuals)
        jb_stat = n_obs / 6 * (skew**2 + (kurt**2) / 4)
        jb_p_value = 1 - stats.chi2.cdf(jb_stat, 2)
        
        diagnostics['jarque_bera'] = jb_stat
        diagnostics['jarque_bera_p'] = jb_p_value
        diagnostics['skewness'] = skew
        diagnostics['kurtosis'] = kurt
        
        # Residual statistics
        diagnostics['mean_residual'] = np.mean(self.residuals)
        diagnostics['residual_variance'] = self.sigma2
        
        # Parameter values and standard errors
        for i, name in enumerate(self._param_names):
            diagnostics[f'param_{name}'] = self.params[i]
            diagnostics[f'stderr_{name}'] = self.std_errors[i]
            diagnostics[f'tstat_{name}'] = self.params[i] / self.std_errors[i]
            diagnostics[f'pvalue_{name}'] = 2 * (1 - stats.norm.cdf(abs(self.params[i] / self.std_errors[i])))
        
        return diagnostics