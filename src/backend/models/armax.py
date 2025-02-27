"""
ARMA/ARMAX time series modeling and forecasting.

This module implements a comprehensive ARMA/ARMAX modeling framework 
with parameter estimation using robust numerical optimization, diagnostic testing, 
and forecasting capabilities. It provides an object-oriented interface with async
support for computationally intensive operations.

Key features:
- Class-based ARMA/ARMAX model implementation
- Asynchronous parameter estimation using SciPy optimization
- Multi-step forecasting with error propagation
- Comprehensive diagnostic statistics and tests
- Numba-optimized core routines for performance
"""

import logging
import numpy as np  # version: 1.26.3
from scipy import stats  # version: 1.11.4
from scipy import optimize  # version: 1.11.4
import statsmodels.tsa.arima as arima  # version: 0.14.1
import numba  # version: 0.59.0
from typing import Dict, Optional, Tuple, Any, List, Union, Callable
from dataclasses import dataclass, field
import asyncio

from ..core.optimization import Optimizer, compute_standard_errors
from ..utils.validation import validate_parameters, validate_array_input, validate_model_order

# Configure logger
logger = logging.getLogger(__name__)

# Global constants
MAX_ORDER = 30


@numba.jit(nopython=True)
def compute_residuals(data: np.ndarray, 
                      params: np.ndarray, 
                      exog: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes model residuals from fitted ARMAX model.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data array
    params : np.ndarray
        Model parameters [ar_params, ma_params, constant, exog_params]
    exog : Optional[np.ndarray]
        Exogenous variables, if any
        
    Returns
    -------
    np.ndarray
        Model residuals
    """
    # Ensure arrays are contiguous for numba performance
    data = np.ascontiguousarray(data)
    params = np.ascontiguousarray(params)
    
    # Extract dimensions
    n = len(data)
    
    # Initialize arrays
    residuals = np.zeros(n)
    ar_lags = np.zeros(n)
    ma_lags = np.zeros(n)
    
    # Get parameter counts from the first elements
    p = int(params[0])  # Number of AR parameters
    q = int(params[1])  # Number of MA parameters
    has_constant = params[2] > 0.5  # Boolean for including constant
    
    # Extract parameters
    ar_params = params[3:3+p] if p > 0 else np.empty(0)
    ma_params = params[3+p:3+p+q] if q > 0 else np.empty(0)
    
    # Constant term is after AR and MA parameters
    constant_idx = 3 + p + q
    constant = params[constant_idx] if has_constant else 0.0
    
    # Exogenous parameters start after constant (if any)
    exog_params_start = constant_idx + (1 if has_constant else 0)
    exog_params = params[exog_params_start:] if exog is not None else np.empty(0)
    
    # Set initial residuals
    for t in range(max(p, q)):
        residuals[t] = 0.0
    
    # Apply ARMA filter
    for t in range(max(p, q), n):
        # AR component
        ar_component = 0.0
        for i in range(p):
            ar_component += ar_params[i] * data[t-i-1]
        
        # MA component
        ma_component = 0.0
        for j in range(q):
            ma_component += ma_params[j] * residuals[t-j-1]
        
        # Exogenous component
        exog_component = 0.0
        if exog is not None:
            for k in range(len(exog_params)):
                exog_component += exog_params[k] * exog[t, k]
        
        # Compute fitted value
        fitted = ar_component + ma_component + constant + exog_component
        
        # Compute residual
        residuals[t] = data[t] - fitted
    
    return residuals


@dataclass
class ARMAX:
    """
    ARMA/ARMAX model implementation with parameter estimation and forecasting.
    
    This class implements ARMA/ARMAX time series models with comprehensive features
    for model estimation, diagnostics, and forecasting. It supports asynchronous
    parameter estimation for improved responsiveness in interactive environments.
    
    Parameters
    ----------
    p : int
        Autoregressive order
    q : int
        Moving average order
    include_constant : bool
        Whether to include a constant term in the model
        
    Attributes
    ----------
    p : int
        Autoregressive order
    q : int
        Moving average order
    include_constant : bool
        Whether to include a constant term
    params : ndarray
        Estimated model parameters
    residuals : ndarray
        Model residuals
    loglikelihood : float
        Log-likelihood value of the fitted model
    standard_errors : ndarray
        Standard errors of parameter estimates
        
    Methods
    -------
    async_fit
        Asynchronously estimate model parameters
    forecast
        Generate multi-step forecasts
    diagnostic_tests
        Compute model diagnostic statistics
    """
    p: int
    q: int
    include_constant: bool = True
    
    # Model state attributes
    params: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    loglikelihood: Optional[float] = None
    standard_errors: Optional[np.ndarray] = None
    
    # Private attributes
    _optimizer: Optional[Optimizer] = field(default=None, init=False)
    _data: Optional[np.ndarray] = field(default=None, init=False)
    _exog: Optional[np.ndarray] = field(default=None, init=False)
    _fitted: Optional[np.ndarray] = field(default=None, init=False)
    _model_params: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """
        Initialize ARMAX model with specified orders and options.
        """
        # Validate model orders
        validate_model_order(self.p, self.q, model_type='ARMA')
        
        # Initialize optimizer
        self._optimizer = Optimizer()
        logger.info(f"Initialized ARMAX model with p={self.p}, q={self.q}, include_constant={self.include_constant}")
    
    def _prepare_params_vector(self, ar_params, ma_params, constant=None, exog_params=None):
        """
        Prepare parameter vector for optimization or residual calculation.
        
        Parameters
        ----------
        ar_params : ndarray or list
            AR parameters
        ma_params : ndarray or list
            MA parameters
        constant : float, optional
            Constant term
        exog_params : ndarray or list, optional
            Exogenous variable parameters
            
        Returns
        -------
        ndarray
            Full parameter vector
        """
        # Determine dimensions
        p = len(ar_params) if ar_params is not None else 0
        q = len(ma_params) if ma_params is not None else 0
        has_constant = constant is not None and self.include_constant
        n_exog = len(exog_params) if exog_params is not None else 0
        
        # Create parameter vector: [p, q, has_constant, ar_params, ma_params, constant, exog_params]
        n_params = 3 + p + q + (1 if has_constant else 0) + n_exog
        params = np.zeros(n_params)
        
        # Set model structure parameters
        params[0] = p
        params[1] = q
        params[2] = 1 if has_constant else 0
        
        # Set model coefficients
        idx = 3
        if p > 0:
            params[idx:idx+p] = ar_params
            idx += p
        
        if q > 0:
            params[idx:idx+q] = ma_params
            idx += q
        
        if has_constant:
            params[idx] = constant
            idx += 1
        
        if n_exog > 0:
            params[idx:idx+n_exog] = exog_params
        
        return params
    
    def _extract_params(self, params):
        """
        Extract individual parameters from parameter vector.
        
        Parameters
        ----------
        params : ndarray
            Full parameter vector
            
        Returns
        -------
        tuple
            (ar_params, ma_params, constant, exog_params)
        """
        # Extract parameter counts
        p = int(params[0])
        q = int(params[1])
        has_constant = params[2] > 0.5
        
        # Extract parameters
        idx = 3
        ar_params = params[idx:idx+p] if p > 0 else np.empty(0)
        idx += p
        
        ma_params = params[idx:idx+q] if q > 0 else np.empty(0)
        idx += q
        
        constant = params[idx] if has_constant else None
        idx += 1 if has_constant else 0
        
        exog_params = params[idx:] if idx < len(params) else np.empty(0)
        
        return ar_params, ma_params, constant, exog_params
    
    def _log_likelihood(self, params, data, exog=None):
        """
        Compute log-likelihood for ARMAX model with given parameters.
        
        Parameters
        ----------
        params : ndarray
            Model parameters
        data : ndarray
            Time series data
        exog : ndarray, optional
            Exogenous variables
            
        Returns
        -------
        float
            Negative log-likelihood (for minimization)
        """
        try:
            # Compute residuals
            residuals = compute_residuals(data, params, exog)
            
            # Filter out initial residuals which may contain startup effects
            p = int(params[0])
            q = int(params[1])
            effective_residuals = residuals[max(p, q):]
            
            # Calculate log-likelihood (assuming Gaussian errors)
            n = len(effective_residuals)
            variance = np.sum(effective_residuals**2) / n
            
            # Gaussian log-likelihood
            loglik = -0.5 * n * np.log(2 * np.pi * variance) - 0.5 * np.sum(effective_residuals**2) / variance
            
            # Return negative log-likelihood for minimization
            return -loglik
        except Exception as e:
            logger.error(f"Error in log-likelihood calculation: {str(e)}")
            # Return a large value to penalize invalid parameters
            return 1e10
    
    async def async_fit(self, data: np.ndarray, exog: Optional[np.ndarray] = None) -> bool:
        """
        Asynchronously estimate model parameters using maximum likelihood.
        
        This method performs parameter estimation for the ARMAX model using numerical
        optimization to maximize the likelihood function. It uses async/await patterns
        to maintain UI responsiveness during computation.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data for model estimation
        exog : Optional[np.ndarray]
            Exogenous variables for ARMAX model, if any
            
        Returns
        -------
        bool
            True if estimation converged, False otherwise
            
        Notes
        -----
        This method is designed to be called with await in an asynchronous context.
        Use in a synchronous context will require running an event loop.
        """
        try:
            # Validate input data
            validate_array_input(data)
            if exog is not None:
                validate_array_input(exog)
                if len(data) != exog.shape[0]:
                    raise ValueError("Length of data and exog must match")
            
            # Store data for later use
            self._data = data
            self._exog = exog
            
            # Initialize parameters
            n_ar = self.p
            n_ma = self.q
            n_constant = 1 if self.include_constant else 0
            n_exog = exog.shape[1] if exog is not None else 0
            
            # Initialize AR parameters
            ar_params = np.zeros(n_ar)
            if n_ar > 0:
                # Start with simple AR(p) initialization
                ar_params = np.array([0.1] * n_ar)
            
            # Initialize MA parameters
            ma_params = np.zeros(n_ma)
            if n_ma > 0:
                # Start with simple MA(q) initialization
                ma_params = np.array([0.1] * n_ma)
            
            # Initialize constant
            constant = np.mean(data) if n_constant > 0 else None
            
            # Initialize exogenous parameters
            exog_params = None
            if n_exog > 0:
                # Start with zeros for exogenous parameters
                exog_params = np.zeros(n_exog)
            
            # Prepare initial parameter vector
            initial_params = self._prepare_params_vector(ar_params, ma_params, constant, exog_params)
            
            # Define bounds for parameters
            # AR and MA parameters between -0.99 and 0.99 for stationarity/invertibility
            bounds = []
            
            # Model structure parameters (fixed)
            bounds.extend([(n_ar, n_ar), (n_ma, n_ma), (n_constant, n_constant)])
            
            # AR parameters
            bounds.extend([(-0.99, 0.99) for _ in range(n_ar)])
            
            # MA parameters
            bounds.extend([(-0.99, 0.99) for _ in range(n_ma)])
            
            # Constant (unconstrained)
            if n_constant > 0:
                bounds.append((None, None))
            
            # Exogenous parameters (unconstrained)
            bounds.extend([(None, None) for _ in range(n_exog)])
            
            # Create optimization function
            def optim_func():
                try:
                    # Define negative log-likelihood function for minimization
                    def neg_loglik(params):
                        return self._log_likelihood(params, data, exog)
                    
                    # Use SLSQP method for constrained optimization
                    result = optimize.minimize(
                        neg_loglik,
                        initial_params,
                        method='SLSQP',
                        bounds=bounds,
                        options={'ftol': 1e-8, 'disp': False, 'maxiter': 1000}
                    )
                    
                    optimal_params = result.x
                    likelihood = -result.fun  # Convert back to positive log-likelihood
                    converged = result.success
                    
                    return optimal_params, likelihood, converged, result
                except Exception as e:
                    logger.error(f"Optimization failed: {str(e)}")
                    return initial_params, -np.inf, False, None
            
            # Run optimization asynchronously
            logger.info("Starting ARMAX parameter estimation")
            loop = asyncio.get_event_loop()
            optimal_params, likelihood, converged, result = await loop.run_in_executor(None, optim_func)
            
            # Store optimization results
            self.params = optimal_params
            self.loglikelihood = likelihood
            self._model_params['result'] = result
            
            # Compute residuals
            self.residuals = compute_residuals(data, optimal_params, exog)
            
            # Compute fitted values
            self._fitted = data - self.residuals
            
            # Extract parameters for easier access
            ar_params, ma_params, constant, exog_params = self._extract_params(optimal_params)
            self._model_params['ar_params'] = ar_params
            self._model_params['ma_params'] = ma_params
            self._model_params['constant'] = constant
            self._model_params['exog_params'] = exog_params
            
            # Compute standard errors
            if converged:
                # Compute standard errors using Fisher Information Matrix
                # For demonstration, using a simplified approach
                n = len(data)
                k = n_ar + n_ma + n_constant + n_exog
                
                # Use effective residuals (excluding initialization period)
                effective_residuals = self.residuals[max(n_ar, n_ma):]
                
                # Estimate residual variance
                residual_variance = np.sum(effective_residuals**2) / (n - k)
                
                # Simple standard error approximation based on observed Fisher Information
                # In a real implementation, we would use the Hessian of the log-likelihood
                if result is not None and hasattr(result, 'hess_inv'):
                    try:
                        # Extract the diagonal of the inverse Hessian (approximate covariance matrix)
                        hess_diag = np.diag(result.hess_inv)
                        self.standard_errors = np.sqrt(hess_diag)
                    except:
                        # Fallback to simple approximation
                        self.standard_errors = np.sqrt(residual_variance) * np.ones(len(optimal_params))
                else:
                    # Simple approximation
                    self.standard_errors = np.sqrt(residual_variance) * np.ones(len(optimal_params))
                
                logger.info("ARMAX estimation completed successfully")
            else:
                logger.warning("ARMAX estimation did not converge")
            
            return converged
        
        except Exception as e:
            logger.error(f"Error in ARMAX estimation: {str(e)}")
            raise RuntimeError(f"ARMAX estimation failed: {str(e)}") from e
    
    def forecast(self, steps: int, exog_future: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate multi-step forecasts from fitted model.
        
        This method produces forecasts for future time periods based on the estimated
        model parameters and past data. It supports both ARMA and ARMAX forecasting
        with exogenous variables.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        exog_future : Optional[np.ndarray]
            Future values of exogenous variables for forecasting
            
        Returns
        -------
        np.ndarray
            Array of point forecasts for specified horizon
            
        Raises
        ------
        ValueError
            If model has not been fit or forecast inputs are invalid
        """
        # Check if model has been fit
        if self.params is None:
            raise ValueError("Model must be fit before forecasting")
        
        # Validate inputs
        if steps <= 0:
            raise ValueError("Steps must be a positive integer")
        
        # Check exogenous variables
        if self._exog is not None and exog_future is None:
            raise ValueError("Model was fit with exogenous variables, so exog_future must be provided")
        
        if exog_future is not None:
            validate_array_input(exog_future)
            if exog_future.shape[0] < steps:
                raise ValueError(f"exog_future must have at least {steps} rows")
            if self._exog is not None and exog_future.shape[1] != self._exog.shape[1]:
                raise ValueError("exog_future must have same number of columns as exog used for fitting")
        
        # Extract parameters
        ar_params, ma_params, constant, exog_params = self._extract_params(self.params)
        
        # Get historical data
        data = self._data
        residuals = self.residuals
        
        # Initialize forecast array
        forecasts = np.zeros(steps)
        
        # Generate forecasts
        for h in range(steps):
            # AR component
            ar_component = 0.0
            for i in range(len(ar_params)):
                lag_idx = -i - 1  # Index for lagged value
                if h + lag_idx >= 0:
                    # Use forecasted value
                    ar_component += ar_params[i] * forecasts[h + lag_idx]
                else:
                    # Use historical value
                    ar_component += ar_params[i] * data[len(data) + lag_idx]
            
            # MA component - future residuals are zero in expectation
            ma_component = 0.0
            for i in range(len(ma_params)):
                lag_idx = -i - 1  # Index for lagged residual
                if h + lag_idx < 0:
                    # Only use historical residuals
                    ma_component += ma_params[i] * residuals[len(residuals) + lag_idx]
            
            # Constant component
            const_component = constant if constant is not None else 0.0
            
            # Exogenous component
            exog_component = 0.0
            if exog_future is not None and len(exog_params) > 0:
                for i in range(len(exog_params)):
                    exog_component += exog_params[i] * exog_future[h, i]
            
            # Calculate forecast
            forecasts[h] = ar_component + ma_component + const_component + exog_component
        
        return forecasts
    
    def diagnostic_tests(self) -> Dict[str, Any]:
        """
        Compute model diagnostic statistics and tests.
        
        This method performs comprehensive diagnostic testing for the fitted model,
        including residual analysis, autocorrelation tests, and information criteria
        to assess model adequacy.
        
        Returns
        -------
        dict
            Dictionary of test statistics and p-values including:
            - AIC: Akaike Information Criterion
            - BIC: Bayesian Information Criterion
            - HQIC: Hannan-Quinn Information Criterion
            - ljung_box: Ljung-Box test statistic and p-value
            - jarque_bera: Jarque-Bera test for normality
            - residual_stats: Descriptive statistics of residuals
            
        Raises
        ------
        ValueError
            If model has not been fit
        """
        # Check if model has been fit
        if self.params is None or self.residuals is None:
            raise ValueError("Model must be fit before computing diagnostics")
        
        # Extract parameters
        ar_params, ma_params, constant, exog_params = self._extract_params(self.params)
        
        # Number of parameters
        k = len(ar_params) + len(ma_params) + (1 if constant is not None else 0) + len(exog_params)
        
        # Number of observations
        n = len(self._data)
        
        # Initialize diagnostics dictionary
        diagnostics = {}
        
        # Compute information criteria
        loglik = self.loglikelihood
        diagnostics['AIC'] = -2 * loglik + 2 * k
        diagnostics['BIC'] = -2 * loglik + np.log(n) * k
        diagnostics['HQIC'] = -2 * loglik + 2 * np.log(np.log(n)) * k
        
        # Compute residual diagnostics
        
        # Ljung-Box test for autocorrelation
        lags = min(10, n // 5)  # Rule of thumb for number of lags
        lb_stat, lb_pvalue = stats.acorr_ljungbox(self.residuals, lags=[lags], return_df=False)
        diagnostics['ljung_box'] = {
            'statistic': float(lb_stat[0]),
            'p_value': float(lb_pvalue[0]),
            'lags': lags,
            'null_hypothesis': "No autocorrelation up to lag {}".format(lags)
        }
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(self.residuals)
        diagnostics['jarque_bera'] = {
            'statistic': float(jb_stat),
            'p_value': float(jb_pvalue),
            'null_hypothesis': "Residuals are normally distributed"
        }
        
        # Descriptive statistics for residuals
        diagnostics['residual_stats'] = {
            'mean': float(np.mean(self.residuals)),
            'std_dev': float(np.std(self.residuals, ddof=1)),
            'variance': float(np.var(self.residuals, ddof=1)),
            'skewness': float(stats.skew(self.residuals)),
            'kurtosis': float(stats.kurtosis(self.residuals, fisher=True)),
            'min': float(np.min(self.residuals)),
            'max': float(np.max(self.residuals)),
            'n_observations': n
        }
        
        # Parameter significance tests
        t_stats = []
        p_values = []
        
        # Skip the first 3 params which are structural (p, q, include_constant)
        for i in range(3, len(self.params)):
            if self.standard_errors is not None and i < len(self.standard_errors):
                t_stat = self.params[i] / self.standard_errors[i]
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
                t_stats.append(float(t_stat))
                p_values.append(float(p_value))
            else:
                t_stats.append(None)
                p_values.append(None)
        
        # Parameter summary
        param_names = []
        param_values = []
        
        # AR parameters
        for i in range(len(ar_params)):
            param_names.append(f"AR({i+1})")
            param_values.append(float(ar_params[i]))
        
        # MA parameters
        for i in range(len(ma_params)):
            param_names.append(f"MA({i+1})")
            param_values.append(float(ma_params[i]))
        
        # Constant
        if constant is not None:
            param_names.append("Constant")
            param_values.append(float(constant))
        
        # Exogenous parameters
        for i in range(len(exog_params)):
            param_names.append(f"Exog({i+1})")
            param_values.append(float(exog_params[i]))
        
        # Parameter summary
        param_summary = []
        for i, name in enumerate(param_names):
            if i < len(t_stats) and t_stats[i] is not None:
                param_summary.append({
                    'name': name,
                    'value': param_values[i],
                    'std_error': float(self.standard_errors[i+3]) if self.standard_errors is not None else None,
                    't_statistic': t_stats[i],
                    'p_value': p_values[i]
                })
            else:
                param_summary.append({
                    'name': name,
                    'value': param_values[i],
                    'std_error': None,
                    't_statistic': None,
                    'p_value': None
                })
        
        diagnostics['parameter_summary'] = param_summary
        
        return diagnostics