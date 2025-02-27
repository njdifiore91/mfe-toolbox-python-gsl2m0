"""
Univariate volatility model implementations with Numba optimization.

This module provides a unified framework for univariate GARCH models including
AGARCH, EGARCH, FIGARCH, IGARCH and TARCH variants. It leverages Numba's JIT
compilation for performance-critical functions and implements robust parameter
estimation through SciPy's optimization routines.

Key features:
- Unified API for multiple volatility model types
- Flexible error distribution specification
- Numba-optimized volatility computation
- Asynchronous parameter estimation
- Robust forecasting capabilities
"""

import logging
import numpy as np
import numba
from scipy import optimize, stats
from typing import Dict, Optional, Tuple, List, Union, Any, Callable
from dataclasses import dataclass, field

# Internal imports
from ..core.optimization import Optimizer
from ..core.distributions import GED, SkewedT
from ..utils.validation import validate_parameters, validate_array_input, VALID_GARCH_TYPES, validate_model_order

# Configure logger
logger = logging.getLogger(__name__)

# List of supported model types
SUPPORTED_MODELS = ['GARCH', 'EGARCH', 'AGARCH', 'FIGARCH', 'IGARCH', 'TARCH']


@numba.jit(nopython=True)
def compute_volatility(returns: np.ndarray, 
                      parameters: np.ndarray, 
                      model_type: str) -> np.ndarray:
    """
    Numba-optimized volatility computation for univariate GARCH models.
    
    This function applies the volatility recursion formula for various GARCH-type
    models using efficient Numba compilation. It supports GARCH, EGARCH, AGARCH, 
    FIGARCH, IGARCH, and TARCH model specifications.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of return data
    parameters : np.ndarray
        Model parameters appropriate for the specified model type
    model_type : str
        Type of GARCH model ('GARCH', 'EGARCH', 'AGARCH', etc.)
        
    Returns
    -------
    np.ndarray
        Conditional volatility estimates
        
    Notes
    -----
    The parameter structure varies by model type:
    - GARCH: [omega, alpha, beta]
    - EGARCH: [omega, alpha, gamma, beta]
    - AGARCH/TARCH: [omega, alpha, beta, gamma]
    - FIGARCH: [omega, d, beta, phi]
    - IGARCH: [omega, alpha] (beta = 1-alpha)
    """
    n = len(returns)
    volatility = np.zeros(n)
    
    # Initialize with unconditional variance for first observation
    volatility[0] = np.var(returns)
    
    # Use model_type to determine which volatility recursion to apply
    if model_type == 'GARCH':
        omega, alpha, beta = parameters[0], parameters[1], parameters[2]
        
        for t in range(1, n):
            volatility[t] = omega + alpha * returns[t-1]**2 + beta * volatility[t-1]
    
    elif model_type == 'EGARCH':
        omega, alpha, gamma, beta = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Initialize log-variance
        log_volatility = np.zeros(n)
        log_volatility[0] = np.log(volatility[0])
        
        for t in range(1, n):
            # Standardized residual
            z_t_1 = returns[t-1] / np.sqrt(volatility[t-1])
            
            # EGARCH recursion
            log_volatility[t] = omega + beta * log_volatility[t-1] + \
                              alpha * (np.abs(z_t_1) - np.sqrt(2/np.pi)) + \
                              gamma * z_t_1
            
            # Convert back to variance
            volatility[t] = np.exp(log_volatility[t])
    
    elif model_type in ['AGARCH', 'TARCH']:
        omega, alpha, beta, gamma = parameters[0], parameters[1], parameters[2], parameters[3]
        
        for t in range(1, n):
            # Asymmetric term: for TARCH, it's an indicator function
            # for AGARCH, it's a quadratic function
            if model_type == 'TARCH':
                asymmetric = gamma * (returns[t-1] < 0) * returns[t-1]**2
                volatility[t] = omega + alpha * returns[t-1]**2 + beta * volatility[t-1] + asymmetric
            else:  # AGARCH
                volatility[t] = omega + alpha * (returns[t-1] - gamma)**2 + beta * volatility[t-1]
    
    elif model_type == 'FIGARCH':
        omega, d, beta, phi = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # FIGARCH uses fractional differencing
        # This is a simplified implementation
        for t in range(1, n):
            # Simple fractional differencing approximation
            long_memory = d * returns[t-1]**2
            volatility[t] = omega + long_memory + beta * volatility[t-1] + phi * returns[t-1]**2
    
    elif model_type == 'IGARCH':
        omega, alpha = parameters[0], parameters[1]
        beta = 1.0 - alpha  # IGARCH constraint: alpha + beta = 1
        
        for t in range(1, n):
            volatility[t] = omega + alpha * returns[t-1]**2 + beta * volatility[t-1]
    
    # Apply lower bound to avoid numerical issues
    volatility = np.maximum(volatility, 1e-6)
    
    return volatility


@dataclass
class UnivariateGARCH:
    """
    Base class for univariate GARCH model implementations.
    
    This class provides a unified framework for estimating and forecasting
    various univariate GARCH models, with support for different error
    distributions and parameter constraints. It uses Numba-optimized
    computations for performance-critical operations.
    
    Parameters
    ----------
    p : int
        ARCH order
    q : int
        GARCH order
    distribution : Optional[str]
        Error distribution type, default is 'normal'
    model_type : str
        Type of GARCH model, default is 'GARCH'
        
    Attributes
    ----------
    p : int
        ARCH order
    q : int
        GARCH order
    distribution : str
        Error distribution ('normal', 'student-t', 'ged', 'skewed-t')
    model_type : str
        GARCH model type
    parameters : Optional[np.ndarray]
        Estimated model parameters
    std_errors : Optional[np.ndarray]
        Standard errors of estimated parameters
    volatility : Optional[np.ndarray]
        Estimated conditional volatility series
    likelihood : float
        Log-likelihood of the estimated model
    converged : bool
        Indicator of optimization convergence
    """
    p: int
    q: int
    distribution: Optional[str] = 'normal'
    model_type: str = 'GARCH'
    
    # Model state
    parameters: Optional[np.ndarray] = None
    std_errors: Optional[np.ndarray] = None
    volatility: Optional[np.ndarray] = None
    likelihood: float = 0.0
    converged: bool = False
    
    # Data
    _returns: Optional[np.ndarray] = field(default=None, repr=False)
    _optimizer: Optional[Optimizer] = field(default=None, repr=False)
    _distribution_instance: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self):
        """
        Initialize model components and validate inputs.
        """
        # Validate model order
        validate_model_order(self.p, self.q, model_type='GARCH')
        
        # Validate model type
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Model type '{self.model_type}' not supported. "
                           f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        
        # Set up distribution
        self._setup_distribution()
        
        # Initialize optimizer
        self._optimizer = Optimizer()
        
        logger.debug(f"Initialized {self.model_type}({self.p},{self.q}) model with {self.distribution} distribution")
    
    def _setup_distribution(self):
        """
        Set up error distribution instance.
        """
        if not hasattr(self, '_distribution_instance') or self._distribution_instance is None:
            if self.distribution.lower() == 'ged':
                # Default parameter for GED
                self._distribution_instance = GED(nu=1.5)
            elif self.distribution.lower() == 'skewed-t':
                # Default parameters for Skewed-T
                self._distribution_instance = SkewedT(nu=8.0, lambda_=0.0)
            elif self.distribution.lower() == 'student-t':
                # We don't have a direct class for Student-t,
                # but we can use a non-skewed SkewedT
                self._distribution_instance = SkewedT(nu=8.0, lambda_=0.0)
            elif self.distribution.lower() == 'normal':
                # No specific instance needed for normal
                self._distribution_instance = None
            else:
                raise ValueError(f"Distribution '{self.distribution}' not supported")
    
    def _get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameters for model optimization.
        
        Returns
        -------
        np.ndarray
            Initial parameter values
        """
        # Default initial parameters depend on model type
        if self.model_type == 'GARCH':
            # [omega, alpha, beta]
            return np.array([0.01, 0.1, 0.8])
        
        elif self.model_type == 'EGARCH':
            # [omega, alpha, gamma, beta]
            return np.array([-0.1, 0.1, 0.0, 0.9])
        
        elif self.model_type in ['AGARCH', 'TARCH']:
            # [omega, alpha, beta, gamma]
            return np.array([0.01, 0.05, 0.8, 0.1])
        
        elif self.model_type == 'FIGARCH':
            # [omega, d, beta, phi]
            return np.array([0.01, 0.4, 0.3, 0.2])
        
        elif self.model_type == 'IGARCH':
            # [omega, alpha]
            return np.array([0.01, 0.2])
        
        else:
            # Generic fallback
            return np.array([0.01, 0.1, 0.8])
    
    def _loglikelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for optimization.
        
        Parameters
        ----------
        params : np.ndarray
            Model parameters
        returns : np.ndarray
            Return data
            
        Returns
        -------
        float
            Negative log-likelihood value
        """
        try:
            # Compute conditional volatility
            vol = compute_volatility(returns, params, self.model_type)
            
            # Standardized residuals
            z = returns / np.sqrt(vol)
            
            # Log-likelihood calculation depends on the distribution
            if self.distribution.lower() == 'normal':
                # Normal log-likelihood
                ll = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(vol) - 0.5 * z**2
                
            elif hasattr(self, '_distribution_instance') and self._distribution_instance is not None:
                # Use distribution-specific PDF
                pdf_values = self._distribution_instance.pdf(z)
                ll = np.log(pdf_values) - 0.5 * np.log(vol)
                
            else:
                # Fallback to normal if distribution instance not available
                ll = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(vol) - 0.5 * z**2
            
            # Return negative sum for minimization (excluding first observation)
            return -np.sum(ll[1:])
            
        except Exception as e:
            logger.warning(f"Error in log-likelihood calculation: {str(e)}")
            return np.inf  # Return infinity for invalid parameters
    
    async def async_fit(self, returns: np.ndarray) -> bool:
        """
        Asynchronously estimate model parameters using maximum likelihood.
        
        This method fits the GARCH model to the provided return data using
        maximum likelihood estimation with SciPy's optimization routines.
        It supports asynchronous execution through Python's async/await pattern.
        
        Parameters
        ----------
        returns : np.ndarray
            Time series of returns
            
        Returns
        -------
        bool
            True if estimation converged
            
        Notes
        -----
        The function estimates parameters, standard errors, and conditional
        volatility. Results are stored in the model's attributes.
        """
        try:
            # Validate inputs
            validate_array_input(returns)
            
            # Store returns for later use
            self._returns = returns
            
            # Get initial parameters
            initial_params = self._get_initial_parameters()
            
            logger.info(f"Starting {self.model_type} model estimation with {self.distribution} distribution")
            
            # Run optimization asynchronously
            optimal_params, likelihood = await self._optimizer.async_optimize(
                returns, initial_params, self.model_type, self.distribution)
            
            # Store results
            self.parameters = optimal_params
            self.likelihood = -likelihood  # Convert back to positive log-likelihood
            self.converged = self._optimizer.converged
            
            # Compute conditional volatility
            self.volatility = compute_volatility(returns, optimal_params, self.model_type)
            
            # Update status
            if self.converged:
                logger.info(f"{self.model_type} model estimation converged successfully")
            else:
                logger.warning(f"{self.model_type} model estimation did not converge")
                
            return self.converged
            
        except Exception as e:
            logger.error(f"Error in model estimation: {str(e)}")
            self.converged = False
            raise RuntimeError(f"Model estimation failed: {str(e)}") from e
    
    def forecast(self, horizon: int = 1, method: str = 'analytic') -> np.ndarray:
        """
        Generate volatility forecasts using estimated parameters.
        
        This method produces forecasts of conditional volatility for a specified
        horizon using either analytic formulas or simulation methods.
        
        Parameters
        ----------
        horizon : int, optional
            Forecast horizon, by default 1
        method : str, optional
            Forecasting method ('analytic' or 'simulation'), by default 'analytic'
            
        Returns
        -------
        np.ndarray
            Array of volatility forecasts with length equal to horizon
            
        Raises
        ------
        ValueError
            If model parameters haven't been estimated
        """
        if self.parameters is None or self.volatility is None:
            raise ValueError("Model must be estimated before forecasting")
        
        if horizon < 1:
            raise ValueError("Forecast horizon must be at least 1")
        
        # Initialize forecasts array
        forecasts = np.zeros(horizon)
        
        if method == 'analytic':
            # Use analytic formula for GARCH models
            # The formula depends on the model type
            if self.model_type == 'GARCH':
                omega, alpha, beta = self.parameters[0], self.parameters[1], self.parameters[2]
                persistence = alpha + beta
                
                # Long-run variance
                long_run_var = omega / (1 - persistence)
                
                # Last estimated volatility
                last_vol = self.volatility[-1]
                last_return = self._returns[-1]
                
                # Compute forecasts
                for h in range(horizon):
                    if h == 0:
                        forecasts[h] = omega + alpha * last_return**2 + beta * last_vol
                    else:
                        forecasts[h] = omega + persistence * forecasts[h-1]
                        
            elif self.model_type == 'IGARCH':
                omega, alpha = self.parameters[0], self.parameters[1]
                beta = 1.0 - alpha
                
                # Last estimated volatility
                last_vol = self.volatility[-1]
                last_return = self._returns[-1]
                
                # Compute forecasts
                for h in range(horizon):
                    if h == 0:
                        forecasts[h] = omega + alpha * last_return**2 + beta * last_vol
                    else:
                        forecasts[h] = omega + (alpha + beta) * forecasts[h-1]
                        
            elif self.model_type == 'EGARCH':
                omega, alpha, gamma, beta = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
                
                # Last log-volatility and return
                last_vol = self.volatility[-1]
                last_log_vol = np.log(last_vol)
                last_return = self._returns[-1]
                last_z = last_return / np.sqrt(last_vol)
                
                # Expected value of abs(z) - sqrt(2/pi)
                expected_abs_z = np.sqrt(2/np.pi)
                
                # Compute log-variance forecasts
                log_forecasts = np.zeros(horizon)
                for h in range(horizon):
                    if h == 0:
                        next_term = alpha * (np.abs(last_z) - expected_abs_z) + gamma * last_z
                        log_forecasts[h] = omega + beta * last_log_vol + next_term
                    else:
                        log_forecasts[h] = omega + beta * log_forecasts[h-1]
                    
                # Convert to variance forecasts
                forecasts = np.exp(log_forecasts)
                
            elif self.model_type in ['AGARCH', 'TARCH']:
                omega, alpha, beta, gamma = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
                
                # Last volatility and return
                last_vol = self.volatility[-1]
                last_return = self._returns[-1]
                
                # Compute forecasts
                for h in range(horizon):
                    if h == 0:
                        if self.model_type == 'TARCH':
                            asymmetric = gamma * (last_return < 0) * last_return**2
                            forecasts[h] = omega + alpha * last_return**2 + beta * last_vol + asymmetric
                        else:  # AGARCH
                            forecasts[h] = omega + alpha * (last_return - gamma)**2 + beta * last_vol
                    else:
                        # For h > 1, we use the persistence
                        if self.model_type == 'TARCH':
                            # Expected value of asymmetric term is gamma/2 * previous volatility
                            persistence = alpha + beta + gamma/2
                            forecasts[h] = omega + persistence * forecasts[h-1]
                        else:  # AGARCH
                            persistence = alpha + beta
                            forecasts[h] = omega + persistence * forecasts[h-1]
                
            elif self.model_type == 'FIGARCH':
                omega, d, beta, phi = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
                
                # Simplified forecast for FIGARCH
                # In practice, this would use fractional differencing operators
                last_vol = self.volatility[-1]
                
                for h in range(horizon):
                    # Simple persistence-based forecast
                    if h == 0:
                        forecasts[h] = last_vol
                    else:
                        persistence = beta + d
                        forecasts[h] = omega + persistence * forecasts[h-1]
                        
            else:
                raise ValueError(f"Analytic forecasting not implemented for {self.model_type}")
                
        elif method == 'simulation':
            # Use Monte Carlo simulation for forecasts
            n_sims = 1000
            sim_forecasts = np.zeros((n_sims, horizon))
            
            # Compute the forecasts through simulation
            for sim in range(n_sims):
                # Start with the last actual volatility
                last_vol = self.volatility[-1]
                
                for h in range(horizon):
                    # Generate random residual
                    if self.distribution.lower() == 'normal':
                        z = np.random.normal(0, 1)
                    elif self.distribution.lower() == 'student-t':
                        # Assuming nu=8 for Student's t
                        z = np.random.standard_t(8)
                    else:
                        # Fallback to normal
                        z = np.random.normal(0, 1)
                    
                    # Compute return based on volatility
                    sim_return = z * np.sqrt(last_vol)
                    
                    # Update volatility based on model type
                    if self.model_type == 'GARCH':
                        omega, alpha, beta = self.parameters[0], self.parameters[1], self.parameters[2]
                        new_vol = omega + alpha * sim_return**2 + beta * last_vol
                    
                    elif self.model_type == 'EGARCH':
                        omega, alpha, gamma, beta = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
                        log_last_vol = np.log(last_vol)
                        log_new_vol = omega + beta * log_last_vol + alpha * (np.abs(z) - np.sqrt(2/np.pi)) + gamma * z
                        new_vol = np.exp(log_new_vol)
                    
                    elif self.model_type in ['AGARCH', 'TARCH']:
                        omega, alpha, beta, gamma = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
                        if self.model_type == 'TARCH':
                            asymmetric = gamma * (sim_return < 0) * sim_return**2
                            new_vol = omega + alpha * sim_return**2 + beta * last_vol + asymmetric
                        else:  # AGARCH
                            new_vol = omega + alpha * (sim_return - gamma)**2 + beta * last_vol
                    
                    elif self.model_type == 'FIGARCH':
                        # Simplified simulation for FIGARCH
                        omega, d, beta, phi = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
                        new_vol = omega + d * sim_return**2 + beta * last_vol
                    
                    elif self.model_type == 'IGARCH':
                        omega, alpha = self.parameters[0], self.parameters[1]
                        beta = 1.0 - alpha
                        new_vol = omega + alpha * sim_return**2 + beta * last_vol
                    
                    # Store forecast and update last_vol for next iteration
                    sim_forecasts[sim, h] = new_vol
                    last_vol = new_vol
            
            # Average across simulations
            forecasts = np.mean(sim_forecasts, axis=0)
            
        else:
            raise ValueError(f"Unknown forecasting method: {method}")
        
        return forecasts
    
    def summary(self) -> str:
        """
        Generate a summary of the estimated model.
        
        Returns
        -------
        str
            Formatted summary string
        """
        if self.parameters is None:
            return "Model not yet estimated"
        
        summary_lines = []
        summary_lines.append(f"{self.model_type}({self.p},{self.q}) with {self.distribution} distribution")
        summary_lines.append("-" * 50)
        
        param_names = self._get_parameter_names()
        
        if hasattr(self, 'std_errors') and self.std_errors is not None:
            # If we have standard errors, include them
            summary_lines.append(f"{'Parameter':<15} {'Estimate':<12} {'Std. Error':<12} {'t-value':<12} {'p-value':<12}")
            summary_lines.append("-" * 65)
            
            for i, (name, param) in enumerate(zip(param_names, self.parameters)):
                stderr = self.std_errors[i]
                t_value = param / stderr
                # Simple two-sided p-value approximation
                p_value = 2 * (1 - stats.norm.cdf(np.abs(t_value)))
                
                summary_lines.append(f"{name:<15} {param:12.6f} {stderr:12.6f} {t_value:12.6f} {p_value:12.6f}")
        else:
            # Just show parameter estimates
            summary_lines.append(f"{'Parameter':<15} {'Estimate':<12}")
            summary_lines.append("-" * 30)
            
            for name, param in zip(param_names, self.parameters):
                summary_lines.append(f"{name:<15} {param:12.6f}")
        
        summary_lines.append("-" * 50)
        summary_lines.append(f"Log-likelihood: {self.likelihood:.6f}")
        
        # Include information criteria if we have estimated the model
        if hasattr(self, 'likelihood') and self.likelihood != 0:
            n_params = len(self.parameters)
            n_obs = len(self._returns) if self._returns is not None else 0
            
            if n_obs > 0:
                aic = -2 * self.likelihood + 2 * n_params
                bic = -2 * self.likelihood + n_params * np.log(n_obs)
                
                summary_lines.append(f"AIC: {aic:.6f}")
                summary_lines.append(f"BIC: {bic:.6f}")
        
        summary_lines.append(f"Convergence: {'Yes' if self.converged else 'No'}")
        
        return "\n".join(summary_lines)
    
    def _get_parameter_names(self) -> List[str]:
        """
        Get parameter names based on model type.
        
        Returns
        -------
        List[str]
            List of parameter names
        """
        if self.model_type == 'GARCH':
            return ['omega', 'alpha', 'beta']
        elif self.model_type == 'EGARCH':
            return ['omega', 'alpha', 'gamma', 'beta']
        elif self.model_type in ['AGARCH', 'TARCH']:
            return ['omega', 'alpha', 'beta', 'gamma']
        elif self.model_type == 'FIGARCH':
            return ['omega', 'd', 'beta', 'phi']
        elif self.model_type == 'IGARCH':
            return ['omega', 'alpha']
        else:
            # Generic fallback
            return [f'param{i}' for i in range(len(self.parameters))] if self.parameters is not None else []


@dataclass
class GARCH(UnivariateGARCH):
    """
    Standard GARCH(p,q) model implementation.
    
    The GARCH model (Generalized Autoregressive Conditional Heteroskedasticity)
    models conditional variance as a function of past squared returns and
    past conditional variance.
    
    Parameters
    ----------
    p : int
        ARCH order
    q : int
        GARCH order
    distribution : Optional[str]
        Error distribution type, default is 'normal'
    """
    
    def __post_init__(self):
        """Initialize with 'GARCH' model type."""
        self.model_type = 'GARCH'
        super().__post_init__()


@dataclass
class EGARCH(UnivariateGARCH):
    """
    EGARCH(p,q) model implementation.
    
    The EGARCH model (Exponential GARCH) models the logarithm of conditional
    variance, allowing for asymmetric effects and removing the need for
    positivity constraints on parameters.
    
    Parameters
    ----------
    p : int
        ARCH order
    q : int
        GARCH order
    distribution : Optional[str]
        Error distribution type, default is 'normal'
    """
    
    def __post_init__(self):
        """Initialize with 'EGARCH' model type."""
        self.model_type = 'EGARCH'
        super().__post_init__()


@dataclass
class AGARCH(UnivariateGARCH):
    """
    AGARCH(p,q) model implementation.
    
    The AGARCH model (Asymmetric GARCH) allows for asymmetric effects of
    positive and negative returns on conditional variance through a shift
    in the squared return term.
    
    Parameters
    ----------
    p : int
        ARCH order
    q : int
        GARCH order
    distribution : Optional[str]
        Error distribution type, default is 'normal'
    """
    
    def __post_init__(self):
        """Initialize with 'AGARCH' model type."""
        self.model_type = 'AGARCH'
        super().__post_init__()


@dataclass
class TARCH(UnivariateGARCH):
    """
    TARCH(p,q) model implementation.
    
    The TARCH model (Threshold ARCH, also known as GJR-GARCH) allows for
    asymmetric effects of positive and negative returns on conditional variance
    using an indicator function for negative returns.
    
    Parameters
    ----------
    p : int
        ARCH order
    q : int
        GARCH order
    distribution : Optional[str]
        Error distribution type, default is 'normal'
    """
    
    def __post_init__(self):
        """Initialize with 'TARCH' model type."""
        self.model_type = 'TARCH'
        super().__post_init__()


@dataclass
class FIGARCH(UnivariateGARCH):
    """
    FIGARCH(p,d,q) model implementation.
    
    The FIGARCH model (Fractionally Integrated GARCH) allows for long memory
    in volatility through a fractional differencing parameter d.
    
    Parameters
    ----------
    p : int
        ARCH order
    q : int
        GARCH order
    d : float
        Fractional differencing parameter, default is 0.5
    distribution : Optional[str]
        Error distribution type, default is 'normal'
    """
    d: float = 0.5
    
    def __post_init__(self):
        """Initialize with 'FIGARCH' model type."""
        self.model_type = 'FIGARCH'
        super().__post_init__()
    
    def _get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameters for FIGARCH optimization.
        
        Returns
        -------
        np.ndarray
            Initial parameter values [omega, d, beta, phi]
        """
        # Use the provided d value
        return np.array([0.01, self.d, 0.3, 0.2])


@dataclass
class IGARCH(UnivariateGARCH):
    """
    IGARCH(p,q) model implementation.
    
    The IGARCH model (Integrated GARCH) imposes the constraint that the
    persistence parameters sum to one, implying infinite persistence of
    volatility shocks.
    
    Parameters
    ----------
    p : int
        ARCH order
    q : int
        GARCH order
    distribution : Optional[str]
        Error distribution type, default is 'normal'
    """
    
    def __post_init__(self):
        """Initialize with 'IGARCH' model type."""
        self.model_type = 'IGARCH'
        super().__post_init__()