"""
GARCH model implementations with Numba optimization for MFE Toolbox.

This module provides a unified framework for univariate GARCH models, 
supporting multiple variants including GARCH, EGARCH, GJR-GARCH (TARCH), 
AGARCH, and FIGARCH. All implementations leverage Numba's JIT compilation
for performance-critical operations and support asynchronous execution
patterns for efficient computation.

Key features:
- Class-based implementations with dataclasses for parameter representation
- Numba-optimized likelihood computation for estimation
- Flexible error distributions (normal, Student's t, GED, skewed-t)
- Asynchronous model estimation with async/await patterns
- Monte Carlo simulation capabilities
- Multi-step volatility forecasting

The module integrates with the broader MFE Toolbox architecture by utilizing
the optimization module for parameter estimation and the distributions module
for flexible error distributions.
"""

import logging
import numpy as np
import numba
from typing import Dict, Optional, Tuple, Any, List, Union, Callable
from dataclasses import dataclass, field

# Internal imports
from ..core.optimization import Optimizer
from ..core.distributions import GED, SkewedT

# Configure logger
logger = logging.getLogger(__name__)

# Global constants
VALID_GARCH_TYPES = ['GARCH', 'EGARCH', 'GJR-GARCH', 'TARCH', 'AGARCH', 'FIGARCH']
VALID_DISTRIBUTIONS = ['normal', 'student-t', 'ged', 'skewed-t']


@numba.jit(nopython=True)
def compute_garch_likelihood(returns: np.ndarray, 
                            parameters: np.ndarray, 
                            model_type_id: int, 
                            distribution_id: int) -> float:
    """
    Computes log-likelihood for GARCH model estimation using Numba optimization.
    
    This function calculates the negative log-likelihood for a given GARCH model
    specification and parameter set, optimized for performance with Numba's
    just-in-time compilation.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of financial returns
    parameters : np.ndarray
        Model parameters [omega, alpha, beta, ...] depending on model type
    model_type_id : int
        Integer ID for model type:
        0=GARCH, 1=EGARCH, 2=GJR/TARCH, 3=AGARCH, 4=FIGARCH
    distribution_id : int
        Integer ID for distribution type:
        0=normal, 1=student-t, 2=ged, 3=skewed-t
        
    Returns
    -------
    float
        Negative log-likelihood value (for minimization)
    """
    # Input validation
    T = len(returns)
    if T < 2:
        return np.inf
    
    # Initialize volatility array
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)  # Initialize with sample variance
    
    # Process different model types
    if model_type_id == 0:  # GARCH
        # For standard GARCH(1,1): sigma^2_t = omega + alpha*r^2_{t-1} + beta*sigma^2_{t-1}
        omega, alpha, beta = parameters[0], parameters[1], parameters[2]
        
        # Parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return np.inf
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
    elif model_type_id == 1:  # EGARCH
        # For EGARCH(1,1): log(sigma^2_t) = omega + beta*log(sigma^2_{t-1}) + 
        #                                    alpha*(|z_{t-1}| - E[|z|]) + gamma*z_{t-1}
        omega, alpha, gamma, beta = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Parameter constraints
        if beta >= 1:
            return np.inf
            
        # Initialize log-variance process
        h = np.zeros(T)
        h[0] = np.log(sigma2[0])
        
        for t in range(1, T):
            z_t_1 = returns[t-1] / np.sqrt(np.exp(h[t-1]))
            h[t] = omega + beta * h[t-1] + alpha * (np.abs(z_t_1) - np.sqrt(2/np.pi)) + gamma * z_t_1
            sigma2[t] = np.exp(h[t])
            
    elif model_type_id == 2:  # GJR-GARCH / TARCH
        # For GJR-GARCH/TARCH: sigma^2_t = omega + alpha*r^2_{t-1} + gamma*I(r_{t-1}<0)*r^2_{t-1} + beta*sigma^2_{t-1}
        omega, alpha, beta, gamma = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0 or alpha + beta + 0.5*gamma >= 1:
            return np.inf
            
        for t in range(1, T):
            # Indicator function for negative returns
            indicator = 1.0 if returns[t-1] < 0 else 0.0
            sigma2[t] = omega + alpha * returns[t-1]**2 + gamma * indicator * returns[t-1]**2 + beta * sigma2[t-1]
            
    elif model_type_id == 3:  # AGARCH
        # For AGARCH: sigma^2_t = omega + alpha*(r_{t-1} - theta)^2 + beta*sigma^2_{t-1}
        omega, alpha, beta, theta = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return np.inf
            
        for t in range(1, T):
            sigma2[t] = omega + alpha * (returns[t-1] - theta)**2 + beta * sigma2[t-1]
            
    elif model_type_id == 4:  # FIGARCH
        # For FIGARCH: sigma^2_t = omega + beta*sigma^2_{t-1} + [1 - beta*L - (1-phi*L)*(1-L)^d]*r^2_t
        # Simplified implementation for Numba compatibility
        omega, beta, phi, d = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Parameter constraints
        if omega <= 0 or beta < 0 or phi < 0 or d <= 0 or d >= 1:
            return np.inf
            
        # Truncation order for fractional differencing
        trunc_lag = min(1000, T-1)
        weights = np.zeros(trunc_lag+1)
        weights[0] = phi  # Initialize with first weight
        
        # Compute fractional differencing weights recursively
        for i in range(1, trunc_lag+1):
            weights[i] = (i - d - 1) / i * weights[i-1]
            
        for t in range(1, T):
            max_lag = min(t, trunc_lag)
            weighted_sum = 0.0
            
            for i in range(max_lag):
                weighted_sum += weights[i] * returns[t-i-1]**2
                
            sigma2[t] = omega + beta * sigma2[t-1] + weighted_sum
    else:
        # Unknown model type
        return np.inf
    
    # Ensure positive variances
    sigma2 = np.maximum(sigma2, 1e-6)
    
    # Compute log-likelihood based on distribution
    ll = 0.0
    
    if distribution_id == 0:  # Normal
        # Normal (Gaussian) distribution
        ll = -0.5 * T * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        
    elif distribution_id == 1:  # Student-t
        # Student's t distribution
        if len(parameters) < 4:
            return np.inf
            
        nu = parameters[-1]  # Last parameter is degrees of freedom
        
        # Parameter constraint for degrees of freedom
        if nu <= 2:
            return np.inf
            
        # Student's t log-likelihood (simplified for Numba compatibility)
        const = 0.5 * np.log((nu - 2) / np.pi) - 0.5 * np.log(nu)
        ll = const * T - 0.5 * np.sum(np.log(sigma2) + (nu + 1) * np.log(1 + returns**2 / (sigma2 * (nu - 2))))
    
    elif distribution_id == 2:  # GED
        # Generalized Error Distribution
        if len(parameters) < 4:
            return np.inf
            
        nu = parameters[-1]  # Last parameter is shape parameter
        
        if nu <= 0:
            return np.inf
            
        # For Numba compatibility, simplified approximation of GED log-likelihood
        lambda_term = np.sqrt(np.exp(np.log(2) - 2 * np.log(3) / nu) * 
                             np.exp(np.log(3) / nu - np.log(1) / nu))
        const_term = np.log(nu) - np.log(2 * lambda_term) - np.log(np.exp(np.log(1) / nu))
        
        ll = T * const_term - np.sum(np.log(sigma2) / 2 + np.power(np.abs(returns) / np.sqrt(sigma2) / lambda_term, nu))
        
    elif distribution_id == 3:  # Skewed-t
        # Hansen's (1994) Skewed t-distribution
        if len(parameters) < 5:
            return np.inf
            
        nu = parameters[-2]  # Second last parameter is degrees of freedom
        lambda_ = parameters[-1]  # Last parameter is skewness
        
        if nu <= 2 or lambda_ <= -1 or lambda_ >= 1:
            return np.inf
            
        # For Numba compatibility, simplified approximation of skewed-t log-likelihood
        a = 4 * lambda_ * ((nu - 2) / (nu - 1))
        b = np.sqrt(1 + a * a)
        
        # Approximate implementation that works with Numba
        ll = -0.5 * np.sum(np.log(sigma2))
        
        for i in range(T):
            z = returns[i] / np.sqrt(sigma2[i])
            if z < -a/b:
                term = 1 + (1 / (nu - 2)) * np.square(b * z + a)
                ll += np.log(b) - 0.5 * (nu + 1) * np.log(term)
            else:
                term = 1 + (1 / (nu - 2)) * np.square(b * z + a)
                ll += np.log(b) - 0.5 * (nu + 1) * np.log(term)
    
    else:
        # Other distributions not directly implemented in Numba
        # For simplicity, fall back to normal in JIT context
        ll = -0.5 * T * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
    
    # Return negative log-likelihood for minimization
    return -ll


@numba.jit(nopython=True)
def simulate_garch(n_samples: int, 
                  parameters: np.ndarray, 
                  model_type_id: int, 
                  distribution_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates simulated data from specified GARCH process.
    
    This function simulates returns and conditional volatility from a GARCH process
    with specified parameters and innovation distribution using Numba for performance.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    parameters : np.ndarray
        Model parameters specific to the GARCH variant
    model_type_id : int
        Integer ID for model type:
        0=GARCH, 1=EGARCH, 2=GJR/TARCH, 3=AGARCH, 4=FIGARCH
    distribution_id : int
        Integer ID for distribution type:
        0=normal, 1=student-t, 2=ged, 3=skewed-t
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (returns, volatility) arrays
    """
    # Validate inputs
    if n_samples <= 0:
        # In Numba context, we can't raise exceptions, so return empty arrays
        return np.array([0.0]), np.array([0.0])
    
    # Initialize arrays for returns and volatility
    returns = np.zeros(n_samples)
    sigma2 = np.zeros(n_samples)
    
    # Initial variance depends on model type
    if model_type_id == 0:  # GARCH
        omega, alpha, beta = parameters[0], parameters[1], parameters[2]
        # Unconditional variance for GARCH
        sigma2[0] = omega / (1 - alpha - beta)
    elif model_type_id == 1:  # EGARCH
        # For EGARCH, just use a reasonable initial value
        sigma2[0] = 1.0
    else:
        # For other models, use the first parameter as a scale factor
        sigma2[0] = parameters[0] * 10.0
    
    # Generate innovations based on distribution
    if distribution_id == 0:  # Normal
        # Standard normal innovations
        z = np.random.normal(0, 1, n_samples)
    elif distribution_id == 1:  # Student-t
        # Student's t innovations
        nu = parameters[-1]  # Last parameter is degrees of freedom
        if nu <= 2:
            nu = 5.0  # Fallback to reasonable value for simulation
        
        # Generate standardized Student's t random variables
        # For Numba compatibility, we'll use a simple transformation of uniform variables
        u = np.random.random(n_samples)
        z = np.zeros(n_samples)
        for i in range(n_samples):
            # Simplified t-distributed random generation
            # In practice would use a better approximation
            normal = np.random.normal(0, 1)
            chi2 = 0.0
            for j in range(int(nu)):
                chi2 += np.random.normal(0, 1)**2
            z[i] = normal / np.sqrt(chi2 / nu)
            
        # Standardize to unit variance
        z = z / np.sqrt(nu / (nu - 2))
    else:
        # Default to normal for other distributions in JIT context
        z = np.random.normal(0, 1, n_samples)
    
    # Calculate returns and volatility based on model type
    if model_type_id == 0:  # GARCH
        omega, alpha, beta = parameters[0], parameters[1], parameters[2]
        
        # Simulate GARCH(1,1) process
        returns[0] = np.sqrt(sigma2[0]) * z[0]
        
        for t in range(1, n_samples):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            returns[t] = np.sqrt(sigma2[t]) * z[t]
            
    elif model_type_id == 1:  # EGARCH
        omega, alpha, gamma, beta = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Initialize log-variance
        h = np.zeros(n_samples)
        h[0] = np.log(sigma2[0])
        
        # Simulate EGARCH(1,1) process
        returns[0] = np.sqrt(np.exp(h[0])) * z[0]
        
        for t in range(1, n_samples):
            z_t_1 = returns[t-1] / np.sqrt(np.exp(h[t-1]))
            h[t] = omega + beta * h[t-1] + alpha * (np.abs(z_t_1) - np.sqrt(2/np.pi)) + gamma * z_t_1
            sigma2[t] = np.exp(h[t])
            returns[t] = np.sqrt(sigma2[t]) * z[t]
            
    elif model_type_id == 2:  # GJR-GARCH / TARCH
        omega, alpha, beta, gamma = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Simulate GJR-GARCH/TARCH process
        returns[0] = np.sqrt(sigma2[0]) * z[0]
        
        for t in range(1, n_samples):
            indicator = 1.0 if returns[t-1] < 0 else 0.0
            sigma2[t] = omega + alpha * returns[t-1]**2 + gamma * indicator * returns[t-1]**2 + beta * sigma2[t-1]
            returns[t] = np.sqrt(sigma2[t]) * z[t]
            
    elif model_type_id == 3:  # AGARCH
        omega, alpha, beta, theta = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Simulate AGARCH process
        returns[0] = np.sqrt(sigma2[0]) * z[0]
        
        for t in range(1, n_samples):
            sigma2[t] = omega + alpha * (returns[t-1] - theta)**2 + beta * sigma2[t-1]
            returns[t] = np.sqrt(sigma2[t]) * z[t]
            
    elif model_type_id == 4:  # FIGARCH
        omega, beta, phi, d = parameters[0], parameters[1], parameters[2], parameters[3]
        
        # Truncation order for fractional differencing
        trunc_lag = min(1000, n_samples-1)
        weights = np.zeros(trunc_lag+1)
        weights[0] = phi  # Initialize with first weight
        
        # Compute fractional differencing weights recursively
        for i in range(1, trunc_lag+1):
            weights[i] = (i - d - 1) / i * weights[i-1]
            
        # Simulate FIGARCH process
        returns[0] = np.sqrt(sigma2[0]) * z[0]
        
        for t in range(1, n_samples):
            max_lag = min(t, trunc_lag)
            weighted_sum = 0.0
            
            for i in range(max_lag):
                weighted_sum += weights[i] * returns[t-i-1]**2
                
            sigma2[t] = omega + beta * sigma2[t-1] + weighted_sum
            returns[t] = np.sqrt(sigma2[t]) * z[t]
    else:
        # Default to GARCH(1,1) for unknown model types
        omega, alpha, beta = parameters[0], parameters[1], parameters[2]
        
        # Simulate GARCH(1,1) process
        returns[0] = np.sqrt(sigma2[0]) * z[0]
        
        for t in range(1, n_samples):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            returns[t] = np.sqrt(sigma2[t]) * z[t]
    
    return returns, sigma2


@dataclass
class GARCHModel:
    """
    Base class for GARCH model implementations with async estimation support.
    
    This class provides a unified interface for estimating, forecasting, and
    simulating various GARCH model variants. It leverages Numba for performance
    optimization and supports asynchronous execution patterns for efficient
    computation.
    
    Parameters
    ----------
    p : int
        Order of GARCH term
    q : int
        Order of ARCH term
    model_type : str
        Type of GARCH model (e.g., 'GARCH', 'EGARCH', 'GJR-GARCH')
    distribution : str
        Error distribution specification (e.g., 'normal', 'student-t')
        
    Attributes
    ----------
    parameters : np.ndarray
        Estimated model parameters
    std_errors : np.ndarray
        Standard errors of the estimated parameters
    likelihood : float
        Log-likelihood value at the optimum
    converged : bool
        Flag indicating whether optimization converged
        
    Methods
    -------
    async_fit
        Asynchronously estimate model parameters
    forecast
        Generate volatility forecasts
    simulate
        Simulate returns from the estimated model
    """
    
    p: int
    q: int
    model_type: str
    distribution: str = 'normal'
    
    # Attributes to be set during/after estimation
    parameters: np.ndarray = None
    std_errors: np.ndarray = None
    likelihood: float = None
    converged: bool = False
    
    # Additional state variables
    last_returns: np.ndarray = field(default=None, init=False)
    last_variance: float = field(default=None, init=False)
    
    def __post_init__(self):
        """
        Initialize GARCH model with specified order and type.
        
        Validates model specification and prepares for estimation.
        """
        # Validate model type
        if self.model_type.upper() not in [t.upper() for t in VALID_GARCH_TYPES]:
            raise ValueError(f"Invalid model type: {self.model_type}. "
                             f"Valid options are: {', '.join(VALID_GARCH_TYPES)}")
        
        # Validate distribution type
        if self.distribution.lower() not in [d.lower() for d in VALID_DISTRIBUTIONS]:
            raise ValueError(f"Invalid distribution: {self.distribution}. "
                             f"Valid options are: {', '.join(VALID_DISTRIBUTIONS)}")
        
        # Validate orders
        if self.p < 0 or self.q < 0:
            raise ValueError("Orders p and q must be non-negative integers")
        
        if self.p == 0 and self.q == 0:
            raise ValueError("At least one of p or q must be positive")
        
        # Initialize optimizer
        self.optimizer = Optimizer()
        
        # Determine number of parameters based on model type
        if self.model_type.upper() == 'GARCH':
            self.n_params = 3  # omega, alpha, beta
        elif self.model_type.upper() in ['EGARCH', 'GJR-GARCH', 'TARCH', 'AGARCH', 'FIGARCH']:
            self.n_params = 4  # omega, alpha, beta, gamma (or theta or d)
        else:
            self.n_params = 3  # Default for unknown models
            
        # Add parameter for distribution if not normal
        if self.distribution.lower() == 'student-t':
            self.n_params += 1  # Add degrees of freedom parameter
        elif self.distribution.lower() == 'ged':
            self.n_params += 1  # Add shape parameter
        elif self.distribution.lower() == 'skewed-t':
            self.n_params += 2  # Add degrees of freedom and skewness parameters
            
        # Initialize parameter arrays
        self.parameters = np.zeros(self.n_params)
        self.std_errors = np.zeros(self.n_params)
        
        # Map model type to ID for Numba functions
        self.model_type_id = self._get_model_type_id()
        
        # Map distribution type to ID for Numba functions
        self.distribution_id = self._get_distribution_id()
        
        logger.info(f"Initialized {self.model_type} model with {self.distribution} distribution")
    
    def _get_model_type_id(self) -> int:
        """
        Map string model type to integer ID for Numba functions.
        
        Returns
        -------
        int
            Integer ID representing the model type
        """
        model_type_upper = self.model_type.upper()
        if model_type_upper == 'GARCH':
            return 0
        elif model_type_upper == 'EGARCH':
            return 1
        elif model_type_upper in ['GJR-GARCH', 'TARCH']:
            return 2
        elif model_type_upper == 'AGARCH':
            return 3
        elif model_type_upper == 'FIGARCH':
            return 4
        else:
            logger.warning(f"Unknown model type {self.model_type}, defaulting to GARCH")
            return 0
    
    def _get_distribution_id(self) -> int:
        """
        Map string distribution type to integer ID for Numba functions.
        
        Returns
        -------
        int
            Integer ID representing the distribution type
        """
        dist_lower = self.distribution.lower()
        if dist_lower == 'normal':
            return 0
        elif dist_lower == 'student-t':
            return 1
        elif dist_lower == 'ged':
            return 2
        elif dist_lower == 'skewed-t':
            return 3
        else:
            logger.warning(f"Unknown distribution {self.distribution}, defaulting to normal")
            return 0
    
    async def async_fit(self, returns: np.ndarray) -> bool:
        """
        Asynchronously estimate GARCH model parameters.
        
        This method performs maximum likelihood estimation of the GARCH model
        parameters using Numba-optimized likelihood calculation and SciPy's
        optimization routines, executed asynchronously for better responsiveness.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of financial returns
            
        Returns
        -------
        bool
            True if optimization converged, False otherwise
        """
        # Input validation
        if not isinstance(returns, np.ndarray):
            returns = np.asarray(returns)
            
        if returns.ndim != 1:
            raise ValueError("Returns must be a 1-dimensional array")
            
        if len(returns) < 10:
            raise ValueError("Insufficient data for estimation: requires at least 10 observations")
            
        # Store last observations for forecasting
        self.last_returns = returns[-self.p:] if self.p > 0 else np.array([])
        self.last_variance = np.var(returns)
            
        # Set initial parameter values
        initial_params = np.zeros(self.n_params)
        
        # Common initial values
        var_r = np.var(returns)
        
        if self.model_type.upper() == 'GARCH':
            # omega, alpha, beta
            initial_params[0] = 0.05 * var_r
            initial_params[1] = 0.1
            initial_params[2] = 0.8
            
        elif self.model_type.upper() == 'EGARCH':
            # omega, alpha, gamma, beta
            initial_params[0] = np.log(0.05 * var_r)
            initial_params[1] = 0.1
            initial_params[2] = -0.05
            initial_params[3] = 0.9
            
        elif self.model_type.upper() in ['GJR-GARCH', 'TARCH']:
            # omega, alpha, beta, gamma
            initial_params[0] = 0.05 * var_r
            initial_params[1] = 0.05
            initial_params[2] = 0.8
            initial_params[3] = 0.05
            
        elif self.model_type.upper() == 'AGARCH':
            # omega, alpha, beta, theta
            initial_params[0] = 0.05 * var_r
            initial_params[1] = 0.1
            initial_params[2] = 0.8
            initial_params[3] = 0.0
            
        elif self.model_type.upper() == 'FIGARCH':
            # omega, beta, phi, d
            initial_params[0] = 0.05 * var_r
            initial_params[1] = 0.2
            initial_params[2] = 0.4
            initial_params[3] = 0.4
            
        # Add distribution parameters if needed
        if self.distribution.lower() == 'student-t':
            initial_params[-1] = 8.0  # Typical degrees of freedom
        elif self.distribution.lower() == 'ged':
            initial_params[-1] = 1.5  # Typical GED shape parameter
        elif self.distribution.lower() == 'skewed-t':
            initial_params[-2] = 8.0  # Degrees of freedom
            initial_params[-1] = 0.0  # No skewness
        
        logger.debug(f"Starting estimation with initial parameters: {initial_params}")
        
        try:
            # Perform asynchronous optimization
            optimal_params, likelihood = await self.optimizer.async_optimize(
                returns,  # data
                initial_params,  # initial parameters
                self.model_type,  # model type (string)
                self.distribution  # distribution type (string)
            )
            
            # Store results
            self.parameters = optimal_params
            self.likelihood = likelihood
            self.converged = self.optimizer.converged
            
            # Update last variance based on estimated parameters
            if self.converged:
                # Compute variance at the last observation
                if self.model_type.upper() == 'GARCH':
                    omega, alpha, beta = self.parameters[0], self.parameters[1], self.parameters[2]
                    self.last_variance = omega + alpha * returns[-1]**2 + beta * np.var(returns)
                else:
                    # For other models, use sample variance as approximation
                    self.last_variance = np.var(returns)
            
            logger.info(f"Estimation {'converged' if self.converged else 'did not converge'} "
                        f"with likelihood {self.likelihood:.4f}")
            
            if self.converged:
                logger.debug(f"Estimated parameters: {self.parameters}")
            
            return self.converged
            
        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            self.converged = False
            return False
    
    def forecast(self, horizon: int) -> np.ndarray:
        """
        Generate volatility forecasts from estimated model.
        
        This method computes multi-step ahead volatility forecasts based on
        the estimated GARCH model parameters.
        
        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast
            
        Returns
        -------
        np.ndarray
            Array of volatility forecasts
        """
        # Check if model has been estimated
        if self.parameters is None or not self.converged:
            raise RuntimeError("Model must be successfully estimated before forecasting")
            
        # Input validation
        if horizon <= 0:
            raise ValueError("Forecast horizon must be a positive integer")
            
        # Check if we have last returns and variance
        if self.last_returns is None or self.last_variance is None:
            raise RuntimeError("Model state not properly initialized, rerun estimation")
            
        # Initialize forecast array
        forecasts = np.zeros(horizon)
        
        # Get last return (or zero if not available)
        last_return = self.last_returns[-1] if len(self.last_returns) > 0 else 0.0
        
        # Extract parameters based on model type
        if self.model_type.upper() == 'GARCH':
            omega, alpha, beta = self.parameters[0], self.parameters[1], self.parameters[2]
            
            # Compute long-run variance (unconditional variance)
            long_run_var = omega / (1 - alpha - beta)
            
            # Generate forecasts recursively
            forecasts[0] = omega + alpha * last_return**2 + beta * self.last_variance
            
            for h in range(1, horizon):
                forecasts[h] = omega + (alpha + beta) * forecasts[h-1]
                
        elif self.model_type.upper() == 'EGARCH':
            omega, alpha, gamma, beta = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
            
            # Compute last standardized return
            z_last = last_return / np.sqrt(self.last_variance)
            
            # Initialize with one-step forecast in log-variance
            log_var = omega + beta * np.log(self.last_variance) + \
                      alpha * (np.abs(z_last) - np.sqrt(2/np.pi)) + gamma * z_last
            
            # First forecast
            forecasts[0] = np.exp(log_var)
            
            # Generate remaining forecasts - EGARCH has constant expected future z contributions
            expected_abs_z = np.sqrt(2/np.pi)
            for h in range(1, horizon):
                log_var = omega + beta * np.log(forecasts[h-1]) + alpha * expected_abs_z
                forecasts[h] = np.exp(log_var)
                
        elif self.model_type.upper() in ['GJR-GARCH', 'TARCH']:
            omega, alpha, beta, gamma = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
            
            # Compute indicator for last return
            indicator = 1.0 if last_return < 0 else 0.0
            
            # First forecast
            forecasts[0] = omega + alpha * last_return**2 + \
                           gamma * indicator * last_return**2 + beta * self.last_variance
            
            # For multi-step forecasts, the expected value of the indicator is 0.5
            # assuming symmetric distribution
            for h in range(1, horizon):
                forecasts[h] = omega + (alpha + 0.5 * gamma) * forecasts[h-1] + beta * forecasts[h-1]
                
        elif self.model_type.upper() == 'AGARCH':
            omega, alpha, beta, theta = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
            
            # First forecast
            forecasts[0] = omega + alpha * (last_return - theta)**2 + beta * self.last_variance
            
            # For multi-step forecasts
            for h in range(1, horizon):
                # The expected value of (r-theta)^2 is sigma^2 + theta^2
                forecasts[h] = omega + alpha * (forecasts[h-1] + theta**2) + beta * forecasts[h-1]
                
        elif self.model_type.upper() == 'FIGARCH':
            omega, beta, phi, d = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
            
            # For FIGARCH, we need to compute the truncated fractional differencing weights
            trunc_lag = min(1000, horizon)
            weights = np.zeros(trunc_lag+1)
            weights[0] = phi
            
            # Compute fractional differencing weights recursively
            for i in range(1, trunc_lag+1):
                weights[i] = (i - d - 1) / i * weights[i-1]
                
            # Initialize with empirical values for recently observed returns and variances
            # (This is a simplification; in practice would need more historical data)
            
            # Generate forecasts
            for h in range(horizon):
                # For simplicity, we'll use a constant term for historical contributions
                hist_contrib = omega / (1 - beta)
                
                forecasts[h] = hist_contrib + beta * self.last_variance
                
        else:
            # Default to GARCH(1,1) for unknown models
            omega, alpha, beta = self.parameters[0], self.parameters[1], self.parameters[2]
            
            # Generate forecasts recursively
            forecasts[0] = omega + alpha * last_return**2 + beta * self.last_variance
            
            for h in range(1, horizon):
                forecasts[h] = omega + (alpha + beta) * forecasts[h-1]
        
        return forecasts
    
    def simulate(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simulated data from estimated model.
        
        This method simulates return series and volatility paths from the
        estimated GARCH model using the specified error distribution.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (returns, volatility) arrays
        """
        # Check if model has been estimated
        if self.parameters is None or not self.converged:
            raise RuntimeError("Model must be successfully estimated before simulation")
            
        # Input validation
        if n_samples <= 0:
            raise ValueError("Number of samples must be a positive integer")
            
        # Call Numba-optimized simulation function
        returns, volatility = simulate_garch(
            n_samples, 
            self.parameters, 
            self.model_type_id, 
            self.distribution_id
        )
        
        return returns, volatility