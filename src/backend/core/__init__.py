"""
Core statistical and computational module for MFE Toolbox.

This module provides access to the core statistical and computational functionality
of the MFE Toolbox including distributions, optimization routines, bootstrap methods,
and statistical tests through a unified interface.

The module serves as the main entry point for accessing the core functionality
of the toolbox and enables a streamlined syntax for common econometric operations.
"""

import logging
from typing import Union, Tuple, Optional, Dict, Any, Callable
import numpy as np
import scipy.stats as stats

# Configure logger
logger = logging.getLogger(__name__)

# Version information
__version__ = '4.0.0'

# Import distribution components
from .distributions import (
    GED,
    SkewedT,
    jarque_bera,
    kurtosis,
    skewness
)

# Import optimization components
from .optimization import (
    optimize_garch,
    Optimizer
)

# Import bootstrap components
from .bootstrap import (
    block_bootstrap,
    stationary_bootstrap,
    Bootstrap
)

# Import test components
from .tests import (
    ljung_box,
    arch_lm,
    durbin_watson,
    white_test,
    breusch_pagan,
    validate_and_prepare_residuals
)

# Define wrappers for functions not directly exposed by modules

def estimate_distribution(data: np.ndarray, distribution_type: str, **kwargs) -> Tuple[Any, float]:
    """
    Estimate parameters of a specified distribution.
    
    Parameters
    ----------
    data : np.ndarray
        Input data for distribution fitting
    distribution_type : str
        Type of distribution to fit ('normal', 'student-t', 'ged', 'skewed-t')
    **kwargs
        Additional parameters for specific distributions
        
    Returns
    -------
    Tuple[Any, float]
        Tuple containing (distribution_object, log_likelihood)
        
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.core import estimate_distribution
    >>> data = np.random.randn(1000)
    >>> ged_dist, loglik = estimate_distribution(data, 'ged', nu=1.5)
    >>> print(f"Log-likelihood: {loglik:.2f}")
    """
    logger.info(f"Estimating {distribution_type} distribution parameters")
    
    if distribution_type.lower() == 'ged':
        from scipy import optimize
        
        # Define objective function (negative log-likelihood)
        def neg_loglik(params):
            nu = params[0]
            try:
                dist = GED(nu=nu)
                return -dist.loglikelihood(data)
            except ValueError:
                return np.inf
        
        # Initial guess
        initial_nu = kwargs.get('nu', 2.0)
        
        # Optimize
        bounds = [(1.1, 100.0)]  # nu > 1 for finite variance
        result = optimize.minimize(neg_loglik, [initial_nu], bounds=bounds)
        
        if result.success:
            optimal_nu = result.x[0]
            dist = GED(nu=optimal_nu)
            loglik = -result.fun
            logger.debug(f"GED estimation succeeded with nu={optimal_nu:.4f}")
            return dist, loglik
        else:
            # Fallback to initial guess if optimization fails
            dist = GED(nu=initial_nu)
            loglik = dist.loglikelihood(data)
            logger.warning(f"GED estimation did not converge, using initial nu={initial_nu}")
            return dist, loglik
        
    elif distribution_type.lower() == 'skewed-t':
        from scipy import optimize
        
        # Define objective function (negative log-likelihood)
        def neg_loglik(params):
            nu, lambda_ = params
            try:
                dist = SkewedT(nu=nu, lambda_=lambda_)
                return -dist.loglikelihood(data)
            except ValueError:
                return np.inf
        
        # Initial guess
        initial_nu = kwargs.get('nu', 5.0)
        initial_lambda = kwargs.get('lambda_', 0.0)
        
        # Optimize
        bounds = [(2.1, 100.0), (-0.99, 0.99)]  # nu > 2, |lambda| < 1
        result = optimize.minimize(neg_loglik, [initial_nu, initial_lambda], bounds=bounds)
        
        if result.success:
            optimal_nu, optimal_lambda = result.x
            dist = SkewedT(nu=optimal_nu, lambda_=optimal_lambda)
            loglik = -result.fun
            logger.debug(f"Skewed-t estimation succeeded with nu={optimal_nu:.4f}, lambda={optimal_lambda:.4f}")
            return dist, loglik
        else:
            # Fallback to initial guess if optimization fails
            dist = SkewedT(nu=initial_nu, lambda_=initial_lambda)
            loglik = dist.loglikelihood(data)
            logger.warning(f"Skewed-t estimation did not converge, using initial values")
            return dist, loglik
        
    elif distribution_type.lower() == 'normal':
        # Normal distribution estimation is straightforward
        mean = np.mean(data)
        std = np.std(data)
        loglik = np.sum(stats.norm.logpdf(data, mean, std))
        logger.debug(f"Normal estimation: mean={mean:.4f}, std={std:.4f}")
        return (mean, std), loglik
        
    elif distribution_type.lower() == 'student-t':
        from scipy import optimize
        
        # Define objective function (negative log-likelihood)
        def neg_loglik(params):
            df = params[0]
            try:
                # Compute t-distribution log-likelihood
                logpdf = stats.t.logpdf(data, df)
                return -np.sum(logpdf)
            except:
                return np.inf
        
        # Initial guess
        initial_df = kwargs.get('df', 5.0)
        
        # Optimize
        bounds = [(2.1, 100.0)]  # df > 2 for finite variance
        result = optimize.minimize(neg_loglik, [initial_df], bounds=bounds)
        
        if result.success:
            optimal_df = result.x[0]
            loglik = -result.fun
            logger.debug(f"Student-t estimation succeeded with df={optimal_df:.4f}")
            return optimal_df, loglik
        else:
            # Fallback to initial guess
            loglik = -neg_loglik([initial_df])
            logger.warning(f"Student-t estimation did not converge, using df={initial_df}")
            return initial_df, loglik
    else:
        logger.warning(f"Distribution type {distribution_type} not directly supported, using normal approximation")
        
        # Fallback to normal distribution
        mean = np.mean(data)
        std = np.std(data)
        loglik = np.sum(stats.norm.logpdf(data, mean, std))
        return (mean, std), loglik


class BootstrapInference:
    """
    Statistical inference using bootstrap methods.
    
    This class provides methods for computing confidence intervals and standard errors
    using bootstrap resampling techniques for time series data.
    
    Attributes
    ----------
    bootstrap : Bootstrap
        Underlying bootstrap implementation
    
    Methods
    -------
    compute_confidence_interval
        Compute bootstrap confidence interval for a statistic
    compute_standard_errors
        Compute bootstrap standard errors for model parameters
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.core import BootstrapInference
    >>> data = np.random.randn(1000)
    >>> bootstrap = BootstrapInference(method="block")
    >>> # Define a statistic function (e.g., mean)
    >>> async def compute_ci():
    ...     ci = await bootstrap.compute_confidence_interval(data, np.mean)
    ...     print(f"95% CI for mean: ({ci[0]:.4f}, {ci[1]:.4f})")
    """
    
    def __init__(self, method: str = "block", options: Optional[dict] = None):
        """
        Initialize BootstrapInference with the specified method and options.
        
        Parameters
        ----------
        method : str, default "block"
            Bootstrap method to use, either "block" or "stationary"
        options : Optional[dict], default None
            Configuration options for the bootstrap method
        """
        self.bootstrap = Bootstrap(method=method, options=options)
        logger.debug(f"Initialized BootstrapInference with method={method}")
    
    async def compute_confidence_interval(
        self, 
        data: np.ndarray, 
        statistic_fn: Callable, 
        num_bootstraps: int = 1000, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a statistic.
        
        Parameters
        ----------
        data : np.ndarray
            Original time series data
        statistic_fn : Callable
            Function that computes the statistic of interest
        num_bootstraps : int, default 1000
            Number of bootstrap samples to generate
        confidence_level : float, default 0.95
            Confidence level for the interval
            
        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds of the confidence interval
        """
        # Generate bootstrap samples
        bootstrap_samples = await self.bootstrap.async_bootstrap(data, num_bootstraps)
        
        # Compute statistic for each bootstrap sample
        bootstrap_statistics = np.array([statistic_fn(sample) for sample in bootstrap_samples])
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
        upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
        
        logger.debug(f"Computed {confidence_level*100}% confidence interval: ({lower_bound:.4f}, {upper_bound:.4f})")
        return lower_bound, upper_bound
    
    async def compute_standard_errors(
        self, 
        data: np.ndarray, 
        estimator_fn: Callable, 
        num_bootstraps: int = 1000
    ) -> np.ndarray:
        """
        Compute bootstrap standard errors for model parameters.
        
        Parameters
        ----------
        data : np.ndarray
            Original time series data
        estimator_fn : Callable
            Function that estimates model parameters
        num_bootstraps : int, default 1000
            Number of bootstrap samples to generate
            
        Returns
        -------
        np.ndarray
            Standard errors for each parameter
        """
        # Generate bootstrap samples
        bootstrap_samples = await self.bootstrap.async_bootstrap(data, num_bootstraps)
        
        # Compute parameters for each bootstrap sample
        bootstrap_params = np.array([estimator_fn(sample) for sample in bootstrap_samples])
        
        # Compute standard deviation of parameter estimates
        std_errors = np.std(bootstrap_params, axis=0)
        
        logger.debug(f"Computed bootstrap standard errors: {std_errors}")
        return std_errors


def test_normality(data: np.ndarray) -> Tuple[float, float]:
    """
    Test for normality of residuals using Jarque-Bera test.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data or residuals to test
        
    Returns
    -------
    Tuple[float, float]
        Test statistic and p-value
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.core import test_normality
    >>> data = np.random.randn(1000)
    >>> stat, p_value = test_normality(data)
    >>> print(f"Jarque-Bera statistic: {stat:.4f}, p-value: {p_value:.4f}")
    """
    return jarque_bera(data)


def test_autocorrelation(residuals: np.ndarray, lags: int = 10) -> Tuple[float, float]:
    """
    Test for autocorrelation in residuals using Ljung-Box test.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression or time series model
    lags : int, default 10
        Number of lags to include in the test
        
    Returns
    -------
    Tuple[float, float]
        Test statistic and p-value
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.core import test_autocorrelation
    >>> residuals = np.random.randn(1000)
    >>> stat, p_value = test_autocorrelation(residuals, lags=15)
    >>> print(f"Ljung-Box Q({15}): {stat:.4f}, p-value: {p_value:.4f}")
    """
    # Validate and prepare inputs
    residuals = validate_and_prepare_residuals(residuals)
    
    if not isinstance(lags, int) or lags <= 0:
        raise ValueError(f"lags must be a positive integer, got {lags}")
    
    if lags >= residuals.shape[0]:
        raise ValueError(f"lags ({lags}) must be less than sample size ({residuals.shape[0]})")
    
    # Call Numba-optimized function
    q_stat, p_value_approx = ljung_box(residuals, lags)
    
    # Check for error codes
    if q_stat < 0:
        raise ValueError("Error computing Ljung-Box statistic")
    
    # Calculate accurate p-value using scipy
    p_value = stats.chi2.sf(q_stat, lags)
    
    logger.debug(f"Ljung-Box test: Q({lags}) = {q_stat:.4f}, p-value = {p_value:.4f}")
    
    return q_stat, p_value


def test_heteroskedasticity(residuals: np.ndarray, regressors: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Test for heteroskedasticity in residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a regression or time series model
    regressors : Optional[np.ndarray]
        Matrix of regressors. If None, ARCH-LM test is used with 5 lags.
        
    Returns
    -------
    Tuple[float, float]
        Test statistic and p-value
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.core import test_heteroskedasticity
    >>> residuals = np.random.randn(1000)
    >>> # ARCH-LM test (no regressors)
    >>> stat1, p_value1 = test_heteroskedasticity(residuals)
    >>> print(f"ARCH-LM: {stat1:.4f}, p-value: {p_value1:.4f}")
    >>>
    >>> # White test (with regressors)
    >>> X = np.random.randn(1000, 3)  # Example regressors
    >>> stat2, p_value2 = test_heteroskedasticity(residuals, X)
    >>> print(f"White: {stat2:.4f}, p-value: {p_value2:.4f}")
    """
    # Validate input
    residuals = validate_and_prepare_residuals(residuals)
    
    if regressors is None:
        # Use ARCH-LM test with 5 lags
        lags = 5
        try:
            stat, p_value = arch_lm(residuals, lags)
            
            # In case the Numba function returns an error code
            if stat < 0:
                raise ValueError("Error in ARCH-LM computation")
            
            logger.debug(f"ARCH-LM test: statistic = {stat:.4f}, p-value = {p_value:.4f}")
            return stat, p_value
        except Exception as e:
            logger.error(f"Error in heteroskedasticity test: {str(e)}")
            raise
    else:
        # Use White's test if regressors are provided
        try:
            # Ensure regressors is a 2D array
            if not isinstance(regressors, np.ndarray):
                raise TypeError(f"regressors must be a NumPy array, got {type(regressors).__name__}")
                
            if regressors.ndim == 1:
                regressors = regressors.reshape(-1, 1)
            
            stat, p_value = white_test(residuals, regressors)
            
            # In case the Numba function returns an error code
            if stat < 0:
                raise ValueError("Error in White test computation")
            
            logger.debug(f"White test: statistic = {stat:.4f}, p-value = {p_value:.4f}")
            return stat, p_value
        except Exception as e:
            logger.error(f"Error in heteroskedasticity test: {str(e)}")
            raise


def test_stationarity(data: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    """
    Test for stationarity of time series data using ADF test.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data to test
        
    Returns
    -------
    Tuple[float, float, Dict[str, float]]
        Test statistic, p-value, critical values dictionary
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.core import test_stationarity
    >>> data = np.cumsum(np.random.randn(1000))  # Non-stationary random walk
    >>> stat, p_value, crit_values = test_stationarity(data)
    >>> print(f"ADF statistic: {stat:.4f}, p-value: {p_value:.4f}")
    >>> print(f"Critical values: 1%: {crit_values['1%']:.4f}, 5%: {crit_values['5%']:.4f}")
    """
    try:
        # Validate input
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        # Check if statsmodels is available
        import statsmodels.api as sm
        
        # Perform ADF test
        result = sm.tsa.stattools.adfuller(data, autolag='AIC')
        
        # Extract results
        adf_stat = result[0]
        p_value = result[1]
        crit_values = result[4]
        
        logger.debug(f"ADF test: statistic = {adf_stat:.4f}, p-value = {p_value:.4f}")
        return adf_stat, p_value, crit_values
        
    except ImportError:
        logger.warning("statsmodels not available, using simplified stationarity test")
        
        # Simplified test based on first differences
        data = np.asarray(data)
        diff = np.diff(data)
        
        # Compute ratio of variances (diff vs. levels)
        var_level = np.var(data)
        var_diff = np.var(diff)
        
        if var_level < 1e-10:  # Avoid division by zero
            return 0.0, 1.0, {'1%': -3.5, '5%': -2.9, '10%': -2.6}
        
        # Calculate test statistic (negative to match ADF convention)
        ratio = var_diff / var_level
        stat = -np.log(ratio)
        
        # Simple p-value approximation
        p_value = np.exp(-abs(stat))
        
        # Mock critical values (approximation)
        crit_values = {
            '1%': -3.5,
            '5%': -2.9,
            '10%': -2.6
        }
        
        logger.warning("Simplified stationarity test result may not be accurate")
        return stat, p_value, crit_values


class DiagnosticTester:
    """
    Comprehensive diagnostic testing class for regression and time series models.
    
    This class provides methods for running and formatting various diagnostic tests
    to assess model adequacy and assumptions.
    
    Methods
    -------
    run_tests
        Run a battery of diagnostic tests on model residuals
    format_results
        Format test results as a string or dictionary
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.core import DiagnosticTester
    >>> residuals = np.random.randn(1000)  # Example residuals
    >>> tester = DiagnosticTester()
    >>> results = tester.run_tests(residuals)
    >>> print(tester.format_results(results))
    """
    
    def __init__(self):
        """Initialize the DiagnosticTester."""
        logger.debug("Initialized DiagnosticTester")
    
    def run_tests(
        self, 
        residuals: np.ndarray, 
        regressors: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        lags: int = 10,
        significance_level: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a battery of diagnostic tests on model residuals.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from a regression or time series model
        regressors : Optional[np.ndarray]
            Matrix of regressors for heteroskedasticity tests
        data : Optional[np.ndarray]
            Original time series data for stationarity test
        lags : int, default 10
            Number of lags for autocorrelation tests
        significance_level : float, default 0.05
            Significance level for determining test rejection
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of test results with format:
            {test_name: {'statistic': value, 'p_value': value, 'rejected': bool}}
        """
        results = {}
        
        # Test for normality
        try:
            jb_stat, jb_pval = test_normality(residuals)
            results['Normality'] = {
                'statistic': jb_stat,
                'p_value': jb_pval,
                'rejected': jb_pval < significance_level,
                'test': 'Jarque-Bera'
            }
        except Exception as e:
            logger.error(f"Normality test failed: {str(e)}")
            results['Normality'] = {
                'error': str(e),
                'test': 'Jarque-Bera'
            }
        
        # Test for autocorrelation
        try:
            lb_stat, lb_pval = test_autocorrelation(residuals, lags)
            results['Autocorrelation'] = {
                'statistic': lb_stat,
                'p_value': lb_pval,
                'rejected': lb_pval < significance_level,
                'test': f'Ljung-Box Q({lags})'
            }
        except Exception as e:
            logger.error(f"Autocorrelation test failed: {str(e)}")
            results['Autocorrelation'] = {
                'error': str(e),
                'test': f'Ljung-Box Q({lags})'
            }
        
        # Test for heteroskedasticity
        try:
            het_stat, het_pval = test_heteroskedasticity(residuals, regressors)
            test_name = 'White' if regressors is not None else f'ARCH-LM({5})'
            results['Heteroskedasticity'] = {
                'statistic': het_stat,
                'p_value': het_pval,
                'rejected': het_pval < significance_level,
                'test': test_name
            }
        except Exception as e:
            logger.error(f"Heteroskedasticity test failed: {str(e)}")
            results['Heteroskedasticity'] = {
                'error': str(e),
                'test': 'White/ARCH-LM'
            }
        
        # Test for stationarity (if data provided)
        if data is not None:
            try:
                stat_stat, stat_pval, crit_values = test_stationarity(data)
                results['Stationarity'] = {
                    'statistic': stat_stat,
                    'p_value': stat_pval,
                    'critical_values': crit_values,
                    'rejected': stat_pval < significance_level,
                    'test': 'ADF'
                }
            except Exception as e:
                logger.error(f"Stationarity test failed: {str(e)}")
                results['Stationarity'] = {
                    'error': str(e),
                    'test': 'ADF'
                }
        
        return results
    
    def format_results(self, results: Dict[str, Dict[str, Any]], format_type: str = 'string') -> Union[str, Dict]:
        """
        Format test results as a string or dictionary.
        
        Parameters
        ----------
        results : Dict[str, Dict[str, Any]]
            Dictionary of test results from run_tests
        format_type : str, default 'string'
            Format type, either 'string' or 'dict'
            
        Returns
        -------
        Union[str, Dict]
            Formatted test results
        """
        if format_type.lower() == 'string':
            output = "Diagnostic Test Results:\n"
            output += "=======================\n\n"
            
            for test_name, result in results.items():
                test_type = result.get('test', test_name)
                output += f"{test_name} Test ({test_type}):\n"
                
                if 'error' in result:
                    output += f"  Error: {result['error']}\n\n"
                    continue
                
                output += f"  Statistic: {result['statistic']:.4f}\n"
                output += f"  P-value:   {result['p_value']:.4f}\n"
                
                if 'critical_values' in result:
                    output += "  Critical Values:\n"
                    for level, value in result['critical_values'].items():
                        output += f"    {level}: {value:.4f}\n"
                
                output += f"  Result:    {'Rejected' if result['rejected'] else 'Not rejected'}\n\n"
            
            return output
            
        elif format_type.lower() == 'dict':
            return results
            
        else:
            raise ValueError(f"Invalid format_type: {format_type}. Must be 'string' or 'dict'")


# Export all components
__all__ = [
    # Distributions
    'GED',
    'SkewedT',
    'jarque_bera',
    'kurtosis',
    'skewness',
    'estimate_distribution',
    
    # Optimization
    'optimize_garch',
    'Optimizer',
    
    # Bootstrap
    'block_bootstrap',
    'stationary_bootstrap',
    'Bootstrap',
    'BootstrapInference',
    
    # Tests
    'test_normality',
    'test_autocorrelation',
    'test_heteroskedasticity',
    'test_stationarity',
    'DiagnosticTester',
    'ljung_box',
    'arch_lm',
    'durbin_watson',
    'white_test',
    'breusch_pagan'
]