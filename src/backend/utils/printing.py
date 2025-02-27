"""
Formatted printing and display utilities for econometric model results.

This module provides standardized formatting utilities for printing parameter
estimates, test statistics, and diagnostic results from econometric models,
including GARCH and ARMAX models. It implements clean, consistent output
formatting using Python's modern string formatting capabilities.
"""

import numpy as np  # version: 1.26.3
from typing import Dict, List, Optional, Union
import logging
from scipy import stats  # version: 1.11.4

# Internal imports
from ..models.garch import GARCHModel
from ..models.armax import ARMAX

# Configure logger
logger = logging.getLogger(__name__)

# Global formatting constants
FLOAT_FORMAT = '.4f'
PVALUE_FORMAT = '.3f'
TABLE_WIDTH = 80


def format_parameter_table(parameters: np.ndarray, 
                          std_errors: np.ndarray, 
                          param_names: List[str]) -> str:
    """
    Creates formatted table of parameter estimates with standard errors and t-statistics.
    
    Parameters
    ----------
    parameters : np.ndarray
        Array of parameter estimates
    std_errors : np.ndarray
        Array of standard errors corresponding to parameters
    param_names : List[str]
        List of parameter names
        
    Returns
    -------
    str
        Formatted table string with parameter estimates and statistics
    """
    # Validate input arrays and dimensions
    if len(parameters) != len(std_errors) or len(parameters) != len(param_names):
        raise ValueError("Parameters, standard errors, and names must have same length")
    
    # Calculate t-statistics and p-values
    t_stats = np.zeros_like(parameters)
    p_values = np.zeros_like(parameters)
    
    for i in range(len(parameters)):
        if std_errors[i] > 0:
            t_stats[i] = parameters[i] / std_errors[i]
            # Use normal distribution for p-values (two-tailed test)
            p_values[i] = 2 * (1 - stats.norm.cdf(np.abs(t_stats[i])))
        else:
            t_stats[i] = np.nan
            p_values[i] = np.nan
    
    # Format header row with column names
    header = f"{'Parameter':<15} {'Value':>12} {'Std.Error':>12} {'t-stat':>12} {'p-value':>12}"
    separator = "-" * min(len(header), TABLE_WIDTH)
    
    # Format each parameter row with estimates and statistics
    rows = []
    for i, name in enumerate(param_names):
        value_str = f"{parameters[i]:{FLOAT_FORMAT}}".rjust(12)
        stderr_str = f"{std_errors[i]:{FLOAT_FORMAT}}".rjust(12)
        
        # Handle NaN values in t-stats and p-values
        if np.isnan(t_stats[i]):
            tstat_str = "N/A".rjust(12)
            pval_str = "N/A".rjust(12)
        else:
            tstat_str = f"{t_stats[i]:{FLOAT_FORMAT}}".rjust(12)
            pval_str = f"{p_values[i]:{PVALUE_FORMAT}}".rjust(12)
        
        rows.append(f"{name:<15} {value_str} {stderr_str} {tstat_str} {pval_str}")
    
    # Return complete formatted table string
    return f"{header}\n{separator}\n" + "\n".join(rows)


def format_diagnostic_tests(test_results: Dict) -> str:
    """
    Formats diagnostic test results into readable output.
    
    Parameters
    ----------
    test_results : Dict
        Dictionary containing test statistics and results
        
    Returns
    -------
    str
        Formatted string of diagnostic test results
    """
    if not test_results:
        return "No diagnostic test results available."
    
    # Format test names and statistics
    formatted_results = ["Diagnostic Tests:"]
    formatted_results.append("-" * min(len(formatted_results[0]), TABLE_WIDTH))
    
    # Process each test
    for test_name, test_data in test_results.items():
        # Skip if not a test object (e.g., parameter summary)
        if not isinstance(test_data, dict):
            continue
            
        # Handle different test formats
        if 'statistic' in test_data and 'p_value' in test_data:
            stat_value = test_data['statistic']
            p_value = test_data['p_value']
            
            # Format test statistic
            test_str = f"{test_name}: {stat_value:{FLOAT_FORMAT}} "
            
            # Add p-value where applicable
            test_str += f"[p-value: {p_value:{PVALUE_FORMAT}}]"
            
            # Add null hypothesis if available
            if 'null_hypothesis' in test_data:
                test_str += f"\n  H0: {test_data['null_hypothesis']}"
                
            formatted_results.append(test_str)
        
        # Format critical values if provided
        elif 'critical_values' in test_data:
            formatted_results.append(f"{test_name}:")
            for level, value in test_data['critical_values'].items():
                formatted_results.append(f"  {level}% critical value: {value:{FLOAT_FORMAT}}")
        
        # Generic format for other test results
        else:
            formatted_results.append(f"{test_name}:")
            for key, value in test_data.items():
                if isinstance(value, (int, float)):
                    formatted_results.append(f"  {key}: {value:{FLOAT_FORMAT}}")
                else:
                    formatted_results.append(f"  {key}: {value}")
    
    # Return formatted test results string
    return "\n".join(formatted_results)


def format_model_summary(model: Union[GARCHModel, ARMAX]) -> str:
    """
    Creates comprehensive model summary including parameters and diagnostics.
    
    Parameters
    ----------
    model : Union[GARCHModel, ARMAX]
        Fitted model object
        
    Returns
    -------
    str
        Complete model summary string
    """
    sections = []
    
    # Format model specification details
    if isinstance(model, GARCHModel):
        model_type = model.model_type
        distribution = model.distribution
        spec_title = f"{model_type}({model.p},{model.q}) Model with {distribution} distribution"
        sections.append(spec_title)
        sections.append("=" * min(len(spec_title), TABLE_WIDTH))
        
        # Check if model was successfully estimated
        if not model.converged or model.parameters is None:
            sections.append("Model estimation did not converge.")
            return "\n".join(sections)
        
        # Get parameter names
        param_names = []
        if model.model_type.upper() == 'GARCH':
            param_names = ['omega', 'alpha', 'beta']
        elif model.model_type.upper() == 'EGARCH':
            param_names = ['omega', 'alpha', 'gamma', 'beta']
        elif model.model_type.upper() in ['GJR-GARCH', 'TARCH']:
            param_names = ['omega', 'alpha', 'beta', 'gamma']
        elif model.model_type.upper() == 'AGARCH':
            param_names = ['omega', 'alpha', 'beta', 'theta']
        elif model.model_type.upper() == 'FIGARCH':
            param_names = ['omega', 'beta', 'phi', 'd']
        
        # Add distribution parameter if applicable
        if model.distribution.lower() == 'student-t':
            param_names.append('nu')
        elif model.distribution.lower() == 'ged':
            param_names.append('nu')
        elif model.distribution.lower() == 'skewed-t':
            param_names.extend(['nu', 'lambda'])
            
    elif isinstance(model, ARMAX):
        p, q = model.p, model.q
        spec_title = f"ARMAX({p},{q}) Model"
        if model.include_constant:
            spec_title += " with constant"
        sections.append(spec_title)
        sections.append("=" * min(len(spec_title), TABLE_WIDTH))
        
        # Check if model was successfully estimated
        if model.params is None:
            sections.append("Model has not been estimated.")
            return "\n".join(sections)
        
        # Get parameter names from ARMAX model
        param_names = []
        for i in range(p):
            param_names.append(f"AR({i+1})")
        for i in range(q):
            param_names.append(f"MA({i+1})")
        if model.include_constant:
            param_names.append("Constant")
        
        # Add exogenous parameters if applicable
        if hasattr(model, '_exog') and model._exog is not None:
            for i in range(model._exog.shape[1]):
                param_names.append(f"Exog({i+1})")
    else:
        sections.append("Unsupported model type")
        return "\n".join(sections)
    
    # Add parameter estimates table
    sections.append("\nParameter Estimates:")
    try:
        if isinstance(model, GARCHModel):
            # For GARCH models
            sections.append(format_parameter_table(
                model.parameters, 
                model.std_errors, 
                param_names
            ))
        elif isinstance(model, ARMAX):
            # For ARMAX models, extract relevant parameters
            # Skip the first 3 parameters [p, q, has_constant]
            ar_params, ma_params, constant, exog_params = model._extract_params(model.params)
            
            # Combine parameters in the right order
            param_list = []
            if len(ar_params) > 0:
                param_list.append(ar_params)
            if len(ma_params) > 0:
                param_list.append(ma_params)
            if constant is not None:
                param_list.append(np.array([constant]))
            if len(exog_params) > 0:
                param_list.append(exog_params)
                
            if param_list:
                parameters = np.concatenate(param_list)
                
                # Extract standard errors (skip the structural parameters)
                if model.standard_errors is not None and len(model.standard_errors) > 3:
                    std_errors = model.standard_errors[3:3+len(parameters)]
                    # Ensure length matches
                    if len(std_errors) < len(parameters):
                        std_errors = np.concatenate([std_errors, np.zeros(len(parameters) - len(std_errors))])
                else:
                    std_errors = np.ones_like(parameters) * np.nan
                
                sections.append(format_parameter_table(parameters, std_errors, param_names))
            else:
                sections.append("No parameters to display")
    except Exception as e:
        logger.error(f"Error formatting parameter table: {str(e)}")
        sections.append("Error formatting parameter estimates.")
    
    # Add model fit statistics
    sections.append("\nModel Fit:")
    
    # Log-likelihood
    if isinstance(model, GARCHModel) and model.likelihood is not None:
        sections.append(f"Log-Likelihood: {model.likelihood:{FLOAT_FORMAT}}")
    elif isinstance(model, ARMAX) and model.loglikelihood is not None:
        sections.append(f"Log-Likelihood: {model.loglikelihood:{FLOAT_FORMAT}}")
    
    # Include diagnostic test results
    try:
        if isinstance(model, ARMAX) and hasattr(model, 'diagnostic_tests'):
            test_results = model.diagnostic_tests()
            # Add information criteria
            if 'AIC' in test_results:
                sections.append(f"AIC: {test_results['AIC']:{FLOAT_FORMAT}}")
            if 'BIC' in test_results:
                sections.append(f"BIC: {test_results['BIC']:{FLOAT_FORMAT}}")
            if 'HQIC' in test_results:
                sections.append(f"HQIC: {test_results['HQIC']:{FLOAT_FORMAT}}")
            
            # Add other diagnostic tests
            sections.append("\n" + format_diagnostic_tests(test_results))
    except Exception as e:
        logger.error(f"Error formatting diagnostic tests: {str(e)}")
        sections.append("Error formatting diagnostic tests.")
    
    # Return complete summary string
    return "\n".join(sections)


class ResultsPrinter:
    """
    Handles formatted printing of model estimation results and diagnostics.
    
    This class provides methods for printing model results with consistent 
    formatting, including parameter estimates, diagnostic tests, and other
    model information.
    
    Parameters
    ----------
    width : Optional[int]
        Width of output tables
    float_format : Optional[str]
        Format specifier for floating point values
    pvalue_format : Optional[str]
        Format specifier for p-values
    
    Methods
    -------
    print_model_results
        Print complete model estimation results
    print_estimation_progress
        Print progress updates during model estimation
    """
    
    def __init__(self, 
                width: Optional[int] = None, 
                float_format: Optional[str] = None, 
                pvalue_format: Optional[str] = None):
        """
        Initialize printer with formatting options.
        
        Parameters
        ----------
        width : Optional[int]
            Width of output tables
        float_format : Optional[str]
            Format specifier for floating point values
        pvalue_format : Optional[str]
            Format specifier for p-values
        """
        # Set default formatting options if not provided
        self.width = width if width is not None else TABLE_WIDTH
        self.float_format = float_format if float_format is not None else FLOAT_FORMAT
        self.pvalue_format = pvalue_format if pvalue_format is not None else PVALUE_FORMAT
        
        # Initialize printer properties
        self._summary_cache = {}
        
        # Configure logging format
        global FLOAT_FORMAT, PVALUE_FORMAT, TABLE_WIDTH
        FLOAT_FORMAT = self.float_format
        PVALUE_FORMAT = self.pvalue_format
        TABLE_WIDTH = self.width
        
        logger.debug(f"Initialized ResultsPrinter with width={self.width}, "
                    f"float_format={self.float_format}, "
                    f"pvalue_format={self.pvalue_format}")
    
    def print_model_results(self, model: Union[GARCHModel, ARMAX]) -> None:
        """
        Print complete model estimation results.
        
        Parameters
        ----------
        model : Union[GARCHModel, ARMAX]
            Fitted model object
        
        Returns
        -------
        None
            Prints formatted results to output
        """
        # Generate model summary string
        summary = format_model_summary(model)
        
        # Print formatted output
        print("\n" + summary)
        
        # Log summary to debug log
        logger.debug("Printed model results summary")
    
    def print_estimation_progress(self, iteration: float, likelihood: float) -> None:
        """
        Print progress updates during model estimation.
        
        Parameters
        ----------
        iteration : float
            Current iteration number
        likelihood : float
            Current log-likelihood value
        
        Returns
        -------
        None
            Prints progress update to output
        """
        # Format progress message
        message = f"Iteration {int(iteration)}: Log-likelihood = {likelihood:{self.float_format}}"
        
        # Print update to output
        print(message, end="\r")
        
        # Log progress to debug log
        logger.debug(message)