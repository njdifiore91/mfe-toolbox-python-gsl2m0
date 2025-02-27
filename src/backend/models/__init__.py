"""
Package initialization module that exposes core time series and volatility modeling
functionality from the MFE Toolbox. Provides a unified interface to ARMA/ARMAX models,
univariate GARCH variants, and multivariate volatility models with async support and
type hints.
"""

import numpy as np  # version 1.26.3
from typing import Union, Optional, Dict, List, Tuple, Any

# Import core time series model
from .timeseries import ARMAX as ARMAModel

# Import univariate volatility models
from .univariate import UnivariateGARCH, AGARCH

# Import multivariate volatility models
from .volatility import BEKK

# Import base volatility model as an alias
from .univariate import UnivariateGARCH as BaseVolatilityModel

# Define version and exported symbols
__version__ = "4.0"
__all__ = ["ARMAModel", "UnivariateGARCH", "AGARCH", "BaseVolatilityModel", "BEKK"]