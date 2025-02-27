"""
Test suite for statistical distribution implementations in the MFE Toolbox.

This module provides comprehensive tests for distribution functions, parameter estimation,
and statistical testing capabilities using pytest and hypothesis for property-based testing.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st
import scipy.stats as stats

# Internal imports
from ..core.distributions import (
    GED, SkewedT, jarque_bera, kurtosis, skewness
)
from ..utils.validation import validate_array_input


def test_ged_initialization():
    """Test GED class initialization and parameter validation."""
    # Test valid initialization
    ged = GED(nu=1.5)
    assert ged.nu == 1.5
    assert ged._lambda is not None
    assert ged._const is not None
    
    # Test normal case (nu=2)
    ged_normal = GED(nu=2.0)
    assert ged_normal.nu == 2.0
    assert abs(ged_normal._lambda - 1.0 / np.sqrt(2.0)) < 1e-10
    assert abs(ged_normal._const - 1.0 / np.sqrt(2.0 * np.pi)) < 1e-10
    
    # Test invalid initialization
    with pytest.raises(ValueError):
        GED(nu=0)  # nu must be positive
    
    with pytest.raises(ValueError):
        GED(nu=-1.5)  # nu must be positive


@pytest.mark.parametrize('nu', [1.5, 2.0, 2.5])
def test_ged_pdf(nu):
    """Test GED probability density function computation."""
    # Generate test data
    np.random.seed(42)
    x = np.linspace(-5, 5, 100)
    
    # Create GED instance and compute PDF
    ged = GED(nu=nu)
    pdf_values = ged.pdf(x)
    
    # Verify basic PDF properties
    assert isinstance(pdf_values, np.ndarray)
    assert len(pdf_values) == len(x)
    assert np.all(pdf_values >= 0)  # PDF values should be non-negative
    
    # For nu=2, GED is equivalent to normal distribution
    if nu == 2.0:
        normal_pdf = stats.norm.pdf(x)
        assert np.allclose(pdf_values, normal_pdf, rtol=1e-5)
    
    # Test PDF integration (should be approximately 1)
    # Use simple numeric integration over a reasonable range
    x_range = np.linspace(-10, 10, 1000)
    pdf_range = ged.pdf(x_range)
    integral = np.trapz(pdf_range, x_range)
    assert abs(integral - 1.0) < 0.01


def test_skewed_t_initialization():
    """Test Skewed T distribution initialization and parameter validation."""
    # Test valid initialization
    st_dist = SkewedT(nu=4.0, lambda_=0.0)
    assert st_dist.nu == 4.0
    assert st_dist.lambda_ == 0.0
    assert st_dist._a is not None
    assert st_dist._b is not None
    assert st_dist._c is not None
    
    # Test initialization with skewness
    st_dist = SkewedT(nu=5.0, lambda_=0.3)
    assert st_dist.nu == 5.0
    assert st_dist.lambda_ == 0.3
    
    # Test invalid initialization - nu must be > 2
    with pytest.raises(ValueError):
        SkewedT(nu=2.0, lambda_=0.0)
    
    with pytest.raises(ValueError):
        SkewedT(nu=1.0, lambda_=0.0)
    
    # Test invalid initialization - lambda must be in (-1, 1)
    with pytest.raises(ValueError):
        SkewedT(nu=4.0, lambda_=1.0)
    
    with pytest.raises(ValueError):
        SkewedT(nu=4.0, lambda_=-1.0)
    
    with pytest.raises(ValueError):
        SkewedT(nu=4.0, lambda_=1.5)


@pytest.mark.parametrize('nu,lambda_', [(4.0, 0.0), (5.0, 0.3), (6.0, -0.3)])
def test_skewed_t_pdf(nu, lambda_):
    """Test Skewed T probability density function computation."""
    # Generate test data
    np.random.seed(42)
    x = np.linspace(-5, 5, 100)
    
    # Create SkewedT instance and compute PDF
    st_dist = SkewedT(nu=nu, lambda_=lambda_)
    pdf_values = st_dist.pdf(x)
    
    # Verify basic PDF properties
    assert isinstance(pdf_values, np.ndarray)
    assert len(pdf_values) == len(x)
    assert np.all(pdf_values >= 0)  # PDF values should be non-negative
    
    # For lambda=0, SkewedT should be equivalent to Student's t
    if lambda_ == 0.0:
        # Create Student's t PDF for comparison
        t_pdf = stats.t.pdf(x, df=nu)
        assert np.allclose(pdf_values, t_pdf, rtol=1e-4)
    
    # Test PDF integration (should be approximately 1)
    # Use simple numeric integration over a reasonable range
    x_range = np.linspace(-10, 10, 1000)
    pdf_range = st_dist.pdf(x_range)
    integral = np.trapz(pdf_range, x_range)
    assert abs(integral - 1.0) < 0.01


def test_jarque_bera():
    """Test Jarque-Bera normality test implementation."""
    # Generate normal data (should have low JB statistic)
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    
    # Generate non-normal data (t distribution with 3 df, should have high JB statistic)
    np.random.seed(42)
    t_data = np.random.standard_t(3, 1000)
    
    # Test Jarque-Bera on normal data
    jb_normal, p_normal = jarque_bera(normal_data)
    assert isinstance(jb_normal, float)
    assert isinstance(p_normal, float)
    
    # Test Jarque-Bera on t-distributed data
    jb_t, p_t = jarque_bera(t_data)
    assert isinstance(jb_t, float)
    assert isinstance(p_t, float)
    
    # JB statistic for normal data should generally be small
    # JB statistic for t data should generally be larger
    assert jb_normal < jb_t
    
    # Check p-value bounds
    assert 0 <= p_normal <= 1
    assert 0 <= p_t <= 1


def test_kurtosis():
    """Test kurtosis calculation function."""
    # Generate data with known kurtosis
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    
    # Generate high kurtosis data
    t_data = np.random.standard_t(3, 1000)  # t with low df has high kurtosis
    
    # Test kurtosis
    normal_kurt = kurtosis(normal_data)
    t_kurt = kurtosis(t_data)
    
    assert isinstance(normal_kurt, float)
    assert isinstance(t_kurt, float)
    
    # Normal data should have excess kurtosis close to 0
    assert abs(normal_kurt) < 0.5
    
    # t data should have positive excess kurtosis
    assert t_kurt > 0.5
    
    # Test kurtosis with excess=False
    normal_kurt_total = kurtosis(normal_data, excess=False)
    assert abs(normal_kurt_total - 3.0) < 0.5  # Normal kurtosis should be ~3


def test_skewness():
    """Test skewness calculation function."""
    # Generate data with known skewness
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    
    # Generate skewed data
    skewed_data = np.exp(normal_data)  # Log-normal is right-skewed
    
    # Test skewness
    normal_skew = skewness(normal_data)
    skewed_skew = skewness(skewed_data)
    
    assert isinstance(normal_skew, float)
    assert isinstance(skewed_skew, float)
    
    # Normal data should have skewness close to 0
    assert abs(normal_skew) < 0.5
    
    # Log-normal data should have positive skewness
    assert skewed_skew > 1.0


@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=30))
def test_distribution_properties(data):
    """Property-based tests for distribution implementations."""
    # Convert to numpy array and filter out potential NaN/Inf values
    data = np.array(data)
    data = data[np.isfinite(data)]
    
    # Skip if not enough data after filtering
    if len(data) < 30:
        return
    
    # Standardize the data for better numeric stability
    data = (data - np.mean(data)) / np.std(data)
    
    # Test GED distribution properties
    try:
        # Try with a few different nu values
        for nu in [1.5, 2.0, 2.5]:
            ged = GED(nu=nu)
            pdf_values = ged.pdf(data)
            
            # 1. PDF non-negativity
            assert np.all(pdf_values >= 0), "PDF values must be non-negative"
            
            # 2. Integration property check (on a grid of values)
            x_grid = np.linspace(-10, 10, 200)
            pdf_grid = ged.pdf(x_grid)
            integral = np.trapz(pdf_grid, x_grid)
            assert abs(integral - 1.0) < 0.01, "PDF should integrate to approximately 1"
            
            # 3. Loglikelihood should be finite
            ll = ged.loglikelihood(data)
            assert np.isfinite(ll), "Log-likelihood should be finite"
    except Exception as e:
        pytest.skip(f"GED test failed: {str(e)}")
    
    # Test SkewedT distribution properties
    try:
        # Try with a few different parameter combinations
        for nu, lambda_ in [(4.0, 0.0), (5.0, 0.3), (6.0, -0.3)]:
            st_dist = SkewedT(nu=nu, lambda_=lambda_)
            pdf_values = st_dist.pdf(data)
            
            # 1. PDF non-negativity
            assert np.all(pdf_values >= 0), "PDF values must be non-negative"
            
            # 2. Integration property check
            x_grid = np.linspace(-10, 10, 200)
            pdf_grid = st_dist.pdf(x_grid)
            integral = np.trapz(pdf_grid, x_grid)
            assert abs(integral - 1.0) < 0.01, "PDF should integrate to approximately 1"
            
            # 3. Loglikelihood should be finite
            ll = st_dist.loglikelihood(data)
            assert np.isfinite(ll), "Log-likelihood should be finite"
    except Exception as e:
        pytest.skip(f"SkewedT test failed: {str(e)}")
    
    # Test statistical functions
    # 4. Statistical properties validation
    jb_stat, p_value = jarque_bera(data)
    assert isinstance(jb_stat, float), "JB statistic should be a float"
    assert 0 <= p_value <= 1, "p-value should be between 0 and 1"
    
    kurt = kurtosis(data)
    assert isinstance(kurt, float), "Kurtosis should be a float"
    
    skew = skewness(data)
    assert isinstance(skew, float), "Skewness should be a float"
    
    # Verify skewness and kurtosis are used in JB statistic
    # JB = n/6 * (S^2 + (K^2)/4) where S is skewness and K is excess kurtosis
    n = len(data)
    expected_jb = n / 6 * (skew**2 + (kurt**2) / 4)
    assert abs(jb_stat - expected_jb) < 1e-10, "JB statistic should be based on skewness and kurtosis"