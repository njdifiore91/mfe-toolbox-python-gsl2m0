# MFE Toolbox Testing Standards

## Table of Contents
- [Introduction](#introduction)
- [Testing Framework](#testing-framework)
- [Test Organization](#test-organization)
- [Test Categories](#test-categories)
  - [Unit Testing](#unit-testing)
  - [Statistical Testing](#statistical-testing)
  - [Integration Testing](#integration-testing)
  - [Performance Testing](#performance-testing)
- [Code Coverage Requirements](#code-coverage-requirements)
- [Setting Up Test Environment](#setting-up-test-environment)
- [Running Tests](#running-tests)
- [Writing Effective Tests](#writing-effective-tests)
- [Test Reporting and Analysis](#test-reporting-and-analysis)
- [Performance Benchmarking Standards](#performance-benchmarking-standards)
- [Memory Profiling Requirements](#memory-profiling-requirements)
- [Appendices](#appendices)
  - [Example Test Cases](#example-test-cases)
  - [Reference Configurations](#reference-configurations)

## Introduction

This document defines the comprehensive testing standards, procedures, and requirements for the MFE Toolbox project. It covers all aspects of testing including unit testing, statistical validation, integration testing, and performance benchmarking. These standards ensure the reliability, performance, and correctness of the MFE Toolbox implementation.

## Testing Framework

The MFE Toolbox project uses the following testing tools and frameworks:

- **pytest (7.4.3)**: Primary testing framework for all test types
- **pytest-asyncio (0.21.1)**: Support for testing asynchronous code using async/await patterns
- **pytest-cov (4.1.0)**: Code coverage measurement and reporting
- **pytest-benchmark (4.0.0)**: Performance benchmarking for critical functions
- **hypothesis (6.92.1)**: Property-based testing for statistical validations
- **numba.testing**: Specialized testing tools for Numba-optimized functions

The testing framework is configured via `pytest.ini` and `coverage.ini` to provide consistent test execution and reporting.

## Test Organization

Tests are organized following a modular structure that mirrors the Python package layout:

```
tests/
├── test_core/
│   ├── test_bootstrap.py
│   ├── test_distributions.py
│   └── test_optimization.py
├── test_models/
│   ├── test_garch.py
│   ├── test_realized.py
│   └── test_volatility.py
├── test_utils/
│   ├── test_validation.py
│   └── test_printing.py
└── conftest.py
```

Each test module corresponds to a specific module in the MFE Toolbox, and test functions within these modules test specific functionality. The `conftest.py` file contains shared pytest fixtures and configurations.

## Test Categories

### Unit Testing

Unit tests verify the correctness of individual functions and classes. They should:

- Focus on testing one function or method at a time
- Validate correct behavior for valid inputs
- Verify proper error handling for invalid inputs
- Use appropriate test fixtures for setup and teardown
- Be fast and independent of each other

**Requirements:**
- All public functions must have corresponding unit tests
- Each conditional branch should be tested
- Edge cases must be explicitly tested
- Mocks should be used to isolate the unit under test

**Example:**

```python
def test_garch_initialization():
    model = garch.GARCH(p=1, q=1)
    assert model.p == 1
    assert model.q == 1
    assert model.estimate is not None
    
@pytest.mark.parametrize("p,q", [(-1, 1), (1, -1), (0, 0)])
def test_garch_invalid_params(p, q):
    with pytest.raises(ValueError):
        garch.GARCH(p=p, q=q)
```

### Statistical Testing

Statistical tests verify the correctness of statistical algorithms and properties. They should:

- Validate distribution properties
- Verify numerical accuracy against known reference values
- Test statistical hypotheses on generated data
- Ensure convergence properties on large samples

**Requirements:**
- Use hypothesis for property-based testing
- Validate results against theoretical properties
- Include tests for edge case distributions
- Verify stability across different data scales

**Example:**

```python
@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=30))
def test_jarque_bera(data):
    data = np.array(data)
    statistic, pval = jarque_bera(data)
    assert isinstance(statistic, float)
    assert 0 <= pval <= 1
```

### Integration Testing

Integration tests verify the interaction between different components. They should:

- Test combinations of multiple modules
- Verify end-to-end workflows
- Validate cross-component data flow
- Ensure proper error propagation

**Requirements:**
- All major workflows must have integration tests
- Test data sharing between components
- Verify consistent behavior across modules
- Include cross-module error handling tests

**Example:**

```python
@pytest.mark.asyncio
async def test_garch_volatility_workflow():
    # Test the full workflow from data loading to volatility prediction
    data = load_test_data()
    model = garch.GARCH(p=1, q=1)
    result = await model.estimate(data)
    forecasts = await model.forecast(horizon=5)
    assert result.converged
    assert len(forecasts) == 5
    assert np.all(forecasts > 0)
```

### Performance Testing

Performance tests verify the computational efficiency of critical functions. They should:

- Measure execution time for key operations
- Compare optimized vs. non-optimized implementations
- Validate memory usage efficiency
- Verify scalability with increasing data size

**Requirements:**
- All Numba-optimized functions must have performance tests
- Include benchmarks for different data sizes
- Verify memory consumption stays within bounds
- Test performance on realistic workloads

**Example:**

```python
@pytest.mark.benchmark
def test_garch_optimization_performance(benchmark):
    data = np.random.randn(10000)
    result = benchmark(lambda: garch.optimize_garch_likelihood(data))
    assert result is not None
    assert benchmark.stats.stats.mean < 0.5  # Execution time < 500ms
```

## Code Coverage Requirements

The MFE Toolbox project enforces strict code coverage requirements:

- Minimum overall coverage: 90%
- Minimum function coverage: 95%
- Minimum branch coverage: 85%
- Coverage excluded patterns: Test files, `__pycache__`, boilerplate code

Coverage is tracked via pytest-cov with configuration in `coverage.ini`. All pull requests and merges to main branches must maintain or improve coverage ratios. Coverage reports are generated in HTML and XML formats for review and CI integration.

## Setting Up Test Environment

To set up the test environment:

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

2. Install the required testing dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Verify the test environment:
   ```bash
   pytest --collect-only
   ```

The test environment should be compatible with Python 3.12 and include all dependencies specified in the project requirements.

## Running Tests

To run the complete test suite:

```bash
pytest
```

To run specific test categories:

```bash
# Run only unit tests
pytest tests/test_core tests/test_models tests/test_utils

# Run statistical validation tests
pytest -m hypothesis

# Run integration tests
pytest -m asyncio

# Run performance benchmarks
pytest -m benchmark

# Generate coverage report
pytest --cov=mfe
```

Test execution should follow CI/CD integration principles, with tests running automatically on each commit and pull request.

## Writing Effective Tests

When writing tests for the MFE Toolbox, follow these guidelines:

1. **Naming Convention**: Use descriptive test names that indicate the functionality being tested
   ```python
   def test_garch_converges_with_valid_data():
   ```

2. **Arrange-Act-Assert**: Structure tests with clear setup, action, and verification phases
   ```python
   # Arrange
   model = GARCH(p=1, q=1)
   data = np.random.randn(1000)
   
   # Act
   result = model.fit(data)
   
   # Assert
   assert result.converged
   assert result.aic < 0
   ```

3. **Test Isolation**: Each test should run independently of others
   ```python
   @pytest.fixture
   def model():
       return GARCH(p=1, q=1)
   
   def test_model_fitting(model):
       # Test uses the fixture for isolation
   ```

4. **Parameterization**: Use pytest.mark.parametrize for testing multiple cases
   ```python
   @pytest.mark.parametrize("p,q,expected", [
       (1, 1, True),
       (2, 2, True),
       (0, 0, False)
   ])
   def test_garch_validation(p, q, expected):
       assert garch.is_valid_order(p, q) == expected
   ```

5. **Statistical Validation**: Use hypothesis for property-based testing
   ```python
   @given(st.lists(st.floats(min_value=-10, max_value=10), min_size=50))
   def test_distribution_properties(data):
       # Test statistical properties
   ```

6. **Test Documentation**: Include docstrings describing the test purpose
   ```python
   def test_realized_volatility():
       """Verify that realized volatility is always positive and handles edge cases."""
   ```

## Test Reporting and Analysis

Test results should be analyzed and reported systematically:

1. **CI Integration**: Tests run automatically via CI pipelines
2. **Coverage Reports**: HTML and XML reports generated after test execution
3. **Benchmark Comparisons**: Performance test results compared against baselines
4. **Regression Analysis**: Changes in performance tracked over time
5. **Test Failure Analysis**: Detailed analysis of test failures with remediation plans

Reports should include:
- Overall pass/fail status
- Coverage percentage by module
- Performance benchmark results
- Memory usage metrics
- Statistical validation results

## Performance Benchmarking Standards

Performance benchmarks ensure the computational efficiency of the MFE Toolbox:

1. **Benchmark Configuration**:
   - Minimum rounds: 100 (configured in pytest.ini)
   - Warmup enabled
   - Timer: time.perf_counter
   - Garbage collection disabled during benchmarks

2. **Benchmark Categories**:
   - Core algorithm performance
   - Numba-optimized function speed
   - Memory consumption patterns
   - Scaling with data size

3. **Acceptance Criteria**:
   - Critical path functions must execute within specified time bounds
   - Numba-optimized functions must show significant speedup over pure Python
   - Memory usage must remain within defined limits
   - Performance must scale appropriately with input size

4. **Benchmark Storage**:
   - Results stored in `.benchmarks` directory
   - Results compared across commits
   - Significant regressions block merges

Example benchmark test:

```python
@pytest.mark.benchmark
def test_bootstrap_performance(benchmark):
    data = np.random.randn(1000)
    result = benchmark.pedantic(
        lambda: bootstrap.stationary_bootstrap(data, 100),
        rounds=200,
        iterations=3
    )
    assert result is not None
    # Verify performance is within acceptable range
    assert benchmark.stats.stats.mean < 0.1
```

## Memory Profiling Requirements

Memory usage is profiled to ensure efficient resource utilization:

1. **Memory Profiling Tools**:
   - pytest-memray for memory tracking
   - Threshold of 100MB for alerting

2. **Memory Test Markers**:
   - `@pytest.mark.memray` for memory profiling
   - `@pytest.mark.high_memory` for tests expected to use significant memory

3. **Memory Requirements**:
   - Core functions should operate within defined memory limits
   - Memory scaling should be predictable with input size
   - No memory leaks in long-running operations
   - Proper cleanup after large operations

Example memory test:

```python
@pytest.mark.memray
def test_large_matrix_memory_usage():
    # Test memory usage with large matrices
    data = np.random.randn(10000, 100)
    result = volatility.multivariate_garch(data)
    assert result is not None
    # Memory usage checked by pytest-memray
```

## Appendices

### Example Test Cases

**Unit Test Example:**

```python
def test_bootstrap_block_size():
    """Test that block bootstrap respects block size parameter."""
    np.random.seed(42)
    data = np.random.randn(1000)
    block_size = 10
    
    # Generate bootstrap samples
    samples = bootstrap.block_bootstrap(data, 100, block_size)
    
    # Verify samples shape
    assert samples.shape == (100, 1000)
    
    # Verify block structure (check for repeated sequences)
    for sample in samples:
        # Confirm blocks of size 'block_size' exist in the resampled data
        found_blocks = 0
        for i in range(len(data) - block_size):
            block = data[i:i+block_size]
            for j in range(len(sample) - block_size):
                if np.array_equal(block, sample[j:j+block_size]):
                    found_blocks += 1
                    break
        assert found_blocks > 0
```

**Statistical Test Example:**

```python
@pytest.mark.hypothesis
@given(
    st.floats(min_value=0.1, max_value=0.9),
    st.floats(min_value=0.1, max_value=0.9),
    st.integers(min_value=500, max_value=2000)
)
def test_garch_parameter_recovery(alpha, beta, n_samples):
    """Test that GARCH model can recover true parameters."""
    assume(alpha + beta < 0.99)  # Ensure stationarity
    
    # Generate data from GARCH process with known parameters
    omega = 0.1
    data = generate_garch_process(n_samples, omega, alpha, beta)
    
    # Fit model
    model = garch.GARCH(p=1, q=1)
    result = model.fit(data)
    
    # Check parameter recovery within tolerance
    assert abs(result.params['omega'] - omega) < 0.1
    assert abs(result.params['alpha'] - alpha) < 0.1
    assert abs(result.params['beta'] - beta) < 0.1
```

**Integration Test Example:**

```python
@pytest.mark.asyncio
async def test_arma_forecast_integration():
    """Test integration between ARMA estimation and forecasting."""
    # Generate test data
    np.random.seed(42)
    ar_params = np.array([0.75])
    ma_params = np.array([0.25])
    data = arma_generate_sample(ar_params, ma_params, 1000)
    
    # Fit ARMA model
    model = armax.ARMAX(p=1, q=1)
    await model.fit_async(data)
    
    # Generate forecasts
    forecasts = await model.forecast_async(steps=10)
    
    # Validate results
    assert len(forecasts) == 10
    assert isinstance(forecasts, np.ndarray)
    
    # Check forecast properties
    expected_mean = data.mean()
    assert abs(forecasts.mean() - expected_mean) < 0.5
```

**Performance Test Example:**

```python
@pytest.mark.benchmark
@pytest.mark.parametrize("size", [1000, 10000, 100000])
def test_realized_volatility_scaling(benchmark, size):
    """Test performance scaling of realized volatility calculations."""
    # Generate random price data
    np.random.seed(42)
    price = np.cumsum(np.random.randn(size)) + 1000
    time = np.arange(size)
    
    # Benchmark execution
    result = benchmark(
        realized.realized_variance,
        price, time, 'BusinessTime', (1, 50)
    )
    
    # Check result validity
    assert isinstance(result[0], float)
    assert result[0] > 0
    
    # Access timing stats
    mean_time = benchmark.stats.stats.mean
    
    # Optional: Check scaling behavior
    if hasattr(benchmark, 'previous_stats') and size > 1000:
        previous_mean = benchmark.previous_stats.stats.mean
        scaling_factor = mean_time / previous_mean
        expected_factor = size / benchmark.previous_size
        assert scaling_factor < expected_factor  # Sub-linear scaling
    
    # Store stats for next comparison
    benchmark.previous_stats = benchmark.stats
    benchmark.previous_size = size
```

### Reference Configurations

**pytest.ini**:
```ini
[pytest]
testpaths = src/backend/tests src/web/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts = --strict-markers -v --cov=mfe --cov-report=term-missing --cov-report=html --benchmark-only --benchmark-storage=.benchmarks --benchmark-autosave --memray --memray-threshold=100MB

markers =
    asyncio: mark test as async/await test
    benchmark: mark test as performance benchmark
    slow: mark test as slow running (>30s)
    numba: mark test as requiring Numba optimization
    numba_parallel: mark test as using parallel Numba optimization
    hypothesis: mark test as property-based test
    distribution: mark test as distribution property test
    memray: mark test for memory profiling
    high_memory: mark test as memory intensive

# Plugin configurations
cov_fail_under = 90

benchmark_min_rounds = 100
benchmark_warmup = True
benchmark_timer = time.perf_counter
benchmark_disable_gc = True

asyncio_mode = auto

memray_threshold = 100MB
memray_output = html
```

**coverage.ini**:
```ini
[run]
branch = True
source = mfe/core,mfe/models,mfe/ui,mfe/utils
omit = **/tests/*,**/__pycache__/*,**/.pytest_cache/*
parallel = True

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == '__main__':
    pass
    def main\(\):
ignore_errors = True
fail_under = 90

[html]
directory = coverage_html
show_contexts = True
title = MFE Toolbox Coverage Report

[xml]
output = coverage.xml
```