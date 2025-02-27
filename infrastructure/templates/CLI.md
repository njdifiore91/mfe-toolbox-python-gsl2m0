# MFE Toolbox Command-Line Interface

## Overview

The MFE Toolbox Command-Line Interface (CLI) provides access to the powerful financial econometrics capabilities directly from your terminal. This interface enables time series analysis, econometric model estimation, and diagnostic testing without writing Python code.

The CLI leverages the same high-performance Python modules that power the MFE Toolbox, including Numba-optimized routines for maximum computational efficiency.

## Installation

To use the MFE Toolbox CLI, ensure you have Python 3.12 or later installed with the package:

```bash
pip install mfe
```

The installation includes all dependencies required for CLI functionality:
- NumPy (1.26.3+) 
- SciPy (1.11.4+)
- Pandas (2.1.4+)
- Statsmodels (0.14.1+)
- Numba (0.59.0+)

## Basic Usage

The MFE Toolbox CLI follows a command-subcommand structure:

```bash
mfe <command> <subcommand> [options]
```

For general help:

```bash
mfe --help
```

For command-specific help:

```bash
mfe <command> --help
```

## Command Reference

### Time Series Commands

#### ARMA/ARMAX Modeling

```bash
mfe timeseries armax [options]
```

Estimate ARMA or ARMAX models with optional exogenous variables.

**Options:**
- `--data FILE` - Path to CSV file containing time series data
- `--p INT` - AR order (default: 1)
- `--q INT` - MA order (default: 1)
- `--exog FILE` - Path to CSV file containing exogenous variables
- `--constant` - Include constant term (default: True)
- `--output FILE` - Path to save results (default: stdout)
- `--forecast INT` - Number of periods to forecast (default: 0)
- `--diagnostics` - Compute diagnostic statistics (default: False)
- `--async` - Use asynchronous estimation to prevent terminal blocking (default: False)

**Example:**
```bash
mfe timeseries armax --data returns.csv --p 2 --q 1 --constant --output results.json --diagnostics
```

#### Forecasting

```bash
mfe timeseries forecast [options]
```

Generate forecasts from a previously estimated model.

**Options:**
- `--model FILE` - Path to saved model or JSON results
- `--steps INT` - Number of steps to forecast
- `--exog FILE` - Path to CSV file containing future exogenous variables
- `--output FILE` - Path to save forecast results (default: stdout)
- `--interval FLOAT` - Confidence interval width (default: 0.95)

**Example:**
```bash
mfe timeseries forecast --model arma_model.json --steps 10 --output forecast.csv
```

### Volatility Modeling Commands

#### GARCH Estimation

```bash
mfe volatility garch [options]
```

Estimate GARCH family models for volatility.

**Options:**
- `--data FILE` - Path to CSV file containing return data
- `--type STRING` - GARCH model type (default: "GARCH", options: "GARCH", "EGARCH", "GJR-GARCH", "TGARCH", "AGARCH")
- `--p INT` - ARCH order (default: 1)
- `--q INT` - GARCH order (default: 1)
- `--dist STRING` - Error distribution (default: "normal", options: "normal", "student-t", "ged", "skewed-t")
- `--output FILE` - Path to save results (default: stdout)
- `--forecast INT` - Number of periods to forecast volatility (default: 0)
- `--async` - Use asynchronous estimation (default: False)

**Example:**
```bash
mfe volatility garch --data returns.csv --type "GJR-GARCH" --p 1 --q 1 --dist "student-t" --output garch_results.json
```

#### Realized Volatility

```bash
mfe volatility realized [options]
```

Compute realized volatility measures from high-frequency data.

**Options:**
- `--data FILE` - Path to CSV file containing high-frequency data
- `--time-col STRING` - Column name for timestamps (default: "time")
- `--price-col STRING` - Column name for price (default: "price")
- `--sampling STRING` - Sampling scheme (default: "calendar", options: "calendar", "business", "fixed")
- `--interval FLOAT` - Sampling interval in minutes (default: 5)
- `--method STRING` - Realized measure (default: "rv", options: "rv", "bv", "rv-subsampled", "kernel")
- `--output FILE` - Path to save results (default: stdout)

**Example:**
```bash
mfe volatility realized --data intraday_prices.csv --sampling "calendar" --interval 15 --method "rv" --output realized_vol.csv
```

### Statistical Testing Commands

#### Unit Root Tests

```bash
mfe test unitroot [options]
```

Perform unit root and stationarity tests on time series data.

**Options:**
- `--data FILE` - Path to CSV file containing time series data
- `--test STRING` - Test type (default: "adf", options: "adf", "pp", "kpss", "ers")
- `--lags INT` - Number of lags (default: 10)
- `--trend STRING` - Trend specification (default: "c", options: "n", "c", "ct", "ctt")
- `--output FILE` - Path to save results (default: stdout)

**Example:**
```bash
mfe test unitroot --data gdp.csv --test "adf" --lags 4 --trend "ct" --output unitroot_results.txt
```

#### Autocorrelation Tests

```bash
mfe test autocorr [options]
```

Perform autocorrelation and partial autocorrelation tests.

**Options:**
- `--data FILE` - Path to CSV file containing time series data
- `--lags INT` - Number of lags to compute (default: 20)
- `--test STRING` - Test type (default: "ljungbox", options: "ljungbox", "boxpierce", "breusch")
- `--plot` - Generate autocorrelation plots (default: False)
- `--output FILE` - Path to save results (default: stdout)

**Example:**
```bash
mfe test autocorr --data returns.csv --lags 30 --test "ljungbox" --output autocorr_results.json
```

### Bootstrap Commands

```bash
mfe bootstrap [options]
```

Perform bootstrap analysis for time series data.

**Options:**
- `--data FILE` - Path to CSV file containing time series data
- `--method STRING` - Bootstrap method (default: "stationary", options: "stationary", "block", "moving-block")
- `--block-size INT` - Block size for block bootstrap methods (default: 10)
- `--replications INT` - Number of bootstrap replications (default: 1000)
- `--statistic STRING` - Statistic to compute (default: "mean", options: "mean", "variance", "percentile")
- `--percentile FLOAT` - Percentile for percentile statistic (default: 0.95)
- `--output FILE` - Path to save results (default: stdout)
- `--async` - Use asynchronous processing for better responsiveness (default: False)

**Example:**
```bash
mfe bootstrap --data returns.csv --method "block" --block-size 20 --replications 5000 --statistic "percentile" --percentile 0.99 --output bootstrap_results.json
```

### Utility Commands

#### Data Preprocessing

```bash
mfe utils preprocess [options]
```

Preprocess financial data for analysis.

**Options:**
- `--data FILE` - Path to CSV file containing raw data
- `--returns` - Compute returns from price data (default: False)
- `--log-returns` - Compute log returns from price data (default: False)
- `--standardize` - Standardize data (default: False)
- `--winsorize FLOAT` - Winsorize data at specified percentile (default: 0)
- `--fill-method STRING` - Method for handling missing data (default: "none", options: "none", "ffill", "bfill", "linear", "cubic")
- `--output FILE` - Path to save processed data (default: processed_data.csv)

**Example:**
```bash
mfe utils preprocess --data prices.csv --log-returns --standardize --fill-method "linear" --output clean_returns.csv
```

#### Data Summary

```bash
mfe utils summary [options]
```

Generate statistical summary of financial data.

**Options:**
- `--data FILE` - Path to CSV file containing data
- `--statistics` - Include descriptive statistics (default: True)
- `--normality` - Test for normality (default: False)
- `--autocorr` - Test for autocorrelation (default: False)
- `--heterosked` - Test for heteroskedasticity (default: False)
- `--output FILE` - Path to save summary (default: stdout)

**Example:**
```bash
mfe utils summary --data returns.csv --statistics --normality --autocorr --output data_summary.json
```

## Common Usage Patterns

### Basic Time Series Analysis Workflow

1. Preprocess your data:
```bash
mfe utils preprocess --data raw_prices.csv --log-returns --output returns.csv
```

2. Check data properties:
```bash
mfe utils summary --data returns.csv --statistics --normality --autocorr --output returns_summary.json
```

3. Test for stationarity:
```bash
mfe test unitroot --data returns.csv --test "adf" --lags 4 --trend "c" --output unitroot_results.txt
```

4. Estimate ARMA model:
```bash
mfe timeseries armax --data returns.csv --p 2 --q 1 --constant --diagnostics --output arma_model.json
```

5. Generate forecasts:
```bash
mfe timeseries forecast --model arma_model.json --steps 10 --output forecast.csv
```

### Volatility Modeling Workflow

1. Preprocess your data:
```bash
mfe utils preprocess --data raw_prices.csv --log-returns --output returns.csv
```

2. Test for ARCH effects:
```bash
mfe test autocorr --data returns.csv --sq --lags 10 --test "ljungbox" --output arch_test.txt
```

3. Estimate GARCH model:
```bash
mfe volatility garch --data returns.csv --type "GJR-GARCH" --p 1 --q 1 --dist "student-t" --output garch_model.json --async
```

4. Forecast volatility:
```bash
mfe volatility forecast --model garch_model.json --steps 10 --output vol_forecast.csv
```

### High-Frequency Analysis Workflow

1. Preprocess intraday data:
```bash
mfe utils preprocess --data tick_data.csv --output clean_ticks.csv
```

2. Compute realized volatility:
```bash
mfe volatility realized --data clean_ticks.csv --sampling "calendar" --interval 5 --method "rv" --output rv_5min.csv
```

3. Analyze volatility patterns:
```bash
mfe utils summary --data rv_5min.csv --statistics --autocorr --output rv_summary.json
```

## Advanced Examples

### Batch Processing Multiple Series

Create a batch script to process multiple time series:

```bash
#!/bin/bash
# batch_process.sh

for file in data/*.csv; do
    base=$(basename "$file" .csv)
    echo "Processing $base..."
    
    # Preprocess data
    mfe utils preprocess --data "$file" --log-returns --output "processed/${base}_returns.csv"
    
    # Estimate ARMA model
    mfe timeseries armax --data "processed/${base}_returns.csv" --p 2 --q 1 --async --output "models/${base}_arma.json"
    
    # Estimate GARCH model
    mfe volatility garch --data "processed/${base}_returns.csv" --async --output "models/${base}_garch.json"
done
```

### Creating Custom Analysis Pipelines

Combine multiple commands in a comprehensive analysis script:

```bash
#!/bin/bash
# analyze_series.sh

# Input validation
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 data_file.csv"
    exit 1
fi

DATA_FILE=$1
BASE_NAME=$(basename "$DATA_FILE" .csv)
RESULTS_DIR="results/$BASE_NAME"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Step 1: Preprocess data
echo "Preprocessing data..."
mfe utils preprocess --data "$DATA_FILE" --log-returns --output "$RESULTS_DIR/returns.csv"

# Step 2: Generate summary statistics
echo "Generating summary statistics..."
mfe utils summary --data "$RESULTS_DIR/returns.csv" --statistics --normality --autocorr --output "$RESULTS_DIR/summary.json"

# Step 3: Test for stationarity
echo "Testing for stationarity..."
mfe test unitroot --data "$RESULTS_DIR/returns.csv" --test "adf" --output "$RESULTS_DIR/unitroot.txt"

# Step 4: Estimate ARMA model with async processing
echo "Estimating ARMA model..."
mfe timeseries armax --data "$RESULTS_DIR/returns.csv" --p 2 --q 1 --constant --diagnostics --async --output "$RESULTS_DIR/arma.json"

# Step 5: Estimate GARCH model with async processing
echo "Estimating GARCH model..."
mfe volatility garch --data "$RESULTS_DIR/returns.csv" --type "GJR-GARCH" --dist "student-t" --async --output "$RESULTS_DIR/garch.json"

# Step 6: Generate forecasts
echo "Generating forecasts..."
mfe timeseries forecast --model "$RESULTS_DIR/arma.json" --steps 10 --output "$RESULTS_DIR/arma_forecast.csv"
mfe volatility forecast --model "$RESULTS_DIR/garch.json" --steps 10 --output "$RESULTS_DIR/vol_forecast.csv"

echo "Analysis complete. Results saved to $RESULTS_DIR/"
```

## Troubleshooting

### Common Errors

#### Error: "Failed to read input file"

**Problem**: The CLI cannot read the specified input file.

**Solution**: 
- Verify the file path is correct
- Ensure the file exists and is readable
- Check that the file format is valid (e.g., proper CSV format)

#### Error: "Model estimation did not converge"

**Problem**: The optimization algorithm failed to converge to a stable solution.

**Solution**:
- Try a different model specification (e.g., different p, q values)
- Check your data for outliers or structural breaks
- Increase max iterations with `--max-iter`
- Use a different optimizer with `--optimizer`

#### Error: "ValueError: Parameter validation failed"

**Problem**: Input parameters are outside acceptable ranges.

**Solution**:
- Check parameter constraints in documentation
- For GARCH models, ensure persistence parameters sum to less than 1
- For ARMA models, ensure parameter values are within stationary/invertible regions

### Performance Tips

1. **Use async processing**: Enable the `--async` flag for long-running computations to maintain terminal responsiveness.
2. **Optimize input data size**: Trim your datasets to only what's needed for analysis.
3. **Use Numba acceleration**: The MFE Toolbox automatically uses Numba JIT compilation for performance-critical functions.
4. **Use efficient formats**: JSON is more efficient than CSV for complex model results.
5. **Batch process**: Use shell scripts for multiple operations rather than running each command separately.

## Reference

### Input Data Format

The MFE Toolbox CLI expects input data in CSV format by default:

- **Time series data**: Single column containing numeric values, optionally with a header.
- **Price data**: Single column containing numeric price values, optionally with a header.
- **Returns data**: Single column containing numeric return values, optionally with a header.
- **High-frequency data**: At minimum, two columns for time and price information.

Example time series CSV:
```
date,value
2023-01-01,100.25
2023-01-02,101.50
2023-01-03,99.75
...
```

### Output Formats

The MFE Toolbox CLI produces output in various formats:

- **JSON**: Model parameters, statistical tests, and complex results
- **CSV**: Time series data, forecasts, and simple numeric results
- **TXT**: Simple text output and summary statistics

### Environment Variables

The MFE Toolbox CLI respects the following environment variables:

- `MFE_CONFIG_PATH`: Path to a configuration file
- `MFE_DATA_PATH`: Default directory for input data
- `MFE_OUTPUT_PATH`: Default directory for output files
- `MFE_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MFE_NUM_THREADS`: Number of threads for parallel processing with Numba

Example usage:
```bash
export MFE_DATA_PATH="/path/to/data"
export MFE_OUTPUT_PATH="/path/to/results"
export MFE_LOG_LEVEL="INFO"
```

### Configuration File

You can create a configuration file to set default options:

```json
{
  "paths": {
    "data": "/path/to/data",
    "output": "/path/to/results"
  },
  "defaults": {
    "armax": {
      "p": 1,
      "q": 1,
      "constant": true
    },
    "garch": {
      "type": "GARCH",
      "p": 1,
      "q": 1,
      "dist": "normal"
    }
  },
  "numba": {
    "enabled": true,
    "threads": 4,
    "cache": true
  },
  "logging": {
    "level": "INFO",
    "file": "mfe_cli.log"
  }
}
```

### Global Command-Line Options

The following options are available to all commands:

- `--help, -h`: Display help message
- `--version, -v`: Display version information
- `--quiet, -q`: Suppress output to stdout
- `--verbose`: Increase verbosity
- `--log FILE`: Log output to specified file
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--no-progress`: Disable progress bars
- `--format FORMAT`: Specify output format (json, csv, txt)
- `--numba BOOL`: Enable/disable Numba JIT acceleration (default: True)
- `--threads INT`: Number of threads for Numba parallel processing