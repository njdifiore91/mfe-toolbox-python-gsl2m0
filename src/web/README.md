# MFE Toolbox Web UI

A comprehensive PyQt6-based graphical interface for the MFE (MATLAB Financial Econometrics) Toolbox, providing interactive modeling, visualization, and analysis tools for financial time series and econometric models.

## Overview

This web UI component implements a modern, interactive environment for econometric analysis using Python 3.12 and PyQt6. It offers:

- Interactive configuration of time series and volatility models
- Real-time parameter visualization with statistical inference
- Comprehensive diagnostic plots with Matplotlib integration
- Asynchronous processing for responsive user experience
- Theme-aware styling with light and dark mode support

## Features

- **Model Configuration**: Interactive interface for configuring ARMA/ARMAX, GARCH, EGARCH, and other econometric models
- **Asynchronous Estimation**: Non-blocking model estimation with progress tracking
- **Interactive Diagnostics**: Real-time plotting for residual analysis, autocorrelation, and other diagnostics
- **Parameter Inference**: Statistical tables for parameter estimates, standard errors, t-statistics, and p-values
- **Model Equation Display**: Visual representation of estimated model equations
- **Statistical Testing**: Comprehensive display of model fit statistics and diagnostic test results
- **Theme Support**: Consistent styling with light and dark theme options
- **Cross-Platform**: Functions across Windows, macOS, and Linux environments

## Components

The Web UI consists of the following main components:

1. **Main Application Window** (`MainWindow`): Central interface coordinating all UI components
2. **Model Configuration Panel** (`ModelConfig`): Parameter input and validation for econometric models
3. **Diagnostic Plots Widget** (`DiagnosticPlotsWidget`): Interactive visualization for model assessment
4. **Results Viewer** (`ResultsViewer`): Detailed display of estimation results and statistics
5. **Plot Display** (`PlotDisplay`): Matplotlib-backed plotting system with PyQt6 integration

## Installation

### Requirements

- Python 3.12 or later
- PyQt6 6.6.1 or later
- NumPy 1.26.3 or later
- SciPy 1.11.4 or later
- Matplotlib 3.8.2 or later
- Pandas 2.1.4 or later
- Statsmodels 0.14.1 or later
- Numba 0.59.0 or later

### Installation Steps

1. Ensure you have Python 3.12 installed:

   ```bash
   python --version
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv mfe-env
   # On Windows
   mfe-env\Scripts\activate
   # On macOS/Linux
   source mfe-env/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install PyQt6==6.6.1 numpy==1.26.3 scipy==1.11.4 matplotlib==3.8.2 pandas==2.1.4 statsmodels==0.14.1 numba==0.59.0
   ```

4. Install the MFE Toolbox package:

   ```bash
   pip install mfe-toolbox
   ```

   Or install from source:

   ```bash
   git clone https://github.com/username/mfe-toolbox.git
   cd mfe-toolbox
   pip install -e .
   ```

### Development Setup

For development or customization of the web UI:

1. Clone the repository:

   ```bash
   git clone https://github.com/username/mfe-toolbox.git
   cd mfe-toolbox
   ```

2. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. Run the development version:

   ```bash
   python -m src.web.app
   ```

## Usage Guide

### Starting the Application

To launch the MFE Toolbox GUI:

```python
from mfe.ui import start_app

# Start the application
start_app()
```

Or from the command line:

```bash
python -m mfe.ui
```

### Model Configuration

1. **Select Model Type**: Choose from ARMA, GARCH, EGARCH, APARCH, or FIGARCH models in the dropdown menu.

2. **Set Model Parameters**:
   - For ARMA models: Specify AR order, MA order, and whether to include a constant
   - For GARCH models: Set ALPHA and BETA parameters, along with model-specific options

3. **Data Selection**:
   - Import data from CSV/Excel files
   - Use sample datasets for demonstration
   - Input data programmatically

### Estimation Workflow

1. **Configure Model**: Select model type and set parameters as described above.

2. **Estimate Model**: Click the "Estimate Model" button to begin the estimation process.
   - A progress indicator will show the status of the estimation
   - The interface remains responsive during estimation due to async processing

3. **Review Diagnostics**: Once estimation completes, the diagnostic plots will automatically update with:
   - Residual plots
   - Autocorrelation function (ACF) plots
   - QQ plots for residual normality assessment
   - Histogram of residuals with distribution overlay

### Interpreting Results

1. **View Results**: Click "View Results" to open the detailed results viewer.

2. **Model Equation**: The top panel displays the estimated model equation with parameter values.

3. **Parameter Estimates**:
   - Review parameter values, standard errors, t-statistics, and p-values
   - Statistically significant parameters are highlighted

4. **Statistical Metrics**:
   - Information criteria (AIC, BIC) for model comparison
   - Log-likelihood value
   - Diagnostic test results including Ljung-Box and Jarque-Bera tests

5. **Navigate Pages**: Use the "Previous" and "Next" buttons to toggle between parameter tables and diagnostic plots.

### Working with Plots

1. **Interaction**: All plots support interactive features:
   - Zoom in/out with the mouse wheel or zoom tool
   - Pan by clicking and dragging
   - Reset view with the home button

2. **Switching Views**: Use the tab interface to switch between:
   - Residual plots
   - ACF plots
   - Comprehensive diagnostics

3. **Exporting**: Use the plot toolbar to:
   - Save plots as images (PNG, PDF, SVG)
   - Copy to clipboard
   - Print plots

## Examples

### Basic ARMA Model Estimation

1. Launch the application
2. Select "ARMA Model" from the model type dropdown
3. Set AR Order to 1
4. Set MA Order to 1
5. Check "Include Constant"
6. Click "Estimate Model"
7. Review the diagnostic plots for residual patterns
8. Click "View Results" to see parameter estimates and statistics

### GARCH Volatility Modeling

1. Launch the application
2. Select "GARCH Model" from the model type dropdown
3. Set ALPHA parameter to 0.1
4. Set BETA parameter to 0.8
5. Click "Estimate Model" 
6. Examine volatility persistence through parameter estimates
7. Review diagnostic plots for volatility clustering

## Troubleshooting

### Common Issues

1. **PyQt6 Installation Errors**
   - Ensure you have the proper development libraries installed
   - On Linux: `sudo apt-get install python3-dev qt6-dev-tools`
   - On macOS: `brew install qt6`

2. **Plotting Issues**
   - If plots don't display correctly, update Matplotlib: `pip install --upgrade matplotlib`
   - For high-DPI displays, set: `export QT_AUTO_SCREEN_SCALE_FACTOR=1`

3. **Performance Problems**
   - Ensure Numba is properly installed: `pip install --upgrade numba`
   - Check that your NumPy installation uses optimized BLAS/LAPACK libraries

## Contributing

Contributions to improve the Web UI are welcome! Please see our contributing guidelines in the main repository.

## License

This software is released under the MIT License. See the LICENSE file for details.