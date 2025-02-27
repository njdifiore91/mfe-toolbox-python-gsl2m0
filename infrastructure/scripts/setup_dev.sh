#!/bin/bash
# setup_dev.sh
#
# Development environment setup script for MFE Toolbox
# This script sets up a Python 3.12 virtual environment with all required
# dependencies and development tools for the MFE Toolbox project.
#
# Usage: ./setup_dev.sh

set -e  # Exit immediately if a command exits with a non-zero status

# ANSI color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print section header
print_header() {
    echo -e "\n${BLUE}===== $1 =====${NC}\n"
}

# Print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# Check if Python 3.12 is available
check_python_version() {
    print_header "Checking Python Version"
    
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &> /dev/null && python3 --version | grep -q "Python 3.12"; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null && python --version | grep -q "Python 3.12"; then
        PYTHON_CMD="python"
    else
        print_error "Python 3.12 is required but not found."
        print_warning "Please install Python 3.12 and try again."
        print_warning "Visit https://www.python.org/downloads/ for installation instructions."
        return 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version)
    print_success "Found $PYTHON_VERSION"
    return 0
}

# Create Python virtual environment
create_venv() {
    print_header "Creating Virtual Environment"
    
    # Check if .venv directory already exists
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists at .venv"
        read -p "Do you want to recreate it? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping virtual environment creation"
            return 0
        fi
        rm -rf .venv
        print_warning "Removed existing virtual environment"
    fi
    
    # Create virtual environment
    echo "Creating new virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv .venv
    
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created successfully at .venv"
        return 0
    else
        print_error "Failed to create virtual environment"
        return 1
    fi
}

# Install required dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    # Determine the activate script based on platform
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        ACTIVATE=".venv/Scripts/activate"
    else
        ACTIVATE=".venv/bin/activate"
    fi
    
    # Activate virtual environment
    if [ ! -f "$ACTIVATE" ]; then
        print_error "Virtual environment activate script not found at $ACTIVATE"
        return 1
    fi
    
    echo "Activating virtual environment..."
    source "$ACTIVATE"
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    # Install core scientific packages
    echo "Installing core scientific packages..."
    pip install numpy==1.26.3 scipy==1.11.4 pandas==2.1.4 statsmodels==0.14.1
    
    # Install Numba for optimization
    echo "Installing Numba for optimization..."
    pip install numba==0.59.0
    
    # Install PyQt6 for GUI
    echo "Installing PyQt6 for GUI..."
    pip install PyQt6==6.6.1
    
    # Install development tools
    echo "Installing development tools..."
    pip install pytest==7.4.3 pytest-cov pytest-asyncio pytest-benchmark pytest-memray
    pip install mypy==1.7.0 hypothesis
    
    # Install package in development mode
    echo "Installing package in development mode..."
    pip install -e .
    
    # Check for installation errors
    if [ $? -eq 0 ]; then
        print_success "All dependencies installed successfully"
        # Print installed package versions
        echo "Installed package versions:"
        pip list | grep -E "numpy|scipy|pandas|statsmodels|numba|PyQt6|pytest|mypy"
        return 0
    else
        print_error "Failed to install dependencies"
        return 1
    fi
}

# Configure development tools
configure_dev_tools() {
    print_header "Configuring Development Tools"
    
    # Ensure config directory exists
    mkdir -p infrastructure/config
    mkdir -p .mypy_cache
    mkdir -p .pytest_cache
    
    # Configure pytest
    echo "Configuring pytest..."
    if [ ! -f "infrastructure/config/pytest.ini" ]; then
        cat > infrastructure/config/pytest.ini << 'EOL'
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
EOL
        print_success "Created pytest configuration at infrastructure/config/pytest.ini"
    else
        print_success "pytest configuration already exists at infrastructure/config/pytest.ini"
    fi
    
    # Create symlink to pytest.ini in project root for convenience
    if [ ! -f "pytest.ini" ]; then
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            # Windows - copy the file instead of symlink
            cp infrastructure/config/pytest.ini pytest.ini
        else
            # Unix - create symlink
            ln -sf infrastructure/config/pytest.ini pytest.ini
        fi
        print_success "Created pytest.ini reference in project root"
    fi
    
    # Configure mypy
    echo "Configuring mypy..."
    if [ ! -f "infrastructure/config/mypy.ini" ]; then
        cat > infrastructure/config/mypy.ini << 'EOL'
[mypy]
python_version = 3.12
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_return_any = True
warn_unreachable = True
strict_optional = True
strict_equality = True

[mypy.plugins.numpy.*]
plugin_modules = numpy.typing.mypy_plugin

[mypy-numba.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True

[mypy-statsmodels.*]
ignore_missing_imports = True

[mypy-PyQt6.*]
ignore_missing_imports = True
EOL
        print_success "Created mypy configuration at infrastructure/config/mypy.ini"
    else
        print_success "mypy configuration already exists at infrastructure/config/mypy.ini"
    fi
    
    # Create symlink to mypy.ini in project root for convenience
    if [ ! -f "mypy.ini" ]; then
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            # Windows - copy the file instead of symlink
            cp infrastructure/config/mypy.ini mypy.ini
        else
            # Unix - create symlink
            ln -sf infrastructure/config/mypy.ini mypy.ini
        fi
        print_success "Created mypy.ini reference in project root"
    fi
    
    # Set up pre-commit hooks
    echo "Setting up pre-commit hooks..."
    cat > .pre-commit-config.yaml << 'EOL'
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
    -   id: mypy
        additional_dependencies: [numpy>=1.26.3]
EOL
    
    # Install pre-commit if available
    if command -v pip &> /dev/null; then
        pip install pre-commit
        if command -v pre-commit &> /dev/null; then
            pre-commit install
            print_success "Pre-commit hooks installed"
        else
            print_warning "Pre-commit not found after installation, hooks not installed"
        fi
    else
        print_warning "Pip not available, skipping pre-commit installation"
    fi
    
    # Configure development environment variables
    echo "Configuring environment variables..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        cat > .env << 'EOL'
# MFE Toolbox development environment variables
PYTHONPATH=.
NUMBA_THREADING_LAYER=omp
NUMBA_NUM_THREADS=0  # Auto-detect threads
NUMBA_CACHE_DIR=.numba_cache
PYTHONDONTWRITEBYTECODE=1
PYTEST_ADDOPTS="--color=yes"
EOL
    else
        # Unix/Linux/MacOS
        cat > .env << 'EOL'
# MFE Toolbox development environment variables
export PYTHONPATH=.
export NUMBA_THREADING_LAYER=omp
export NUMBA_NUM_THREADS=0  # Auto-detect threads
export NUMBA_CACHE_DIR=.numba_cache
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="--color=yes"
EOL
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Development tools configured successfully"
        return 0
    else
        print_error "Failed to configure development tools"
        return 1
    fi
}

# Main function to run all setup steps
main() {
    print_header "MFE Toolbox Development Environment Setup"
    
    # Step 1: Check Python version
    check_python_version
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # Step 2: Create virtual environment
    create_venv
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # Step 3: Install dependencies
    install_dependencies
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # Step 4: Configure development tools
    configure_dev_tools
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # All steps completed successfully
    print_header "Setup Complete"
    print_success "MFE Toolbox development environment is ready!"
    print_success "To activate the environment, run:"
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo -e "${GREEN}    .venv\\Scripts\\activate${NC}"
    else
        echo -e "${GREEN}    source .venv/bin/activate${NC}"
    fi
    
    print_success "For development helpers, see the scripts in infrastructure/scripts/"
    
    return 0
}

# Run the main function
main

exit 0