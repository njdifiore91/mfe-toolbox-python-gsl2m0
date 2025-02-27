#!/bin/bash
#
# setup_prod.sh - Production Environment Setup for MFE Toolbox
#
# Sets up the production environment for the MFE Toolbox, including:
# - Python 3.12 validation
# - Virtual environment creation
# - Dependency installation
# - Installation validation
#
# Usage: ./setup_prod.sh [venv_path]
#   venv_path: Optional path for virtual environment (default: .venv)

set -e  # Exit immediately if a command exits with non-zero status

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Get virtual environment path from command line or use default
VENV_PATH="${1:-.venv}"
REQUIREMENTS_BACKEND="src/backend/requirements.txt"
REQUIREMENTS_WEB="src/web/requirements.txt"

# Print header
echo -e "${YELLOW}==================================================${NC}"
echo -e "${YELLOW}   MFE Toolbox Production Environment Setup       ${NC}"
echo -e "${YELLOW}==================================================${NC}"
echo

# Function to check if Python 3.12 is available
check_python_version() {
    echo -e "${YELLOW}Checking Python version...${NC}"
    
    if ! command -v python3.12 &> /dev/null; then
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}Error: Python 3 not found. Please install Python 3.12.${NC}"
            return 1
        fi
        
        # Check if python3 is at least version 3.12
        PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
        PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)
        
        if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]); then
            echo -e "${RED}Error: Python 3.12 or newer is required, found $PY_VERSION.${NC}"
            return 1
        fi
        
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python3.12"
    fi
    
    echo -e "${GREEN}Python version check passed. Using $(${PYTHON_CMD} --version)${NC}"
    export PYTHON_CMD
    return 0
}

# Function to set up the virtual environment
setup_virtual_env() {
    echo -e "${YELLOW}Creating virtual environment at $VENV_PATH...${NC}"
    
    # Remove existing virtual environment if it exists
    if [ -d "$VENV_PATH" ]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_PATH"
    fi
    
    # Create virtual environment
    $PYTHON_CMD -m venv "$VENV_PATH"
    
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Failed to create virtual environment.${NC}"
        return 1
    fi
    
    # Determine activation script based on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        ACTIVATE_SCRIPT="$VENV_PATH/Scripts/activate"
    else
        ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
    fi
    
    # Check if activation script exists
    if [ ! -f "$ACTIVATE_SCRIPT" ]; then
        echo -e "${RED}Error: Virtual environment activation script not found at $ACTIVATE_SCRIPT.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    
    # Source the activation script
    source "$ACTIVATE_SCRIPT"
    
    # Check if activation was successful by examining VIRTUAL_ENV variable
    if [ -z "$VIRTUAL_ENV" ]; then
        echo -e "${RED}Error: Failed to activate virtual environment.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Virtual environment activated successfully.${NC}"
    
    # Upgrade pip to the latest version
    echo -e "${YELLOW}Upgrading pip...${NC}"
    pip install --upgrade pip
    
    # Install wheel for binary package installations
    echo -e "${YELLOW}Installing wheel...${NC}"
    pip install wheel
    
    echo -e "${GREEN}Virtual environment setup completed.${NC}"
    return 0
}

# Function to install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    
    # Check if requirements files exist
    if [ ! -f "$REQUIREMENTS_BACKEND" ]; then
        echo -e "${RED}Error: Backend requirements file not found at $REQUIREMENTS_BACKEND.${NC}"
        return 1
    fi
    
    if [ ! -f "$REQUIREMENTS_WEB" ]; then
        echo -e "${RED}Error: Web requirements file not found at $REQUIREMENTS_WEB.${NC}"
        return 1
    fi
    
    # Install backend requirements
    echo -e "${YELLOW}Installing backend requirements...${NC}"
    pip install -r "$REQUIREMENTS_BACKEND"
    
    # Install web/UI requirements
    echo -e "${YELLOW}Installing web/UI requirements...${NC}"
    pip install -r "$REQUIREMENTS_WEB"
    
    echo -e "${GREEN}Dependency installation completed.${NC}"
    return 0
}

# Function to validate installation
validate_installation() {
    echo -e "${YELLOW}Validating installation...${NC}"
    
    # List of required packages to validate
    PACKAGES=(
        "numpy"
        "scipy"
        "pandas"
        "statsmodels"
        "numba"
        "PyQt6"
    )
    
    # Check each package
    for pkg in "${PACKAGES[@]}"; do
        echo -e "${YELLOW}Checking $pkg...${NC}"
        if ! python -c "import $pkg; print(f'$pkg {$pkg.__version__} installed successfully')" 2>/dev/null; then
            echo -e "${RED}Error: Failed to import $pkg. Installation validation failed.${NC}"
            return 1
        fi
    done
    
    # Validate Numba JIT compilation
    echo -e "${YELLOW}Validating Numba JIT compilation...${NC}"
    if ! python -c "
from numba import jit
import numpy as np

@jit(nopython=True)
def test_function(x):
    return np.sum(x)

# Test with a simple array
result = test_function(np.array([1, 2, 3, 4, 5]))
print(f'Numba JIT compilation works! Test result: {result}')
" 2>/dev/null; then
        echo -e "${RED}Error: Numba JIT compilation validation failed.${NC}"
        return 1
    fi
    
    # Basic PyQt6 validation
    echo -e "${YELLOW}Validating PyQt6 initialization...${NC}"
    if ! python -c "
from PyQt6.QtWidgets import QApplication
import sys

# Initialize QApplication
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)
print('PyQt6 initialized successfully')
" 2>/dev/null; then
        echo -e "${RED}Error: PyQt6 initialization validation failed.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Installation validation completed successfully.${NC}"
    return 0
}

# Main execution
main() {
    # Check Python version
    if ! check_python_version; then
        echo -e "${RED}Environment setup failed at Python version check.${NC}"
        exit 1
    fi
    
    # Set up virtual environment
    if ! setup_virtual_env; then
        echo -e "${RED}Environment setup failed at virtual environment creation.${NC}"
        exit 1
    fi
    
    # Install dependencies
    if ! install_dependencies; then
        echo -e "${RED}Environment setup failed at dependency installation.${NC}"
        exit 1
    fi
    
    # Validate installation
    if ! validate_installation; then
        echo -e "${RED}Environment setup failed at installation validation.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}==================================================${NC}"
    echo -e "${GREEN}   MFE Toolbox production environment setup       ${NC}"
    echo -e "${GREEN}   completed successfully.                        ${NC}"
    echo -e "${GREEN}==================================================${NC}"
    echo
    echo -e "${YELLOW}To activate the virtual environment:${NC}"
    echo -e "  source $VENV_PATH/bin/activate  # Linux/macOS"
    echo -e "  $VENV_PATH\\Scripts\\activate     # Windows"
    echo
    
    exit 0
}

# Execute main function
main