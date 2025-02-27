#!/bin/bash
# test.sh - Comprehensive test suite for the MFE Toolbox
# Executes unit tests, integration tests, property-based tests and performance benchmarks
# using pytest as the primary test runner with hypothesis for property-based testing
# and numba.testing for performance validation

set -e  # Exit on error

# Color definitions for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

setup_venv() {
    echo -e "${YELLOW}Setting up virtual environment...${NC}"
    
    # Define virtual environment directory
    VENV_DIR="$PROJECT_ROOT/.venv"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3.12 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to create virtual environment. Is Python 3.12 installed?${NC}"
            return 1
        fi
    else
        echo "Virtual environment already exists."
    fi
    
    # Activate virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        source "$VENV_DIR/Scripts/activate"  # Windows
    fi
    
    # Verify Python version
    python_version=$(python --version)
    if [[ $python_version != *"Python 3.12"* ]]; then
        echo -e "${RED}Error: Python 3.12 is required. Found: $python_version${NC}"
        return 1
    fi
    
    # Install test dependencies with exact versions
    echo "Installing test dependencies..."
    pip install -q pytest==7.4.3 pytest-asyncio==0.21.1 pytest-cov==4.1.0 hypothesis==6.92.1
    pip install -q pytest-benchmark pytest-memray numba
    
    # Install package in development/editable mode
    echo "Installing package in editable mode..."
    pip install -q -e "$PROJECT_ROOT"
    
    # Configure environment variables
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export NUMBA_DISABLE_JIT=0  # Ensure Numba JIT is enabled for tests
    
    echo -e "${GREEN}Virtual environment setup complete.${NC}"
    return 0
}

run_tests() {
    echo -e "${YELLOW}Running test suite...${NC}"
    
    # Create directories for test reports if they don't exist
    mkdir -p "$PROJECT_ROOT/test-reports"
    
    # Run pytest with configuration from pytest.ini
    cd "$PROJECT_ROOT"
    
    echo -e "${BLUE}Executing comprehensive test suite...${NC}"
    echo -e "${BLUE}This includes unit tests, integration tests, property-based tests, and benchmarks${NC}"
    echo -e "${BLUE}as configured in pytest.ini${NC}"
    
    # Run pytest with all tests
    python -m pytest
    test_result=$?
    
    if [ $test_result -ne 0 ]; then
        echo -e "${RED}Tests failed with exit code $test_result${NC}"
        return $test_result
    fi
    
    # Check coverage compliance
    echo -e "${BLUE}Verifying code coverage requirements...${NC}"
    # The coverage requirements are already specified in pytest.ini and coverage.ini
    # So this is just an extra explicit check
    coverage_report=$(coverage report 2>&1)
    if [[ $coverage_report == *"FAIL"* ]]; then
        echo -e "${RED}Coverage requirements not met. Minimum 90% coverage required.${NC}"
        echo "$coverage_report"
        return 1
    fi
    
    echo -e "${GREEN}All tests passed successfully with required coverage.${NC}"
    return 0
}

cleanup() {
    echo -e "${YELLOW}Performing cleanup...${NC}"
    
    # Archive test reports
    if [ -d "$PROJECT_ROOT/coverage_html" ]; then
        echo "Archiving coverage reports..."
        timestamp=$(date +"%Y%m%d_%H%M%S")
        mkdir -p "$PROJECT_ROOT/test-archives"
        tar -czf "$PROJECT_ROOT/test-archives/coverage_${timestamp}.tar.gz" -C "$PROJECT_ROOT" coverage_html
    fi
    
    # Archive benchmark results
    if [ -d "$PROJECT_ROOT/.benchmarks" ]; then
        echo "Archiving benchmark results..."
        timestamp=$(date +"%Y%m%d_%H%M%S")
        mkdir -p "$PROJECT_ROOT/test-archives"
        tar -czf "$PROJECT_ROOT/test-archives/benchmarks_${timestamp}.tar.gz" -C "$PROJECT_ROOT" .benchmarks
    fi
    
    # Clean pytest cache
    echo "Cleaning pytest cache..."
    find "$PROJECT_ROOT" -name ".pytest_cache" -type d -exec rm -rf {} +>/dev/null 2>&1 || true
    
    # Remove hypothesis examples database
    echo "Cleaning hypothesis database..."
    find "$PROJECT_ROOT" -name ".hypothesis" -type d -exec rm -rf {} +>/dev/null 2>&1 || true
    
    # Clean __pycache__ directories
    echo "Cleaning __pycache__ directories..."
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} +>/dev/null 2>&1 || true
    
    # Reset environment variables
    unset NUMBA_DISABLE_JIT
    
    # Deactivate virtual environment if active
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Deactivating virtual environment..."
        deactivate
    fi
    
    echo -e "${GREEN}Cleanup completed.${NC}"
    return 0
}

# Main script execution
main() {
    echo -e "${GREEN}=== MFE Toolbox Test Suite ===${NC}"
    echo "Starting tests at $(date)"
    
    # Setup virtual environment
    setup_venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to set up virtual environment. Exiting.${NC}"
        exit 1
    fi
    
    # Run tests
    run_tests
    test_result=$?
    
    # Cleanup
    cleanup
    
    # Exit with test result
    if [ $test_result -eq 0 ]; then
        echo -e "${GREEN}Test suite completed successfully at $(date)${NC}"
        exit 0
    else
        echo -e "${RED}Test suite failed at $(date)${NC}"
        exit $test_result
    fi
}

# Execute main function or allow sourcing for individual functions
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Run script directly
    main "$@"
else
    # Script is being sourced, export functions
    export -f setup_venv
    export -f run_tests
    export -f cleanup
    echo "Functions exported: setup_venv, run_tests, cleanup"
fi