#!/bin/bash
# clean.sh - Script to clean build artifacts, temporary files, and generated content
# for the MFE Toolbox Python package

# Constants
MFE_ROOT=$(pwd)
PYTHON_VERSION="3.12"
VENV_DIR=".venv"
DIST_DIR="dist"
BUILD_DIR="build"
NUMBA_CACHE_DIR="__pycache__/_numba_cache"

# Colors for terminal output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Initialize counters for summary
REMOVED_FILES=0
REMOVED_DIRS=0

# Function to log error messages
log_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

# Function to log warning messages
log_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}" >&2
}

# Function to log success messages
log_success() {
    echo -e "${GREEN}$1${NC}"
}

# Function to remove Python build artifacts and caches
clean_python_artifacts() {
    echo -e "${YELLOW}Cleaning Python artifacts...${NC}"
    
    # Remove __pycache__ directories recursively
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove Python __pycache__ directories"
        pycache_count=$(find . -type d -name "__pycache__" | wc -l)
        echo "  Found $pycache_count __pycache__ directories"
    else
        pycache_count=0
        while IFS= read -r dir; do
            rm -rf "$dir" 2>/dev/null && ((pycache_count++)) || log_warning "Failed to remove $dir"
        done < <(find . -type d -name "__pycache__")
        log_success "Removed $pycache_count __pycache__ directories"
        REMOVED_DIRS=$((REMOVED_DIRS + pycache_count))
    fi
    
    # Remove .pyc and .pyo files (compiled bytecode)
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove .pyc and .pyo files"
        pyc_count=$(find . -type f -name "*.pyc" | wc -l)
        pyo_count=$(find . -type f -name "*.pyo" | wc -l)
        echo "  Found $pyc_count .pyc files and $pyo_count .pyo files"
    else
        pyc_count=0
        pyo_count=0
        
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((pyc_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name "*.pyc")
        
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((pyo_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name "*.pyo")
        
        log_success "Removed $pyc_count .pyc files and $pyo_count .pyo files"
        REMOVED_FILES=$((REMOVED_FILES + pyc_count + pyo_count))
    fi
    
    # Remove Numba JIT cache directories
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove Numba cache directories"
        numba_count=$(find . -name "_numba_cache" -type d | wc -l)
        echo "  Found $numba_count Numba cache directories"
    else
        numba_count=0
        while IFS= read -r dir; do
            rm -rf "$dir" 2>/dev/null && ((numba_count++)) || log_warning "Failed to remove $dir"
        done < <(find . -name "_numba_cache" -type d)
        log_success "Removed $numba_count Numba cache directories"
        REMOVED_DIRS=$((REMOVED_DIRS + numba_count))
    fi
    
    # Remove .coverage files
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove .coverage files"
        coverage_count=$(find . -type f -name ".coverage" | wc -l)
        echo "  Found $coverage_count .coverage files"
    else
        coverage_count=0
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((coverage_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name ".coverage")
        log_success "Removed $coverage_count .coverage files"
        REMOVED_FILES=$((REMOVED_FILES + coverage_count))
    fi
    
    # Remove .pytest_cache directories
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove .pytest_cache directories"
        pytest_count=$(find . -type d -name ".pytest_cache" | wc -l)
        echo "  Found $pytest_count .pytest_cache directories"
    else
        pytest_count=0
        while IFS= read -r dir; do
            rm -rf "$dir" 2>/dev/null && ((pytest_count++)) || log_warning "Failed to remove $dir"
        done < <(find . -type d -name ".pytest_cache")
        log_success "Removed $pytest_count .pytest_cache directories"
        REMOVED_DIRS=$((REMOVED_DIRS + pytest_count))
    fi
    
    # Remove .mypy_cache directories
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove .mypy_cache directories"
        mypy_count=$(find . -type d -name ".mypy_cache" | wc -l)
        echo "  Found $mypy_count .mypy_cache directories"
    else
        mypy_count=0
        while IFS= read -r dir; do
            rm -rf "$dir" 2>/dev/null && ((mypy_count++)) || log_warning "Failed to remove $dir"
        done < <(find . -type d -name ".mypy_cache")
        log_success "Removed $mypy_count .mypy_cache directories"
        REMOVED_DIRS=$((REMOVED_DIRS + mypy_count))
    fi
    
    # Remove .hypothesis test directories
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove .hypothesis directories"
        hypothesis_count=$(find . -type d -name ".hypothesis" | wc -l)
        echo "  Found $hypothesis_count .hypothesis directories"
    else
        hypothesis_count=0
        while IFS= read -r dir; do
            rm -rf "$dir" 2>/dev/null && ((hypothesis_count++)) || log_warning "Failed to remove $dir"
        done < <(find . -type d -name ".hypothesis")
        log_success "Removed $hypothesis_count .hypothesis directories"
        REMOVED_DIRS=$((REMOVED_DIRS + hypothesis_count))
    fi
    
    log_success "Python artifacts cleaned."
    return 0
}

# Function to clean Python package build artifacts
clean_build_artifacts() {
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    
    # Remove dist directory (wheels and source distributions)
    if [ -d "$DIST_DIR" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove $DIST_DIR directory"
        else
            rm -rf "$DIST_DIR" && log_success "Removed $DIST_DIR directory" || log_warning "Failed to remove $DIST_DIR directory"
            REMOVED_DIRS=$((REMOVED_DIRS + 1))
        fi
    fi
    
    # Remove build directory
    if [ -d "$BUILD_DIR" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove $BUILD_DIR directory"
        else
            rm -rf "$BUILD_DIR" && log_success "Removed $BUILD_DIR directory" || log_warning "Failed to remove $BUILD_DIR directory"
            REMOVED_DIRS=$((REMOVED_DIRS + 1))
        fi
    fi
    
    # Remove *.egg-info directories
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove *.egg-info directories"
        egginfo_count=$(find . -type d -name "*.egg-info" | wc -l)
        echo "  Found $egginfo_count *.egg-info directories"
    else
        egginfo_count=0
        while IFS= read -r dir; do
            rm -rf "$dir" 2>/dev/null && ((egginfo_count++)) || log_warning "Failed to remove $dir"
        done < <(find . -type d -name "*.egg-info")
        log_success "Removed $egginfo_count *.egg-info directories"
        REMOVED_DIRS=$((REMOVED_DIRS + egginfo_count))
    fi
    
    # Remove .eggs directory
    if [ -d ".eggs" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove .eggs directory"
        else
            rm -rf ".eggs" && log_success "Removed .eggs directory" || log_warning "Failed to remove .eggs directory"
            REMOVED_DIRS=$((REMOVED_DIRS + 1))
        fi
    fi
    
    # Remove pip-wheel-metadata
    if [ -d "pip-wheel-metadata" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove pip-wheel-metadata directory"
        else
            rm -rf "pip-wheel-metadata" && log_success "Removed pip-wheel-metadata directory" || log_warning "Failed to remove pip-wheel-metadata directory"
            REMOVED_DIRS=$((REMOVED_DIRS + 1))
        fi
    fi
    
    log_success "Build artifacts cleaned."
    return 0
}

# Function to clean temporary files
clean_temp_files() {
    echo -e "${YELLOW}Cleaning temporary files...${NC}"
    
    # Remove .log files
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove .log files"
        log_count=$(find . -type f -name "*.log" | wc -l)
        echo "  Found $log_count .log files"
    else
        log_count=0
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((log_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name "*.log")
        log_success "Removed $log_count .log files"
        REMOVED_FILES=$((REMOVED_FILES + log_count))
    fi
    
    # Remove .tmp files
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove .tmp files"
        tmp_count=$(find . -type f -name "*.tmp" | wc -l)
        echo "  Found $tmp_count .tmp files"
    else
        tmp_count=0
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((tmp_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name "*.tmp")
        log_success "Removed $tmp_count .tmp files"
        REMOVED_FILES=$((REMOVED_FILES + tmp_count))
    fi
    
    # Remove .bak files
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove .bak files"
        bak_count=$(find . -type f -name "*.bak" | wc -l)
        echo "  Found $bak_count .bak files"
    else
        bak_count=0
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((bak_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name "*.bak")
        log_success "Removed $bak_count .bak files"
        REMOVED_FILES=$((REMOVED_FILES + bak_count))
    fi
    
    # Remove PyQt6 temporary files
    if [ "$DRY_RUN" = true ]; then
        echo "Would remove PyQt6 temporary files"
        qmlc_count=$(find . -type f -name "*.qmlc" | wc -l)
        jsc_count=$(find . -type f -name "*.jsc" | wc -l)
        pyqt6cache_count=$(find . -type f -name "*.pyqt6cache" | wc -l)
        qmake_count=$(find . -type f -name ".qmake.stash" | wc -l)
        echo "  Found $qmlc_count .qmlc files, $jsc_count .jsc files, $pyqt6cache_count .pyqt6cache files, and $qmake_count .qmake.stash files"
    else
        qmlc_count=0
        jsc_count=0
        pyqt6cache_count=0
        qmake_count=0
        
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((qmlc_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name "*.qmlc")
        
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((jsc_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name "*.jsc")
        
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((pyqt6cache_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name "*.pyqt6cache")
        
        while IFS= read -r file; do
            rm -f "$file" 2>/dev/null && ((qmake_count++)) || log_warning "Failed to remove $file"
        done < <(find . -type f -name ".qmake.stash")
        
        log_success "Removed $qmlc_count .qmlc files, $jsc_count .jsc files, $pyqt6cache_count .pyqt6cache files, and $qmake_count .qmake.stash files"
        REMOVED_FILES=$((REMOVED_FILES + qmlc_count + jsc_count + pyqt6cache_count + qmake_count))
    fi
    
    # Clean temporary test files
    if [ -d "./tests" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove temporary test files"
            temp_test_count=$(find ./tests -type f -name "temp_*" 2>/dev/null | wc -l)
            echo "  Found $temp_test_count temporary test files"
        else
            temp_test_count=0
            while IFS= read -r file; do
                rm -f "$file" 2>/dev/null && ((temp_test_count++)) || log_warning "Failed to remove $file"
            done < <(find ./tests -type f -name "temp_*" 2>/dev/null || true)
            log_success "Removed $temp_test_count temporary test files"
            REMOVED_FILES=$((REMOVED_FILES + temp_test_count))
        fi
    fi
    
    # Remove pytest benchmark data
    if [ -d ".benchmarks" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove .benchmarks directory"
        else
            rm -rf ".benchmarks" && log_success "Removed .benchmarks directory" || log_warning "Failed to remove .benchmarks directory"
            REMOVED_DIRS=$((REMOVED_DIRS + 1))
        fi
    fi
    
    log_success "Temporary files cleaned."
    return 0
}

# Function to clean virtual environment
clean_venv() {
    echo -e "${YELLOW}Cleaning virtual environment...${NC}"
    
    # Check if virtual environment exists
    if [ -d "$VENV_DIR" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove $VENV_DIR directory"
        else
            rm -rf "$VENV_DIR" && log_success "Removed $VENV_DIR directory" || log_warning "Failed to remove $VENV_DIR directory"
            REMOVED_DIRS=$((REMOVED_DIRS + 1))
        fi
    else
        log_warning "No virtual environment found at $VENV_DIR."
    fi
    
    # Remove pip cache directory
    if [ -d ".pip_cache" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would remove .pip_cache directory"
        else
            rm -rf ".pip_cache" && log_success "Removed .pip_cache directory" || log_warning "Failed to remove .pip_cache directory"
            REMOVED_DIRS=$((REMOVED_DIRS + 1))
        fi
    fi
    
    return 0
}

# Main function
main() {
    # Parse command line arguments
    DRY_RUN=false
    CLEAN_VENV=false
    
    for arg in "$@"; do
        case $arg in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --clean-venv)
                CLEAN_VENV=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Clean build artifacts, temporary files, and generated content"
                echo ""
                echo "Options:"
                echo "  --dry-run      Show what would be removed without actually removing"
                echo "  --clean-venv   Also remove virtual environment directory"
                echo "  -h, --help     Show this help message"
                exit 0
                ;;
        esac
    done
    
    echo -e "${YELLOW}Starting MFE Toolbox cleanup process...${NC}"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}DRY RUN: No files will be actually removed${NC}"
    fi
    
    # Track overall status
    OVERALL_STATUS=0
    
    # Clean Python artifacts
    clean_python_artifacts
    if [ $? -ne 0 ]; then
        log_error "Failed to clean Python artifacts."
        OVERALL_STATUS=1
    fi
    
    # Clean build artifacts
    clean_build_artifacts
    if [ $? -ne 0 ]; then
        log_error "Failed to clean build artifacts."
        OVERALL_STATUS=1
    fi
    
    # Clean temporary files
    clean_temp_files
    if [ $? -ne 0 ]; then
        log_error "Failed to clean temporary files."
        OVERALL_STATUS=1
    fi
    
    # Clean virtual environment if requested
    if [ "$CLEAN_VENV" = true ]; then
        clean_venv
        if [ $? -ne 0 ]; then
            log_error "Failed to clean virtual environment."
            OVERALL_STATUS=1
        fi
    fi
    
    # Print summary
    if [ "$DRY_RUN" = false ]; then
        echo -e "\n${GREEN}Cleanup completed:${NC}"
        echo -e "  - Removed directories: $REMOVED_DIRS"
        echo -e "  - Removed files: $REMOVED_FILES"
    else
        echo -e "\n${YELLOW}Dry run completed. No files were actually removed.${NC}"
        echo -e "  - Would remove directories: $REMOVED_DIRS"
        echo -e "  - Would remove files: $REMOVED_FILES"
    fi
    
    if [ $OVERALL_STATUS -eq 0 ]; then
        log_success "Cleanup process completed successfully."
    else
        log_error "Cleanup process completed with errors."
    fi
    
    return $OVERALL_STATUS
}

# Execute main function
main "$@"
exit $?