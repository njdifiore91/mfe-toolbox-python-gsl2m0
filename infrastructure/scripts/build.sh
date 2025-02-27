#!/bin/bash
# build.sh - Automated build script for MFE Toolbox Python package
# This script handles building both backend and web components

# Constants
REQUIRED_PYTHON_VERSION="3.12"
BACKEND_DIR="src/backend"
WEB_DIR="src/web"
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Find Python interpreter with version 3.12+
find_python() {
    # Try common Python commands
    for cmd in python3 python python3.12; do
        if command -v $cmd >/dev/null 2>&1; then
            version=$($cmd --version 2>&1 | awk '{print $2}')
            major=$(echo $version | cut -d. -f1)
            minor=$(echo $version | cut -d. -f2)
            
            if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    
    echo ""
    return 1
}

# Compare version strings
version_compare() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "0"
        return
    fi
    echo "$1" "$2" | awk '{if ($1 >= $2) print 1; else print 0}'
}

# Check if Python 3.12+ is available
check_python_version() {
    echo -e "${YELLOW}Checking Python version...${NC}"
    
    # Find Python interpreter
    PYTHON_CMD=$(find_python)
    
    if [ -z "$PYTHON_CMD" ]; then
        echo -e "${RED}Error: Python $REQUIRED_PYTHON_VERSION+ is required but not found.${NC}"
        echo -e "${RED}Please install Python $REQUIRED_PYTHON_VERSION or higher and try again.${NC}"
        return 1
    fi
    
    # Get Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    
    echo -e "${GREEN}Found $PYTHON_CMD version $PYTHON_VERSION.${NC}"
    export PYTHON_CMD
    return 0
}

# Check if required build dependencies are installed
check_build_deps() {
    echo -e "${YELLOW}Checking build dependencies...${NC}"
    
    # Check setuptools
    SETUPTOOLS_VERSION=$($PYTHON_CMD -c "import setuptools; print(setuptools.__version__)" 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: setuptools not found.${NC}"
        echo -e "${RED}Please install with: $PYTHON_CMD -m pip install setuptools>=69.0.2${NC}"
        return 1
    fi
    
    if [ $(version_compare "$SETUPTOOLS_VERSION" "69.0.2") -ne 1 ]; then
        echo -e "${RED}Error: setuptools version $SETUPTOOLS_VERSION is less than required (69.0.2).${NC}"
        echo -e "${RED}Please upgrade with: $PYTHON_CMD -m pip install --upgrade setuptools>=69.0.2${NC}"
        return 1
    fi
    
    # Check wheel
    WHEEL_VERSION=$($PYTHON_CMD -c "import wheel; print(wheel.__version__)" 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: wheel not found.${NC}"
        echo -e "${RED}Please install with: $PYTHON_CMD -m pip install wheel>=0.42.0${NC}"
        return 1
    fi
    
    if [ $(version_compare "$WHEEL_VERSION" "0.42.0") -ne 1 ]; then
        echo -e "${RED}Error: wheel version $WHEEL_VERSION is less than required (0.42.0).${NC}"
        echo -e "${RED}Please upgrade with: $PYTHON_CMD -m pip install --upgrade wheel>=0.42.0${NC}"
        return 1
    fi
    
    # Check build
    BUILD_VERSION=$($PYTHON_CMD -c "import build; print(build.__version__)" 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: build package not found.${NC}"
        echo -e "${RED}Please install with: $PYTHON_CMD -m pip install build>=1.0.3${NC}"
        return 1
    fi
    
    if [ $(version_compare "$BUILD_VERSION" "1.0.3") -ne 1 ]; then
        echo -e "${RED}Error: build version $BUILD_VERSION is less than required (1.0.3).${NC}"
        echo -e "${RED}Please upgrade with: $PYTHON_CMD -m pip install --upgrade build>=1.0.3${NC}"
        return 1
    fi
    
    echo -e "${GREEN}All build dependencies are satisfied.${NC}"
    return 0
}

# Build the backend package
build_backend() {
    echo -e "${YELLOW}Building backend package...${NC}"
    
    # Check if backend directory exists
    if [ ! -d "$BACKEND_DIR" ]; then
        echo -e "${RED}Error: Backend directory '$BACKEND_DIR' not found.${NC}"
        return 1
    fi
    
    # Change to backend directory
    cd "$BACKEND_DIR"
    
    # Run build
    echo "Running Python build..."
    $PYTHON_CMD -m build
    BUILD_STATUS=$?
    
    # Verify build output
    if [ $BUILD_STATUS -ne 0 ] || [ ! -d "dist" ] || [ ! "$(ls -A dist)" ]; then
        echo -e "${RED}Error: Build failed or no output was generated.${NC}"
        cd - > /dev/null
        return 1
    fi
    
    echo -e "${GREEN}Backend package built successfully:${NC}"
    ls -l dist/
    cd - > /dev/null
    return 0
}

# Build the web/GUI package
build_web() {
    echo -e "${YELLOW}Building web/GUI package...${NC}"
    
    # Check if web directory exists
    if [ ! -d "$WEB_DIR" ]; then
        echo -e "${YELLOW}Web directory '$WEB_DIR' not found. Skipping web build.${NC}"
        return 0
    fi
    
    # Change to web directory
    cd "$WEB_DIR"
    
    # Run build
    echo "Running Python build..."
    $PYTHON_CMD -m build
    BUILD_STATUS=$?
    
    # Verify build output
    if [ $BUILD_STATUS -ne 0 ] || [ ! -d "dist" ] || [ ! "$(ls -A dist)" ]; then
        echo -e "${RED}Error: Web build failed or no output was generated.${NC}"
        cd - > /dev/null
        return 1
    fi
    
    echo -e "${GREEN}Web/GUI package built successfully:${NC}"
    ls -l dist/
    cd - > /dev/null
    return 0
}

# Main function
main() {
    echo -e "${YELLOW}Starting MFE Toolbox build process...${NC}"
    
    # Check Python version
    check_python_version
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Check build dependencies
    check_build_deps
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Build backend package
    build_backend
    BACKEND_STATUS=$?
    
    if [ $BACKEND_STATUS -ne 0 ]; then
        echo -e "${RED}Backend build failed. Stopping build process.${NC}"
        return 1
    fi
    
    # Build web package (if it exists)
    build_web
    WEB_STATUS=$?
    
    # Final status
    if [ $BACKEND_STATUS -eq 0 ] && [ $WEB_STATUS -eq 0 ]; then
        echo -e "\n${GREEN}Build completed successfully!${NC}"
        echo -e "${GREEN}Distribution packages are available in:${NC}"
        echo -e "${GREEN}  - $BACKEND_DIR/dist/${NC}"
        
        if [ -d "$WEB_DIR" ]; then
            echo -e "${GREEN}  - $WEB_DIR/dist/${NC}"
        fi
        
        echo -e "\n${YELLOW}Use the following to install the backend package:${NC}"
        echo -e "$PYTHON_CMD -m pip install $BACKEND_DIR/dist/*.whl"
        return 0
    else
        echo -e "${RED}Build process encountered errors. Please check the logs above.${NC}"
        return 1
    fi
}

# Execute main function
main
exit $?