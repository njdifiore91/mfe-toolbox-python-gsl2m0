#!/bin/bash
# Lint script for MFE Toolbox
# This script runs Python code quality checks and static type analysis across the MFE Toolbox codebase.
# It uses flake8 (version 6.1.0) for style and quality checking and mypy (version 1.7.0) for static type analysis.
# Flake8 configuration is loaded from infrastructure/config/flake8.ini and mypy configuration from infrastructure/config/mypy.ini.
# The checks are performed on the src/backend and src/web directories.
#
# Exit code 0 indicates that all checks passed successfully.
# A non-zero exit code indicates one or more errors were detected.

# Exit immediately if a command exits with a non-zero status
set -e

# Change to the repository root directory.
# This script is located in infrastructure/scripts; we move two levels up.
cd "$(dirname "$0")/../.."

echo "Running flake8 code style checks..."
flake8 --config=infrastructure/config/flake8.ini src/backend src/web
flake8_exit=$?

echo "Running mypy static type analysis..."
mypy --config-file=infrastructure/config/mypy.ini src/backend src/web
mypy_exit=$?

# Combine exit codes: if either check fails, exit with error.
if [ $flake8_exit -ne 0 ] || [ $mypy_exit -ne 0 ]; then
    echo "Linting errors detected. Please fix the issues above."
    exit 1
else
    echo "All linting checks passed successfully."
    exit 0
fi