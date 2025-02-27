#!/bin/bash
# This script automates code formatting for the MFE Toolbox codebase.
# It uses isort to organize imports and black to format Python code.
# Configuration is loaded from:
#   - Black: infrastructure/config/black.toml
#   - isort: infrastructure/config/isort.cfg
# A backup of the original Python files is created in the backup directory (.formatting_backup).
# All log output is written to format.log.

# Globals
BLACK_CONFIG_PATH="infrastructure/config/black.toml"
ISORT_CONFIG_PATH="infrastructure/config/isort.cfg"
LOG_FILE="format.log"
BACKUP_DIR=".formatting_backup"

# Function to log messages with timestamps to the log file
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Check if Black is installed
if ! command -v black &> /dev/null; then
    echo "Error: black (v23.11.0) is not installed." >&2
    exit 1
fi

# Check if isort is installed
if ! command -v isort &> /dev/null; then
    echo "Error: isort (v5.12.0) is not installed." >&2
    exit 1
fi

# Determine source directory: use first argument if provided, else use current directory
SRC_DIR="${1:-.}"

log "Starting code formatting in directory: $SRC_DIR"

# Create backup directory if it doesn't exist
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
    log "Created backup directory: $BACKUP_DIR"
fi

# Backup all Python files from SRC_DIR to BACKUP_DIR while preserving directory structure
log "Creating backup of Python files..."
find "$SRC_DIR" -type f -name "*.py" | while read -r file; do
    cp --parents "$file" "$BACKUP_DIR"
done

# Run isort to organize imports using configuration from isort.cfg
log "Running isort with configuration from: $ISORT_CONFIG_PATH"
isort "$SRC_DIR" --settings-path "$ISORT_CONFIG_PATH" >> "$LOG_FILE" 2>&1
RET_CODE=$?
if [ $RET_CODE -ne 0 ]; then
    log "isort failed with exit code: $RET_CODE"
    exit $RET_CODE
fi
log "isort completed successfully."

# Run Black to format Python code using configuration from black.toml
log "Running black with configuration from: $BLACK_CONFIG_PATH"
black "$SRC_DIR" --config "$BLACK_CONFIG_PATH" >> "$LOG_FILE" 2>&1
RET_CODE=$?
if [ $RET_CODE -ne 0 ]; then
    log "black failed with exit code: $RET_CODE"
    exit $RET_CODE
fi
log "black completed successfully."

log "Code formatting finished successfully."
exit 0