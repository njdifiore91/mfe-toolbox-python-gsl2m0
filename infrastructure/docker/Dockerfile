# Builder stage for installing dependencies and compiling packages
FROM python:3.12-slim as builder

# Set build arguments
ARG PYTHON_VERSION=3.12
ARG APP_DIR=/app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=${APP_DIR}/src \
    PIP_NO_CACHE_DIR=1

# Install system dependencies including build tools and Qt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR ${APP_DIR}

# Copy requirements file
COPY src/backend/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # Ensure specific version of NumPy is installed as per spec
    pip install numpy==1.26.3

# Final runtime stage with minimal image size
FROM python:3.12-slim

# Set build arguments
ARG PYTHON_VERSION=3.12
ARG APP_DIR=/app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=${APP_DIR}/src \
    QT_QPA_PLATFORM=offscreen

# Install runtime dependencies for PyQt6 and other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m mfeuser

# Set up working directory
WORKDIR ${APP_DIR}

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages/ /usr/local/lib/python${PYTHON_VERSION}/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY src/ ${APP_DIR}/src/

# Change ownership to non-root user for security
RUN chown -R mfeuser:mfeuser ${APP_DIR}

# Switch to non-root user
USER mfeuser

# Define the entry point
ENTRYPOINT ["python", "-m", "mfe"]

# Default command if no command is provided
CMD ["--help"]