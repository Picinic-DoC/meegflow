# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Build argument for trusted hosts (useful in CI/CD environments with self-signed certs)
ARG PIP_TRUSTED_HOST=""

# Install system dependencies required by MNE and other packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN if [ -n "$PIP_TRUSTED_HOST" ]; then \
        pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy the entire project
COPY . .

# Install the package
RUN if [ -n "$PIP_TRUSTED_HOST" ]; then \
        pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -e .; \
    else \
        pip install --no-cache-dir -e .; \
    fi

# Set the entrypoint to the CLI command
ENTRYPOINT ["meegflow"]

# Default command (shows help if no arguments provided)
CMD ["--help"]
