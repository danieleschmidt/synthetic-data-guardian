# Synthetic Data Guardian Dockerfile
# Multi-stage build for optimal image size and security

# =============================================================================
# BUILD STAGE
# =============================================================================
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata labels
LABEL org.opencontainers.image.title="Synthetic Data Guardian"
LABEL org.opencontainers.image.description="Enterprise-grade synthetic data pipeline with built-in validation, watermarking, and auditable lineage tracking"
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.url="https://github.com/terragon-labs/synthetic-data-guardian"
LABEL org.opencontainers.image.source="https://github.com/terragon-labs/synthetic-data-guardian"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install Poetry
RUN pip install poetry==1.7.1

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-dev && \
    rm -rf $POETRY_CACHE_DIR

# =============================================================================
# RUNTIME STAGE
# =============================================================================
FROM python:3.11-slim as runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /bin/bash -c "App User" appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PORT=8080

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ src/
COPY README.md LICENSE ./

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/logs /app/temp \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app

# Health check script
COPY <<EOF /app/healthcheck.py
#!/usr/bin/env python3
import sys
import urllib.request
import urllib.error

def health_check():
    try:
        response = urllib.request.urlopen('http://localhost:8080/health', timeout=10)
        if response.status == 200:
            sys.exit(0)
        else:
            sys.exit(1)
    except urllib.error.URLError:
        sys.exit(1)

if __name__ == "__main__":
    health_check()
EOF

RUN chmod +x /app/healthcheck.py

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Default command
CMD ["python", "-m", "uvicorn", "synthetic_guardian.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# DEVELOPMENT STAGE
# =============================================================================
FROM builder as development

# Install development dependencies
RUN poetry install --with dev

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    less \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy development files
COPY tests/ tests/
COPY .pre-commit-config.yaml .editorconfig ./

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true

# Development command
CMD ["python", "-m", "uvicorn", "synthetic_guardian.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# =============================================================================
# PRODUCTION STAGE (default)
# =============================================================================
FROM runtime as production

# Production-specific configurations
ENV ENVIRONMENT=production \
    DEBUG=false \
    WORKERS=4

# Use gunicorn for production
RUN /app/.venv/bin/pip install gunicorn[gthread]==21.2.0

# Production command with gunicorn
CMD ["gunicorn", "synthetic_guardian.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--max-requests", "1000", "--max-requests-jitter", "100", "--timeout", "30", "--keep-alive", "2"]

# =============================================================================
# TESTING STAGE
# =============================================================================
FROM development as testing

# Install additional testing tools
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy all test files
COPY tests/ tests/

# Set testing environment
ENV ENVIRONMENT=testing \
    TESTING=true

# Testing command
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src/synthetic_guardian", "--cov-report=html", "--cov-report=term-missing"]

# =============================================================================
# SECURITY SCANNING STAGE
# =============================================================================
FROM runtime as security

# Install security scanning tools
USER root
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Trivy for vulnerability scanning
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - \
    && echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | tee -a /etc/apt/sources.list.d/trivy.list \
    && apt-get update \
    && apt-get install -y trivy

USER appuser

# Security scanning command
CMD ["trivy", "filesystem", "--exit-code", "1", "/app"]