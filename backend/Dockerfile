# Enhanced KOO Platform Backend Dockerfile
# Multi-stage build for production optimization

# ============================================================================
# Base Python Image with System Dependencies
# ============================================================================

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    postgresql-client \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================================================
# Development Stage
# ============================================================================

FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/media /app/static /app/models

# Set up development environment
EXPOSE 8000

# Development command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# ============================================================================
# Production Build Stage
# ============================================================================

FROM base as builder

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Production Stage
# ============================================================================

FROM base as production

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN groupadd -r koo && useradd -r -g koo koo

# Create directories and set permissions
RUN mkdir -p /app/logs /app/media /app/static /app/models && \
    chown -R koo:koo /app

# Copy source code
COPY --chown=koo:koo . .

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/openai && \
    chown -R koo:koo /app/.cache

# Set PATH to include local Python packages
ENV PATH=/root/.local/bin:$PATH

# Install additional production tools
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/nginx.conf /etc/nginx/sites-available/default

# Health check script
COPY docker/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Switch to non-root user
USER koo

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD /healthcheck.sh

# Production command
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]