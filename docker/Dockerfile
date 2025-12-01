# Multi-stage Dockerfile for Parle Backend (Production)
# Python 3.11 slim base image

FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
# Upgrade pip first, then install dependencies
# Use --no-deps for problematic packages and install dependencies separately
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user pydantic-core>=2.18.2 && \
    pip install --no-cache-dir --user -r requirements.txt --no-deps || \
    pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set labels
LABEL maintainer="Parle Team" \
      org.opencontainers.image.title="Parle Backend" \
      org.opencontainers.image.description="Parle Backend - Sistema de conversação multimodal" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 parle && \
    mkdir -p /app /app/logs && \
    chown -R parle:parle /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/parle/.local

# Copy application code
COPY --chown=parle:parle . .

# Set environment variables
ENV PATH=/home/parle/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Switch to non-root user
USER parle

# Expose port
EXPOSE 8000

# Health check (using curl instead of requests)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
