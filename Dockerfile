# syntax=docker/dockerfile:1.6

# ---------------------------------------------------------------------------
# Stage 1: Build — install dependencies into a virtual environment
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for native extensions (scipy, asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency manifests first to leverage Docker layer cache
COPY pyproject.toml README.md ./

# Install all runtime dependencies (no dev extras)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[dev]" --extra-index-url https://pypi.org/simple/ || \
    pip install --no-cache-dir \
        fastapi>=0.110.0 \
        pydantic>=2.6.0 \
        pydantic-settings>=2.2.0 \
        uvicorn[standard]>=0.29.0 \
        httpx>=0.27.0 \
        scipy>=1.12.0 \
        scikit-learn>=1.4.0 \
        evidently>=0.4.0 \
        sqlalchemy[asyncio]>=2.0.0 \
        asyncpg>=0.29.0 \
        confluent-kafka>=2.3.0 \
        structlog>=24.1.0 \
        opentelemetry-api>=1.23.0 \
        opentelemetry-sdk>=1.23.0 \
        numpy>=1.26.0

# Copy source code and install the package itself
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal image with non-root user
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Install only runtime system libraries (libpq for asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 aumos \
    && useradd --uid 10001 --gid aumos --no-create-home --shell /usr/sbin/nologin aumos

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the package source
COPY --chown=aumos:aumos src/ /app/src/

WORKDIR /app

# Drop to non-root user
USER aumos

# Expose the FastAPI port
EXPOSE 8000

# Health check — liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/live').raise_for_status()"

# Run uvicorn
CMD ["uvicorn", "aumos_drift_detector.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-config", "/dev/null"]
