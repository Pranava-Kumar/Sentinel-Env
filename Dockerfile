# ── Stage 1: Build dependencies ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir --prefix=/install uv && \
    /install/bin/uv pip install --no-cache-dir --prefix=/install -r pyproject.toml

# ── Stage 2: Production image ────────────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Install curl for healthcheck and runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports: main app + Prometheus metrics
EXPOSE 7860

# Health check with liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Run the server with production settings
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--log-level", "info", "--workers", "1"]
