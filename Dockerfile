# =============================================================================
# Stage 1 — builder
# Install dependencies into an isolated virtualenv so the final image
# only carries what is needed at runtime.
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the dependency manifest first to leverage layer caching
COPY pyproject.toml ./
# Minimal stub so the editable install resolves the package name
COPY src/ ./src/

# Install runtime dependencies (no dev extras)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

# =============================================================================
# Stage 2 — runtime
# Lean image; copy only the virtualenv and application source.
# =============================================================================
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="rca-agent" \
      org.opencontainers.image.description="AI Production Incident Root Cause Analysis platform" \
      org.opencontainers.image.version="0.1.0"

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --chown=appuser:appgroup src/ ./src/

USER appuser

# Expose the default port (overridable via PORT env var in compose)
EXPOSE 8000

# Health check — Docker will poll this to mark the container healthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Use exec form so signals propagate correctly to uvicorn
CMD ["python", "-m", "uvicorn", "rca_agent.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]
