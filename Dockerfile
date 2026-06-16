# rtsp-warden v1.0.0 — Default (slim) image
# Multi-stage build with python:3.13-slim runtime.
# For the smaller Distroless variant, see Dockerfile.distroless (685 MB vs 1.21 GB).
#
# Build with:      docker build -t rtsp-warden:latest .
# Build distroless: docker build -f Dockerfile.distroless -t rtsp-warden:distroless .

# syntax=docker/dockerfile:1.6

# ---- Stage 1: builder ----
FROM python:3.13-slim AS builder

# Install build deps (gcc for psycopg2, libxml2/libxslt for zeep)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy source, README (required by pyproject.toml), and install the project
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- Stage 2: runtime ----
FROM python:3.13-slim AS runtime

# Install runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxml2 \
    libxslt1.1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 warden

WORKDIR /app

# Copy the venv from the builder stage
COPY --from=builder --chown=warden:warden /app/.venv /app/.venv
COPY --from=builder --chown=warden:warden /app/src /app/src

# Make sure the venv is on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Volume for recordings + config
RUN mkdir -p /app/recordings /app/config /app/data && \
    chown -R warden:warden /app

USER warden

# Expose web UI port
EXPOSE 8080

# Health check (uses the /healthz endpoint absorbed in Sprint 2)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

# Default command
CMD ["rtsp-warden", "serve", "-c", "/app/config/config.yaml", "--web", "--web-port", "8080"]