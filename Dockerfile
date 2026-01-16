# syntax=docker/dockerfile:1

########################################
# CSS builder stage
########################################
FROM node:20 AS css-builder

WORKDIR /app

COPY package*.json ./
# Install with devDependencies using cache mount to speed rebuilds.
# Build tools (@tailwindcss/postcss, autoprefixer, postcss) are only needed in this stage.
RUN --mount=type=cache,target=/root/.npm npm ci

COPY postcss.config.js tailwind.config.js ./
COPY theme ./theme
# Some environments fail to load Tailwind's native binding (oxide) during Docker builds.
# Force the portable JS/WASM path for reliable builds in CI/containers.
ENV TAILWIND_DISABLE_OXIDE=1
# Ensure output directory exists to avoid PostCSS write errors
RUN mkdir -p ./theme/static/css \
    && npm run build:css

########################################
# Python builder stage
########################################
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Ensure system trust store and TLS tooling are present for pip downloads
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl openssl \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Copy dependency manifests and install Python deps
COPY requirements*.txt ./
# Use BuildKit cache for pip wheels to accelerate rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 \
    python -m pip install --upgrade 'pip<24.3' setuptools wheel \
    && python -m pip install --retries 10 --timeout 60 -r requirements.txt -r requirements-dev.txt

# Smoke test TLS path to PyPI to surface trust store issues early
RUN python - <<'PY'
import ssl
import urllib.request

ctx = ssl.create_default_context()
with urllib.request.urlopen("https://pypi.org/simple/", context=ctx, timeout=15) as response:
    print(response.status)
PY

# Copy application source
COPY . .

# Copy built CSS from node builder
COPY --from=css-builder /app/theme/static/css ./theme/static/css

# Minimal env for Django settings during build (collectstatic)
# Provide placeholder DB/Redis URLs so settings can load without .env.
ENV DJANGO_SETTINGS_MODULE=noesis2.settings.production \
    SECRET_KEY=build-secret \
    DATABASE_URL=postgresql://noesis2:noesis2@db:5432/noesis2 \
    REDIS_URL=redis://redis:6379/0

# Collect static files
RUN python manage.py collectstatic --noinput

########################################
# Runner stage
########################################
FROM python:3.12-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=noesis2.settings.production \
    PORT=8000 \
    PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

# Install procps for pgrep used by k8s probes
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends procps ca-certificates openssl curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 10001 appuser

# Copy Python site-packages and scripts from builder
COPY --from=builder /usr/local /usr/local

# Copy application source and collected static from builder.
# The builder stage already respected .dockerignore when copying the context,
# so heavy/dev artefacts (e.g. node_modules, .venv) are not present here.
COPY --from=builder /app /app

# Ensure correct ownership (optional, improves security)
RUN chown -R appuser:appuser /app && chmod +x /app/entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
# Start via Daphne (ASGI) for WebSocket support with OpenTelemetry auto-instrumentation
CMD ["sh", "-c", "opentelemetry-instrument daphne -b 0.0.0.0 -p ${PORT} noesis2.asgi:application"]
