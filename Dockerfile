# syntax=docker/dockerfile:1

########################################
# CSS builder stage
########################################
FROM node:20 AS css-builder

WORKDIR /app

COPY package*.json ./
# Install with devDependencies, since build tools (@tailwindcss/postcss, autoprefixer, postcss)
# are needed only during this stage. Final image does not include node_modules.
RUN npm ci

COPY postcss.config.js tailwind.config.js ./
COPY theme ./theme

RUN npm run build:css

########################################
# Python builder stage
########################################
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests and install Python deps
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Copy built CSS from node builder
COPY --from=css-builder /app/theme/static/css ./theme/static/css

# Minimal env for Django settings during build (collectstatic)
# No database needed for collectstatic; keep a placeholder SECRET_KEY only.
ENV DJANGO_SETTINGS_MODULE=noesis2.settings.production \
    SECRET_KEY=build-secret

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
    PORT=8000

WORKDIR /app

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
# Start via gunicorn (honors $PORT environment variable)
CMD ["sh", "-c", "gunicorn noesis2.wsgi:application --bind 0.0.0.0:${PORT} --workers 3"]
