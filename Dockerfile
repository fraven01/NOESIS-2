# syntax=docker/dockerfile:1

########################################
# Builder stage
########################################
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps and Node.js for PostCSS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests and install Python + Node deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY package*.json ./
RUN npm ci

# Copy application source
COPY . .

# Minimal env for Django settings during build (collectstatic)
ENV DJANGO_SETTINGS_MODULE=noesis2.settings.production \
    SECRET_KEY=build-secret \
    DB_NAME=builddb \
    DB_USER=builduser \
    DB_PASSWORD=buildpass \
    DB_HOST=localhost \
    DB_PORT=5432

# Build CSS once (no watch) and collect static files
RUN npx postcss ./theme/static_src/input.css -o ./theme/static/css/output.css \
    && python manage.py collectstatic --noinput

########################################
# Runner stage
########################################
FROM python:3.12-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=noesis2.settings.production \
    PORT=8080

WORKDIR /app

# Create non-root user
RUN useradd -m -u 10001 appuser

# Copy Python site-packages and scripts from builder
COPY --from=builder /usr/local /usr/local

# Copy app code and collected static
# Copy only necessary application files (avoid node_modules in final image)
COPY --from=builder /app/manage.py /app/manage.py
COPY --from=builder /app/noesis2 /app/noesis2
COPY --from=builder /app/ai_core /app/ai_core
COPY --from=builder /app/core /app/core
COPY --from=builder /app/documents /app/documents
COPY --from=builder /app/workflows /app/workflows
COPY --from=builder /app/users /app/users
COPY --from=builder /app/common /app/common
COPY --from=builder /app/theme /app/theme
COPY --from=builder /app/staticfiles /app/staticfiles

# Ensure correct ownership (optional, improves security)
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Start via gunicorn (honors $PORT environment variable)
CMD ["sh", "-c", "gunicorn noesis2.wsgi:application --bind 0.0.0.0:${PORT} --workers 3"]
