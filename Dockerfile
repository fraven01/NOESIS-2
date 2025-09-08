# syntax=docker/dockerfile:1

# ========================
# Builder stage
# ========================
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system build dependencies and Node.js
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python and Node dependencies
COPY requirements*.txt package*.json ./
RUN pip install --no-cache-dir -r requirements.txt \
    && npm ci

# Copy application code
COPY . .

# Build static assets and collect static files
RUN npm run build:css \
    && python manage.py collectstatic --noinput

# ========================
# Runner stage
# ========================
FROM python:3.12-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Copy installed Python packages
COPY --from=builder /usr/local /usr/local

# Copy application code and static files
COPY --from=builder /app /app

RUN chown -R app:app /app

USER app

EXPOSE 8080

CMD ["gunicorn", "noesis2.wsgi:application", "--bind", "0.0.0.0:8080"]
