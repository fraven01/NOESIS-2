#!/usr/bin/env bash
set -euo
# Enable pipefail when supported (e.g., bash); ignore on shells that don't support it
{ set -o pipefail; } 2>/dev/null || true

# One-command local dev bootstrap: bring up stack, run migrations, ensure tenants.

if [ ! -f .env ]; then
  echo "[dev-up] Fehler: Keine .env im Projektstamm gefunden." >&2
  echo "[dev-up] Bitte .env.example nach .env kopieren und Werte anpassen." >&2
  exit 1
fi

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

# Defaults (override via env when calling the script)
DEV_TENANT_SCHEMA=${DEV_TENANT_SCHEMA:-dev}
DEV_TENANT_NAME=${DEV_TENANT_NAME:-"Dev Tenant"}
DEV_DOMAIN=${DEV_DOMAIN:-dev.localhost}
DEV_SUPERUSER_USERNAME=${DEV_SUPERUSER_USERNAME:-admin}
DEV_SUPERUSER_EMAIL=${DEV_SUPERUSER_EMAIL:-admin@example.com}
DEV_SUPERUSER_PASSWORD=${DEV_SUPERUSER_PASSWORD:-admin123}

export DEV_TENANT_SCHEMA

echo "[dev-up] Bringing up services (db, redis, litellm, web, worker)"
$COMPOSE up -d

echo "[dev-up] Waiting for web to respond (warm-up)"
for i in $(seq 1 20); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ || true)
  # Consider 200..499 as web up; 5xx/000 means not ready yet
  if [ -n "$code" ] && [ "$code" -ge 200 ] && [ "$code" -lt 500 ]; then
    echo "[dev-up] Web responded with HTTP $code"; break
  fi
  sleep 1
done

echo "[dev-up] Init jobs: migrate + bootstrap"
npm run dev:init

echo "[dev-up] Optional tenant ping (nach Bootstrap):"
echo "curl -i \\" 
echo "  -H 'X-Tenant-Schema: \${DEV_TENANT_SCHEMA:-demo}' \\" 
echo "  -H 'X-Tenant-Id: \${DEV_TENANT_ID:-demo}' \\" 
echo "  -H 'X-Case-ID: local' \\" 
echo "  http://localhost:8000/ai/ping/"
