#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

# Allow passing service names, defaults to web, worker, ingestion-worker
if [ "$#" -gt 0 ]; then
  SERVICES=("$@")
else
  SERVICES=(web worker ingestion-worker)
fi

echo "[dev-restart] Restarting: ${SERVICES[*]}"
if ! $COMPOSE restart "${SERVICES[@]}"; then
  echo "[dev-restart] Restart failed (service may not exist yet). Trying up --no-deps --no-build"
  $COMPOSE up -d --no-deps --no-build "${SERVICES[@]}"
fi

echo "[dev-restart] Done"
