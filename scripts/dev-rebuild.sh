#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

# Rebuild backend images without touching named volumes.
# Pass --with-frontend to include the frontend build image as well.

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"
WITH_FRONTEND=${1:-}

echo "[dev-rebuild] Building web and worker images"
$COMPOSE build web worker

if [[ "${WITH_FRONTEND}" == "--with-frontend" ]]; then
  echo "[dev-rebuild] Building frontend image"
  $COMPOSE build frontend
fi
