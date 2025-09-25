#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

echo "[dev-restart] Restarting web + worker"
$COMPOSE restart web worker

echo "[dev-restart] Done"
