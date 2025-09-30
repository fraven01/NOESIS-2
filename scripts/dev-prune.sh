#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

ALL=${1:-}

echo "[dev-prune] Pruning dangling images and builder cache"
docker image prune -f >/dev/null
docker builder prune -f >/dev/null

if [[ "$ALL" == "--all" ]]; then
  echo "[dev-prune] Also pruning unused networks and volumes (destructive)"
  docker network prune -f >/dev/null || true
  docker volume prune -f >/dev/null || true
fi

echo "[dev-prune] Done"

