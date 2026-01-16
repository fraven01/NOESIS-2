#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

# Allow passing service names, defaults to all Django services (web, worker, beat).
# Use -i|--include-infrastructure to also restart redis, litellm, and db.
INCLUDE_INFRA=0
SERVICES=()

for arg in "$@"; do
  case "$arg" in
    -i|--include-infrastructure)
      INCLUDE_INFRA=1
      ;;
    *)
      SERVICES+=("$arg")
      ;;
  esac
done

if [ "${#SERVICES[@]}" -eq 0 ]; then
  SERVICES=(web worker beat)
fi
if [ "$INCLUDE_INFRA" -eq 1 ]; then
  SERVICES+=(db redis litellm)
fi

declare -A seen
unique_services=()
for svc in "${SERVICES[@]}"; do
  if [ -z "${seen[$svc]+x}" ]; then
    unique_services+=("$svc")
    seen[$svc]=1
  fi
done
SERVICES=("${unique_services[@]}")

echo "[dev-restart] Restarting: ${SERVICES[*]}"
if ! $COMPOSE restart "${SERVICES[@]}"; then
  echo "[dev-restart] Restart failed (service may not exist yet). Trying up --no-deps --no-build"
  $COMPOSE up -d --no-deps --no-build "${SERVICES[@]}"
fi

echo "[dev-restart] Done"
