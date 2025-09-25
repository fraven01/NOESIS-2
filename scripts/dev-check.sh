#!/usr/bin/env bash
# Be tolerant to shells without pipefail
set -euo
{ set -o pipefail; } 2>/dev/null || true

# Simple smoketests for local dev stack

DEV_TENANT_SCHEMA=${DEV_TENANT_SCHEMA:-dev}
TENANT_ID=${TENANT_ID:-dev-tenant}
CASE_ID=${CASE_ID:-local}

echo "[dev-check] LiteLLM liveliness/readiness + chat"
bash ./scripts/smoke_litellm.sh || {
  echo "[dev-check] LiteLLM smoke failed (chat requires valid API key)" >&2
}

echo "[dev-check] AI Core ping with tenant headers"
# Retry a few times in case web is still warming up
for i in $(seq 1 5); do
  if curl -fsSI \
  -H "X-Tenant-Schema: ${DEV_TENANT_SCHEMA}" \
  -H "X-Tenant-ID: ${TENANT_ID}" \
  -H "X-Case-ID: ${CASE_ID}" \
  http://localhost:8000/ai/ping/ | head -n 1; then
    break
  fi
  sleep 1
done

echo "[dev-check] POST /ai/scope minimal payload"
RESP=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-Tenant-Schema: ${DEV_TENANT_SCHEMA}" \
    -H "X-Tenant-ID: ${TENANT_ID}" \
    -H "X-Case-ID: ${CASE_ID}" \
    --data '{"hello":"world"}' \
  http://localhost:8000/ai/scope/ || true)
echo "$RESP" | head -c 200; echo

echo "[dev-check] RAG migrate"
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T rag || {
  echo "[dev-check] RAG migrate failed" >&2
}
echo "[dev-check] RAG health"
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T rag-health || {
  echo "[dev-check] RAG health failed" >&2
}

echo "[dev-check] Done"
