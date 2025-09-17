#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
  echo "[rag-enable] .env not found" >&2
  exit 1
fi

if grep -q '^RAG_ENABLED=' "$ENV_FILE"; then
  # Replace existing value with true
  if sed -i.bak 's/^RAG_ENABLED=.*/RAG_ENABLED=true/' "$ENV_FILE"; then
    rm -f "$ENV_FILE.bak"
  fi
else
  echo 'RAG_ENABLED=true' >> "$ENV_FILE"
fi

echo "[rag-enable] Set RAG_ENABLED=true in .env"

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"
echo "[rag-enable] Restarting web and worker"
$COMPOSE restart web worker

echo "[rag-enable] Done"

