#!/usr/bin/env bash
set -euo pipefail

# Bootstrap both the application stack and the local ELK stack for development.

if [ ! -f .env ]; then
  echo "[dev-up-all] Fehler: Keine .env im Projektstamm gefunden." >&2
  echo "[dev-up-all] Bitte .env.example nach .env kopieren und Werte anpassen." >&2
  exit 1
fi

APP_COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"
ELK_COMPOSE="docker compose -f docker/elk/docker-compose.yml"

APP_LOG_PATH=${APP_LOG_PATH:-"$(pwd)/logs/app"}
export APP_LOG_PATH

mkdir -p "$APP_LOG_PATH"
echo "[dev-up-all] Log-Verzeichnis: $APP_LOG_PATH"

echo "[dev-up-all] Building application stack images"
$APP_COMPOSE build

echo "[dev-up-all] Building ELK stack images"
$ELK_COMPOSE build

echo "[dev-up-all] Starting application stack"
$APP_COMPOSE up -d

echo "[dev-up-all] Starting ELK stack"
$ELK_COMPOSE up -d

echo "[dev-up-all] Waiting for web to respond (warm-up)"
for i in {1..20}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ai/ping/ || true)
  if [ -n "$code" ] && [ "$code" -ge 200 ] && [ "$code" -lt 500 ]; then
    echo "[dev-up-all] Web responded with HTTP $code"
    break
  fi
  sleep 1

done

echo "[dev-up-all] Running migrations and bootstrap tasks"
npm run dev:init

echo "[dev-up-all] Done. Kibana läuft unter http://localhost:5601 (ELASTIC_PASSWORD erforderlich)."
