#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

if [ ! -f .env ]; then
  echo "[dev-demo] Fehler: Keine .env im Projektstamm gefunden." >&2
  echo "[dev-demo] Bitte .env.example nach .env kopieren und Werte anpassen." >&2
  exit 1
fi

COMPOSE_CMD=(docker compose -f docker-compose.yml -f docker-compose.dev.yml)
COMPOSE_JOBS_CMD=(docker compose --profile jobs -f docker-compose.yml -f docker-compose.dev.yml)

echo "[dev-demo] Building backend & job images"
"${COMPOSE_JOBS_CMD[@]}" build web worker migrate bootstrap

echo "[dev-demo] Bringing up core services"
"${COMPOSE_CMD[@]}" up -d

echo "[dev-demo] Applying migrations & bootstrap"
npm run dev:init

echo "[dev-demo] Seeding demo tenant data"
npm run seed:demo

echo "[dev-demo] Fertig. Login unter http://demo.localhost:8000/admin/ mit demo/demo."
echo "[dev-demo] Hinweis: Hosts-Eintrag 127.0.0.1 demo.localhost erforderlich."
