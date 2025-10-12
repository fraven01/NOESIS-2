#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

echo "[dev-reset] Down + prune project volumes"
$COMPOSE down -v --remove-orphans || true

echo "[dev-reset] Build fresh images"
$COMPOSE build --no-cache --pull

echo "[dev-reset] Bring up base services"
$COMPOSE up -d

echo "[dev-reset] Run jobs: migrate + bootstrap + rag"
npm run dev:init

echo "[dev-reset] Smoke checks"
npm run dev:check

echo "[dev-reset] Done"
