#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

COMPOSE_CMD=(docker compose -f docker-compose.yml -f docker-compose.dev.yml exec)
if [ ! -t 0 ]; then
  COMPOSE_CMD+=(-T)
fi

ENV_ARGS=()
POSITIONAL=()

while (($#)); do
  case "$1" in
    --)
      shift
      while (($#)); do
        POSITIONAL+=("$1")
        shift
      done
      break
      ;;
    --env)
      if (($# < 2)); then
        echo "Error: --env requires KEY=VALUE" >&2
        exit 1
      fi
      ENV_ARGS+=(-e "$2")
      shift 2
      ;;
    *=*)
      ENV_ARGS+=(-e "$1")
      shift
      ;;
    *)
      POSITIONAL=("$@")
      break
      ;;
  esac
done

if ((${#POSITIONAL[@]} == 0)); then
  echo "Usage: $0 [VAR=value ...] <command> [args...]" >&2
  exit 1
fi

exec "${COMPOSE_CMD[@]}" "${ENV_ARGS[@]}" web python manage.py "${POSITIONAL[@]}"
