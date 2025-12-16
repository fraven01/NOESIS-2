# Docker & compose conventions (observed configuration)

This document summarizes what is configured in the repository’s `Dockerfile` and `docker-compose*.yml` files.

## Build layout (Dockerfile)

The repository `Dockerfile` uses a multi-stage build:

- Node/Tailwind build stage (`css-builder`)
- Python dependency build stage (`builder`)
- Runtime stage (`runner`) running as a non-root user (`appuser`, UID 10001)

See: `Dockerfile`.

## Local service commands (docker-compose)

The compose files define the local commands and queues:

- Web service runs Django/Gunicorn (varies between `docker-compose.yml` and `docker-compose.dev.yml`).
- Celery workers are split by queue:
  - `worker`: `-Q celery,rag_delete,crawler`
  - `agents-worker`: `-Q agents`
  - `ingestion-worker`: `-Q ingestion`

See: `docker-compose.yml`, `docker-compose.dev.yml`.

## Queue usage in code

- Graph execution can enqueue `llm_worker.tasks.run_graph` onto the `agents` queue: `ai_core/services/__init__.py` (search for `queue="agents"`).
- Ingestion tasks run on the `ingestion` queue, e.g. `ai_core/tasks.py:run_ingestion_graph`.
