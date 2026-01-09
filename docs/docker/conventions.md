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
- Celery workers consume queues explicitly:
  - `worker`: `-Q agents-high,agents-low,crawler,celery,ingestion,ingestion-bulk,dead_letter,rag_delete`
- Optional beat scheduler runs periodic jobs (e.g. dead-letter cleanup, DLQ alerts):
  - `beat`: `celery -A noesis2 beat -l info`

See: `docker-compose.yml`, `docker-compose.dev.yml`.

## Queue usage in code

- Graph execution enqueues `llm_worker.tasks.run_graph` on `agents-high` by default (background can use `agents-low`).
- Ingestion tasks run on the `ingestion` queue, e.g. `ai_core/tasks.py:run_ingestion_graph`.
