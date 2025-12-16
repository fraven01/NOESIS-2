# NOESIS-2

Local setup and developer entrypoint.

- LLM/code navigation: `AGENTS.md`
- Explanatory documentation index: `docs/README.md`

## Prerequisites

- Docker + Docker Compose
- Node.js + npm (for local scripts and Tailwind build)

## Quick start (Docker)

1. Create local env file: copy `.env.example` → `.env`
2. Start the stack:
   - Linux/macOS: `npm run dev:stack`
   - Windows (PowerShell): `npm run win:dev:stack`
3. Initialize (migrations + bootstrap + RAG schema): `npm run dev:init`
4. Smoke-check local services: `npm run dev:check`

## Common commands

- Start/stop: `npm run dev:up`, `npm run dev:down`
- Rebuild/restart: `npm run dev:rebuild`, `npm run dev:restart`
- Django manage.py wrapper: `npm run dev:manage`
- Generate OpenAPI schema: `npm run api:schema`

## Tests & lint

- Python tests (in Docker): `npm run dev:test` (Windows: `npm run win:dev:test`)
- Lint: `npm run lint` / `npm run lint:fix`
