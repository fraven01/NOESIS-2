# Contributing

This project represents a minimal AI Core MVP. Contributions should preserve
its lightweight nature while demonstrating best practices.

## Development Loop

1. Install dependencies: `pip install -r requirements-dev.txt`
2. Run services for manual testing:

   ```bash
   uvicorn apps.api.main:app --reload
   celery -A apps.workers.celery_app worker -P solo
   ```

3. Execute tasks or call API endpoints with required tenant headers

## Code Style

- Python formatted with **black** and linted with **ruff** via `npm run lint`
- Keep functions small and pure; prefer dataclasses / Pydantic models for
  structured data

## Tests

- Write unit tests for helpers and orchestrator nodes
- Run tests locally: `pytest --noconftest ai-core/tests -q`
- Ensure `npm run lint` passes before committing

See [README](README.md) for project goals and [AGENTS](AGENTS.md) for node
documentation.

