# NOESIS-2 Architecture — 4-layer “firm hierarchy” (living architecture lens)

This document uses the “firm hierarchy” (Layer 1–4) as a lens to describe how the repository is structured and how execution flows through UI/API → orchestration → workers. Runtime behavior and contracts remain defined by code.

Current, code-backed inventory: `docs/architecture/architecture-reality.md`.
For strict layer boundary definitions, see: `docs/architecture/layer-contracts.md`.

## Layer mapping (code anchors)

### Layer 1 — Customer interface (UI/API)

Primary locations:

- UI templates/static: `theme/`
- HTTP routing: `noesis2/urls.py`
- AI Core HTTP views: `ai_core/views.py`
- Worker API views: `llm_worker/views.py`

### Layer 2 — Business (cases and business context)

Primary locations:

- Cases domain model + APIs: `cases/` (notably `cases/models.py`, `cases/services.py`, `cases/api.py`)
- Case lifecycle integration used by graphs: `cases/integration.py`, `cases/lifecycle.py` (if present in the repo)

Business-heavy graph example (domain logic implemented in a graph module):

- `ai_core/graphs/business/framework_analysis_graph.py` (writes `documents/framework_models.py:FrameworkProfile` / `FrameworkDocument`)

### Layer 3 — Technical orchestration (graphs + services)

Primary locations:

- Graph implementations: `ai_core/graphs/`
- Graph execution orchestration: `ai_core/services/__init__.py:execute_graph`
- Document domain orchestration boundary: `documents/domain_service.py`, `documents/service_facade.py`

### Layer 4 — Workers (task execution + I/O)

Primary locations:

- Agents queue worker task: `llm_worker/tasks.py:run_graph` (`queue="agents-high"` default, `agents-low` for background)
- Ingestion queue task: `ai_core/tasks.py:run_ingestion_graph` (`queue="ingestion"`)
- Task/context plumbing: `common/celery.py` (`ScopedTask`, `with_scope_apply_async`)
- Local queue wiring (compose): `docker-compose.yml`, `docker-compose.dev.yml`

## Related contracts and ID normalization

- Canonical header names: `common/constants.py`
- Request → `ScopeContext`: `ai_core/ids/http_scope.py:normalize_request` and `ai_core/graph/schemas.py:_build_scope_context`
- Tool envelopes: `ai_core/tool_contracts/base.py`
