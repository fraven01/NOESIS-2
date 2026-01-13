# Backlog: ai_core/tasks Refactoring

**Status**: Draft
**Created**: 2026-01-06
**Priority**: P1 - High Value Cleanup (Medium Effort)
**Context**: Follow-up to "Break Up God Files" (P2) - the legacy tasks monolith (now split into `ai_core/tasks/`) was ~3,210 lines

## Problem Statement

The legacy tasks monolith (now split into `ai_core/tasks/`) grew to **~3,210 lines** with multiple responsibilities:
- Legacy ingestion pipeline tasks (ingest_raw, extract_text, pii_mask, chunk, embed, upsert)
- Graph execution orchestration (run_ingestion_graph, build_graph)
- Dead Letter Queue management (cleanup, alerts)
- Monitoring utilities (drift checks, logging)
- ~100+ private helper functions for caching, embedding, chunking

This violates the "no file >500 lines" acceptance criteria from the P2 "Break Up God Files" initiative.

## Goals

1. **Split by responsibility**: Separate ingestion tasks, graph tasks, monitoring, and utilities
2. **Maintain backward compatibility**: Preserve all public task signatures and Celery discovery
3. **Improve navigability**: Reduce cognitive load for developers
4. **Enable parallel development**: Reduce merge conflicts
5. **Align with Layer architecture**: Keep Layer 2 (tasks) vs Layer 1 (helpers) separation clear

## Proposed Structure

```
ai_core/
  tasks/                               # Package (current layout)
    __init__.py                        # Public API - re-exports all tasks for backward compatibility
    ingestion_tasks.py                 # Legacy ingestion: ingest_raw, extract_text, pii_mask, chunk, embed, upsert
    graph_tasks.py                     # Graph orchestration: run_ingestion_graph, build_graph
    monitoring_tasks.py                # DLQ + drift checks (no logging helpers)
    helpers/
      __init__.py
      caching.py                       # Redis caching & deduplication
      embedding.py                     # Embedding normalization, hashing, profile resolution
      chunking.py                      # Semantic chunker helpers, overlap calculation
      task_utils.py                    # Shared utilities: _build_path, _coerce_*, _jsonify_for_task
```

**Historical note**: The legacy single-module tasks file had to be moved atomically with package creation to avoid import shadowing. This is now resolved.

**Estimated total after split**: ~2,850 lines across 9 files (avg ~317 lines/file)

## Implementation Status (2026-01-13)

- Package split done (`ai_core/tasks/` with ingestion/graph/monitoring + helpers)
- Legacy `ai_core/tasks.py` removed; task names pinned via explicit `name=`
- Docs updated to new paths; tests + lint green (`npm run win:test:py`, `npm run lint:fix`)
- Note: `ai_core/tasks/ingestion_tasks.py` remains >500 lines; guideline acknowledged, no further split yet

## Migration Plan

### Phase 0: Choose Split Strategy (Required)

**Acceptance**: Strategy chosen and documented before structural changes

**Decision (locked)**: Option A (package move). Rationale: consistent package layout is easier for agents to
navigate than a flat set of `tasks_*` modules, and keeps the Layer split obvious.

**Option A (package move)**:
- Move the legacy single-module tasks file into `ai_core/tasks/__init__.py` in the same PR
- Add submodules under `ai_core/tasks/` and re-export from `__init__.py`

**Option B (keep module)**:
- Keep the single-module tasks layout (no package)
- Add sibling modules: `ai_core/tasks_ingestion.py`, `ai_core/tasks_graph.py`,
  `ai_core/tasks_monitoring.py`, `ai_core/tasks_helpers_*.py`
- Re-export from the top-level tasks module to preserve `ai_core.tasks.*`

**Notes**:
- Option A is preferred and now selected; rename must be atomic with package creation.

---

### Phase 1: Create Module Structure (Medium Risk)

**Acceptance**: New module exists, no behavior change

- [x] Create module/package skeleton per Phase 0 choice
- [x] Add `__init__.py` re-exports (placeholder)

**Pointers**:
- `ai_core/tasks/` (source package)
- `ai_core/tasks/__init__.py` (new, Option A)
- `ai_core/tasks_helpers_*.py` (new, Option B)

**Tests**: No test changes needed (re-exports preserve API)

---

### Phase 2: Extract Helper Modules (Low Risk)

**Acceptance**: Helpers moved, tasks import from helpers, tests green

#### 2.1 Extract Caching Helpers

**Scope**: Redis caching & deduplication logic

**Functions to move**:
- `_resolve_redis_url()`
- `_redis_client()`
- `_cache_key()`, `_dedupe_key()`
- `_cache_get()`, `_cache_set()`, `_cache_delete()`
- `_dedupe_status()`, `_acquire_dedupe_lock()`, `_mark_dedupe_done()`, `_release_dedupe_lock()`
- `_log_cache_hit()`, `_log_dedupe_hit()`

**Pointers**:
- Source: `ai_core/tasks/` (use `rg -n "def _resolve_redis_url" ai_core/tasks/`)
- Target: `ai_core/tasks/helpers/caching.py` (Option A) or `ai_core/tasks_helpers_caching.py` (Option B)

**Imports to update**: Add `from .helpers.caching import ...` to tasks modules

**Tests**:
- `ai_core/tests/test_tasks_caching.py` (if exists)
- Integration tests still use tasks from `ai_core/tasks/__init__.py`

---

#### 2.2 Extract Embedding Helpers

**Scope**: Embedding normalization, hashing, profile resolution

**Functions to move**:
- `_should_normalise_embeddings()`, `_normalise_embedding()`
- `_coerce_cache_part()`, `_hash_parts()`
- `_resolve_embedding_profile_id()`, `_resolve_embedding_model_version()`
- `_resolve_vector_space_id()`, `_resolve_vector_space_schema()`
- `_extract_chunk_meta_value()`
- `_resolve_upsert_content_hash()`, `_resolve_upsert_vector_space_id()`, `_resolve_upsert_embedding_profile()`
- `_log_embedding_cache_hit()`
- `_observed_embed_section()`
- `_extract_chunk_identifier()`

**Pointers**:
- Source: `ai_core/tasks/` (use `rg -n "def _should_normalise_embeddings" ai_core/tasks/`)
- Target: `ai_core/tasks/helpers/embedding.py` (Option A) or `ai_core/tasks_helpers_embedding.py` (Option B)

**Tests**:
- `ai_core/tests/test_tasks_embed_observability.py` (update imports)

---

#### 2.3 Extract Chunking Helpers

**Scope**: Chunking logic, tokenization, overlap calculation

**Functions to move**:
- `_split_sentences()`
- `_should_use_tiktoken()`, `set_tokenizer_override()`, `force_whitespace_tokenizer()`, `_token_count()`
- `_split_by_limit()`
- `_estimate_overlap_ratio()`, `_resolve_overlap_tokens()`
- `_chunkify()`
- `_build_chunk_prefix()`
- `_resolve_parent_capture_max_depth()`, `_resolve_parent_capture_max_bytes()`
- `_coerce_block_kind()`, `_coerce_section_path()`
- `_build_parsed_blocks()`, `_build_processing_context()`

**Pointers**:
- Source: `ai_core/tasks/` (use `rg -n "def _split_sentences" ai_core/tasks/`)
- Target: `ai_core/tasks/helpers/chunking.py` (Option A) or `ai_core/tasks_helpers_chunking.py` (Option B)

**Tests**:
- `ai_core/tests/test_tasks_chunking.py` (if exists)
- `ai_core/tests/test_tasks_embed_observability.py` (may need import updates)

---

#### 2.4 Extract Task Utils

**Scope**: Generic task utilities (path building, coercion, context extraction)

**Functions to move**:
- `_object_store_path_exists()`
- `_task_context_payload()`
- `_build_path()`, `_resolve_artifact_filename()`
- `_jsonify_for_task()`
- `_coerce_positive_int()`, `_coerce_str()`
- `_resolve_document_id()`, `_resolve_trace_context()`
- `_build_base_span_metadata()`
- `_coerce_transition_result()`, `_collect_transition_attributes()`
- `_ensure_ingestion_phase_spans()`
- `log_ingestion_run_start()`, `log_ingestion_run_end()`

**Pointers**:
- Source: `ai_core/tasks/` (use `rg -n "def _object_store_path_exists" ai_core/tasks/`)
- Target: `ai_core/tasks/helpers/task_utils.py` (Option A) or `ai_core/tasks_helpers_task_utils.py` (Option B)

**Tests**: Update imports in test files

---

### Phase 3: Extract Task Modules (Medium Risk)

**Acceptance**: Tasks split into focused modules, all imports updated, tests green

#### 3.1 Extract Monitoring Tasks

**Scope**: DLQ management, drift checks

**Tasks to move**:
- `cleanup_dead_letter_queue()` (@shared_task, name=`ai_core.tasks.cleanup_dead_letter`)
- `alert_dead_letter_queue()` (@shared_task, name=`ai_core.tasks.alert_dead_letter`)
- `embedding_drift_check()` (@shared_task, name=`ai_core.tasks.embedding_drift_check`)

**Helper functions to move**:
- `_is_redis_broker()`, `_resolve_dlq_queue_key()`, `_decode_dlq_message()`
- `_extract_dead_lettered_at()`, `_is_dlq_message_expired()`
- `_load_drift_ground_truth()`, `_drift_metrics_path()`
- `_load_previous_drift_metrics()`, `_extract_expected_ids()`, `_evaluate_drift_queries()`

**Pointers**:
- Source: `ai_core/tasks/` (use `rg -n "def cleanup_dead_letter_queue" ai_core/tasks/`)
- Target: `ai_core/tasks/monitoring_tasks.py` (Option A) or `ai_core/tasks_monitoring.py` (Option B)

**Tests**:
- `ai_core/tests/test_dead_letter_cleanup.py`

**Note**: `log_ingestion_run_start()` / `log_ingestion_run_end()` are helpers (not tasks)
used by `ai_core/ingestion.py`; move them with ingestion/observability helpers, not monitoring tasks.

---

#### 3.2 Extract Graph Tasks

**Scope**: LangGraph orchestration and execution

**Tasks to move**:
- `run_ingestion_graph()` (@shared_task, name=`ai_core.tasks.run_ingestion_graph`)
- `build_graph()` (legacy shim)

**Helper functions to move**:
- `_resolve_event_emitter()`
- `_callable_accepts_kwarg()`
- `_build_ingestion_graph()`
- `_prepare_working_state()`

**Pointers**:
- Source: `ai_core/tasks/` (use `rg -n "def run_ingestion_graph" ai_core/tasks/`)
- Target: `ai_core/tasks/graph_tasks.py` (Option A) or `ai_core/tasks_graph.py` (Option B)

**Tests**:
- `ai_core/tests/graphs/test_universal_ingestion_graph.py`
- `ai_core/tests/graphs/test_universal_ingestion_graph_hybrid_chunker.py`

---

#### 3.3 Extract Ingestion Tasks (LARGEST)

**Scope**: Legacy ingestion pipeline tasks

**Tasks to move**:
- `ingest_raw()` (@shared_task, no explicit `name=`)
- `extract_text()` (@shared_task, no explicit `name=`)
- `pii_mask()` (@shared_task, no explicit `name=`)
- `chunk()` (@shared_task - **LARGE**: ~640 lines, no explicit `name=`)
- `embed()` (@shared_task - **LARGE**: ~425 lines, no explicit `name=`)
- `upsert()` (@shared_task - **LARGE**: ~254 lines, no explicit `name=`)
- `ingestion_run()` (@shared_task, no explicit `name=`)

**Pointers**:
- Source: `ai_core/tasks/` (use `rg -n "def ingest_raw" ai_core/tasks/`)
- Target: `ai_core/tasks/ingestion_tasks.py` (Option A) or `ai_core/tasks_ingestion.py` (Option B)

**Tests**:
- `ai_core/tests/test_ingestion_task.py`
- `ai_core/tests/test_tasks_embed_observability.py`

**Note**: This is the largest module (~1,900 lines). Consider further splitting if needed:
- `ingestion_tasks_preprocessing.py` (ingest_raw, extract_text, pii_mask)
- `ingestion_tasks_chunking.py` (chunk)
- `ingestion_tasks_embedding.py` (embed, upsert)

---

### Phase 4: Update Public API (Low Risk)

**Acceptance**: `ai_core.tasks` imports work unchanged, Celery discovery works

**Tasks**:
- [x] Update `ai_core/tasks/__init__.py` (Option A) or the single-module tasks file (Option B) with re-exports
- [x] Verify Celery discovery via `noesis2/celery.py` (`app.autodiscover_tasks()`) (verified in Docker)
- [x] Update any hardcoded imports in other modules (search: `from ai_core.tasks import`)
- [x] Preserve task names for string-based callers (see "Task name stability" below)
- [x] Ensure `ai_core/tasks/__init__.py` imports submodules so tasks register with Celery autodiscovery
- [x] Update test monkeypatch targets that reference `ai_core.tasks` internals
- [x] Ensure `ai_core/ingestion.py` can still import `ai_core.tasks` as `pipe`

**Known test updates**:
- `ai_core/tests/test_dead_letter_cleanup.py` (patches `ai_core.tasks` internals like `Redis`, `time`, `settings`)
- `ai_core/tests/test_tasks_embed_observability.py` (patches `_observed_embed_section`, `_redis_client`, etc.)
- `tests/chaos/redis_faults.py` (string-based task name `ai_core.tasks.ingest_raw`)

**Re-export structure** (Option A; Option B uses the single-module tasks file):
```python
# ai_core/tasks/__init__.py
"""
Celery tasks for ingestion, graph execution, and monitoring.

For backward compatibility, all tasks are re-exported from submodules.
"""

# Ingestion tasks
from .ingestion_tasks import (
    chunk,
    embed,
    extract_text,
    ingest_raw,
    ingestion_run,
    pii_mask,
    upsert,
)

# Graph tasks
from .graph_tasks import (
    build_graph,
    run_ingestion_graph,
)

# Monitoring tasks
from .monitoring_tasks import (
    alert_dead_letter_queue,
    cleanup_dead_letter_queue,
    embedding_drift_check,
)

# Helper utilities
from .helpers.task_utils import (
    log_ingestion_run_end,
    log_ingestion_run_start,
)

__all__ = [
    # Ingestion
    "chunk",
    "embed",
    "extract_text",
    "ingest_raw",
    "ingestion_run",
    "pii_mask",
    "upsert",
    # Graph
    "build_graph",
    "run_ingestion_graph",
    # Monitoring
    "alert_dead_letter_queue",
    "cleanup_dead_letter_queue",
    "embedding_drift_check",
    # Helpers
    "log_ingestion_run_end",
    "log_ingestion_run_start",
]
```

**Pointers**:
- `ai_core/tasks/__init__.py` (new, Option A)
- Single-module tasks file (Option B)
- `noesis2/celery.py` (autodiscover)
- `noesis2/settings/base.py` (beat schedules referencing `ai_core.tasks.*`)

**Tests**:
- Run full test suite: `npm run test:py`
- Verify Celery worker startup: `npm run dev:check`

---

### Task Name Stability (Required)

**Why**: Several tasks rely on implicit Celery names (`module.function`). Moving them will change
task names unless `name=` is set or a wrapper remains at `ai_core.tasks.<task>`.

**Known string references** (non-exhaustive):
- `tests/chaos/redis_faults.py` uses `signature("ai_core.tasks.ingest_raw")`
- `noesis2/settings/base.py` schedules `ai_core.tasks.cleanup_dead_letter` / `alert_dead_letter` / `embedding_drift_check`

**Acceptance**:
- Tasks callable by string remain resolvable (`ai_core.tasks.*`)
- Use explicit `name="ai_core.tasks.<task>"` for all public tasks post-split
- Queue routing still matches `ai_core.tasks.*` in `common/celery.py`

---

### Phase 5: Cleanup & Documentation (Low Risk)

**Acceptance**: Old file deleted, imports working, docs updated

**Tasks**:
- [x] Delete the legacy single-module tasks file only if Option A is chosen (after verification)
- [x] Update `ai_core/README.md` to document new structure (N/A: no `ai_core/README.md` in repo)
- [x] Update `AGENTS.md` / `CLAUDE.md` references if needed
- [x] Update docs that reference the legacy tasks monolith to the new package path
- [x] Verify no broken imports: `npm run lint`
- [x] Run full test suite: `npm run test:py`
- [x] Update `docs/audit/architecture-anti-patterns-2025-12-31.md` (mark tasks.py as resolved)

**Known docs with legacy tasks references**:
- `AGENTS.md`
- `doc/rag_audit.md`
- `docs/architecture/4-layer-firm-hierarchy.md`
- `docs/architecture/architecture-reality.md`
- `docs/architecture/graph-onboarding.md`
- `docs/architecture/id-contract-review-checklist.md`
- `docs/architecture/id-propagation.md`
- `docs/architecture/id-semantics.md`
- `docs/architecture/id-usage-plan-graphs-workers.md`
- `docs/architecture/layer-contracts.md`
- `docs/architecture/user-document-integration.md`
- `docs/backlog/refactor-logging-contextvars.md`
- `docs/docker/conventions.md`
- `docs/observability/langfuse.md`
- `docs/rag/crawler_chunking_review.md`
- `ai_core/rag/chunking/README.md`
- `roadmap/SOTA-backlog.md`
- `roadmap/document-repo-user-integration.md`

**Pointers**:
- `ai_core/README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/audit/architecture-anti-patterns-2025-12-31.md`

---

## Verification Checklist

### Celery Discovery
- [x] `celery -A noesis2 inspect registered` shows all tasks (verified via Docker manage shell)
- [ ] Tasks appear in Flower UI (if running)
- [x] `noesis2/celery.py` uses `app.autodiscover_tasks()` and `ai_core` is installed

### Import Compatibility
- [x] `from ai_core.tasks import chunk` works (backward compat)
- [x] `from ai_core.tasks.ingestion_tasks import chunk` works (Option A)
- [x] `from ai_core.tasks_ingestion import chunk` works (Option B; N/A for Option A)
- [x] No circular imports: `npm run lint`

### Tests
- [ ] `npm run test:py:unit` passes (unit tests)
- [ ] `npm run test:py:fast` passes (fast integration)
- [ ] `npm run test:py:parallel` passes (full parallel suite)
- [x] `npm run test:py` passes (full suite with slow tests)

### Runtime
- [ ] Dev stack starts: `npm run dev:stack`
- [ ] Ingestion task runs: trigger via `/rag-tools` workbench
- [ ] Graph execution works: check Langfuse traces
- [ ] DLQ cleanup runs: `python manage.py shell` -> `cleanup_dead_letter_queue.delay()`

---

## Risks & Mitigation

### Risk: Celery discovery breaks
**Likelihood**: Low
**Impact**: High (production tasks not registered)
**Mitigation**:
- Test Celery discovery before deleting old file
- Keep a re-export stub at `ai_core/tasks/` (Option A) or the single-module tasks file (Option B) during transition
- Add CI check for Celery task registration count

### Risk: Module/package name collision
**Likelihood**: Medium
**Impact**: High (imports resolve to the wrong module)
**Mitigation**:
- Do not create the package while the single-module tasks file still exists
- Use Phase 0 to select Option A or Option B before any file moves

### Risk: Circular imports
**Likelihood**: Medium
**Impact**: Medium (import errors at runtime)
**Mitigation**:
- Keep helpers in `helpers/` subpackage (clear import direction)
- Use `from __future__ import annotations` for type hints
- Run linter after each phase: `npm run lint`

### Risk: Test imports break
**Likelihood**: Medium
**Impact**: Low (easy to fix)
**Mitigation**:
- Update test imports in batches
- Use IDE refactoring tools (VSCode: "Update imports")
- Run tests after each phase

### Risk: Missing helper dependencies
**Likelihood**: Low
**Impact**: Low (runtime errors in specific tasks)
**Mitigation**:
- Extract helpers first (Phase 2), verify imports work
- Run integration tests after each helper extraction
- Use `rg -n "def _function_name" ai_core/tasks/` to find all call sites

---

## Effort Estimate

**Total effort**: 8-12 hours (Medium Effort for P1)

| Phase | Estimated Time | Risk Level |
|-------|----------------|------------|
| Phase 1: Module structure | 30 min | Medium |
| Phase 2: Extract helpers | 2-3 hours | Low |
| Phase 3: Extract tasks | 4-6 hours | Medium |
| Phase 4: Update public API | 1 hour | Low |
| Phase 5: Cleanup & docs | 1-2 hours | Low |

**Dependencies**: None (can start immediately)

**Blocking**: None (no other work blocked by this)

---

## Success Metrics

- **File size**: No file in `ai_core/tasks/` >500 lines
- **Note**: 500 lines is a guideline, not a hard limit. If `ingestion_tasks.py` exceeds it,
  either split further or document why the size is acceptable.
- **Test coverage**: No reduction in test coverage (maintain 100% for critical paths)
- **Import time**: `import ai_core.tasks` time unchanged or reduced
- **Developer satisfaction**: Easier to find relevant code (subjective, measured via team feedback)

---

## Future Work (Out of Scope)

- Further split `ingestion_tasks.py` into preprocessing/chunking/embedding (if >800 lines)
- Extract chunking logic to `ai_core/rag/chunking/` (align with adaptive chunking)
- Consolidate monitoring tasks with `ai_core/monitoring/` package (if created)
- Deprecate legacy ingestion tasks in favor of graph-only execution

---

## References

- [docs/audit/architecture-anti-patterns-2025-12-31.md](../docs/audit/architecture-anti-patterns-2025-12-31.md) - "Break Up God Files" initiative
- [roadmap/backlog.md](backlog.md) - P2 "Break Up God Files" (completed for services, views)
- [AGENTS.md](../AGENTS.md) - Layer architecture, import rules
- [ai_core/README.md](../ai_core/README.md) - AI Core module overview
- [ai_core/tasks/](../ai_core/tasks/) - Current split package (legacy monolith removed)

---

**Next Steps**:
1. Get approval from team/stakeholders
2. Create feature branch: `refactor/split-ai-core-tasks`
3. Execute Phase 1-2 (helpers extraction) in single PR
4. Execute Phase 3-5 (task extraction + cleanup) in follow-up PR
5. Merge to `main` after full test suite passes
