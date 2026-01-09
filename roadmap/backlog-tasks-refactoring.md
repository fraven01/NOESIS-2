# Backlog: ai_core/tasks.py Refactoring

**Status**: Draft
**Created**: 2026-01-06
**Priority**: P1 - High Value Cleanup (Medium Effort)
**Context**: Follow-up to "Break Up God Files" (P2) - ai_core/tasks.py still contains 3,488 lines

## Problem Statement

`ai_core/tasks.py` has grown to **3,488 lines** with multiple responsibilities:
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
ai_core/tasks/
├── __init__.py                    # Public API - re-exports all tasks for backward compatibility
├── ingestion_tasks.py             # Legacy ingestion: ingest_raw, extract_text, pii_mask, chunk, embed, upsert (~800 lines)
├── graph_tasks.py                 # Graph orchestration: run_ingestion_graph, build_graph (~500 lines)
├── monitoring_tasks.py            # DLQ, drift checks, logging tasks (~400 lines)
├── helpers/
│   ├── __init__.py
│   ├── caching.py                 # Redis caching & deduplication (~250 lines)
│   ├── embedding.py               # Embedding normalization, hashing, profile resolution (~300 lines)
│   ├── chunking.py                # Semantic chunker helpers, overlap calculation (~400 lines)
│   └── task_utils.py              # Shared utilities: _build_path, _coerce_*, _jsonify_for_task (~200 lines)
```

**Estimated total after split**: ~2,850 lines across 9 files (avg ~317 lines/file)

## Migration Plan

### Phase 1: Create Module Structure (Low Risk)

**Acceptance**: New module exists, no behavior change

- [ ] Create `ai_core/tasks/` package directory
- [ ] Create `ai_core/tasks/helpers/` subpackage
- [ ] Create skeleton files with imports and docstrings
- [ ] Add `__init__.py` with re-exports (placeholder)

**Pointers**:
- `ai_core/tasks.py` (source)
- `ai_core/tasks/__init__.py` (new)
- `ai_core/tasks/helpers/__init__.py` (new)

**Tests**: No test changes needed (re-exports preserve API)

---

### Phase 2: Extract Helper Modules (Low Risk)

**Acceptance**: Helpers moved, `ai_core/tasks.py` imports from helpers/, tests green

#### 2.1 Extract Caching Helpers

**Scope**: Redis caching & deduplication logic

**Functions to move**:
- `_resolve_redis_url()` (L217)
- `_redis_client()` (L227)
- `_cache_key()`, `_dedupe_key()` (L240, L244)
- `_cache_get()`, `_cache_set()`, `_cache_delete()` (L248, L256, L263)
- `_dedupe_status()`, `_acquire_dedupe_lock()`, `_mark_dedupe_done()`, `_release_dedupe_lock()` (L270-303)
- `_log_cache_hit()`, `_log_dedupe_hit()` (L332, L372)

**Pointers**:
- Source: `ai_core/tasks.py:217-391`
- Target: `ai_core/tasks/helpers/caching.py`

**Imports to update**: Add `from .helpers.caching import ...` to tasks modules

**Tests**:
- `ai_core/tests/test_tasks_caching.py` (if exists)
- Integration tests still use tasks from `ai_core/tasks/__init__.py`

---

#### 2.2 Extract Embedding Helpers

**Scope**: Embedding normalization, hashing, profile resolution

**Functions to move**:
- `_should_normalise_embeddings()`, `_normalise_embedding()` (L111, L125)
- `_coerce_cache_part()`, `_hash_parts()` (L149, L162)
- `_resolve_embedding_profile_id()`, `_resolve_embedding_model_version()` (L173, L184)
- `_resolve_vector_space_id()`, `_resolve_vector_space_schema()` (L195, L206)
- `_extract_chunk_meta_value()` (L391)
- `_resolve_upsert_content_hash()`, `_resolve_upsert_vector_space_id()`, `_resolve_upsert_embedding_profile()` (L404-438)
- `_log_embedding_cache_hit()` (L352)
- `_observed_embed_section()` (L521)
- `_extract_chunk_identifier()` (L557)

**Pointers**:
- Source: `ai_core/tasks.py:111-570`
- Target: `ai_core/tasks/helpers/embedding.py`

**Tests**:
- `ai_core/tests/test_tasks_embed_observability.py` (update imports)

---

#### 2.3 Extract Chunking Helpers

**Scope**: Chunking logic, tokenization, overlap calculation

**Functions to move**:
- `_split_sentences()` (L721)
- `_should_use_tiktoken()`, `set_tokenizer_override()`, `force_whitespace_tokenizer()`, `_token_count()` (L783-819)
- `_split_by_limit()` (L820)
- `_estimate_overlap_ratio()`, `_resolve_overlap_tokens()` (L872, L922)
- `_chunkify()` (L953)
- `_build_chunk_prefix()` (L1016)
- `_resolve_parent_capture_max_depth()`, `_resolve_parent_capture_max_bytes()` (L1029, L1043)
- `_coerce_block_kind()`, `_coerce_section_path()` (L1069, L1076)
- `_build_parsed_blocks()`, `_build_processing_context()` (L1083, L1176)

**Pointers**:
- Source: `ai_core/tasks.py:721-1211`
- Target: `ai_core/tasks/helpers/chunking.py`

**Tests**:
- `ai_core/tests/test_tasks_chunking.py` (if exists)
- `ai_core/tests/test_tasks_embed_observability.py` (may need import updates)

---

#### 2.4 Extract Task Utils

**Scope**: Generic task utilities (path building, coercion, context extraction)

**Functions to move**:
- `_object_store_path_exists()` (L306)
- `_task_context_payload()` (L313)
- `_build_path()`, `_resolve_artifact_filename()` (L440, L453)
- `_jsonify_for_task()` (L616)
- `_coerce_positive_int()`, `_coerce_str()` (L2731, L3022)
- `_resolve_document_id()`, `_resolve_trace_context()` (L3037, L3067)
- `_build_base_span_metadata()` (L3101)
- `_coerce_transition_result()`, `_collect_transition_attributes()` (L3125, L3140)
- `_ensure_ingestion_phase_spans()` (L3226)

**Pointers**:
- Source: `ai_core/tasks.py:306-3273`
- Target: `ai_core/tasks/helpers/task_utils.py`

**Tests**: Update imports in test files

---

### Phase 3: Extract Task Modules (Medium Risk)

**Acceptance**: Tasks split into focused modules, all imports updated, tests green

#### 3.1 Extract Monitoring Tasks

**Scope**: DLQ management, drift checks, logging

**Tasks to move**:
- `cleanup_dead_letter_queue()` (L2665, @shared_task)
- `alert_dead_letter_queue()` (L2744, @shared_task)
- `embedding_drift_check()` (L2911, @shared_task)
- `log_ingestion_run_start()`, `log_ingestion_run_end()` (L570, L621)

**Helper functions to move**:
- `_is_redis_broker()`, `_resolve_dlq_queue_key()`, `_decode_dlq_message()` (L2600-2609)
- `_extract_dead_lettered_at()`, `_is_dlq_message_expired()` (L2623, L2650)
- `_load_drift_ground_truth()`, `_drift_metrics_path()` (L2790, L2805)
- `_load_previous_drift_metrics()`, `_extract_expected_ids()`, `_evaluate_drift_queries()` (L2811-2906)

**Pointers**:
- Source: `ai_core/tasks.py:570-2982`
- Target: `ai_core/tasks/monitoring_tasks.py`

**Tests**:
- `ai_core/tests/test_tasks_dlq.py` (if exists)
- `ai_core/tests/test_tasks_drift.py` (if exists)

---

#### 3.2 Extract Graph Tasks

**Scope**: LangGraph orchestration and execution

**Tasks to move**:
- `run_ingestion_graph()` (L3280, @shared_task)
- `build_graph()` (L3016)

**Helper functions to move**:
- `_resolve_event_emitter()` (L2982)
- `_callable_accepts_kwarg()` (L2990)
- `_build_ingestion_graph()` (L3007)
- `_prepare_working_state()` (L3424+, needs verification)

**Pointers**:
- Source: `ai_core/tasks.py:2982-3488`
- Target: `ai_core/tasks/graph_tasks.py`

**Tests**:
- `ai_core/tests/test_tasks.py` (graph execution tests)
- `ai_core/tests/graphs/test_universal_ingestion_graph.py` (integration tests)

---

#### 3.3 Extract Ingestion Tasks (LARGEST)

**Scope**: Legacy ingestion pipeline tasks

**Tasks to move**:
- `ingest_raw()` (L679, @shared_task)
- `extract_text()` (L693, @shared_task)
- `pii_mask()` (L703, @shared_task)
- `chunk()` (L1220, @shared_task - **LARGE**: ~640 lines)
- `embed()` (L1867, @shared_task - **LARGE**: ~425 lines)
- `upsert()` (L2293, @shared_task - **LARGE**: ~254 lines)
- `ingestion_run()` (L2548, @shared_task)

**Pointers**:
- Source: `ai_core/tasks.py:679-2598`
- Target: `ai_core/tasks/ingestion_tasks.py`

**Tests**:
- `ai_core/tests/test_tasks.py` (ingestion task tests)
- `ai_core/tests/test_tasks_embed_observability.py`
- `ai_core/tests/test_rag_ingestion_run.py`

**Note**: This is the largest module (~1,900 lines). Consider further splitting if needed:
- `ingestion_tasks_preprocessing.py` (ingest_raw, extract_text, pii_mask)
- `ingestion_tasks_chunking.py` (chunk)
- `ingestion_tasks_embedding.py` (embed, upsert)

---

### Phase 4: Update Public API (Low Risk)

**Acceptance**: `ai_core.tasks` imports work unchanged, Celery discovery works

**Tasks**:
- [ ] Update `ai_core/tasks/__init__.py` with comprehensive re-exports
- [ ] Verify Celery task discovery (check `CELERY_IMPORTS` in settings)
- [ ] Update any hardcoded imports in other modules (search: `from ai_core.tasks import`)

**Re-export structure**:
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
    "log_ingestion_run_end",
    "log_ingestion_run_start",
]
```

**Pointers**:
- `ai_core/tasks/__init__.py` (new)
- `noesis2/settings/celery.py` (verify `CELERY_IMPORTS`)

**Tests**:
- Run full test suite: `npm run test:py`
- Verify Celery worker startup: `npm run dev:check`

---

### Phase 5: Cleanup & Documentation (Low Risk)

**Acceptance**: Old file deleted, imports working, docs updated

**Tasks**:
- [ ] Delete `ai_core/tasks.py` (after verification)
- [ ] Update `ai_core/README.md` to document new structure
- [ ] Update `AGENTS.md` / `CLAUDE.md` references if needed
- [ ] Verify no broken imports: `npm run lint`
- [ ] Run full test suite: `npm run test:py`
- [ ] Update `docs/audit/architecture-anti-patterns-2025-12-31.md` (mark tasks.py as resolved)

**Pointers**:
- `ai_core/README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/audit/architecture-anti-patterns-2025-12-31.md`

---

## Verification Checklist

### Celery Discovery
- [ ] `celery -A common.celery inspect registered` shows all tasks
- [ ] Tasks appear in Flower UI (if running)
- [ ] `noesis2/settings/celery.py:CELERY_IMPORTS` includes `ai_core.tasks` (package)

### Import Compatibility
- [ ] `from ai_core.tasks import chunk` works (backward compat)
- [ ] `from ai_core.tasks.ingestion_tasks import chunk` works (new API)
- [ ] No circular imports: `npm run lint`

### Tests
- [ ] `npm run test:py:unit` passes (unit tests)
- [ ] `npm run test:py:fast` passes (fast integration)
- [ ] `npm run test:py:parallel` passes (full parallel suite)
- [ ] `npm run test:py` passes (full suite with slow tests)

### Runtime
- [ ] Dev stack starts: `npm run dev:stack`
- [ ] Ingestion task runs: trigger via `/rag-tools` workbench
- [ ] Graph execution works: check Langfuse traces
- [ ] DLQ cleanup runs: `python manage.py shell` → `cleanup_dead_letter_queue.delay()`

---

## Risks & Mitigation

### Risk: Celery discovery breaks
**Likelihood**: Low
**Impact**: High (production tasks not registered)
**Mitigation**:
- Test Celery discovery before deleting old file
- Keep `ai_core/tasks.py` as re-export stub during transition
- Add CI check for Celery task registration count

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
- Use `grep -r "def _function_name"` to find all call sites

---

## Effort Estimate

**Total effort**: 8-12 hours (Medium Effort for P1)

| Phase | Estimated Time | Risk Level |
|-------|----------------|------------|
| Phase 1: Module structure | 30 min | Low |
| Phase 2: Extract helpers | 2-3 hours | Low |
| Phase 3: Extract tasks | 4-6 hours | Medium |
| Phase 4: Update public API | 1 hour | Low |
| Phase 5: Cleanup & docs | 1-2 hours | Low |

**Dependencies**: None (can start immediately)

**Blocking**: None (no other work blocked by this)

---

## Success Metrics

- **File size**: No file in `ai_core/tasks/` >500 lines
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
- [ai_core/tasks.py](../ai_core/tasks.py) - Current monolith (3,488 lines)

---

**Next Steps**:
1. Get approval from team/stakeholders
2. Create feature branch: `refactor/split-ai-core-tasks`
3. Execute Phase 1-2 (helpers extraction) in single PR
4. Execute Phase 3-5 (task extraction + cleanup) in follow-up PR
5. Merge to `main` after full test suite passes
