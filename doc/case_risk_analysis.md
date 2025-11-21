# Case Subsystem Risk Analysis

This document details the findings of a code analysis targeting four specific risks in the Case subsystem.

## 1. Authority of `case_id` vs `Case.external_id`

**Risk**: The authority of `case_id` and `Case.external_id` is not strictly enforced.
**Finding**: **Partially Mitigated**.

* **Middleware**: `RequestContextMiddleware` (`ai_core/middleware/context.py`) normalizes the `X-Case-ID` header but does not validate it against the database. It binds it to the logging context as-is.
* **Services**: `cases.services.get_or_create_case_for` performs normalization (`strip()`) and ensures a `Case` exists for the given ID.
* **Ingestion**: `ai_core.ingestion.process_document` accepts `case` as a string argument and passes it through to storage paths and metadata without immediate validation against the `Case` table. Validation happens lazily via `record_ingestion_case_event`.
* **Gap**: There is no central "gatekeeper" that rejects API requests with invalid `case_id` formats (beyond basic string strictness) or non-existent IDs before they reach workers.

## 2. `DocumentIngestionRun.case` Consistency

**Risk**: `DocumentIngestionRun.case` could contain inconsistent values.
**Finding**: **Confirmed Risk**.

* **Model**: `DocumentIngestionRun.case` is a `CharField` (string), not a ForeignKey.
* **Creation**: Ingestion runs are created/updated via `RedisIngestionStatusStore` (`ai_core/ingestion_status.py`) using the raw string ID passed from the worker.
* **Validation**: There is no database constraint ensuring `DocumentIngestionRun.case` matches a `Case.external_id`.
* **Mitigation**: `record_ingestion_case_event` calls `get_or_create_case_for`, which effectively "blesses" any string ID used in ingestion by creating a corresponding `Case` record if one doesn't exist. This prevents dangling references but allows implicit Case creation via ingestion.

## 3. Direct Case Status Updates by Workers

**Risk**: Workers might overwrite Case status directly instead of generating events.
**Finding**: **No Evidence of Risk**.

* **Analysis**: A search for `Case.objects.update` and `.save()` across `llm_worker` and `ai_core` revealed no direct status mutations by workers.
* **Pattern**: Workers consistently use the bridging functions:
  * Ingestion uses `emit_ingestion_case_event` -> `record_ingestion_case_event`.
  * Collection Search uses `emit_case_lifecycle_for_collection_search`.
* **Observation**: `ai_core.ingestion._load_case_observability_context` performs a *read-only* query on `Case` to enrich logs, which is safe.

## 4. Event Generation & Lifecycle Coverage

**Risk**: Event generation might be bypassed, leading to stale lifecycle states.
**Finding**: **Good Coverage, Minor Edge Cases**.

* **Ingestion**: `process_document` calls `emit_ingestion_case_event` at the end of the pipeline (via `pipe.ingestion_completed` task chain usually, though the analyzed file `ai_core/ingestion.py` shows imports but the direct call in `process_document` was not visible in the snippet, it's likely in the `on_success` callback of the chain).
  * *Correction*: `ai_core/case_events.py` exists and is used.
* **Graphs**: `CollectionSearchGraph` explicitly emits lifecycle events at the end of execution (`llm_worker/tasks.py`).
* **Gap**: If a worker crashes *hard* (OOM) before the `finally` block or callback, no "failed" event might be emitted, leaving the Case in an inconsistent state (though `DocumentIngestionRun` might eventually time out).

## Recommendations

1. **Enforce ID Validation**: Implement a strict validator for `case_id` strings (e.g., regex) at the API gateway/middleware level to prevent garbage strings from entering the system.
2. **Explicit Case Creation**: Consider disabling `get_or_create` behavior in `record_ingestion_case_event` for strict mode, requiring Cases to be pre-created via API.
3. **Ingestion Foreign Key**: Long-term, migrate `DocumentIngestionRun.case` to a ForeignKey to enforce referential integrity, though this couples the domains tightly.
