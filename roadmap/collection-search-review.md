# Collection Search Review Followups

Scope: `ai_core/graphs/technical/collection_search.py`, `ai_core/services/collection_search/strategy.py`,
`llm_worker/graphs/hybrid_search_and_score.py`, `llm_worker/graphs/score_results.py`,
`ai_core/llm/client.py`.

This doc captures the detailed followups referenced by `roadmap/backlog.md`.

## Graph Boundaries (Validated)

- `CollectionSearchGraphRequest` and `CollectionSearchGraphOutput` are Pydantic-validated at the boundary.
- Output fields remain loosely typed (`Mapping[str, Any]`), which is acceptable at the boundary but creates
  internal contract drift when nodes pass dicts instead of models.

## Timeout and Stall Risks (P0)

1) LLM client has no explicit HTTP timeouts (sync + streaming), which can hang indefinitely.
   - Pointers: `ai_core/llm/client.py:391`, `ai_core/llm/client.py:738`
   - Fix: add connect/read timeouts (configurable) and ensure timeout errors surface as retries/failures.

2) Parallel search has no overall timeout and blocks on `future.result()`; a single slow query can stall the node.
   - Pointers: `ai_core/graphs/technical/collection_search.py:561`, `ai_core/graphs/technical/collection_search.py:650`
   - Fix: add `asyncio.wait_for` or equivalent total timeout; return partial results on timeout.

3) Hybrid scoring handles `TimeoutError`, but no upstream timeout is configured, so the branch is effectively dead.
   - Pointers: `llm_worker/graphs/hybrid_search_and_score.py:1116`, `llm_worker/graphs/score_results.py:353`
   - Fix: wire LLM timeouts so this branch can actually fire; log it consistently.

4) Whole-graph timeout must be Windows-safe; avoid `signal.alarm`.
   - Pointers: `llm_worker/tasks.py:run_graph`, `ai_core/services/__init__.py`
   - Fix: rely on worker-level time limits or explicit timeout wrappers.

## Boundary Contract Violations (V1-V8)

| ID | Location | Violation | Severity | Notes / Desired Fix |
| --- | --- | --- | --- | --- |
| V1 | `ai_core/graphs/technical/collection_search.py:140` | Output `search` uses `Mapping[str, Any]` | MEDIUM | Define typed output payload or strongly typed internal envelope before boundary serialization. |
| V2 | `ai_core/graphs/technical/collection_search.py:480` | `strategy.model_dump()` passed through state | HIGH | Keep `SearchStrategy` model in state; serialize only at boundary. |
| V3 | `ai_core/graphs/technical/collection_search.py:636` | Re-parse `SearchStrategy` from dict | HIGH | Avoid dict round-trip; reuse typed model from state. |
| V4 | `ai_core/graphs/technical/collection_search.py:837` | dict -> `SearchCandidate` -> dict -> dict churn | CRITICAL | Keep `SearchCandidate` list in state and serialize once for output. |
| V5 | `llm_worker/graphs/hybrid_search_and_score.py:626` | Dead code branch for `SearchCandidate` input | LOW | Remove dead branch or re-enable by passing typed candidates. |
| V6 | `llm_worker/graphs/hybrid_search_and_score.py:1389` | Redundant `.model_dump()` | MEDIUM | Avoid double-serialization; preserve models. |
| V7 | `llm_worker/graphs/score_results.py:353` | Three config dicts (`control`, `meta`, prompt) | MEDIUM | Consolidate config shape to one typed config or explicit parameters. |
| V8 | `ai_core/graphs/technical/collection_search.py:867-879` | Hardcoded `jurisdiction`/`purpose` | MEDIUM | Derive from `GraphInput` or `BusinessContext` instead of constants. |

## Strategy Quality Improvements (P1)

- Prompt lacks few-shot examples and strict JSON schema adherence.
  - Pointers: `ai_core/services/collection_search/strategy.py:226`
  - Fix: add JSON-only schema block and 1-2 examples; keep policy guidance explicit.

- Fallback strategy is too generic.
  - Pointers: `ai_core/services/collection_search/strategy.py:139`
  - Fix: include domain-aware variants and reduce trivial "overview/guide" repeats.

## Adaptive Embedding Weights (P1)

- Embedding vs heuristic weight is static unless callers override input.
  - Pointer: `ai_core/graphs/technical/collection_search.py:738`
  - Fix: add a quality_mode -> weight profile map and record the chosen profile in telemetry.

## Suggested Acceptance Criteria (Summary)

- Timeouts: explicit HTTP timeouts for sync + streaming LLM calls; overall timeout for parallel search; timeout
  handling in hybrid scoring is reachable.
- Contracts: models pass through internal nodes without dict round-trips; boundary serialization happens once.
- Cleanup: remove dead branches and redundant serialization; avoid hardcoded jurisdiction/purpose.
- Tests: update `ai_core/tests/graphs/test_collection_search_graph.py` and relevant llm_worker tests to
  enforce typed boundaries and timeout behavior.
