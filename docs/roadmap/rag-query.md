# Roadmap: RAG Query and Retrieval (Future Work)

Status: frontbook item, to be scheduled after the current backlog is complete.

This document is intentionally a proposal. Code remains the source of truth.

## Motivation

RagQueryViewV1 currently routes to the technical `rag.default` graph. This is
acceptable as a temporary dev/workbench path, but we want a clearer, capability-
first query stack that is versioned, testable, and consistent with the new
graph I/O contracts.

## Goals

- Define a canonical, versioned graph I/O contract for RAG query.
- Make retrieval and composition capability-first and reusable across graphs.
- Keep UI endpoints decoupled from technical graphs via a service boundary or
  business graph facade.
- Preserve strict ToolContext usage and the runtime context injection pattern.
- Make query behavior explicit (filters, top_k, visibility, guardrails).

## Non-goals

- No immediate behavior changes in production endpoints.
- No naming or ID decisions that bypass `ai_core/graph/registry.py`.

## Proposed Direction (Draft)

### Graph boundary

- A dedicated RAG Query graph (name to be defined in the registry).
- Versioned Pydantic input/output models with `schema_id` and `schema_version`.
- Input model includes:
  - query
  - top_k override
  - retrieval scope/filters
  - answer format preferences
  - optional safety/visibility controls
- Output model includes:
  - answer (or structured response)
  - prompt_version
  - retrieval metadata
  - snippets/citations

### Capabilities

Build the graph by composing explicit capabilities, not ad-hoc logic:

- query normalization and routing
- retrieval (existing `retrieve.run` capability)
- rerank/hybrid scoring where required
- answer composition (existing `compose.run` or a dedicated capability)
- policy/guardrail checks (visibility, scope, safety)

### Execution boundary

- UI endpoints call a GraphExecutor or business facade, not technical graphs.
- `RagQueryViewV1` becomes a thin adapter over the new boundary once available.
- Technical graph implementations remain reusable, but are not invoked directly
  by production endpoints.

## Migration Plan (Draft)

1. Define the graph I/O contract and add tests for schema validation.
2. Implement the new RAG Query graph with capability-first nodes.
3. Route dev/workbench endpoints to the new boundary.
4. Deprecate direct calls to the old `rag.default` path where appropriate.

## Open Questions

- Which capability set is mandatory vs optional for MVP?
- Which filters and visibility rules are required by default?
- Should the query graph expose multiple response formats (plain, structured)?
