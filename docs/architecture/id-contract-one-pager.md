# ID Contract One-Pager (Pre-MVP)

Goal
Establish a single, stable contract for IDs, scope, and tracing as the basis
for user management and permissions. This contract is the target; existing DB
schemas and Pydantic models are not constraints.

Non-Goals
- Preserve current DB/Pydantic structures.
- Avoid migrations or resets (resets are expected if needed).

1) Glossary and ID Semantics

Business IDs:
- tenant_id (UUIDv7): mandatory tenant scope for auth/permissions.
- user_id (UUIDv7): tenant-scoped identity.
- case_id (UUIDv7): business scope and permission primitive (CaseMembership).
- system_case_id (UUIDv7): per-tenant system case for tenant-wide operations.
  Note: "general" is a display name only, not exposed as case_id in APIs.
- collection_id (UUIDv7, optional): only if it is a real permission boundary.
- workflow_id (UUIDv7, optional): workflow definition identifier (business).

Technical/Routing IDs:
- tenant_schema (string): technical routing for django-tenants, derived from
  tenant_id lookup. INVARIANT: tenant_schema is loaded FROM tenant_id, never
  the reverse (except one-time migration path).
- workflow_run_id (UUIDv7): workflow/graph execution identifier (technical).
- ingestion_run_id (UUIDv7): ingestion execution identifier (technical).

Observability IDs:
- trace_id (W3C-compatible): observability correlation across hops.
- invocation_id (UUIDv7): per-hop request/job identifier.
  Hop = HTTP Request OR Celery Task Start.

Identity IDs (for S2S):
- service_id (string, REQUIRED for S2S): identifies the service in S2S hops
  (e.g., "celery-ingestion-worker", "graph-executor").
- initiated_by_user_id (UUIDv7, optional): in audit_meta, tracks which user
  triggered a service-to-service flow. Causal only, never a principal.
- created_by_user_id (UUIDv7, optional): in audit_meta, who owns the entity.
  DISTINCT from initiated_by_user_id - do not conflate.

Idempotency:
- idempotency_key (string): request-level determinism key.
- idempotency_scope_name (string): stable identifier for idempotency scope,
  NOT the URL. Examples: "create_document", "start_ingestion".

2) ScopeContext Contract

Required: tenant_id, trace_id, invocation_id.

Auth rule: when auth is present, user_id is required. Without auth, user_id
must be absent and the endpoint must be explicitly public.

S2S rule (strict):
- User Request Hop: user_id REQUIRED, service_id ABSENT.
- S2S Hop (Celery/Graph): service_id REQUIRED, user_id ABSENT.
- initiated_by_user_id is causal tracking in audit_meta only, never a principal.

Optional: case_id, collection_id, workflow_id.

Runtime IDs: workflow_run_id and ingestion_run_id may co-exist (no XOR).

Invariants: case.tenant_id must equal the context tenant_id.

3) Scoping Rules
- Tenant-scoped: every operation must include tenant_id.
- Case-scoped: case_id is required. No defaulting in the normalizer.
- System case (formerly "general case"):
  - Each tenant has exactly one system_case_id (UUIDv7, created at bootstrap).
  - "general" is a display name only, not exposed as case_id in external APIs.
  - Avoids collision if customer wants a case literally named "General".
  - System Case MUST NOT be used as backdoor for tenant-scoped admin functions.
  - External counsel is never auto-membered to System Case.

4) Runtime Rules
- Starting a workflow requires workflow_run_id.
- Starting ingestion requires ingestion_run_id.
- Workflows may trigger ingestion: both IDs can exist.
- workflow_run_id and ingestion_run_id are technical execution IDs, not
  business identities.

5) Trust Model (tenant_id)
- Source of truth: auth + tenant routing.
- Headers are for plausibility checks/debug only, never authority.

6) Tracing
- trace_id stays constant across hops (end-to-end correlation).
- invocation_id is unique per hop (new ID per request/job).
- Hop definition: HTTP Request OR Celery Task Start.
- Graph Node executions share the invocation_id of their triggering Celery Task.

7) Idempotency
- Request-level only.
- Entry points: HTTP Requests and Celery Task Start.
- Scope key: (tenant_id, idempotency_scope_name, idempotency_key).
- idempotency_scope_name: stable string, NOT the URL.
  Examples: "create_document", "start_ingestion", "send_invite".
- TTL: 24 hours default, per-scope overridable.
- Deduplication store: Redis with TTL, required for Pre-MVP.
- DB constraint: Only for critical writes (Document creation, etc.).
- Celery replay behavior: no side effect, log replay=true,
  optionally return existing result handle.
- Response must mark replay vs fresh consistently:
  - Header: X-Idempotency-Replayed: true|false
  - Optional body field: idempotency_replayed
- Replay responses must be semantically equivalent, even if side effects are
  suppressed.

8) Persistence Guideline (audit_meta)
Pre-MVP: use audit_meta JSON on core entities, not a column explosion.

Minimal keys:
- trace_id
- invocation_id
- workflow_run_id (optional)
- ingestion_run_id (optional)
- idempotency_key (optional)
- created_by_user_id (optional): who OWNS the entity. SET ONCE at creation,
  immutable thereafter.
- initiated_by_user_id (optional): who TRIGGERED the flow (root cause).
- last_hop_service_id (optional): which service last wrote this entity.

IMPORTANT: ScopeContext.service_id != audit_meta.last_hop_service_id
- ScopeContext.service_id = who executes THIS hop (request scope).
- audit_meta.last_hop_service_id = who LAST WROTE this entity (persistence scope).

9) Normalization is the Only Entry Point
All entry points must build ScopeContext via a canonical normalize_request /
normalize_context function. No manual header extraction.

