# ID Contract One-Pager (Pre-MVP)

Goal
Establish a single, stable contract for IDs, scope, and tracing as the basis
for user management and permissions. This contract is the target; existing DB
schemas and Pydantic models are not constraints.

Non-Goals
- Preserve current DB/Pydantic structures.
- Avoid migrations or resets (resets are expected if needed).

1) Glossary and ID Semantics
- tenant_id (UUIDv7): mandatory tenant scope for auth/permissions.
- tenant_schema (string): technical routing identifier, not a business ID.
- user_id (UUIDv7): tenant-scoped identity.
- case_id (UUIDv7): business scope and permission primitive (CaseMembership).
- collection_id (UUIDv7, optional): only if it is a real permission boundary.
- workflow_id (UUIDv7, optional): workflow definition identifier (business).
- workflow_run_id (UUIDv7): workflow/graph execution identifier (technical).
- ingestion_run_id (UUIDv7): ingestion execution identifier (technical).
- trace_id (W3C-compatible): observability correlation across hops.
- invocation_id (UUIDv7): per-hop request/job identifier.
- idempotency_key (string): request-level determinism key.

2) ScopeContext Contract
Required: tenant_id, trace_id, invocation_id.
Auth rule: when auth is present, user_id is required. Without auth, user_id
must be absent and the endpoint must be explicitly public.
Optional: case_id, collection_id, workflow_id.
Runtime IDs: workflow_run_id and ingestion_run_id may co-exist (no XOR).
Invariants: case.tenant_id must equal the context tenant_id.

3) Scoping Rules
- Tenant-scoped: every operation must include tenant_id.
- Case-scoped: case_id is required. No defaulting in the normalizer.
- General case: explicit domain object per tenant. External counsel is never
  auto-membered.

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
- trace_id stays constant across hops.
- invocation_id is unique per hop (new ID per request/job).

7) Idempotency
- Request-level only.
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
- created_by_user_id (optional)

9) Normalization is the Only Entry Point
All entry points must build ScopeContext via a canonical normalize_request /
normalize_context function. No manual header extraction.

