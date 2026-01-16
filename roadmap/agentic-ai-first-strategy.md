# Agentic AI-First Strategy (Target State)

Status: Draft
Owner: Architecture + Product

This file is a planning artifact. It is not a runtime contract. Runtime behavior and contracts are defined in code.

## Goal

Build a deterministic, auditable, agentic AI-first system for regulated use cases.
This is not a chat assistant. It is a controlled planning and execution system.

## Core principle

Agents are execution logic over explicit state.
Truth lives in plans, rules, and artifacts, not in LLM output.

## Key artifacts

### Blueprints

- Machine-readable definitions of structure, allowed variables, constraints, and gates.
- Blueprints bound what the system is allowed to do.

### Implementation plans

- One plan per workflow execution context (per workflow step).
- Plans contain slots (structured values, not free text), tasks, gates, and deviation records.
- Agents read and write plans. Plans control agents, not the other way around.

### Capability layer

- Capabilities expose strict input/output schemas, preconditions, risk and data classification, and cost indicators.
- Execution is abstracted so capabilities can move to services or external protocols without changing agent logic.

### State and evidence

- Plans, evidence, deviations, and reviews are first-class state.
- Every generated output must link to evidence and a confidence level.

### HITL (exception only)

Triggered only by missing required data, conflicting sources, low confidence, risk thresholds, or explicit deviations.
Human input is slot-based and minimal.

### Agents (roles)

Planner, extractor, validator, drafter, QA/review.
These are implemented as graph states and transitions, not personas.

## Terminology guardrails

- In code and contracts, use `workflow_execution` or `workflow_step_execution`, not `case`.
- Do not map workflow execution terminology to `BusinessContext.case_id` (legal or business case).
- Plan scope includes: tenant_id, gremium_identifier, framework_profile_version (or framework_profile_id), optional business.case_id, business.workflow_id, and scope.run_id.
- Plan keys are deterministic (UUIDv5 or hash) over the plan scope tuple; derived, not minted; no new ID category is introduced.
- Use plan_key (derived, not minted) instead of plan_id in planning artifacts.

## Current code anchors (no behavior change)

- Graph runtime: `ai_core/graph/core.py`
- Graph meta normalization: `ai_core/graph/schemas.py`
- Tool context envelope: `ai_core/tool_contracts/base.py`
- Graph I/O contracts: `ai_core/graph/io.py`
- Existing plan output: `ai_core/graphs/technical/collection_search.py` (`CollectionSearchPlan`, `build_plan_node`)
- HITL gateway pattern: `ai_core/services/collection_search/hitl.py`

## Proposed vertical slice (confirmed)

Name: Plan-driven Collection Search (Acquire -> Review -> Ingest)

Purpose: Deliver one end-to-end workflow step with explicit plans, evidence, gates, and minimal HITL.

Plan scope (target state):
- tenant_id (ScopeContext)
- gremium_identifier (plan field)
- framework_profile_version or framework_profile_id (plan field)
- business.case_id (optional, for domain workflows)
- business.workflow_id
- scope.run_id (unique execution run)

Plan key (target state):
- UUIDv5 or hash derived from the scope tuple above (derived, not minted).

Blueprint (target state):
- Allowed slot types (e.g., source_url, document_id), constraints, and gate rules.
- Risk and data classification per task and capability.

Plan structure (target schema):
- schema_version: "v0" or semver string for forward-compatible schema changes
- slots: required inputs with type, status, and provenance
- tasks: ordered steps with preconditions and outputs
- gates: hitl or auto decisions with thresholds and required evidence
- deviations: overrides or missing data with rationale and approvals
- evidence: artifacts linked to tasks and outputs, each with confidence

Execution outline (graph mapping):
- build_plan -> validate_inputs -> acquire_sources -> rank_and_select -> gate_review
  -> execute_ingestion -> record_evidence -> finalize
- Reuse existing collection_search nodes and services; extend `build_plan_node` and HITL handling.

Evidence rules (target state):
- Each selected URL has evidence (source metadata, scoring rationale, confidence).
- Ingestion results link to document_id and chunk ids with confidence.
- Evidence references are structured: ref_type (url | repo_doc | repo_chunk | object_store | confluence | screenshot) plus ref_id.

Optional slot typing (target state):
- slot_type or json_schema_ref is allowed for routing/validation; optional in MVP.

Plan key canonicalization (target state):
- Scope tuple has stable ordering and no JSON-order dependence.
- gremium_identifier normalized before hashing.
- Use either framework_profile_id or framework_profile_version, never both.

## Roadmap (high level)

- Phase 1: Define Blueprint and ImplementationPlan schemas with slots/tasks/gates/deviations/evidence.
- Phase 2: Update collection_search to emit and consume the new plan schema, including HITL gates.
- Phase 3: Add capability metadata (risk, data class, cost) for nodes used by the vertical slice.
- Phase 4: Extend plan-driven execution to the next workflow (framework analysis or retrieval).

## Open decisions (confirm before implementation)

- Blueprint registry location: code-defined vs DB-managed.
- Confidence representation: numeric (0..1) vs discrete labels (low/med/high).

## Locked decisions

- No execution_case_id or other new ID categories.
- Plan is scoped to workflow execution (existing context fields) and keyed by a derived plan_key.
- plan_key is internal only, derived from the plan scope tuple, not a new domain field.
- Persist plans in graph state/object store for the vertical slice; no new DB tables in the slice.

## Registry guardrail

Naming and ID decisions must not bypass `ai_core/graph/registry.py`. Plan keys are derived inside the existing registry/state envelope flow and do not introduce new identifiers.

## Contract location (target state)

Plan-related contracts live under `ai_core/contracts/plans/` to keep plan, blueprint, and evidence contracts versionable without bloating the root contracts package.

- `ai_core/contracts/plans/__init__.py`
- `ai_core/contracts/plans/plan.py` (ImplementationPlan, Slot, Task, Gate, Deviation)
- `ai_core/contracts/plans/blueprint.py` (Blueprint, BlueprintGate, SlotSpec)
- `ai_core/contracts/plans/evidence.py` (Evidence, Confidence)
- `ai_core/contracts/plans/__init__.py:export_plan_schemas` (JSON schema export helper)
