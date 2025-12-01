# Agent Prompt: ID Consistency & Type Safety Review

**Role:** Senior Backend Architect / Code Reviewer
**Objective:** Audit the codebase for identifier inconsistency patterns, specifically focusing on the mismatch between UUID objects, string representations of UUIDs, and other string identifiers (like schema names or slugs).

## Context

We recently resolved a critical bug in the RAG ingestion pipeline where deterministic UUID generation failed because of inconsistent inputs. Specifically, one part of the system used the Tenant's **Schema Name** (string) to generate a collection UUID, while another part used the Tenant's **Primary Key** (UUID). This resulted in two different "deterministic" IDs for the same logical resource, causing lookups to fail.

## Review Instructions

Please analyze the codebase (specifically `ai_core`, `documents`, and `customers` modules) for the following anti-patterns:

### 1. Inconsistent Deterministic ID Generation

* **Pattern:** Usage of `uuid.uuid3` or `uuid.uuid5`.
* **Check:** Are the input seeds consistent across all call sites?
  * *Risk:* Using `tenant.schema_name` in one place and `tenant.id` (PK) in another.
  * *Risk:* Using `case_id` (string) vs `case_obj.id` (UUID).
  * *Risk:* Inconsistent casing (e.g., `slug.lower()` vs raw `slug`).

### 2. Tenant Identifier Ambiguity

* **Pattern:** Functions accepting `tenant_id` or `tenant` as arguments.
* **Check:** Does the code assume `tenant_id` is always a UUID or always a Schema Name?
  * *Risk:* Passing a schema name (e.g., "autotest") to a function expecting a UUID string, or vice-versa.
  * *Fix:* Look for explicit resolution steps (e.g., `TenantContext.resolve_identifier`) at the entry points of service layers.

### 3. Type Hint Lying

* **Pattern:** Type hints saying `UUID` but runtime code handling strings, or `str | UUID` unions that are not normalized early.
* **Check:** Are identifiers normalized to a single type (preferably `str` for external APIs/JSON, `UUID` for internal DB lookups) immediately upon entry?
  * *Risk:* Comparisons like `if id_a == id_b` failing because one is a `UUID` object and the other is a `str`.

### 4. Test Data Divergence

* **Pattern:** Test fixtures generating IDs differently than production code.
* **Check:** Do tests manually construct IDs (e.g., `str(uuid4())`) where the production code uses a deterministic method (e.g., `manual_collection_uuid`)?

## Target Files

Prioritize reviewing these files/directories:
* `ai_core/services/__init__.py` (Service Layer orchestration)
* `documents/domain_service.py` (Core logic)
* `documents/collection_service.py` (ID generation logic)
* `ai_core/rag/collections.py` (Legacy helpers)
* `customers/tenant_context.py` (Resolution logic)

## Output Format

Report findings in the following format:

| Severity | File | Line | Issue Description | Suggested Fix |
| :--- | :--- | :--- | :--- | :--- |
| **High** | `ai_core/services/example.py` | 123 | `generate_id` uses `tenant_schema` but `get_id` uses `tenant_pk`. | Standardize on `tenant_pk` for stability. |
| **Medium** | `documents/models.py` | 45 | `collection_id` comparison without type coercion. | Ensure both sides are `str` or `UUID` before comparing. |

## Goal

Eliminate "ID drift" bugs where resources become inaccessible because the system looks for them under the wrong ID.
