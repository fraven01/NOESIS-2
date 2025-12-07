# collection_id Dual-Identity Review

Date: 2025-12-07  
Author: Codex  
Scope: DocumentCollection.id vs DocumentCollection.collection_id

---

## Executive Summary
- Dual UUIDs are deliberate: `id` is the Django PK; `collection_id` is the logical/vector identifier used by repositories, contracts, and vector store calls.
- Defaults diverge: `ensure_collection` generates two independent `uuid4` values, so `id != collection_id` for most new rows unless explicitly aligned.
- Ingestion dispatcher currently passes the PK, while downstream code expects the logical ID; this breaks ingestion when the IDs differ (the default).
- The modeled DB uniqueness `(tenant, collection_id)` is missing in migrations, so the database cannot prevent logical ID collisions.
- APIs and views accept multiple identifiers (PK, logical ID, key), keeping backward compatibility but increasing caller ambiguity.

---

## 1. Semantik-Klaerung
- `id`: Django primary key, internal to ORM.
- `collection_id`: logical/vector identifier for pgvector and contracts.
- Generation paths:
  - `DocumentDomainService.ensure_collection` (documents/domain_service.py:334-434) uses `collection_id or uuid4()` for the logical ID while the PK is generated separately → divergence by default.
  - `CollectionService.ensure_manual_collection` (documents/collection_service.py:55-79) sets a deterministic uuid5 for `collection_id`; PK remains auto `uuid4`.
  - `ai_core/rag/collections.ensure_manual_collection_model` (ai_core/rag/collections.py:100-112) aligns both fields (`id = collection_id`) for manual collections only.
- Divergence examples:
  - Any call to `ensure_collection` without an explicit `collection_id` yields two unrelated `uuid4` values.
  - Manual collections via `CollectionService` return `id != collection_id` (random `uuid4` vs deterministic `uuid5`).
  - Legacy rows created before constraint updates may already differ.
- Cross-tenant isolation: all observed lookups scope by tenant (or tenant schema). Lack of a DB constraint still allows duplicate logical IDs per tenant.

---

## 2. Usage Matrix
| File:Line | Verwendung | Nutzt | Tenant-Filter? | Risiko |
|-----------|-----------|-------|----------------|--------|
| documents/models.py:31-54 | Model fields + unique constraints | both; `unique_collection_id_per_tenant` modeled | N/A | High (constraint not migrated) |
| documents/migrations/0004_add_document_collection.py:34-53 | Creates model without `collection_id` uniqueness | `collection_id` (no uniqueness) | N/A | High |
| documents/domain_service.py:204-220 | Ingestion dispatcher payload uses PK list | `id` | Implicit via collection FK only | High (downstream expects logical ID) |
| documents/domain_service.py:334-434 | ensure_collection creates/overwrites logical ID | `collection_id` | Yes | Medium (default divergence; override can orphan vector data) |
| documents/domain_service.py:451-458 | _resolve_collection_reference matches by PK then logical ID | both | Yes | Medium (accepts both IDs) |
| documents/domain_service.py:630-716 | delete_collection + vector sync payloads | `collection_id` | Yes | Low |
| documents/services/document_space_service.py:86-114 | Document listing selects by logical ID | `collection_id` | Yes (schema context) | Low |
| documents/services/document_space_service.py:244-274 | Serialization uses PK for selector, logical ID for display | both | N/A | Medium (caller sees two IDs) |
| documents/services.py:27-62 | resolve_collection_for_document_input accepts PK or logical ID | both | Yes | Medium (dual fields) |
| ai_core/adapters/db_documents_repository.py:80-103 | Upsert validates membership using logical ID | `collection_id` | Yes | Low |
| ai_core/adapters/db_documents_repository.py:506-537 | _collection_queryset filters memberships | `collection__collection_id` | Yes | Low |
| ai_core/services/__init__.py:425-460 | _ensure_collection_with_warning resolves/creates via logical ID | `collection_id` | Yes | Low |
| ai_core/services/__init__.py:1354-1425 | _ensure_document_collection_record syncs manual uploads | `collection_id` (PK only for updates) | Yes | Low |
| ai_core/rag/collections.py:100-112 | ensure_manual_collection_model sets PK = logical ID | both (aligned) | Yes | Low |
| ai_core/graphs/framework_analysis_graph.py:633-640 | Looks up collection by PK from `document_collection_id` state | `id` | Yes | Medium (state name suggests logical ID) |
| theme/views.py:516-583 | Document-space view accepts `collection` param as PK/logical/key | both | Yes | Medium (ambiguous input) |
| documents/contracts.py:178-184,1044-1046 | Contracts expose both fields with equality validation | both | N/A | Medium (type confusion) |

---

## 3. Findings
- Ingestion uses the wrong identifier: `ingest_document` queues `collection.id` (documents/domain_service.py:204-220) while repositories and vector sync expect `collection.collection_id`. Because defaults diverge, ingestion can target non-existent logical collections.
- DB uniqueness gap: `unique_collection_id_per_tenant` is absent from migrations (documents/migrations/0004_add_document_collection.py), so logical ID collisions are not prevented at the database layer.
- Identifier ambiguity across layers: UI (theme/views.py) and resolvers (documents/services.py, documents/domain_service.py) accept PK, logical ID, or key; contracts also expose both. Callers can easily mix IDs in downstream payloads.
- Manual collections diverge unless created via ai_core/rag: `CollectionService.ensure_manual_collection` returns `collection_id` while leaving the PK random; only `ai_core/rag/collections.ensure_manual_collection_model` aligns them.
- Framework analysis graph mismatch: state field `document_collection_id` is resolved against the PK (ai_core/graphs/framework_analysis_graph.py:633-640). If callers pass the logical `collection_id`, lookups fail.

---

## 4. Test-Gaps
- Missing coverage for `id != collection_id` through the ingestion dispatcher path.
- No test asserting DB uniqueness of `(tenant, collection_id)` (migration gap).
- No negative test for cross-tenant isolation with colliding `collection_id` values.
- No coverage for endpoints that accept both IDs (e.g., document-space `?collection=` with logical ID vs PK).
- No test ensuring framework analysis graph resolves the intended identifier type.

---

## 5. Empfehlungen
- Short term:
  1) Switch ingestion dispatch to `collection.collection_id` and update consumers.
  2) Add a migration enforcing `unique_collection_id_per_tenant` and backfill duplicates.
  3) Add regression tests for `id != collection_id` flows (ingestion, document_space, framework_analysis_graph).
- Long term:
  1) Choose a single external identifier (`collection_id`) and treat `id` as internal; deprecate public PK exposure and the duplicate `document_collection_id` contract field.
  2) Consider aligning PK and logical ID at creation time (`id = collection_id`) or drop PK exposure entirely to avoid dual identity drift.
  3) Simplify client inputs so public surfaces accept only `collection_id` (logical) plus `key` as a friendly alias.
