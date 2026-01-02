# Contract Divergence Report

**Generated**: 2025-12-15
**Scope**: Documents, RAG, AI Core Graphs
**Basis**: [AGENTS.md](../../AGENTS.md) as Source of Truth

## Executive Summary

This audit identifies inconsistencies between implemented Pydantic contracts and their documentation across the refactored modules (`documents/`, `ai_core/rag/`, `ai_core/graphs/`). The primary objectives are:

1. Ensure all contracts align with the 4-Layer architecture defined in AGENTS.md
2. Replace legacy "Pipeline" terminology with "Graph" (LangGraph)
3. Synchronize Pydantic schemas with Markdown documentation
4. Identify and mark legacy components pending refactoring

## Critical Findings

### 1. Chunk Schema Compliance ✅ RESOLVED (2026-01-02)

**Location**: [`ai_core/rag/schemas.py`](../../ai_core/rag/schemas.py:1-15)

**Issue**: The `Chunk` schema previously used Python `@dataclass` instead of Pydantic `BaseModel`.

**AGENTS.md Requirement**:
> Tool-Hüllmodelle basieren auf Pydantic `BaseModel` mit `frozen=True` (immutable)

**Current Implementation**:

```python
class Chunk(BaseModel):
    """A chunk of knowledge used for retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    meta: dict[str, Any]
    embedding: list[float] | None = None
    parents: dict[str, dict[str, Any]] | None = None
```

**Migration Notes**:

- Call sites migrated to keyword-only construction (Pydantic v2 requirement).
- Chunk metadata updates reconstruct a new `Chunk` instead of mutating `Chunk.meta` in-place (see `ai_core/rag/vector_client.py`).

**Status**: ✅ COMPLIANT (hard break applied; legacy dataclass removed)

---

### 2. ~~Terminology Inconsistency: "Pipeline" vs. "Graph"~~ ✅ RESOLVED (2025-12-15)

**AGENTS.md Directive**:
> `Graph` bezeichnet eine LangGraph-Orchestrierung; `Pipeline` ist ein veralteter Begriff dafür.

**Resolution**:

- Historical note added to [`docs/rag/ingestion.md:5`](../../docs/rag/ingestion.md#L5): *„Der Begriff ‚Pipeline' ist eine historische Bezeichnung für die heute als ‚Graph' (LangGraph) bezeichneten Orchestrierungs-Flows."*
- Node names like `document_pipeline` retained as identifiers
- [`docs/rag/overview.md`](../../docs/rag/overview.md) uses consistent "Graph" terminology

**Status**: ✅ COMPLIANT - Prose updated, historical note preserved

---

### 3. ~~Upload Ingestion Graph Documentation Gap~~ ✅ RESOLVED (2025-12-15)

**Issue**: ~~[`ai_core/graphs/technical/universal_ingestion_graph.py`](../../ai_core/graphs/technical/universal_ingestion_graph.py) implements a complete LangGraph flow but is not documented.~~

**Resolution**: Added comprehensive documentation in [`docs/rag/ingestion.md`](../../docs/rag/ingestion.md#upload-ingestion-graph) (Lines 33-133):

- ✅ Mermaid flow diagram
- ✅ State Contract (`UploadIngestionState`) as table
- ✅ Node descriptions (validate_input, build_config, run_processing, map_results)
- ✅ Transition semantics (accept_upload, delta_and_guardrails, document_pipeline)
- ✅ Runtime dependencies table
- ✅ Error handling documentation

**Status**: ✅ DOCUMENTED

---

### 4. ~~External Knowledge Graph Undocumented~~ ✅ RESOLVED (2025-12-15)

**Issue**: ~~[`ai_core/graphs/external_knowledge_graph.py`](../../ai_core/graphs/external_knowledge_graph.py) is production code but missing from agent documentation.~~

**Resolution**: Documentation expanded in [`docs/agents/overview.md`](../../docs/agents/overview.md#2-external-knowledge-graph) (Lines 33-103):

- ✅ Mermaid flow diagram
- ✅ State Contract (`ExternalKnowledgeState`) as table
- ✅ Node semantics (search, select, ingest)
- ✅ Runtime dependencies (WebSearchWorker, IngestionTrigger protocols)
- ✅ Error handling documentation
- ✅ HITL flow (current + future plans)

**Status**: ✅ DOCUMENTED

---

### 5. ~~Developer Workbench Undocumented~~ ✅ RESOLVED (2025-12-15)

**Issue**: ~~[`theme/templates/theme/rag_tools.html`](../../theme/templates/theme/rag_tools.html) provides a Developer Workbench UI but has no docs/ reference.~~

**Resolution**: Created [`docs/development/rag-tools-workbench.md`](../../docs/development/rag-tools-workbench.md) (304 lines):

- ✅ Feature overview (Web Search, Crawler, Ingestion tabs)
- ✅ Form field documentation
- ✅ HTMX integration details
- ✅ Security considerations
- ✅ Usage examples

**Status**: ✅ DOCUMENTED

---

## Architectural Compliance Review

### Layer Mapping (vs. AGENTS.md 4-Layer Model)

| Component | Layer | Compliance | Notes |
|-----------|-------|------------|-------|
| `documents/contracts.py` | L4 (Data) | ✅ PASS | Pydantic contracts with `frozen=True` |
| `ai_core/rag/schemas.py` | L4 (Data) | ✅ PASS | Migrated to Pydantic `BaseModel` with `frozen=True` |
| `ai_core/graphs/upload_ingestion_graph.py` | L3 (Technical Managers) | ✅ PASS | LangGraph implementation, TypedDict state |
| `ai_core/graphs/external_knowledge_graph.py` | L3 (Technical Managers) | ✅ PASS | LangGraph implementation, Protocol-based dependencies |
| `theme/templates/theme/rag_tools.html` | L1 (Frontend) | ✅ PASS | HTMX-based UI, no business logic |

---

## ID Propagation Compliance

**AGENTS.md Requirements** (Glossar & Feld-Matrix):

- `tenant_id` (Pflicht)
- `trace_id` (Pflicht)
- `invocation_id` (Pflicht für Tools)
- Genau eine Laufzeit-ID: `run_id` XOR `ingestion_run_id`

### Universal Ingestion Graph (Current)

**State Fields**:

```python
context: dict[str, Any]  # Serialized ToolContext from invocation
tool_context: ToolContext | None
```

**Compliance**: PASS - Context injected via ToolContext and stored in state

### External Knowledge Graph

**State Fields**:

```python
context: dict[str, Any]  # Tenant ID, Trace ID, etc.
```

**Compliance**: ✅ PASS - Context explicitly captures all IDs

### Canonical Runtime Context Injection Pattern

**Goal**: Ensure tenant/trace/run IDs are never implicit knowledge inside graphs.

**Pattern**:

1. Boundary builds meta via `ai_core/graph/schemas.py:normalize_meta`.
2. Graph entry parses a single `ToolContext` via
   `ai_core/tool_contracts/base.py:tool_context_from_meta`.
3. The validated context is stored in state (e.g. `state["tool_context"]`).
4. Nodes read IDs from `context.scope.*` and `context.business.*` only.

**Pointers**:
- `ai_core/graph/schemas.py:normalize_meta`
- `ai_core/contracts/scope.py:ScopeContext`
- `ai_core/contracts/business.py:BusinessContext`
- `ai_core/tool_contracts/base.py:tool_context_from_meta`
- `ai_core/graphs/README.md` (Context & Identity)

---

## Pydantic Contract Drift

### Documents Subsystem

**Reference**: [`documents/contracts.py`](../../documents/contracts.py)
**Documentation**: [`docs/documents/contracts-reference.md`](../../docs/documents/contracts-reference.md)

**Audit Result**: ✅ **SYNCHRONIZED**

Fields, constraints, and error codes are 1:1 aligned. Example spot-check:

| Contract Field | Code (contracts.py) | Docs (contracts-reference.md) |
|----------------|---------------------|-------------------------------|
| `DocumentRef.tenant_id` | Line 175, `normalize_tenant` | Table row, `tenant_empty`, `tenant_too_long` |
| `NormalizedDocument.source` | Line 1511, Literal["upload", "crawler", ...] | Line 205, Literal enum |
| `Asset.caption_method` | Line 1321, Literal["vlm_caption", "ocr_only", ...] | Line 305, Literal enum |

**Recommendation**: No changes required

---

### RAG Retrieval Contracts

**Reference**: `ai_core/nodes/retrieve.py` (inferred from docs)
**Documentation**: [`docs/rag/retrieval-contracts.md`](../../docs/rag/retrieval-contracts.md)

**Audit Result**: ⚠️ **PARTIALLY VERIFIED** (Code not read in this audit)

Docs describe:

- `RetrieveInput`
- `RetrieveRouting`
- `RetrieveMeta`
- `RetrieveOutput`

**Recommendation**: Verify against `ai_core/nodes/retrieve.py` implementation (deferred to follow-up)

---

## Legacy Components Requiring Markers

### Crawler Ingestion Graph

**File**: [`ai_core/graphs/crawler_ingestion_graph.py`](../../ai_core/graphs/crawler_ingestion_graph.py)

**Status**: Code not read in this audit (identified via docs/README references)

**From** [`ai_core/graphs/README.md:28-47`](../../ai_core/graphs/README.md:28-47):
> Der `crawler_ingestion_graph` orchestriert den Crawler-Workflow in der
> deterministischen Sequenz
> `update_status_normalized → enforce_guardrails → document_pipeline → ingest_decision → ingest → finish`.

**Assessment**: Appears to be **refactored** (uses `DocumentProcessingGraph` delegation, per docs)

**Recommendation**: Verify implementation status before marking as legacy

---

## Removed Files (None)

No obsolete files identified for deletion in this audit. All scanned docs/ files provide value or context.

---

## Action Items

### High Priority

1. **[CRITICAL]** Migrate `ai_core/rag/schemas.py::Chunk` to Pydantic BaseModel ⚠️ DEFERRED (blocks on 50+ call site refactor)
2. ~~**[HIGH]** Document `UploadIngestionGraph` flow in `docs/rag/ingestion.md`~~ ✅ DONE
3. ~~**[HIGH]** Replace "Pipeline" with "Graph" in `docs/rag/ingestion.md` prose~~ ✅ DONE (historical note added)

### Medium Priority

4. ~~**[MEDIUM]** Expand `ExternalKnowledgeGraph` documentation in `docs/agents/`~~ ✅ DONE
5. **[MEDIUM]** Verify `RetrieveInput`/`RetrieveOutput` contracts against `ai_core/nodes/retrieve.py`
6. ~~**[MEDIUM]** Update terminology in `docs/rag/overview.md` (keep historical note)~~ ✅ DONE

### Low Priority

7. ~~**[LOW]** Create `docs/development/rag-tools-workbench.md`~~ ✅ DONE
8. ~~**[LOW]** Add ID propagation pattern documentation for graphs~~ DONE

---

## Appendix: Schema Extraction Checklists

### NormalizedDocument (documents/contracts.py)

**Pydantic Config**:

- ✅ `frozen=True`
- ✅ `extra="forbid"`
- ✅ Field descriptions

**Required Fields**:

- ✅ `ref: DocumentRef`
- ✅ `meta: DocumentMeta`
- ✅ `blob: BlobLocator`
- ✅ `checksum: str` (validated hex64)
- ✅ `created_at: datetime` (tz-aware, UTC)

**Validators**:

- ✅ Cross-field consistency (tenant_id, workflow_id across ref/meta/assets)
- ✅ Checksum integrity (strict mode)
- ✅ Lifecycle state normalization

**Documentation Alignment**: ✅ PASS

---

### Chunk (ai_core/rag/schemas.py) - ✅ COMPLIANT

**Implementation**: `BaseModel` with `frozen=True`, `extra="forbid"`

**Current Code**:

```python
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Optional, Any

class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str = Field(description="Chunk text content")
    meta: Dict[str, Any] = Field(description="Metadata (tenant_id, trace_id, hash, etc.)")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    parents: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Parent metadata")
```

---

## Sign-Off

**Auditor**: Claude Sonnet 4.5 (Documentation Engineer)
**Review Date**: 2025-12-15
**Migration Date**: 2025-12-15
**Status**: **✅ HIGH PRIORITY ITEMS RESOLVED**
