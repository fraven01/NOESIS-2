# L-Track Deep Review: Vollständige Typsicherheit

**Datum:** 2026-01-23
**Scope:** Datenfluss-Architektur & Agentic-Optimierung
**Status:** Analyse abgeschlossen

---

## Executive Summary

| Bereich | Score | Status |
|---------|-------|--------|
| Graph State-Typisierung | 6.5/10 | ⚠️ Schwach intern |
| Celery Task-Context | 4/10 | ❌ Kritisch |
| Domain Models | 8/10 | ✅ Gut strukturiert |
| AX-Score (Agenten-Lesbarkeit) | 62% | ⚠️ Lückenhaft |

**Kernproblem:** Grenzen (API/Tool I/O) sind stark typisiert, interne State-Übergaben nutzen `dict[str, Any]`.

---

## 1. Data Lineage Map: Wo Daten "im Dunkeln" fließen

### 1.1 Kritische Schattenzonen

```
┌─────────────────────────────────────────────────────────────────┐
│ TYPISIERT (Licht)          │ UNTYPISIERT (Dunkel)              │
├─────────────────────────────┼──────────────────────────────────┤
│ API Input → Pydantic        │ Graph State["context"]: dict     │
│ Tool Input → RetrieveInput  │ Node Returns: dict[str, Any]     │
│ Tool Output → RetrieveOutput│ Celery args[0]: Any              │
│ ScopeContext → Pydantic     │ meta.get("key_alias"): Any       │
│ BusinessContext → Pydantic  │ working_state["question"]: Any   │
└─────────────────────────────┴──────────────────────────────────┘
```

### 1.2 Quantitative Analyse

| Kategorie | Typisiert | Untypisiert | Quote |
|-----------|-----------|-------------|-------|
| Graph State-Keys | 22 | 25 | 47% |
| Node-Rückgaben | 5 | 13 | 28% |
| Celery kwargs-Felder | 0 | 17 | 0% |
| Pydantic-Modelle Fields | 142 | 58 | 71% |

### 1.3 Hotspot-Dateien (Untypisierte dict-Zugriffe)

| Datei | dict-Ops | Risiko |
|-------|----------|--------|
| [common/celery.py](../common/celery.py#L267-559) | 18 | KRITISCH |
| [ai_core/graphs/technical/retrieval_augmented_generation.py](../ai_core/graphs/technical/retrieval_augmented_generation.py#L883-911) | 12 | KRITISCH |
| [ai_core/graphs/technical/universal_ingestion_graph.py](../ai_core/graphs/technical/universal_ingestion_graph.py#L199-474) | 8 | HOCH |
| [ai_core/graphs/technical/collection_search.py](../ai_core/graphs/technical/collection_search.py#L655-1094) | 6 | MITTEL |

---

## 2. Refactoring-Blaupause: Hierarchische Pydantic-Struktur

### 2.1 Ziel-Architektur

```
BaseContext (ABC)
├── ScopeContext ✅ (implementiert)
│   ├── tenant_id: str [PFLICHT]
│   ├── trace_id: str [PFLICHT]
│   ├── invocation_id: str [PFLICHT]
│   └── run_id | ingestion_run_id [mind. 1]
│
├── BusinessContext ✅ (implementiert)
│   └── case_id, collection_id, workflow_id, document_id [alle optional]
│
└── TaskContext 🆕 (zu erstellen)
    ├── scope: ScopeContext
    ├── business: BusinessContext
    └── metadata: TaskContextMetadata
        ├── session_salt: str | None
        ├── priority: Literal["high", "low", "background"] | None
        └── retry_count: int | None

BaseParams (ABC)
├── SearchParams 🆕
│   ├── query: str
│   ├── top_k: int
│   └── filters: FilterSpec | None
│
├── HybridSearchParams 🆕 (exists as dataclass → Pydantic)
│   └── alpha, min_sim, vec_limit, lex_limit, trgm_limit, max_candidates
│
└── FilterSpec 🆕
    ├── tenant_id: str [PFLICHT]
    ├── case_id: str | None
    └── metadata: dict[str, Any] [Extension Point]

NodeReturns (ABC) 🆕
├── ValidateInputNodeOutput
├── DeduplicationNodeOutput
├── PersistNodeOutput
└── ProcessNodeOutput
```

### 2.2 Neue Pydantic-Modelle (zu erstellen)

#### TaskContext (für common/celery.py)

```python
class TaskScopeContext(BaseModel):
    """Infrastructure IDs aus ScopeContext."""
    tenant_id: str = Field(description="Mandant-UUID")
    trace_id: str = Field(description="Distributed Tracing ID")
    invocation_id: str = Field(description="Einzelner API-Aufruf")
    run_id: str | None = Field(None, description="LangGraph Execution ID")
    ingestion_run_id: str | None = Field(None, description="Document Ingestion ID")

    @model_validator(mode="after")
    def require_runtime_id(self):
        if not self.run_id and not self.ingestion_run_id:
            raise ValueError("At least one runtime ID required")
        return self

class TaskContextMetadata(BaseModel):
    """Runtime-Metadaten für Celery Tasks."""
    key_alias: str | None = None
    session_salt: str | None = None
    priority: Literal["high", "low", "background"] | None = None
    task_id: str | None = None
    queue: str | None = None
    retry_count: int | None = None

class TaskContext(BaseModel):
    """Vollständiger Task-Kontext für Celery."""
    scope: TaskScopeContext
    business: BusinessContext
    metadata: TaskContextMetadata = Field(default_factory=TaskContextMetadata)

    model_config = ConfigDict(frozen=True)
```

#### Graph Node Returns (für ai_core/graphs/)

```python
class ValidateInputNodeOutput(TypedDict):
    """Typisierte Rückgabe von validate_input_node."""
    error: str | None
    tool_context: ToolContext | None
    normalized_document: NormalizedDocument | None

class DeduplicationNodeOutput(TypedDict):
    """Typisierte Rückgabe von dedup_node."""
    dedup_status: Literal["new", "duplicate"]
    existing_document_ref: DocumentRef | None

class PersistNodeOutput(TypedDict):
    """Typisierte Rückgabe von persist_node."""
    ingestion_result: IngestionResult
    normalized_document: NormalizedDocument
```

---

## 3. Breaking-Change-Analyse

### 3.1 Tests die bei strict=True sofort fehlschlagen

| Test-Datei | Grund | Impact |
|------------|-------|--------|
| `test_tool_context.py` | Legacy args[0] meta passing | 5 Tests |
| `test_meta_normalization.py` | dict ohne scope_context | 8 Tests |
| `test_ingestion_orchestration.py` | Untypisierte kwargs | 3 Tests |
| `test_graph_tasks.py` | session_scope Tuple ohne Validierung | 2 Tests |
| `test_request_context_middleware.py` | key_alias raw dict access | 2 Tests |

**Gesamt: ~20 Tests müssen migriert werden**

### 3.2 Breaking Points in common/celery.py

| Zeile | Code | Problem bei Strict |
|-------|------|-------------------|
| 285-287 | `args[0]` als meta | TypeError: Expected Mapping |
| 307 | `_from_meta(meta: Any)` | Muss Mapping sein |
| 361 | `meta.get("key_alias")` | Umgeht Pydantic |
| 478 | `pop("session_scope")` | Keine Tuple-Element-Prüfung |
| 517 | `"||".join([None, ...])` | TypeError bei None |

### 3.3 Migrations-Phasen

| Phase | Änderung | Tests betroffen |
|-------|----------|-----------------|
| **1** | Remove args[0]/args[1] fallback | ~5 |
| **2** | Enforce tool_context_from_meta() | ~15 |
| **3** | Validate session_scope tuple | ~3 |
| **4** | Remove key_alias raw dict | ~2 |
| **5** | Enforce BusinessContext separation | ~8 |

---

## 4. AX-Score: Agenten-Freundlichkeit

### 4.1 Scoring-Kriterien

- **0-25%**: Keine Beschreibungen, nur Typ-Hints
- **26-50%**: Teilweise Beschreibungen
- **51-75%**: Gute Beschreibungen, fehlende Literal Types
- **76-100%**: Vollständig dokumentiert, Literal Types, JSON Schema Export

### 4.2 Bewertung kritischer Modelle

| Modell | Felder | Mit Description | Literal | AX-Score |
|--------|--------|-----------------|---------|----------|
| ScopeContext | 10 | 10 | ✅ | **100%** A+ |
| BusinessContext | 6 | 6 | ❌ | **95%** A |
| ToolContext | 9 | 6 | ✅ | **85%** A |
| ChunkMeta | 21 | 0 | ❌ | **5%** F |
| RetrieveMeta | 13 | 0 | ❌ | **10%** D |
| ComposeOutput | 9 | 0 | ❌ | **10%** D |
| RetrieveInput | 7 | 0 | ❌ | **20%** D |

### 4.3 Kritische Lücken

**F-Grade (sofort beheben):**
- `ChunkMeta` (21 Felder ohne Beschreibung) → Agents raten bei embedding_profile, chunker_mode
- `CrawlerIngestionPayload` (13 Felder) → Kernvertrag für Ingestion

**D-Grade (kurzfristig):**
- `RetrieveMeta` → alpha, min_sim, top_k_effective undokumentiert
- `ComposeOutput` → reasoning, used_sources, suggested_followups unklar
- `RetrieveInput` → query, filters, process, visibility ohne Kontext

### 4.4 Muster für Exzellenz (zu kopieren)

```python
# ScopeContext-Pattern (AX-Score 100%)
class ScopeContext(BaseModel):
    user_id: UserId = Field(
        default=None,
        description="User identity for User Request Hops. Must be absent for S2S."
    )

    @model_validator(mode="after")
    def validate_identity(self) -> "ScopeContext":
        """Ensure user_id and service_id are mutually exclusive."""
```

---

## 5. Empfehlungen & Roadmap

### 5.1 Sofortmaßnahmen (Woche 1)

1. **ChunkMeta dokumentieren** (21 Felder) - höchstes AX-Impact
2. **RetrieveMeta dokumentieren** (13 Felder) - Hybrid Search verständlich machen
3. **TaskContext Pydantic erstellen** - common/celery.py typisieren

### 5.2 Kurzfristig (Woche 2-3)

4. **Node-Return TypedDicts** für Universal Ingestion Graph
5. **ComposeOutput dokumentieren** - LLM-Generierung transparent
6. **args[0] Fallback entfernen** in common/celery.py

### 5.3 Mittelfristig (Woche 4-6)

7. **HybridSearchParams → Pydantic** (von dataclass)
8. **FilterSpec einführen** für typisierte Filter
9. **Graph State context: ToolContext** statt dict[str, Any]

### 5.4 Langfristig (Monat 2+)

10. **Pre-commit Hook** für undokumentierte Pydantic-Felder
11. **Pydantic Style Guide** in docs/
12. **100% AX-Score** für alle öffentlichen Modelle

---

## 6. Aufwandsschätzung

| Track | Aufwand | Story Points | Scope |
|-------|---------|--------------|-------|
| **S-Track** | 2-3 PT | 5 SP | ChunkMeta + RetrieveMeta dokumentieren |
| **M-Track** | 8-10 PT | 21 SP | + TaskContext + Node Returns |
| **L-Track** | 20-25 PT | 55 SP | Vollständige Typsicherheit |

---

## 7. Dateien für Refactoring

### Priorität KRITISCH

| Datei | Änderung |
|-------|----------|
| [common/celery.py:267-559](../common/celery.py#L267-559) | TaskContext einführen |
| [ai_core/rag/ingestion_contracts.py:69](../ai_core/rag/ingestion_contracts.py#L69) | ChunkMeta dokumentieren |
| [ai_core/nodes/retrieve.py:100](../ai_core/nodes/retrieve.py#L100) | RetrieveMeta dokumentieren |

### Priorität HOCH

| Datei | Änderung |
|-------|----------|
| [ai_core/graphs/technical/universal_ingestion_graph.py](../ai_core/graphs/technical/universal_ingestion_graph.py) | Node-Return TypedDicts |
| [ai_core/graphs/technical/retrieval_augmented_generation.py:883](../ai_core/graphs/technical/retrieval_augmented_generation.py#L883) | working_state typisieren |
| [ai_core/nodes/compose.py](../ai_core/nodes/compose.py) | ComposeOutput dokumentieren |

### Priorität MITTEL

| Datei | Änderung |
|-------|----------|
| [ai_core/graphs/technical/collection_search.py:373](../ai_core/graphs/technical/collection_search.py#L373) | search State typisieren |
| [ai_core/rag/query_planner.py](../ai_core/rag/query_planner.py) | QueryPlan dokumentieren |

---

## 8. Metriken-Dashboard (Ziel-Werte)

| Metrik | Aktuell | Ziel M-Track | Ziel L-Track |
|--------|---------|--------------|--------------|
| Graph State-Typisierung | 47% | 75% | 95% |
| Node-Return Typisierung | 28% | 80% | 100% |
| Celery kwargs typisiert | 0% | 90% | 100% |
| AX-Score Durchschnitt | 62% | 80% | 95% |
| Tests ohne dict[str, Any] | ~20 | 0 | 0 |

