# Refactor: Modernize Logging Pattern (structlog contextvars)

**Status**: Backlog
**Priority**: Medium (Quality-of-Life)
**Effort**: 5 Wochen (phasiert, ~40 Vorkommen über 14 Dateien)
**Labels**: refactoring, technical-debt, observability, contract-compliance

---

## Hintergrund: Logging-Infrastruktur in NOESIS-2

NOESIS-2 nutzt **structlog** als zentrale Logging-Basis (konfiguriert in `config/logging.py`, Abhängigkeit in `requirements.txt`). Standard-Python-Logging (`logging.getLogger()`) wird in einzelnen Modulen verwendet und ist mit structlog integriert.

**Beispiele**:
- Structlog: `views.py`, `authentication.py` (nutzen `structlog.stdlib.get_logger`)
- Standard-Logging: `late_chunker.py`, `hybrid_chunker.py`, `llm_judge.py` (nutzen `logging.getLogger(__name__)`)

Beide Patterns unterstützen `extra={}` für strukturiertes Logging, aber structlog bietet zusätzlich `contextvars`-basierte automatische Kontext-Propagation.

---

## Problem

In `ai_core/tasks.py` (und möglicherweise anderen Modulen) wird ein **manueller Pattern** verwendet, bei dem ein `extra`-Dictionary für jeden Log-Call gebaut wird:

```python
# ai_core/tasks.py:260-284 (aktuell)
extra = {
    "tenant_id": tenant_id,
    "document_id": str(document_id),
    "trace_id": trace_id,
    "span_id": span_id,
    "workflow_id": workflow_id,
    "ingestion_run_id": ingestion_run_id,
    "collection_id": str(collection_id) if collection_id else None,
    "vector_space_id": vector_space_id,
    "case_status": case_status,
    "case_phase": case_phase,
    "collection_scope": collection_scope,
    "document_collection_key": document_collection_key,
    # ... weitere Felder
}
logger.info("ingestion.start", extra=extra)
```

**Probleme:**

1. **~200 LOC Boilerplate**: Jeder Log-Call muss `extra=extra` übergeben
2. **Fehleranfällig**: Leicht vergisst man Pflichtfelder oder nutzt inkonsistente Naming
3. **Nicht DRY**: Scope/Business-Context-Felder werden überall wiederholt
4. **Schwer wartbar**: Neue Pflichtfelder (gemäß [AGENTS.md](../AGENTS.md#glossar--feld-matrix)) müssen in jedem Call manuell ergänzt werden
5. **Contract-Verletzungen**: Inkonsistente Verwendung von Scope-Context-Feldern (`tenant_id`, `trace_id`, `run_id`/`ingestion_run_id`)

---

## Lösung

**Nutze `structlog.contextvars`** für automatisches Context-Binding. Dies ist der **moderne Standard** für strukturiertes Logging in Python 3.7+.

### Pattern: Context Binding am Task-Start

```python
# ai_core/tasks.py (refactored)
from structlog import contextvars

def run_ingestion_graph(
    tenant_id: str,
    document_id: UUID,
    ingestion_run_id: UUID,
    trace_id: str,
    # ... weitere Parameter
):
    # 1. Binde Scope + Business Context EINMAL am Task-Start
    contextvars.bind_contextvars(
        # Scope Context (Layer 1 Norm - AGENTS.md)
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=str(ingestion_run_id),  # Standardisiert
        ingestion_run_id=str(ingestion_run_id),

        # Business Context (Layer 2 Norm - AGENTS.md)
        workflow_id=workflow_id,
        document_id=str(document_id),
        collection_id=str(collection_id) if collection_id else None,

        # Technische Metadaten
        vector_space_id=vector_space_id,
    )

    # 2. Logge OHNE extra - Kontext ist automatisch dabei
    logger.info("ingestion.start")
    logger.info("ingestion.parse", parser="unstructured")
    logger.error("ingestion.failed", error=str(exc))

    # 3. Unbinde am Task-Ende (automatisch bei Context-Exit)
    try:
        # ... Task-Logik
        pass
    finally:
        contextvars.unbind_contextvars(
            "tenant_id", "trace_id", "ingestion_run_id", "document_id",
            "workflow_id", "collection_id", "vector_space_id"
        )
```

### Pattern: Lokales Binding mit `logger.bind()`

Für **Graph-Nodes** oder **Tool-Calls** mit lokalem Scope:

```python
# ai_core/graphs/technical/universal_ingestion_graph.py (refactored)
def normalize_node(state: GraphState, config: RunnableConfig) -> GraphTransition:
    # Lokaler Logger mit zusätzlichem Kontext
    log = logger.bind(
        node="normalize",
        graph="universal_ingestion",
    )

    log.info("node.start")  # Scope-Context automatisch von contextvars
    # ... Node-Logik
    log.info("node.complete", duration_ms=123)

    return GraphTransition(to="parse")
```

---

## Codebase Audit: `extra=` Pattern Usage

Gesamtzahl: **~40 Vorkommen** über 14 Dateien (Stand: 2025-12-29)

### High-Impact Kandidaten (>3 Calls)

#### 1. **ai_core/tasks.py** (3 Calls)
- **Aktuell**: Manueller `extra={...}` Dict-Bau mit 10+ Feldern
- **Impact**: ~200 LOC Boilerplate
- **Priority**: 🔴 **Critical** (bereits dokumentiert oben)

#### 2. **ai_core/llm/client.py** (4 Calls)
```python
# Zeilen 296, 321, 346, 380, 402
logger.warning("llm 5xx response", extra={**log_extra, "status": status})
logger.warning("llm rate limited", extra={**log_extra, "status": status})
logger.warning("llm 4xx response", extra={**log_extra, "status": status})
logger.warning("llm.response_missing_content", extra=log_extra)
```
- **Pattern**: Wiederholtes `log_extra` mit HTTP-Status
- **Refactor**: `logger.bind(**log_extra)` einmal, dann `log.warning(..., status=status)`
- **Priority**: 🟡 **High**

#### 3. **llm_worker/domain_policies.py** (5 Calls)
```python
# Zeilen 239, 255, 290, 357, 393, 405
logger.debug("hybrid.policy_action_invalid", extra={"action": action_raw})
logger.debug("hybrid.policy_regex_invalid", extra={"error": str(exc)})
logger.warning("hybrid.policy_yaml_load_failed", extra={"error": str(exc)})
```
- **Pattern**: Viele Error-Logs mit `{"error": str(exc)}`
- **Refactor**: Structlog serialisiert Exceptions automatisch, kann einfach `exc_info=exc` nutzen
- **Priority**: 🟡 **High**

#### 4. **ai_core/rag/vector_client.py** (3 Calls)
```python
# Zeilen 1519, 4239, 4877
logger.info("ingestion.doc.result", extra=doc_payload)
logger.info("ingestion.doc.near_duplicate_skipped", extra=log_extra)
logger.info("ingestion.doc.metadata_repaired", extra=extra)
```
- **Pattern**: Document-Level-Logging mit großen Payloads
- **Refactor**: `logger.bind(**doc_payload)` für Document-Context
- **Priority**: 🟡 **High**

#### 5. **llm_worker/graphs/hybrid_search_and_score.py** (3 Calls)
```python
# Zeilen 146, 1016, 1157
logger.warning("hybrid.scoring_context_invalid", extra={"error": str(exc)})
logger.debug("hybrid.candidate_invalid", extra={"error": str(exc)})
logger.warning("hybrid.llm_failure", extra={"error": str(exc)})
```
- **Priority**: 🟢 **Medium**

### Medium-Impact Kandidaten (2 Calls)

#### 6. **ai_core/services/__init__.py** (2 Calls)
```python
# Zeilen 1106, 1122
logger.warning("llm.rate_limited", extra=extra)
logger.warning("llm.client_error", extra=extra)
```
- **Priority**: 🟢 **Medium**

#### 7. **documents/domain_service.py** (3 Calls)
```python
# Zeilen 777, 805, 837
logger.info("documents.collection.vector_sync_success", extra=payload)
logger.info("documents.collection.vector_delete_success", extra=payload)
logger.info("documents.collection.delete_dispatched", extra=extra)
```
- **Priority**: 🟢 **Medium**

### Low-Impact Kandidaten (1 Call)

- **crawler/tasks.py** (1x): `extra={"url": url}`
- **ai_core/api.py** (1x): `extra=filtered_payload`
- **ai_core/graphs/technical/universal_ingestion_graph.py** (2x): `extra={"errors": ...}`, `extra={"query": ...}`
- **ai_core/rag/hard_delete.py** (1x): `extra=log_payload`
- **documents/processing_graph.py** (1x): `extra={"error": str(exc)}`
- **documents/repository.py** (1x): `extra=_ingestion_log_fields(record)`
- **ai_core/rag/ingestion_contracts.py** (1x): `extra=metadata`
- **ai_core/rag/profile_resolver.py** (1x): `extra=metadata`
- **ai_core/rag/vector_store.py** (2x): `extra={"scope": ...}`
- **ai_core/rag/vector_space_resolver.py** (1x): `extra=metadata`

### Priorisierung (nach ROI)

**Phase 1 (Wochen 1-2):**
1. ✅ ai_core/tasks.py (~200 LOC gespart)
2. ai_core/llm/client.py (~50 LOC gespart)
3. ai_core/rag/vector_client.py (~40 LOC gespart)

**Phase 2 (Wochen 3-4):**
4. llm_worker/domain_policies.py
5. llm_worker/graphs/hybrid_search_and_score.py
6. ai_core/services/__init__.py
7. documents/domain_service.py

**Phase 3 (Week 5):**
8. Alle Low-Impact Kandidaten (Cleanup)

**Gesamt-Ersparnis**: ~350-400 LOC + garantierte Contract-Compliance

---

## Implementation Plan

### Phase 1: Setup (0.5 Tag)

1. **Prüfe structlog-Konfiguration**:
   ```bash
   grep -r "structlog" config/ ai_core/
   ```
   - Falls nicht vorhanden: Setup in `config/settings.py` oder `ai_core/logging.py`
   - Context Processors konfigurieren: `merge_contextvars`, `merge_threadlocal`

2. **Definiere Standard-Felder** (gemäß [AGENTS.md](../AGENTS.md#glossar--feld-matrix)):
   ```python
   # ai_core/logging.py (neu)
   REQUIRED_SCOPE_FIELDS = [
       "tenant_id",
       "trace_id",
       "invocation_id",  # run_id ODER ingestion_run_id
   ]

   OPTIONAL_BUSINESS_FIELDS = [
       "workflow_id",
       "document_id",
       "collection_id",
       "case_id",
   ]
   ```

3. **Helper-Funktion** für Task-Start:
   ```python
   # ai_core/logging.py
   def bind_task_context(
       tenant_id: str,
       trace_id: str,
       invocation_id: str | UUID,
       **extra_fields,
   ) -> None:
       """Binde Scope + Business Context für Task-Logging."""
       contextvars.bind_contextvars(
           tenant_id=tenant_id,
           trace_id=trace_id,
           invocation_id=str(invocation_id),
           **extra_fields,
       )
   ```

### Phase 2: Refactor Tasks (0.5 Tag)

1. **ai_core/tasks.py**:
   - Ersetze `extra`-Dict-Aufbau durch `bind_task_context()` am Task-Start
   - Entferne `extra=extra` aus allen `logger.*()` Calls
   - **Dateien**: `run_ingestion_graph()`, `embed_chunks()`, `chunk_document()`

2. **Regression-Tests**:
   - Prüfe, dass alle Tests noch laufen
   - Validiere Log-Output enthält alle Pflichtfelder

### Phase 3: Refactor Graphs (0.5 Tag)

1. **ai_core/graphs/technical/universal_ingestion_graph.py**:
   - Graph-Level Binding am Graph-Start (in `__call__` oder Runner)
   - Node-Level Binding mit `logger.bind(node=...)`

2. **ai_core/graphs/business/framework_analysis_graph.py**:
   - Analog zu universal_ingestion_graph

### Phase 4: Documentation + Cleanup (0.5 Tag)

1. **Update [CLAUDE.md](../CLAUDE.md)**:
   - Ergänze Best Practices: "Logging mit structlog contextvars"
   - Verweise auf [AGENTS.md#Logging-Contracts](../AGENTS.md)

2. **Code-Review + Linting**:
   - Suche nach verbliebenem `extra=` Pattern:
     ```bash
     grep -r "logger\.\(info\|warning\|error\).*extra=" ai_core/
     ```

3. **Optional: Pre-commit Hook**:
   - Warne bei `logger.*(extra=` Pattern (veraltet)

---

## Benefits

### Codequalität

- **50% weniger Logging-Code** (~100-150 LOC eingespart in `ai_core/tasks.py`)
- **100% konsistente Kontext-Felder** (garantiert durch contextvars)
- **Einfachere Wartung**: Neue Pflichtfelder einmal in `bind_task_context()` ergänzen

### Contract-Compliance

- **Automatische Einhaltung** von [AGENTS.md#Glossar & Feld-Matrix](../AGENTS.md#glossar--feld-matrix)
- **Konsistente Scope/Business-Trennung** (Layer 1/2 Norm)
- **Besseres Tracing**: `trace_id`, `tenant_id` in jedem Log

### Observability

- **Langfuse**: Präzisere Korrelation von Logs → Traces
- **ELK**: Bessere Filterbarkeit (alle Logs haben garantiert `tenant_id`, `trace_id`)
- **Debugging**: Einfacher, da Kontext immer vollständig

---

## Risks & Mitigations

### Risk 1: Thread-Safety in Celery

**Problem**: Celery nutzt Thread-Pools, `contextvars` ist per-Task isoliert.

**Mitigation**:
- Celery ≥5.0 unterstützt `contextvars` nativ
- Tests schreiben: Parallele Tasks dürfen Kontext nicht mischen
- Fallback: `contextvars.unbind_contextvars()` im `finally` Block

### Risk 2: Bestehende Log-Parser brechen

**Problem**: ELK/Langfuse erwarten `extra`-Dict-Format.

**Mitigation**:
- Structlog serialisiert identisch wie `extra=` (dict → JSON)
- Test: Log-Output vor/nach Refactoring vergleichen
- Rollback-Plan: Feature-Flag `USE_CONTEXTVARS_LOGGING`

### Risk 3: Performance-Overhead

**Problem**: `contextvars` könnte langsamer sein als `extra=dict`.

**Mitigation**:
- Benchmark: `contextvars.bind()` vs. dict-Bau (≪1ms erwartet)
- Profiling: `pytest --durations=10`
- **Erwartung**: Kein messbarer Overhead (Python 3.7+ optimiert)

---

## Acceptance Criteria

- [ ] `ai_core/tasks.py` nutzt `bind_task_context()` statt `extra={...}`
- [ ] Alle `logger.*()` Calls ohne `extra=` Parameter
- [ ] Tests laufen ohne Regression
- [ ] Log-Output enthält **alle** Scope-Context-Felder (`tenant_id`, `trace_id`, `invocation_id`)
- [ ] [CLAUDE.md](../CLAUDE.md) enthält Logging-Best-Practices
- [ ] Optional: Pre-commit Hook gegen `extra=` Pattern

---

## Related Issues

- [AGENTS.md#Glossar & Feld-Matrix](../AGENTS.md#glossar--feld-matrix) - Contract-Definition
- [docs/observability/langfuse.md](../observability/langfuse.md) - Trace-Korrelation
- [docs/observability/elk.md](../observability/elk.md) - Log-Aggregation

---

## References

- [structlog contextvars Docs](https://www.structlog.org/en/stable/contextvars.html)
- [Python contextvars (PEP 567)](https://peps.python.org/pep-0567/)
- Beispiel: [ai_core/tasks.py:260-284](../../ai_core/tasks.py#L260-L284) (aktueller Pattern)
