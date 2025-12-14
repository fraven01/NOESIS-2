# Upload Ingestion Graph → LangGraph Refactoring Plan

**Ziel:** Refaktorisierung von `upload_ingestion_graph.py` von einer Klassen-basierten Wrapper-Implementierung zu einem originären LangGraph (analog zu `external_knowledge_graph.py`)

**Status:** Pre-MVP, Breaking Changes erlaubt, Datenbank-Reset möglich

**Erstellt:** 2025-12-14

---

## Executive Summary

`UploadIngestionGraph` ist aktuell ein **OOP-Wrapper** um den inneren `build_document_processing_graph()` LangGraph. Die Refaktorisierung transformiert ihn zu einem **originären LangGraph** mit:

- **TypedDict State** statt Dict-Mapping
- **Nodes als Funktionen** statt Klassen-Methoden
- **StateGraph-Compilation** mit Conditional Edges
- **Runtime-Config via State** statt Constructor DI

---

## Kontext & Problemanalyse

### Aktuelle Architektur

```python
# Aktuell: OOP-Wrapper-Pattern
class UploadIngestionGraph:
    def __init__(self, repository, embedding_handler, ...):
        self._repository = repository
        self._document_graph = build_document_processing_graph(...)

    def run(self, state, meta, run_until=None):
        # 1. Validate Input
        # 2. Build Config & Context
        # 3. Invoke inner graph: self._document_graph.invoke()
        # 4. Map results to transitions
        return {"decision": ..., "transitions": {...}}
```

**Innerer Graph** (`build_document_processing_graph()`) ist **bereits ein echter LangGraph** mit:
- Nodes: `accept_upload`, `check_delta_guardrails`, `parse_document`, `persist_document`, `caption_assets`, `chunk_document`, `embed_chunks`, `cleanup`
- Conditional edges mit Checkpoints (`persist_complete`, `caption_complete`, etc.)
- State: `DocumentProcessingState` (@dataclass)

### Zwei Upload-Flows Identifiziert

1. **Synchroner Flow (Legacy):**
   - `ai_core/services/__init__.py` Zeile 1782
   - Direkter Aufruf: `graph = UploadIngestionGraph(repository=repository)`
   - Nutzt `run_until="persist_complete"` → stoppt vor Embedding

2. **Asynchroner Flow (Modern):**
   - `UploadWorker` → `run_ingestion_graph.delay()` Celery Task
   - Task nutzt **CrawlerIngestionGraph** (nicht UploadIngestionGraph!)
   - Full pipeline mit Embedding

**→ UploadIngestionGraph wird nur im synchronen Legacy-Flow genutzt**

### Warum Refaktorisierung?

1. **Konsistenz**: Alle Graphs sollten echte LangGraphs sein (siehe `external_knowledge_graph.py`)
2. **Testability**: LangGraph-Nodes sind einfacher zu mocken/testen
3. **Observability**: Native LangGraph-Integration mit Langfuse
4. **Wartbarkeit**: Klarer State-Flow, keine versteckten Klassen-Abhängigkeiten
5. **Zukunftssicherheit**: Ermöglicht Sub-Graph-Composition, Parallelisierung, HITL-Integration

---

## Zielarchitektur (LangGraph-Pattern)

### State Definition

```python
from typing import TypedDict, Any, Optional
from documents.contracts import NormalizedDocument
from documents.processing_graph import DocumentProcessingState

class UploadIngestionState(TypedDict):
    """State für Upload Ingestion Graph."""

    # Input (Required)
    normalized_document_input: dict[str, Any]  # JSON-serialized NormalizedDocument

    # Input (Optional)
    trace_id: Optional[str]
    case_id: Optional[str]
    run_until: Optional[str]  # "persist_complete" | "full" | etc.

    # Context/Config (Runtime)
    context: dict[str, Any]  # Runtime-Dependencies (repository, embedder, etc.)

    # Intermediate/Output
    document: Optional[Any]  # NormalizedDocument (validated)
    config: Optional[Any]  # DocumentPipelineConfig
    processing_context: Optional[Any]  # DocumentProcessingContext
    processing_result: Optional[DocumentProcessingState]  # Result from inner graph

    # Transitions (Output)
    transitions: dict[str, Any]

    # Final Decision (Output)
    decision: str  # "completed" | "skip_guardrail" | "skip_duplicate" | "error"
    reason: str
    severity: str  # "info" | "error"
    document_id: Optional[str]
    version: Optional[str]
    telemetry: dict[str, Any]

    # Error
    error: Optional[str]
```

### Graph Structure

```
START
  → validate_input
  → build_config_and_context
  → run_document_processing  # Invokes build_document_processing_graph()
  → map_results_to_transitions
  → END
```

**Nodes:**

1. **`validate_input`**: Validiert `normalized_document_input`, hydrated zu `NormalizedDocument`
2. **`build_config_and_context`**: Erstellt `DocumentPipelineConfig` und `DocumentProcessingContext`
3. **`run_document_processing`**: Ruft inneren Graph auf (`build_document_processing_graph().invoke()`)
4. **`map_results_to_transitions`**: Mapped Results zu strukturierten Transitions (wie aktuell `_map_result()`)

**Conditional Edges:**
- Nach `validate_input`: Wenn Error → springe zu `map_results_to_transitions` (Error-Handling)
- Nach `run_document_processing`: Wenn Error → springe zu `map_results_to_transitions`

### Runtime Dependencies

**Pattern: Context-Injection** (wie `external_knowledge_graph.py`)

```python
state["context"] = {
    "runtime_repository": repository,
    "runtime_embedder": embedding_handler,
    "runtime_guardrail_enforcer": guardrail_enforcer,
    "runtime_delta_decider": delta_decider,
    "runtime_quarantine_scanner": quarantine_scanner,
}
```

Nodes extrahieren Dependencies aus `state["context"]`.

**Alternative Pattern: Closure-Capture** (wie `build_document_processing_graph()`)

```python
def build_upload_ingestion_graph(
    repository,
    embedding_handler=ai_core_api.trigger_embedding,
    guardrail_enforcer=ai_core_api.enforce_guardrails,
    delta_decider=ai_core_api.decide_delta,
    quarantine_scanner=None,
):
    """Factory function that captures dependencies in closures."""

    def validate_input_node(state: UploadIngestionState) -> dict:
        # Accesses captured dependencies directly
        ...

    graph = StateGraph(UploadIngestionState)
    graph.add_node("validate_input", validate_input_node)
    ...
    return graph.compile()
```

**→ Empfehlung: Closure-Capture-Pattern** (sauberer, type-safe)

---

## Implementierungsplan

### Phase 1: State & Node Definition

**Schritte:**

1. **Definiere `UploadIngestionState` TypedDict** in `upload_ingestion_graph.py`
   - Alle Input/Output-Felder
   - `context` dict für Runtime-Config
   - `error` field für Error-Propagation

2. **Erstelle Node-Funktionen:**
   - `validate_input(state: UploadIngestionState) -> dict`
   - `build_config_and_context(state: UploadIngestionState) -> dict`
   - `run_document_processing(state: UploadIngestionState) -> dict`
   - `map_results_to_transitions(state: UploadIngestionState) -> dict`

3. **Error-Handling-Wrapper:**
   ```python
   def _with_error_capture(name: str, runner: Callable):
       def _wrapped(state: UploadIngestionState) -> UploadIngestionState:
           if state.get("error"):
               return {}  # Skip if already errored
           try:
               return runner(state)
           except Exception as exc:
               return {"error": str(exc)}
       return _wrapped
   ```

**Files:**
- `ai_core/graphs/upload_ingestion_graph.py` (komplett neu strukturiert)

**Tests:**
- Unit-Tests für jede Node-Funktion
- Integration-Test für ganzen Graph

---

### Phase 2: Graph Construction (Factory Pattern)

**Schritte:**

1. **Factory Function `build_upload_ingestion_graph()`:**
   ```python
   def build_upload_ingestion_graph(
       repository=None,
       embedding_handler=ai_core_api.trigger_embedding,
       guardrail_enforcer=ai_core_api.enforce_guardrails,
       delta_decider=ai_core_api.decide_delta,
       quarantine_scanner=None,
       storage=None,
   ):
       # Build dependencies (parser, chunker, etc.)
       components = require_document_components()
       ...

       # Build inner document processing graph
       document_graph = build_document_processing_graph(
           parser=parser_dispatcher,
           repository=repository,
           storage=storage,
           captioner=captioner,
           chunker=chunker,
           embedder=embedding_handler,
           delta_decider=delta_decider,
           guardrail_enforcer=guardrail_enforcer,
           quarantine_scanner=quarantine_scanner,
       )

       # Define nodes (closures capture dependencies)
       def validate_input(state):
           normalized_input = state.get("normalized_document_input")
           if not normalized_input:
               raise ValueError("input_missing:normalized_document_input")
           try:
               doc = NormalizedDocument.model_validate(normalized_input)
           except Exception as exc:
               raise ValueError(f"input_invalid:{exc}")
           return {"document": doc}

       def build_config_and_context(state):
           doc = state["document"]
           config = DocumentPipelineConfig(
               enable_upload_validation=True,
               max_bytes=25 * 1024 * 1024,
               mime_allowlist=(...),
               enable_asset_captions=False,
               enable_embedding=True,
           )
           context = DocumentProcessingContext.from_document(
               doc,
               case_id=state.get("case_id"),
               trace_id=state.get("trace_id"),
           )
           return {"config": config, "processing_context": context}

       def run_document_processing(state):
           run_until = DocumentProcessingPhase.coerce(state.get("run_until"))
           graph_state = DocumentProcessingState(
               document=state["document"],
               config=state["config"],
               context=state["processing_context"],
               storage=storage,
               run_until=run_until,
           )
           result_state = document_graph.invoke(graph_state)
           if isinstance(result_state, dict):
               result_state = DocumentProcessingState(**result_state)
           return {"processing_result": result_state}

       def map_results_to_transitions(state):
           # Existing _map_result() logic
           result_state = state.get("processing_result")
           if not result_state:
               return {
                   "decision": "error",
                   "reason": "processing_failed",
                   "severity": "error",
                   "transitions": {},
                   "telemetry": {},
               }

           # Extract document_id, version, etc.
           doc = result_state.document
           ...

           # Build transitions dict (accept_upload, delta_and_guardrails, document_pipeline)
           transitions = {...}

           # Determine final decision
           decision = "completed"
           reason = "ingestion_finished"
           severity = "info"

           if guardrail and not guardrail.allowed:
               decision = "skip_guardrail"
               ...
           elif delta and delta.decision in {"skip", "unchanged", ...}:
               decision = "skip_duplicate"
               ...

           return {
               "decision": decision,
               "reason": reason,
               "severity": severity,
               "document_id": document_id,
               "version": version,
               "transitions": transitions,
               "telemetry": {...},
           }

       # Build graph
       graph = StateGraph(UploadIngestionState)
       graph.add_node("validate_input", _with_error_capture("validate", validate_input))
       graph.add_node("build_config", _with_error_capture("config", build_config_and_context))
       graph.add_node("process", _with_error_capture("process", run_document_processing))
       graph.add_node("map_results", map_results_to_transitions)

       # Edges
       graph.add_edge(START, "validate_input")
       graph.add_conditional_edges(
           "validate_input",
           lambda s: "error" if s.get("error") else "continue",
           {"continue": "build_config", "error": "map_results"}
       )
       graph.add_edge("build_config", "process")
       graph.add_edge("process", "map_results")
       graph.add_edge("map_results", END)

       return graph.compile()
   ```

2. **Default Graph Instance:**
   ```python
   # For backward compatibility
   _default_graph = None

   def get_default_graph():
       global _default_graph
       if _default_graph is None:
           _default_graph = build_upload_ingestion_graph()
       return _default_graph
   ```

3. **Convenience `run()` Function:**
   ```python
   def run(
       state: Mapping[str, Any],
       meta: Mapping[str, Any] | None = None,
       run_until: str | None = None,
   ) -> Mapping[str, Any]:
       """Convenience function matching old API."""
       graph = get_default_graph()

       # Merge meta into state
       input_state = dict(state)
       if meta:
           if "trace_id" in meta:
               input_state.setdefault("trace_id", meta["trace_id"])
           if "case_id" in meta:
               input_state.setdefault("case_id", meta["case_id"])
       if run_until:
           input_state["run_until"] = run_until

       result_state = graph.invoke(input_state)

       # Extract output fields
       return {
           "decision": result_state.get("decision", "error"),
           "reason": result_state.get("reason", "unknown"),
           "severity": result_state.get("severity", "error"),
           "document_id": result_state.get("document_id"),
           "version": result_state.get("version"),
           "telemetry": result_state.get("telemetry", {}),
           "transitions": result_state.get("transitions", {}),
       }
   ```

**Files:**
- `ai_core/graphs/upload_ingestion_graph.py` (neue Factory-Funktion)

**Tests:**
- Factory-Function-Tests (verschiedene Dependency-Combinations)
- Graph-Invocation-Tests

---

### Phase 3: Backward Compatibility & Migration

**Schritte:**

1. **Legacy `UploadIngestionGraph` Klasse als Wrapper:**
   ```python
   class UploadIngestionGraph:
       """Legacy wrapper for backward compatibility. DEPRECATED."""

       def __init__(
           self,
           *,
           document_service=None,
           repository=None,
           document_persistence=None,
           persistence_handler=None,
           guardrail_enforcer=ai_core_api.enforce_guardrails,
           delta_decider=ai_core_api.decide_delta,
           quarantine_scanner=None,
           embedding_handler=None,
           lifecycle_hook=None,
           storage=None,
       ):
           import warnings
           warnings.warn(
               "UploadIngestionGraph class is deprecated. "
               "Use build_upload_ingestion_graph() instead.",
               DeprecationWarning,
               stacklevel=2,
           )

           if embedding_handler is None:
               embedding_handler = ai_core_api.trigger_embedding

           self._graph = build_upload_ingestion_graph(
               repository=repository,
               embedding_handler=embedding_handler,
               guardrail_enforcer=guardrail_enforcer,
               delta_decider=delta_decider,
               quarantine_scanner=quarantine_scanner,
               storage=storage,
           )

       def run(self, state, meta=None, run_until=None):
           """Delegate to new graph."""
           return run(state, meta, run_until)
   ```

2. **Update Imports in `ai_core/services/__init__.py`:**
   ```python
   # Before:
   from ai_core.graphs.upload_ingestion_graph import UploadIngestionGraph

   # After (Option A - keep class):
   from ai_core.graphs.upload_ingestion_graph import UploadIngestionGraph  # Uses new impl

   # After (Option B - use factory):
   from ai_core.graphs.upload_ingestion_graph import build_upload_ingestion_graph

   # In code:
   graph = build_upload_ingestion_graph(repository=repository)
   result = graph.invoke(graph_payload)
   ```

3. **Error-Klasse beibehalten:**
   ```python
   class UploadIngestionError(RuntimeError):
       """Raised for unexpected internal errors in the upload ingestion graph."""
   ```

**Files:**
- `ai_core/graphs/upload_ingestion_graph.py` (Legacy-Wrapper)
- `ai_core/services/__init__.py` (Migration zu Factory oder Wrapper)

**Tests:**
- Backward-Compatibility-Tests (alte API funktioniert noch)

---

### Phase 4: Test Migration

**Schritte:**

1. **Update `test_upload_ingestion_graph.py`:**
   ```python
   # Before:
   graph = UploadIngestionGraph(repository=repository, embedding_handler=mock_embedder)
   result = graph.run(payload)

   # After (Option A - Factory):
   graph = build_upload_ingestion_graph(repository=repository, embedding_handler=mock_embedder)
   result_state = graph.invoke(payload)
   result = {
       "decision": result_state["decision"],
       "reason": result_state["reason"],
       ...
   }

   # After (Option B - Convenience run()):
   from ai_core.graphs.upload_ingestion_graph import run
   result = run(payload, meta=None)
   ```

2. **Neue Tests für Nodes:**
   ```python
   def test_validate_input_node():
       graph = build_upload_ingestion_graph()
       # Access specific node for unit testing?
       # Or test via full graph with early exit
       ...

   def test_error_propagation():
       """Test that errors in validate_input skip processing."""
       payload = {"normalized_document_input": None}  # Invalid
       graph = build_upload_ingestion_graph()
       result_state = graph.invoke(payload)
       assert result_state["error"]
       assert result_state["decision"] == "error"
   ```

3. **Tests für run_until:**
   ```python
   def test_run_until_persist_complete():
       payload = {...}
       result = run(payload, run_until="persist_complete")
       # Should stop before embedding
       assert result["decision"] in ("completed", "skip_duplicate")
   ```

**Files:**
- `ai_core/tests/graphs/test_upload_ingestion_graph.py`

---

### Phase 5: Documentation & Cleanup

**Schritte:**

1. **Update `ai_core/graphs/README.md`:**
   - Dokumentiere neuen Upload Ingestion Graph
   - LangGraph-Pattern mit State, Nodes, Factory
   - Migration-Guide von alter zu neuer API

2. **Docstrings:**
   - `build_upload_ingestion_graph()` mit Parametern
   - `UploadIngestionState` TypedDict
   - Node-Funktionen

3. **Deprecation Notice:**
   - `UploadIngestionGraph` Klasse als deprecated markieren
   - Timeline für Removal (z.B. nach 3 Monaten)

4. **Cleanup:**
   - Remove Debug-Prints in `processing_graph.py` (Zeile 392, 682, 859, 882)
   - Logging statt Prints

**Files:**
- `ai_core/graphs/README.md`
- `ai_core/graphs/upload_ingestion_graph.py` (Docstrings)
- `documents/processing_graph.py` (Cleanup)

---

## Breaking Changes & Migration

### Breaking Changes

1. **API-Signatur (wenn Factory statt Klasse):**
   ```python
   # Alt:
   graph = UploadIngestionGraph(repository=repo)
   result = graph.run(state, meta, run_until="persist_complete")

   # Neu:
   graph = build_upload_ingestion_graph(repository=repo)
   result_state = graph.invoke(state)  # run_until muss in state sein
   result = {
       "decision": result_state["decision"],
       ...
   }
   ```

2. **State-Input:**
   - `run_until` muss jetzt in `state` sein (nicht als Parameter)
   - `meta` wird in `state` gemerged (nicht separater Parameter)

3. **Output:**
   - Graph gibt `UploadIngestionState` zurück (dict)
   - Convenience `run()` behält altes Format

### Migration Strategy

**Option A: Non-Breaking (Empfohlen für Pre-MVP):**
- Behalte `UploadIngestionGraph` Klasse als Wrapper
- Interne Implementierung nutzt neuen LangGraph
- Tests funktionieren ohne Änderungen

**Option B: Breaking mit Deprecation:**
- Neue Factory-Funktion als primäre API
- Klasse deprecated mit Warning
- 3-Monats-Übergangszeit

**Option C: Hard Break (Pre-MVP erlaubt):**
- Entferne Klasse komplett
- Nur Factory-Funktion
- Update alle Aufrufer (nur 1 Stelle: `ai_core/services/__init__.py`)

**→ Empfehlung: Option A** (Non-Breaking mit internem Refactoring)

### Migration Checklist

- [ ] Update `ai_core/services/__init__.py` (Zeile 1782)
- [ ] Update `ai_core/tests/graphs/test_upload_ingestion_graph.py`
- [ ] Verify Celery Task Integration (indirekt via CrawlerIngestionGraph)
- [ ] Check Documentation References
- [ ] Run Full Test Suite
- [ ] Manual E2E Test (Upload → Persist → Check DB)

---

## Offene Fragen & Entscheidungen

### Frage 1: UploadIngestionGraph vs. CrawlerIngestionGraph

**Beobachtung:**
- Moderne asynchrone Uploads nutzen **CrawlerIngestionGraph** (via `run_ingestion_graph` Task)
- **UploadIngestionGraph** wird nur im synchronen Legacy-Flow genutzt (`ai_core/services/__init__.py`)

**Optionen:**

A. **Refaktorisiere UploadIngestionGraph** wie geplant
   - Pro: Konsistenz, alle Graphs sind LangGraphs
   - Contra: Wird wenig genutzt, Aufwand vs. Nutzen?

B. **Deprecate UploadIngestionGraph** zugunsten von CrawlerIngestionGraph
   - Pro: Weniger Code, ein Graph für beide Use Cases
   - Contra: Synchroner Flow muss migriert werden

C. **Harmonisiere beide Graphs** (gemeinsamer Core)
   - Pro: DRY, konsistente Logik
   - Contra: Komplexer Refactoring-Scope

**→ Empfehlung: A** (Refaktorisiere wie geplant, dann später Deprecation erwägen)

### Frage 2: Closure-Capture vs. Context-Injection

**Option A: Closure-Capture** (wie `build_document_processing_graph()`)
```python
def build_graph(repository, embedder):
    def node(state):
        repository.get(...)  # Captured from outer scope
    ...
```
- Pro: Type-safe, kein Runtime-Lookup
- Contra: Weniger flexibel für Tests

**Option B: Context-Injection** (wie `external_knowledge_graph.py`)
```python
state["context"]["runtime_repository"] = repository

def node(state):
    repository = state["context"]["runtime_repository"]
```
- Pro: Flexibel, einfach zu mocken
- Contra: Type-unsicher, Runtime-Fehler möglich

**→ Empfehlung: A** (Closure-Capture für Production-Stabilität)

### Frage 3: Sub-Graph-Integration

**Aktuell:** `build_document_processing_graph()` wird als Sub-Graph invoked

**Optionen:**

A. **Node-basierte Invocation** (aktuell):
```python
def run_document_processing(state):
    result = document_graph.invoke(DocumentProcessingState(...))
    return {"processing_result": result}
```

B. **Separate Graphs mit Shared State**:
```python
# UploadIngestionGraph endet mit State
# Weiterer Graph konsumiert dieses State
```

C. **Flatten:** Alle Nodes direkt in UploadIngestionGraph
```python
# Kein Sub-Graph, alle Nodes auf einer Ebene
graph.add_node("accept_upload", ...)
graph.add_node("parse", ...)
...
```

**→ Empfehlung: A** (Node-basierte Invocation, bewahrt Separation of Concerns)

---

## Risiken & Mitigation

| Risiko | Impact | Wahrscheinlichkeit | Mitigation |
|--------|--------|-------------------|------------|
| Breaking Changes in Production | High | Low (nur 1 Aufrufer) | Backward-Compatible Wrapper |
| Test-Failures | Medium | Medium | Schrittweise Migration, Parallel-Tests |
| Performance-Regression | Low | Low | LangGraph ist performant, Benchmarks |
| Observability-Gap | Medium | Low | @observe_span auf Nodes |
| Sub-Graph-Integration-Fehler | High | Medium | Unit-Tests für State-Transformation |

---

## Success Criteria

- [ ] `build_upload_ingestion_graph()` Factory-Funktion existiert
- [ ] Graph ist ein kompilierter `StateGraph(UploadIngestionState)`
- [ ] Alle Nodes sind Funktionen (keine Klassen-Methoden)
- [ ] Tests laufen mit > 90% Coverage
- [ ] Backward Compatibility: Alte API funktioniert (via Wrapper)
- [ ] Performance: Graph-Invocation ≤ alte run()-Zeit
- [ ] Observability: Langfuse-Spans für alle Nodes
- [ ] Documentation: README mit Migration-Guide

---

## Timeline & Effort

**Gesamt-Aufwand:** ~8-12 Stunden

| Phase | Aufwand | Priorität |
|-------|---------|----------|
| Phase 1: State & Nodes | 3h | High |
| Phase 2: Graph Construction | 3h | High |
| Phase 3: Backward Compatibility | 1h | Medium |
| Phase 4: Test Migration | 2h | High |
| Phase 5: Documentation | 1h | Low |
| Review & Iteration | 2h | Medium |

**Empfehlung: 2-3 Sessions à 3-4h**

---

## Next Steps

1. **User-Feedback einholen:**
   - Bestätigen: Refaktorisierung trotz Legacy-Status sinnvoll?
   - Entscheiden: Option A/B/C für UploadIngestionGraph vs. CrawlerIngestionGraph
   - Klären: Closure-Capture vs. Context-Injection Präferenz

2. **Implementierung starten (nach Approval):**
   - Phase 1: State & Node-Definitionen
   - Phase 2: Graph-Factory
   - Phase 3: Tests
   - Phase 4: Integration

3. **Parallel: Cleanup-Tasks**
   - Remove Debug-Prints in `processing_graph.py`
   - Harmonisiere Error-Handling zwischen Graphs

---

## Anhang: Code-Referenzen

### Betroffene Dateien

**Hauptdateien:**
- `ai_core/graphs/upload_ingestion_graph.py` - Refactoring Target
- `documents/processing_graph.py` - Innerer LangGraph (Cleanup)

**Aufrufer:**
- `ai_core/services/__init__.py` Zeile 1782 - Direkter Aufruf
- `documents/upload_worker.py` Zeile 146 - Indirekt via run_ingestion_graph Task

**Tests:**
- `ai_core/tests/graphs/test_upload_ingestion_graph.py`

**Dokumentation:**
- `ai_core/graphs/README.md`
- `AGENTS.md` (Update Graph-Übersicht)

### Referenz-Implementierungen

**Vorbild (Echter LangGraph):**
- `ai_core/graphs/external_knowledge_graph.py` - TypedDict State, Context-Injection

**Best Practice (Closure-Capture):**
- `documents/processing_graph.py` - Factory-Pattern, Dataclass State

**Legacy-Pattern (zu refaktorisieren):**
- `ai_core/graphs/crawler_ingestion_graph.py` - OOP-Wrapper mit Sub-Graph-Invocation

---

**Ende des Plans**
