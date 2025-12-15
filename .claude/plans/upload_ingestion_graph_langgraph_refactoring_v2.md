# Upload Ingestion Graph → LangGraph Refactoring Plan v2

**Ziel:** Refaktorisierung von `upload_ingestion_graph.py` zu einem originären LangGraph nach dem Pattern von `external_knowledge_graph.py`

**Vorbild:** `ai_core/graphs/external_knowledge_graph.py`

**Scope:** Nur UploadIngestionGraph (CrawlerIngestionGraph out of scope)

**Status:** Pre-MVP, Breaking Changes erlaubt

---

## Pattern-Analyse: external_knowledge_graph

### ✅ Kernmuster (zu übernehmen)

1. **TypedDict State** - Klare Input/Output/Intermediate Felder
   ```python
   class ExternalKnowledgeState(TypedDict):
       # Input
       query: str
       context: dict[str, Any]
       # Output/Intermediate
       search_results: list[dict[str, Any]]
       error: str | None
   ```

2. **Context-Injection für Runtime-Dependencies**
   ```python
   worker = context.get("runtime_worker")
   trigger = context.get("runtime_trigger")
   ```

3. **Standalone Node-Funktionen mit @observe_span**
   ```python
   @observe_span(name="node.search")
   def search_node(state: ExternalKnowledgeState) -> dict[str, Any]:
       return {"search_results": results}
   ```

4. **Error-Handling via Return-Values**
   ```python
   if not worker:
       return {"error": "No search worker configured"}
   ```

5. **Module-Level Graph-Definition**
   ```python
   workflow = StateGraph(ExternalKnowledgeState)
   workflow.add_node("search", search_node)
   graph = workflow.compile()  # Module-level, nicht in Factory
   ```

6. **Protocols für Dependencies**
   ```python
   class IngestionTrigger(Protocol):
       def trigger(self, *, url: str, ...) -> Mapping[str, Any]:
           ...
   ```

7. **Kein run() Wrapper** - Caller nutzen direkt `graph.invoke(state)`

---

## Zielarchitektur

### 1. State Definition

```python
from typing import TypedDict, Any, NotRequired
from documents.processing_graph import DocumentProcessingState

class UploadIngestionState(TypedDict):
    """State for upload ingestion graph."""

    # Input (Required)
    normalized_document_input: dict[str, Any]

    # Input (Optional)
    trace_id: NotRequired[str]
    case_id: NotRequired[str]
    run_until: NotRequired[str]  # "persist_complete" | "full" | etc.

    # Runtime Context (Injected Dependencies)
    context: NotRequired[dict[str, Any]]

    # Intermediate State
    document: NotRequired[Any]  # NormalizedDocument (validated)
    config: NotRequired[Any]  # DocumentPipelineConfig
    processing_context: NotRequired[Any]  # DocumentProcessingContext
    processing_result: NotRequired[DocumentProcessingState]

    # Output
    decision: NotRequired[str]  # "completed" | "skip_guardrail" | "skip_duplicate" | "error"
    reason: NotRequired[str]
    severity: NotRequired[str]
    document_id: NotRequired[str]
    version: NotRequired[str]
    telemetry: NotRequired[dict[str, Any]]
    transitions: NotRequired[dict[str, Any]]

    # Error
    error: NotRequired[str]
```

**Anmerkungen:**
- Alle optionalen Felder nutzen `NotRequired[]` (Python 3.11+)
- `context` enthält Runtime-Dependencies: `runtime_repository`, `runtime_storage`, etc.

### 2. Protocols für Dependencies

```python
from typing import Protocol, Mapping, Any

class DocumentRepository(Protocol):
    """Protocol for document repository."""
    def get(self, tenant_id: str, document_id: str, **kwargs) -> Any: ...
    def upsert(self, document: Any, **kwargs) -> Any: ...

class EmbeddingHandler(Protocol):
    """Protocol for embedding handler."""
    def __call__(self, *, normalized_document: Any, **kwargs) -> Any: ...

class GuardrailEnforcer(Protocol):
    """Protocol for guardrail enforcement."""
    def __call__(self, *, normalized_document: Any, **kwargs) -> Any: ...

class DeltaDecider(Protocol):
    """Protocol for delta decision."""
    def __call__(self, *, normalized_document: Any, baseline: Any, **kwargs) -> Any: ...
```

### 3. Node-Funktionen

#### Node 1: validate_input

```python
@observe_span(name="upload.validate_input")
def validate_input_node(state: UploadIngestionState) -> dict[str, Any]:
    """Validate and hydrate normalized_document_input."""
    from documents.contracts import NormalizedDocument

    normalized_input = state.get("normalized_document_input")
    if not normalized_input:
        return {"error": "input_missing:normalized_document_input"}

    try:
        doc = NormalizedDocument.model_validate(normalized_input)
    except Exception as exc:
        return {"error": f"input_invalid:{exc}"}

    return {"document": doc, "error": None}
```

#### Node 2: build_config

```python
@observe_span(name="upload.build_config")
def build_config_node(state: UploadIngestionState) -> dict[str, Any]:
    """Build pipeline config and processing context."""
    from documents.pipeline import DocumentPipelineConfig, DocumentProcessingContext
    from django.conf import settings

    doc = state.get("document")
    if not doc:
        return {"error": "document_missing"}

    # Build config (upload-specific settings)
    config = DocumentPipelineConfig(
        enable_upload_validation=True,
        max_bytes=int(getattr(settings, "UPLOAD_MAX_BYTES", 25 * 1024 * 1024)),
        mime_allowlist=tuple(getattr(settings, "UPLOAD_ALLOWED_MIME_TYPES", (
            "text/plain", "text/markdown", "text/html",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ))),
        enable_asset_captions=False,  # Disabled for upload
        enable_embedding=True,
    )

    # Build processing context
    context = DocumentProcessingContext.from_document(
        doc,
        case_id=state.get("case_id"),
        trace_id=state.get("trace_id"),
    )

    return {
        "config": config,
        "processing_context": context,
        "error": None,
    }
```

#### Node 3: run_processing

```python
@observe_span(name="upload.run_processing")
def run_processing_node(state: UploadIngestionState) -> dict[str, Any]:
    """Invoke inner document processing graph."""
    from documents.processing_graph import (
        DocumentProcessingState,
        DocumentProcessingPhase,
        build_document_processing_graph,
    )

    doc = state.get("document")
    config = state.get("config")
    proc_context = state.get("processing_context")
    runtime_context = state.get("context", {})

    if not all([doc, config, proc_context]):
        return {"error": "missing_required_state"}

    # Extract runtime dependencies from context
    repository = runtime_context.get("runtime_repository")
    storage = runtime_context.get("runtime_storage")
    embedder = runtime_context.get("runtime_embedder")
    delta_decider = runtime_context.get("runtime_delta_decider")
    guardrail_enforcer = runtime_context.get("runtime_guardrail_enforcer")
    quarantine_scanner = runtime_context.get("runtime_quarantine_scanner")

    # Build inner graph (with runtime dependencies)
    from documents.parsers import create_default_parser_dispatcher
    from documents.cli import SimpleDocumentChunker
    from documents.pipeline import require_document_components

    components = require_document_components()
    parser = create_default_parser_dispatcher()
    chunker = SimpleDocumentChunker()

    # Captioner
    captioner_cls = components.captioner
    try:
        captioner = captioner_cls()
    except Exception:
        captioner = captioner_cls

    # Storage fallback
    if not storage:
        try:
            storage = components.storage()
        except Exception:
            from documents.storage import ObjectStoreStorage
            storage = ObjectStoreStorage()

    inner_graph = build_document_processing_graph(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=captioner,
        chunker=chunker,
        embedder=embedder,
        delta_decider=delta_decider,
        guardrail_enforcer=guardrail_enforcer,
        quarantine_scanner=quarantine_scanner,
    )

    # Determine run_until
    run_until_str = state.get("run_until")
    run_until = DocumentProcessingPhase.coerce(run_until_str)

    # Build inner state
    inner_state = DocumentProcessingState(
        document=doc,
        config=config,
        context=proc_context,
        storage=storage,
        run_until=run_until,
    )

    # Invoke inner graph
    try:
        result_state = inner_graph.invoke(inner_state)
        if isinstance(result_state, dict):
            result_state = DocumentProcessingState(**result_state)
    except Exception as exc:
        logger.exception("Document processing graph failed")
        return {"error": f"processing_failed:{exc}"}

    return {
        "processing_result": result_state,
        "error": None,
    }
```

#### Node 4: map_results

```python
@observe_span(name="upload.map_results")
def map_results_node(state: UploadIngestionState) -> dict[str, Any]:
    """Map processing results to output transitions."""
    from ai_core.graphs.transition_contracts import (
        StandardTransitionResult,
        PipelineSection,
        build_delta_section,
        build_guardrail_section,
    )

    result_state = state.get("processing_result")
    if not result_state:
        return {
            "decision": "error",
            "reason": "processing_result_missing",
            "severity": "error",
            "transitions": {},
            "telemetry": {},
        }

    # Extract data
    doc = result_state.document
    normalized_doc = getattr(doc, "document", doc)
    ref = getattr(normalized_doc, "ref", None)
    document_id = getattr(ref, "document_id", None)
    version = getattr(ref, "version", None)

    delta = result_state.delta_decision
    guardrail = result_state.guardrail_decision

    # Build transitions
    def _transition(*, phase, decision, reason, severity="info", context=None,
                    pipeline=None, delta=None, guardrail=None):
        ctx = {k: v for k, v in (context or {}).items() if v is not None}
        result = StandardTransitionResult(
            phase=phase,
            decision=decision,
            reason=reason,
            severity=severity,
            context=ctx,
            pipeline=pipeline,
            delta=build_delta_section(delta) if delta is not None else None,
            guardrail=build_guardrail_section(guardrail) if guardrail is not None else None,
        )
        return result.model_dump()

    transitions = {}

    # accept_upload transition
    blob = getattr(normalized_doc, "blob", None)
    mime_type = getattr(blob, "media_type", None)
    transitions["accept_upload"] = _transition(
        phase="accept_upload",
        decision="accepted",
        reason="upload_validated",
        context={
            "mime": mime_type,
            "size_bytes": getattr(blob, "size", None),
        },
    )

    # delta_and_guardrails transition
    if guardrail and not getattr(guardrail, "allowed", False):
        dg_decision = guardrail.decision
        dg_reason = guardrail.reason
        dg_severity = "error"
    elif delta:
        dg_decision = delta.decision
        dg_reason = delta.reason
        dg_severity = "info"
    else:
        dg_decision = "unknown"
        dg_reason = "guardrail_delta_missing"
        dg_severity = "info"

    transitions["delta_and_guardrails"] = _transition(
        phase="delta_and_guardrails",
        decision=dg_decision,
        reason=dg_reason,
        severity=dg_severity,
        context={"document_id": str(document_id) if document_id else None},
        delta=delta,
        guardrail=guardrail,
    )

    # document_pipeline transition
    pipeline_section = PipelineSection(
        phase=result_state.phase,
        run_until=state.get("run_until"),
        error=repr(result_state.error) if result_state.error else None,
    )
    transitions["document_pipeline"] = _transition(
        phase="document_pipeline",
        decision="processed" if result_state.error is None else "error",
        reason="document_pipeline_completed" if result_state.error is None else "document_pipeline_failed",
        severity="error" if result_state.error else "info",
        context={"phase": result_state.phase},
        pipeline=pipeline_section,
    )

    # Determine final decision
    decision = "completed"
    reason = "ingestion_finished"
    severity = "info"

    if guardrail and not getattr(guardrail, "allowed", False):
        decision = "skip_guardrail"
        reason = guardrail.reason or "guardrail_denied"
        severity = "error"
    elif delta:
        delta_flag = delta.decision.strip().lower()
        if delta_flag in {"skip", "unchanged", "duplicate", "near_duplicate"}:
            decision = "skip_duplicate"
            reason = delta.reason or "delta_skip"

    if result_state.error is not None:
        decision = "error"
        reason = "document_pipeline_failed"
        severity = "error"

    telemetry = {
        "phase": result_state.phase,
        "run_until": state.get("run_until"),
        "delta_decision": getattr(delta, "decision", None) if delta else None,
        "guardrail_decision": getattr(guardrail, "decision", None) if guardrail else None,
    }

    return {
        "decision": decision,
        "reason": reason,
        "severity": severity,
        "document_id": str(document_id) if document_id else None,
        "version": version,
        "telemetry": {k: v for k, v in telemetry.items() if v is not None},
        "transitions": transitions,
    }
```

### 4. Graph Definition (Module-Level)

```python
# --------------------------------------------------------------------- Graph Definition

workflow = StateGraph(UploadIngestionState)

workflow.add_node("validate_input", validate_input_node)
workflow.add_node("build_config", build_config_node)
workflow.add_node("run_processing", run_processing_node)
workflow.add_node("map_results", map_results_node)

# Edges
workflow.add_edge(START, "validate_input")

# Conditional: Skip processing if validation failed
def _check_validation_error(state: UploadIngestionState) -> Literal["continue", "error"]:
    return "error" if state.get("error") else "continue"

workflow.add_conditional_edges(
    "validate_input",
    _check_validation_error,
    {"continue": "build_config", "error": "map_results"}
)

workflow.add_edge("build_config", "run_processing")
workflow.add_edge("run_processing", "map_results")
workflow.add_edge("map_results", END)

# Compiled Graph (module-level)
graph = workflow.compile()
```

### 5. Error-Klasse (beibehalten für Backward Compatibility)

```python
class UploadIngestionError(RuntimeError):
    """Raised for unexpected internal errors in the upload ingestion graph."""
```

### 6. Exports

```python
__all__ = [
    "UploadIngestionState",
    "UploadIngestionError",
    "graph",  # Compiled graph for direct invocation
    "validate_input_node",
    "build_config_node",
    "run_processing_node",
    "map_results_node",
]
```

---

## Caller-Integration

### Aktuell (ai_core/services/__init__.py)

```python
from ai_core.graphs.upload_ingestion_graph import UploadIngestionGraph

repository = _get_documents_repository()
graph = UploadIngestionGraph(repository=repository)
graph_result = graph.run(graph_payload, run_until="persist_complete")
```

### Nach Refactoring

```python
from ai_core.graphs.upload_ingestion_graph import graph as upload_graph

repository = _get_documents_repository()

# Prepare state with runtime context
state = {
    "normalized_document_input": normalized_document.model_dump(),
    "trace_id": meta["trace_id"],
    "case_id": meta["case_id"],
    "run_until": "persist_complete",
    "context": {
        "runtime_repository": repository,
        "runtime_storage": None,  # Will use fallback
        "runtime_embedder": None,  # Will use default ai_core_api.trigger_embedding
        "runtime_delta_decider": None,  # Will use default ai_core_api.decide_delta
        "runtime_guardrail_enforcer": None,  # Will use default ai_core_api.enforce_guardrails
        "runtime_quarantine_scanner": None,  # Optional
    }
}

# Invoke graph
result_state = upload_graph.invoke(state)

# Extract output
graph_result = {
    "decision": result_state.get("decision", "error"),
    "reason": result_state.get("reason", "unknown"),
    "severity": result_state.get("severity", "error"),
    "document_id": result_state.get("document_id"),
    "version": result_state.get("version"),
    "telemetry": result_state.get("telemetry", {}),
    "transitions": result_state.get("transitions", {}),
}
```

---

## Implementierungsplan

### Phase 1: State & Protocols (2h)

**Schritte:**
1. Definiere `UploadIngestionState` TypedDict
2. Definiere Protocols: `DocumentRepository`, `EmbeddingHandler`, `GuardrailEnforcer`, `DeltaDecider`
3. Importiere benötigte Types

**Dateien:**
- `ai_core/graphs/upload_ingestion_graph.py` (neu strukturiert)

**Tests:**
- Type-Checking mit mypy
- Validiere State-Schema

---

### Phase 2: Node-Implementierung (4h)

**Schritte:**
1. Implementiere `validate_input_node`
   - NormalizedDocument validation
   - Error-Return bei Fehler
   - @observe_span decorator

2. Implementiere `build_config_node`
   - DocumentPipelineConfig mit Upload-Settings
   - DocumentProcessingContext
   - @observe_span decorator

3. Implementiere `run_processing_node`
   - Dependency-Extraction aus context
   - Inner-Graph-Building
   - Inner-Graph-Invocation
   - Error-Handling
   - @observe_span decorator

4. Implementiere `map_results_node`
   - Transitions-Building (accept_upload, delta_and_guardrails, document_pipeline)
   - Final decision logic
   - Telemetry
   - @observe_span decorator

**Dateien:**
- `ai_core/graphs/upload_ingestion_graph.py`

**Tests:**
- Unit-Tests für jede Node-Funktion (mit Mocks)
- Test Error-Paths

---

### Phase 3: Graph-Definition (1h)

**Schritte:**
1. Erstelle `StateGraph(UploadIngestionState)`
2. Add Nodes
3. Add Edges (inkl. conditional edge für validation error)
4. Compile Graph
5. Export als module-level `graph`

**Dateien:**
- `ai_core/graphs/upload_ingestion_graph.py`

**Tests:**
- Graph-Structure-Tests (Nodes, Edges vorhanden)
- Mock-basierte End-to-End-Tests

---

### Phase 4: Caller-Migration (2h)

**Schritte:**
1. Update `ai_core/services/__init__.py`
   - Import `graph` statt `UploadIngestionGraph`
   - Build State mit context
   - Invoke graph
   - Extract output

2. Error-Handling
   - Catch Exceptions aus graph.invoke()
   - Map zu HTTP-Errors

**Dateien:**
- `ai_core/services/__init__.py` (Zeile 1780-1810)

**Tests:**
- Integration-Test: Upload-Flow End-to-End
- Error-Cases

---

### Phase 5: Test-Migration (2h)

**Schritte:**
1. Update `ai_core/tests/graphs/test_upload_ingestion_graph.py`
   - Mock repository, storage, etc.
   - Build State mit context
   - Invoke graph
   - Assert output

2. Neue Tests
   - Test each node individually
   - Test conditional routing
   - Test error propagation

**Dateien:**
- `ai_core/tests/graphs/test_upload_ingestion_graph.py`

---

### Phase 6: Cleanup & Documentation (1h)

**Schritte:**
1. Remove alte `UploadIngestionGraph` Klasse (komplett)
2. Update `__all__` exports
3. Docstrings für alle Nodes
4. Update `ai_core/graphs/README.md`
5. Update `AGENTS.md` (Graph-Übersicht)

**Dateien:**
- `ai_core/graphs/upload_ingestion_graph.py`
- `ai_core/graphs/README.md`
- `AGENTS.md`

---

## Offene Fragen

### 1. Default-Werte für Runtime-Dependencies

**Frage:** Sollen Nodes Default-Werte für fehlende Runtime-Dependencies nutzen?

**Beispiel:**
```python
embedder = context.get("runtime_embedder") or ai_core_api.trigger_embedding
```

**Optionen:**
- A: **Defaults in Nodes** - Nodes nutzen ai_core_api.* Defaults wenn nicht injected
- B: **Caller muss alles übergeben** - Nodes returnen Error wenn Dependencies fehlen
- C: **Hybrid** - Nur kritische Dependencies required, Rest hat Defaults

**Empfehlung basierend auf external_knowledge_graph:**
- Option A - external_knowledge_graph nutzt auch Defaults/Fallbacks in Nodes
- Kritische Dependencies (worker, trigger) werden als Error gemeldet wenn fehlend
- Optionale Config-Werte nutzen Defaults

**Vorschlag:** Option C (Hybrid)
- `runtime_repository` → Optional (Inner Graph hat graceful None-Handling)
- `runtime_storage` → Fallback zu `ObjectStoreStorage()`
- `runtime_embedder` → Fallback zu `ai_core_api.trigger_embedding`
- Andere → Fallback zu ai_core_api.* Defaults

---

### 2. Error-Handling-Strategie

**Frage:** Wie sollen schwere Fehler behandelt werden?

**Optionen:**
- A: **Return Error-Dict** (wie external_knowledge_graph) - `return {"error": "..."}`
- B: **Raise Exception** - `raise UploadIngestionError("...")`
- C: **Hybrid** - Validation-Errors als Return, Processing-Errors als Exception

**Empfehlung basierend auf external_knowledge_graph:**
- Option A - external_knowledge_graph nutzt Error-Returns
- Caller prüft `state.get("error")` und handhabt entsprechend

**Vorschlag:** Option A (Error-Returns)
- Nodes returnen `{"error": "reason"}` bei Fehlern
- Graph propagiert Error durch alle Nodes (Skip-Logic via conditional edges)
- Caller extrahiert Error aus final state
- `UploadIngestionError` bleibt für Backward-Compatibility, wird aber nur vom Caller geraised

---

### 3. Observability-Spans

**Frage:** Sollen alle Nodes @observe_span haben?

**Antwort basierend auf external_knowledge_graph:**
- Ja, alle Nodes haben `@observe_span(name="upload.node_name")`
- Naming-Convention: `"upload.validate_input"`, `"upload.build_config"`, etc.

**Vorschlag:** Ja, alle Nodes mit Spans

---

### 4. Context-Injection vs. Module-Level Defaults

**Frage:** Sollen Components (parser, chunker, captioner) auch via context injected werden?

**Optionen:**
- A: **Alles via context** - Auch parser, chunker, etc.
- B: **Nur Services via context** - parser/chunker/captioner werden im Node gebaut
- C: **Hybrid** - Services via context, Components im Node

**Empfehlung basierend auf external_knowledge_graph:**
- external_knowledge_graph injected nur Services (worker, trigger), nicht utilities
- Components wie parser sind eher utilities als runtime-dependencies

**Vorschlag:** Option B
- Nur Services via context: repository, storage, embedder, delta_decider, guardrail_enforcer, quarantine_scanner
- Components werden in `run_processing_node` gebaut (wie aktuell)

---

## Breaking Changes

### 1. API-Änderung

**Alt:**
```python
graph = UploadIngestionGraph(repository=repository)
result = graph.run(state, meta=None, run_until="persist_complete")
```

**Neu:**
```python
state = {
    "normalized_document_input": doc.model_dump(),
    "run_until": "persist_complete",
    "context": {"runtime_repository": repository},
}
result_state = graph.invoke(state)
result = {
    "decision": result_state["decision"],
    ...
}
```

**Betroffene Caller:**
- `ai_core/services/__init__.py` (1 Stelle)

---

### 2. Import-Änderung

**Alt:**
```python
from ai_core.graphs.upload_ingestion_graph import UploadIngestionGraph
```

**Neu:**
```python
from ai_core.graphs.upload_ingestion_graph import graph as upload_graph
```

---

### 3. Output-Format

**Keine Änderung** - Output-Format bleibt identisch:
```python
{
    "decision": str,
    "reason": str,
    "severity": str,
    "document_id": str | None,
    "version": str | None,
    "telemetry": dict,
    "transitions": dict,
}
```

---

## Success Criteria

- [ ] `UploadIngestionState` TypedDict definiert
- [ ] 4 Node-Funktionen implementiert (validate, build_config, run_processing, map_results)
- [ ] Graph kompiliert als module-level `graph`
- [ ] Alle Nodes haben `@observe_span` decorator
- [ ] Error-Handling via Return-Values
- [ ] Context-Injection für Runtime-Dependencies
- [ ] Tests laufen (>90% Coverage)
- [ ] Caller (`ai_core/services/__init__.py`) migriert
- [ ] alte `UploadIngestionGraph` Klasse entfernt
- [ ] Documentation aktualisiert

---

## Timeline

**Gesamt-Aufwand:** ~12 Stunden

| Phase | Aufwand | Beschreibung |
|-------|---------|--------------|
| Phase 1: State & Protocols | 2h | TypedDict + Protocols definieren |
| Phase 2: Node-Implementierung | 4h | 4 Nodes + Error-Handling |
| Phase 3: Graph-Definition | 1h | StateGraph + Compilation |
| Phase 4: Caller-Migration | 2h | Update ai_core/services |
| Phase 5: Test-Migration | 2h | Update + neue Tests |
| Phase 6: Cleanup & Docs | 1h | Cleanup + README |

**Empfehlung: 2 Sessions à 6h**

---

## Dateien-Übersicht

**Zu ändern:**
- `ai_core/graphs/upload_ingestion_graph.py` - **Komplett neu strukturiert**
- `ai_core/services/__init__.py` - Caller-Update (Zeile 1780-1810)
- `ai_core/tests/graphs/test_upload_ingestion_graph.py` - Test-Migration
- `ai_core/graphs/README.md` - Dokumentation
- `AGENTS.md` - Graph-Übersicht

**Referenz:**
- `ai_core/graphs/external_knowledge_graph.py` - **Vorbild**
- `documents/processing_graph.py` - Inner Graph (unverändert)

---

**Ende des Plans v2**
