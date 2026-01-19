# Framework Analysis Graph Convergence

**Status:** In Planning
**Target:** Pre-MVP (Breaking Changes erlaubt)
**Referenz-Implementierungen:**
- [collection_search.py](../ai_core/graphs/technical/collection_search.py)
- [retrieval_augmented_generation.py](../ai_core/graphs/technical/retrieval_augmented_generation.py)
- [web_acquisition_graph.py](../ai_core/graphs/technical/web_acquisition_graph.py)

## Ziel

Migration des Framework Analysis Business-Graphen von der Custom DSL (`GraphNode` + Sequential Loop) zur LangGraph `StateGraph`-Architektur der technischen Graphen. Einheitliche Patterns für Error Handling, Observability und Service Abstraction.

---

## Ist-Zustand (Analyse)

### Architektur-Gaps

| Aspekt | Technical Graphs | Framework Analysis | Gap |
|--------|------------------|-------------------|-----|
| **Framework** | LangGraph `StateGraph` | Custom DSL | Komplett anders |
| **State** | `TypedDict` + Reducers | `Dict[str, Any]` | Keine Type Safety |
| **Execution** | Compiled `.invoke()` | Sequential Loop | Kein Parallelismus |
| **Error Handling** | Strukturierte Payloads | Exception → Abbruch | Keine Graceful Degradation |
| **Async** | `asyncio.gather`, Timeouts | Synchron | Keine Parallelität |
| **Services** | Protocols + DI | Direkte Calls | Nicht testbar |
| **Observability** | `emit_event()` + Reducers | `logger.info()` | Weniger granular |

### Aktuelle Datei-Struktur

```
ai_core/graphs/business/
├── __init__.py
└── framework_analysis_graph.py  # 558 Zeilen, Custom DSL
```

### Code-Referenzen (Ist-Zustand)

- **State Definition:** [framework_analysis_graph.py:32](../ai_core/graphs/business/framework_analysis_graph.py#L32) - `StateMapping = Dict[str, Any]`
- **Node DSL:** [framework_analysis_graph.py:70-85](../ai_core/graphs/business/framework_analysis_graph.py#L70-L85) - `GraphNode` dataclass
- **Sequential Execution:** [framework_analysis_graph.py:186-192](../ai_core/graphs/business/framework_analysis_graph.py#L186-L192) - Loop über nodes
- **Error Handling:** [framework_analysis_graph.py:257-320](../ai_core/graphs/business/framework_analysis_graph.py#L257-L320) - Exception → Graph-Abbruch

---

## Soll-Zustand

### Ziel-Architektur

```
ai_core/graphs/business/
├── __init__.py
├── framework_analysis/
│   ├── __init__.py
│   ├── graph.py           # LangGraph StateGraph
│   ├── state.py           # TypedDict + Reducers
│   ├── nodes.py           # Node-Funktionen
│   ├── io.py              # I/O Spezifikationen (schema_id, schema_version)
│   └── protocols.py       # Service Abstractions
└── framework_analysis_graph.py  # Deprecated wrapper (optional)
```

### Pattern-Übernahme von Technical Graphs

**Von Collection Search:**
- `TypedDict` State mit `Annotated[..., reducer]` ([collection_search.py:359-389](../ai_core/graphs/technical/collection_search.py#L359-L389))
- `_get_ids()` Helper für ID-Propagation ([collection_search.py:563-577](../ai_core/graphs/technical/collection_search.py#L563-L577))
- `_record_transition()` für strukturiertes Tracking ([collection_search.py:537-554](../ai_core/graphs/technical/collection_search.py#L537-L554))
- Protocols für Service Abstraction ([collection_search.py:277-326](../ai_core/graphs/technical/collection_search.py#L277-L326))
- Boundary Validation mit schema_version ([collection_search.py:203-214](../ai_core/graphs/technical/collection_search.py#L203-L214))

**Von RAG Graph:**
- `emit_event()` Integration ([retrieval_augmented_generation.py:495-500](../ai_core/graphs/technical/retrieval_augmented_generation.py#L495-L500))
- `update_observation()` für Metriken ([retrieval_augmented_generation.py:672-678](../ai_core/graphs/technical/retrieval_augmented_generation.py#L672-L678))

---

## Implementierungsplan

### Phase 1: Foundation (State & I/O)

#### FA-1.1: TypedDict State mit Reducers
- **Task:** Migriere `StateMapping = Dict[str, Any]` zu `FrameworkAnalysisState(TypedDict)`
- **Akzeptanz:**
  - Alle State-Keys explizit typisiert
  - Reducer für `transitions`, `components`, `errors`
  - Immutable Input-Felder (`input`, `tool_context`)
- **Pointers:**
  - Neu: `ai_core/graphs/business/framework_analysis/state.py`
  - Referenz: [collection_search.py:333-389](../ai_core/graphs/technical/collection_search.py#L333-L389)

#### FA-1.2: I/O Spezifikation mit Boundary Validation
- **Task:** Extrahiere I/O Models in separates Modul mit strikter schema_version Validation
- **Akzeptanz:**
  - `FrameworkAnalysisGraphInput` enforced schema_version (nicht nur Default)
  - `FrameworkAnalysisGraphOutput` mit allen Feldern typisiert
  - `FRAMEWORK_ANALYSIS_IO` Spec exportiert
  - Boundary Validator wie [collection_search.py:203-214](../ai_core/graphs/technical/collection_search.py#L203-L214)
- **Pointers:**
  - Neu: `ai_core/graphs/business/framework_analysis/io.py`
  - Aktuell: [framework_analysis_graph.py:34-68](../ai_core/graphs/business/framework_analysis_graph.py#L34-L68)

### Phase 2: Service Abstraction (Protocols)

#### FA-2.1: FrameworkRetrieveService Protocol
- **Task:** Extrahiere `retrieve.run()` Calls in abstrahierten Service
- **Akzeptanz:**
  - `FrameworkRetrieveService` Protocol mit `retrieve()` Method
  - Production Implementation mit echtem `retrieve.run()`
  - Mock Implementation für Tests
  - Dependency Injection via `runtime` Dict
- **Pointers:**
  - Neu: `ai_core/graphs/business/framework_analysis/protocols.py`
  - Calls: [framework_analysis_graph.py:259-268](../ai_core/graphs/business/framework_analysis_graph.py#L259-L268)

#### FA-2.2: FrameworkLLMService Protocol
- **Task:** Extrahiere `call_llm_json_prompt()` Calls in abstrahierten Service
- **Akzeptanz:**
  - `FrameworkLLMService` Protocol mit `detect_type()`, `extract_components()`, etc.
  - Timeout-Management pro Call
  - Retry-Mechanismus (Optional: Exponential Backoff)
- **Pointers:**
  - LLM Calls: [framework_analysis_graph.py:269-285](../ai_core/graphs/business/framework_analysis_graph.py#L269-L285)
  - Referenz Retry: [collection_search.py:945-979](../ai_core/graphs/technical/collection_search.py#L945-L979)

### Phase 3: LangGraph Migration

#### FA-3.1: Node-Funktionen extrahieren
- **Task:** Konvertiere `_detect_type_and_gremium`, `_extract_components`, etc. zu standalone Funktionen
- **Akzeptanz:**
  - Funktionen in `nodes.py` mit `@observe_span` Decorator
  - Jede Funktion nimmt `FrameworkAnalysisState`, returned `dict[str, Any]`
  - `_record_transition()` Helper wie in Collection Search
- **Pointers:**
  - Aktuell: [framework_analysis_graph.py:252-520](../ai_core/graphs/business/framework_analysis_graph.py#L252-L520)
  - Referenz: [collection_search.py:652-1400](../ai_core/graphs/technical/collection_search.py#L652-L1400)

#### FA-3.2: StateGraph Construction
- **Task:** Ersetze Sequential Loop durch LangGraph `StateGraph`
- **Akzeptanz:**
  - `workflow = StateGraph(FrameworkAnalysisState)`
  - Explicit Edges zwischen Nodes
  - Conditional Edges für Fehler-Handling (HITL Fallback statt Abbruch)
  - Compiled Graph mit `.invoke()` / `.ainvoke()`
- **Pointers:**
  - Aktuell: [framework_analysis_graph.py:186-192](../ai_core/graphs/business/framework_analysis_graph.py#L186-L192)
  - Referenz: [collection_search.py:1529-1551](../ai_core/graphs/technical/collection_search.py#L1529-L1551)

### Phase 4: Error Handling & Resilience

#### FA-4.1: Strukturierte Fehler-Payloads
- **Task:** Ersetze Exception → Abbruch durch strukturierte Fehler im State
- **Akzeptanz:**
  - `FrameworkAnalysisError` Dataclass (analog zu `SearchError`)
  - Fehler werden in `state["errors"]` gesammelt
  - Graph läuft bis zum Ende (Graceful Degradation)
  - Output enthält `partial_results` + `errors`
- **Pointers:**
  - Aktuell: [framework_analysis_graph.py:314-320](../ai_core/graphs/business/framework_analysis_graph.py#L314-L320)
  - Referenz: [collection_search.py:697-770](../ai_core/graphs/technical/collection_search.py#L697-L770)

#### FA-4.2: Timeout-Management
- **Task:** Füge Timeouts für LLM-Calls und Retrieve-Operationen hinzu
- **Akzeptanz:**
  - Per-Node Timeout konfigurierbar via `runtime`
  - Gesamter Graph-Timeout (Worker-Safe, kein `signal.alarm`)
  - Timeout → Graceful Degradation (nicht Abbruch)
- **Pointers:**
  - Referenz: [collection_search.py:772-874](../ai_core/graphs/technical/collection_search.py#L772-L874) (Async mit Timeout)

### Phase 5: Async & Parallelismus (Optional)

#### FA-5.1: Async Node Support
- **Task:** Konvertiere sync Nodes zu async für parallele LLM-Calls
- **Akzeptanz:**
  - `_extract_components` kann mehrere Komponenten parallel analysieren
  - `asyncio.gather` für unabhängige Operations
  - Cancellation bei Timeout
- **Pointers:**
  - Referenz: [collection_search.py:804-823](../ai_core/graphs/technical/collection_search.py#L804-L823)

### Phase 6: Observability & Telemetry

#### FA-6.1: emit_event() Integration
- **Task:** Ersetze `logger.info()` durch strukturierte Events
- **Akzeptanz:**
  - `emit_event()` für Milestone-Events (graph_started, type_detected, components_extracted, etc.)
  - `update_observation()` für Metriken (latency, token_count, etc.)
  - Telemetry Reducer im State
- **Pointers:**
  - Aktuell: [framework_analysis_graph.py:157-168](../ai_core/graphs/business/framework_analysis_graph.py#L157-L168)
  - Referenz: [retrieval_augmented_generation.py:495-500](../ai_core/graphs/technical/retrieval_augmented_generation.py#L495-L500)

#### FA-6.2: _get_ids() Helper
- **Task:** Zentralisierte ID-Extraction für konsistente Propagation
- **Akzeptanz:**
  - `_get_ids(tool_context)` returned alle relevanten IDs
  - Alle Nodes nutzen denselben Helper
  - IDs in jedem `_record_transition()` Call
- **Pointers:**
  - Referenz: [collection_search.py:563-577](../ai_core/graphs/technical/collection_search.py#L563-L577)

---

## Migration-Strategie

### Breaking Changes (erlaubt da Pre-MVP)

1. **State Shape:** `Dict[str, Any]` → `TypedDict`
2. **Output Format:** Feld-Änderungen in `FrameworkAnalysisGraphOutput`
3. **Error Handling:** Graph läuft durch statt Abbruch
4. **Service Calls:** Direkte Calls → Protocol Injection

### Backward Compatibility (Optional)

Falls gewünscht, kann ein Wrapper `framework_analysis_graph.py` die alte API beibehalten:

```python
# Deprecated wrapper für Übergangsphase
def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
    warnings.warn("Use FrameworkAnalysisAdapter.run() instead", DeprecationWarning)
    return FrameworkAnalysisAdapter().run(state)
```

---

## Abhängigkeiten

- **Keine externen Abhängigkeiten** - LangGraph bereits im Stack
- **Interne Abhängigkeiten:**
  - `ai_core/graph/io.py` (GraphIOSpec, GraphIOVersion)
  - `ai_core/tools/context.py` (ToolContext)
  - `ai_core/nodes/retrieve.py` (Retrieve Service)

---

## Tests

### Unit Tests
- State Reducers (merge behavior)
- I/O Validation (schema_version enforcement)
- Error Payload construction

### Integration Tests
- Full Graph execution mit Mock Services
- Graceful Degradation bei Service-Fehlern
- Timeout handling

### E2E Tests (Optional)
- Real LLM Calls mit Langfuse Traces
- Performance Baseline

---

## Effort Estimate

| Phase | Tasks | Komplexität |
|-------|-------|-------------|
| Phase 1 | FA-1.1, FA-1.2 | S-M |
| Phase 2 | FA-2.1, FA-2.2 | M |
| Phase 3 | FA-3.1, FA-3.2 | M-L |
| Phase 4 | FA-4.1, FA-4.2 | M |
| Phase 5 | FA-5.1 | S (Optional) |
| Phase 6 | FA-6.1, FA-6.2 | S |

**Gesamt:** ~2-3 Sprints (abhängig von Async-Scope)

---

## Referenzen

- [Collection Search Review](collection-search-review.md)
- [AGENTS.md - Tool-Verträge](../AGENTS.md#tool-verträge-layer-2--norm)
- [Graph I/O Spec](../ai_core/graph/io.py)
