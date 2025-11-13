# Graph-Orchestrierung im AI Core

Die Graphen im `ai_core` kapseln unsere Business-Orchestrierung. Jeder Graph
besteht aus benannten Knoten (`GraphNode`), die eine State-Mutation ausführen und
ein standardisiertes Übergabeobjekt (`GraphTransition`) zurückgeben. Die
Transitions enthalten `decision`, `reason` und ein `attributes`-Mapping und
ermöglichen damit konsistente Auswertung in Tests, Telemetrie und im
Observability-Layer.

## Übersicht

NOESIS 2 enthält folgende Produktions-Graphen:

| Graph | Zweck | Status |
|-------|-------|--------|
| `retrieval_augmented_generation` | Produktions-RAG-Flow (retrieve → compose) | ✅ Produktiv |
| `upload_ingestion_graph` | Upload-Processing mit Guardrails & Delta | ✅ Produktiv |
| `crawler_ingestion_graph` | Crawler-basierte Ingestion | ✅ Produktiv |
| `software_documentation_collection` | Dokumentations-Akquise mit Hybrid-Reranking | ✅ Produktiv |
| `external_knowledge_graph` | Web-Suche, Selektion & Ingestion | ✅ Produktiv |
| `info_intake` | Meta-Recording (Demo/Test) | 🧪 Experimental |
| `rag_demo` | Legacy-RAG-Demo | ⚠️ Deprecated |

Zusätzliche Module:
- `cost_tracking.py` - Kostentracking-Utilities
- `document_service.py` - Service-Protokolle für Dokumenten-Lifecycle
- `transition_contracts.py` - Transition-Datenverträge

## DSL

```python
@dataclass(frozen=True)
class GraphNode:
    name: str
    runner: Callable[[dict[str, Any]], tuple[GraphTransition, bool]]
```

* Der Runner erhält ein `dict[str, Any]`, das den Crawl-/RAG-Kontext,
  Zwischenartefakte und Kontrollflags kapselt.
* Jeder Knoten darf das Mapping in-place erweitern und liefert gemeinsam mit dem
  Transition-Objekt ein `bool`, das den Kontrollfluss steuert.

Ein `GraphTransition` validiert seine Felder beim Erzeugen und erzwingt
mindestens die Severity `info`. Dadurch können nachgelagerte Komponenten die
Transitions ohne zusätzliche Guards interpretieren.

## Crawler-Ingestion-Graph

Der `crawler_ingestion_graph` orchestriert den Crawler-Workflow in der
deterministischen Sequenz
`update_status_normalized → enforce_guardrails → document_pipeline → ingest_decision → ingest → finish`.

* **update_status_normalized** sorgt für initiale Status-Updates des Dokuments.
* **enforce_guardrails** prüft mithilfe von `ai_core.api.enforce_guardrails`,
  ob Inhalte verarbeitet werden dürfen.
* **document_pipeline** führt Normalisierung und Chunking durch; sie nutzt
  `ai_core.api.decide_delta`, um Delta-Entscheidungen zu treffen.
* **ingest_decision** entscheidet über UPSERT- oder RETIRE-Pfade basierend auf
  den Guardrail- und Delta-Ergebnissen.
* **ingest** persistiert Daten und löst `ai_core.api.trigger_embedding` aus, um
  Embeddings für gültige Dokumente zu erzeugen.
* **finish** bündelt den Abschlussstatus und Telemetrie-Metadaten.

   Die Tests in `ai_core/tests/graphs/test_crawler_ingestion_graph.py` decken
   Guardrail-, Lifecycle- und Embedding-Entscheidungen ab, während
   `ai_core/tests/test_crawler_delta.py` die Delta-Heuristiken prüft.

Jeder Schritt emittiert eine Transition mit deterministischer Struktur. Diese
Transitions landen im aggregierten Ergebnis des Graph-Laufs und bilden die
Grundlage für Unit-Tests sowie Langfuse-Spans.

## Software-Documentation-Collection-Graph

Der `software_documentation_collection`-Graph koordiniert Query-Expansion,
Websuche, Hybrid-Reranking, HITL-Review und Ingestion für Software-Dokumentation
innerhalb eines Tenants. Er nutzt den Worker-Graph
`hybrid_search_and_score`, um Web-Treffer und vorhandene RAG-Daten zu
verschmelzen und erzeugt dabei eine deterministische Telemetrie pro Knoten.

Die Knotenfolge lautet:

1. **generate_search_strategy** — erzeugt Query-Expansions und wendet
   Tenant-Policies an.
2. **parallel_web_search** — sammelt Treffer aus 3–7 Websuchen (derzeit
   sequentiell).
3. **execute_hybrid_score** — ruft den Worker-Graph mit einem
   `ScoringContext` (jurisdiction = DE) auf.
4. **hitl_approval_gate** — baut die Review-Payload samt Scores, Gap-Tags und
   Coverage-Delta.
5. **trigger_ingestion** — übergibt approvte URLs an den Crawler.
6. **verify_coverage** — pollt den Coverage-Status (30 s Intervall, 10 min
   Timeout).

Die Tests in
`ai_core/tests/graphs/test_software_documentation_collection_graph.py`
validieren einen End-to-End-Lauf mit Mock-Komponenten sowie die HITL-Warte- und
Fehlerpfade.

## Retrieval-Augmented-Generation-Graph

Der `retrieval_augmented_generation`-Graph implementiert den produktiven
RAG-Flow mit zwei Hauptknoten: `retrieve` und `compose`. Er ist das zentrale
Herzstück der Query-Verarbeitung in NOESIS 2.

**Knotenfolge**:
1. **retrieve** — Nutzt `ai_core.nodes.retrieve.run()`, um relevante Snippets
   aus dem RAG-Store zu holen. Verwendet `ToolContext` für Multi-Tenancy und
   `RetrieveInput` für Query-Parameter.
2. **compose** — Führt `ai_core.nodes.compose.run()` aus, um die Retrieved
   Snippets mit einem LLM-Prompt zu kombinieren und eine finale Antwort zu
   generieren.

**Eingabe**: `state` (mit Query) und `meta` (mit `tenant_id`, `case_id`,
`trace_id`).

**Ausgabe**: Tuple `(final_state, result_payload)` mit:
- `answer` — Die generierte Antwort
- `prompt_version` — Verwendete Prompt-Version
- `retrieval` — Retrieval-Metadaten (z. B. `took_ms`)
- `snippets` — Liste der verwendeten Snippets mit Normalisierung

**Besonderheiten**:
- Nutzt `ToolContext` für Tenant-Isolation und Visibility-Overrides
- Normalisiert Snippets mit Fallback-Logik für fehlende Felder
- Warnt, wenn `prompt_version` fehlt
- Singleton-Instanz `GRAPH` für Shared Use

Tests: `ai_core/tests/graphs/test_retrieval_augmented_generation.py`

## Upload-Ingestion-Graph

Der `upload_ingestion_graph` orchestriert die Verarbeitung hochgeladener
Dokumente mit vollständiger Guardrail- und Delta-Prüfung. Implementiert als
`UploadIngestionGraph`-Klasse mit konfigurierbaren Hooks.

**Knotenfolge**:
1. **accept_upload** — Validiert Input (Größe, MIME-Type, Pflichtfelder),
   erzeugt `content_hash`.
2. **quarantine_scan** — Optional: Malware-/Inhalts-Scan (konfigurierbar).
3. **deduplicate** — Prüft `content_hash` gegen In-Memory-Index, setzt
   Versionsnummer bei `source_key`.
4. **parse** — Extrahiert Text aus Binary (UTF-8/Latin-1 Fallback).
5. **normalize** — Normalisiert Whitespace im extrahierten Text.
6. **delta_and_guardrails** — Führt Guardrail- und Delta-Checks aus:
   - Guardrails: Inhaltsprüfung (PII, Policy-Events)
   - Delta: Vergleich mit Baseline (unchanged/near_duplicate/changed)
7. **persist_document** — Persistiert Dokument, erzeugt `document_id`.
8. **chunk_and_embed** — Chunking (max. 128 Wörter) und Embedding-Trigger.
9. **lifecycle_hook** — Optional: Post-Ingestion-Hook.
10. **finalize** — Aggregiert finalen Status.

**Eingabe**: Payload mit:
- `tenant_id`, `uploader_id`, `trace_id`, `ingestion_run_id` (Pflicht)
- `file_bytes` oder `file_uri`
- `filename`, `declared_mime`, `visibility`, `tags` (Optional)

**Ausgabe**: Result mit:
- `decision` — z. B. `completed`, `skip_guardrail`, `skip_delta`
- `document_id`, `version`, `snippets`, `warnings`
- `transitions` — Dict aller Knoten-Transitions
- `telemetry` — Timing pro Knoten

**Besonderheiten**:
- Deduplizierung über In-Memory-Index (Session-Scope)
- Versionierung über `source_key`
- Guardrail- und Delta-Entscheidungen nutzen `ai_core.api`
- Stop-Early bei `run_until`-Parameter
- Strukturierte `StandardTransitionResult` für jeden Knoten

Tests: `ai_core/tests/graphs/test_upload_ingestion_graph.py`

## External-Knowledge-Graph

Der `external_knowledge_graph` koordiniert Web-Suche, URL-Selektion, optionale
HITL-Review und anschließende Crawler-Ingestion für externe Wissensquellen.

**Knotenfolge**:
1. **k_search** — Führt Web-Suche über `WebSearchWorker` aus (Google Custom
   Search), speichert Ergebnisse.
2. **k_filter_and_select** — Filtert Ergebnisse (Snippet-Länge, Blocked
   Domains, Noindex-Hinweise), wählt beste Kandidat aus (prefer PDF).
3. **k_hitl_gate** (optional) — Pausiert für Human-in-the-Loop-Review:
   - Emittiert Review-Payload mit `review_token`
   - Wartet auf `review_response` mit `approved`/`override_url`
   - Validiert `override_url` gegen Blocked Domains
4. **k_trigger_ingestion** — Triggert Crawler-Ingestion über
   `CrawlerIngestionAdapter`.

**Eingabe**:
- `state`: `query`, `collection_id`, `enable_hitl`, `run_until`
- `meta`: `tenant_id`, `workflow_id`, `case_id`, `trace_id`, `run_id`

**Ausgabe**: Tuple `(state, result)` mit:
- `outcome` — z. B. `ingested`, `rejected`, `no_results`, `error`
- `document_id` — Bei erfolgreicher Ingestion
- `selected_url` — Gewählte URL
- `telemetry` — Nodes, IDs, Review-Status

**Besonderheiten**:
- Unterstützt Pause/Resume für HITL-Flow
- Blocked-Domain-Check für URL-Validierung
- PDF-Präferenz konfigurierbar
- Idempotent über State-Caching
- Strukturiertes `Transition`-Objekt pro Knoten

Tests: `ai_core/tests/graphs/test_external_knowledge_graph.py`

## Info-Intake-Graph

Der `info_intake`-Graph ist ein minimaler Demo/Test-Graph, der nur Meta-Daten
aufzeichnet und weiterreicht.

**Knotenfolge**:
- Einziger Knoten: Kopiert `meta` nach `state["meta"]`

**Eingabe**: `state` (dict), `meta` (dict mit `tenant_id`, `case_id`)

**Ausgabe**: Tuple `(new_state, result)` mit:
- `received: true`
- `tenant_id`, `case_id` aus Meta

**Zweck**: Unit-Test-Fixture, Demo für minimale Graph-Struktur.

## Deprecated: RAG-Demo-Graph

Der `rag_demo`-Graph wurde entfernt und ist seit MVP nicht mehr verfügbar.

**Status**: Deprecated, raises `RuntimeError` bei Import.

**Migration**: Nutze `retrieval_augmented_generation` für produktive RAG-Flows.

## Utilities & Contracts

### Cost Tracking (`cost_tracking.py`)

Utilities für Kostentracking während Graph-Läufen:

- **`GraphCostTracker`** — Sammelt Kostenkomponenten mit Reconciliation-IDs
  - `add_component()` — Fügt Kosten hinzu (USD)
  - `record_ledger_meta()` — Extrahiert Kosten aus Ledger-Metadaten
  - `summary()` — Aggregiert Total + Komponenten

- **`track_ledger_costs()`** — Context Manager für automatisches Tracking

**Verwendung**: In Graph-Läufen, um LLM-Costs zu aggregieren.

### Document Service (`document_service.py`)

Service-Protokolle für Dokumenten-Lifecycle in Ingestion-Graphen:

- **`DocumentLifecycleService`** — Normalisierung & Lifecycle-Updates
  - `normalize_from_raw()` — Raw → `NormalizedDocumentPayload`
  - `update_lifecycle_status()` — Persistiert Status-Transitions

- **`DocumentPersistenceService`** — Persistierung normalisierter Dokumente
  - `upsert_normalized()` — Speichert Dokument

- **`DocumentsApiLifecycleService`** — Default-Impl, delegiert an
  `documents.api`

**Verwendung**: Dependency Injection in `upload_ingestion_graph` und
`crawler_ingestion_graph`.

### Transition Contracts (`transition_contracts.py`)

Definiert `GraphTransition` und `StandardTransitionResult` für strukturierte
Knoten-Ausgaben:

- **`GraphTransition`** — Wrapper mit `decision`, `reason`, `severity`,
  `context`
- **`StandardTransitionResult`** — Pydantic-Model mit:
  - `phase`, `decision`, `reason`, `severity`
  - Optional: `guardrail`, `delta`, `lifecycle`, `embedding`

**Builder-Funktionen**:
- `build_guardrail_section()`
- `build_delta_section()`
- `build_lifecycle_section()`
- `build_embedding_section()`

**Zweck**: Konsistente Transition-Struktur für Tests und Observability.

## Best Practices

### Graph-Entwicklung

1. **State-Management**:
   - State ist `MutableMapping[str, Any]`, wird in-place modifiziert
   - Jeder Knoten erweitert State um eigene Keys (z. B. `search`, `selection`)
   - Telemetrie in `state["telemetry"]["nodes"]` speichern

2. **Transition-Struktur**:
   - Immer `GraphTransition` oder `Transition` zurückgeben
   - `decision` beschreibt Pfad-Entscheidung (z. B. `proceed`, `skip_*`, `error`)
   - `reason` oder `rationale` erklärt Entscheidung
   - `meta`/`context` enthält Diagnostics (IDs, Counts, Latencies)

3. **Observability**:
   - `@observe_span(name="graph.*.node_name")` für jeden Knoten
   - `update_observation(metadata={...})` für Span-Annotation
   - Telemetrie-IDs propagieren: `tenant_id`, `trace_id`, `run_id`, etc.

4. **Idempotenz**:
   - State-Caching: Prüfe `state.get("node_name", {}).get("completed")`
   - Vermeide redundante Aufrufe bei Pause/Resume
   - Speichere Transition in `state["node_name"]["transition"]`

5. **Error-Handling**:
   - Fange spezifische Exceptions (z. B. `SearchProviderError`)
   - Emittiere `error`-Transition mit `meta["error"]`
   - Nutze `emit_event()` für kritische Fehler

6. **Testing**:
   - Mock alle externen Abhängigkeiten (Services, APIs, LLMs)
   - Teste alle Transition-Pfade (success, skip, error)
   - Validiere Telemetrie-Struktur
   - Prüfe Idempotenz bei doppeltem Run

### Konfiguration

Graphen konfigurieren über:
- **Dependency Injection**: Services/Adapters als Constructor-Parameter
- **Config-Objekte**: `*Config`-Dataclasses für Tuning-Parameter
- **Settings**: Django-Settings für Feature-Flags (z. B.
  `UPLOAD_QUARANTINE_ENABLED`)

### Deployment

- Graphen sind stateless → horizontal skalierbar
- Runner-Funktionen über Celery-Tasks aufrufen
- State in Redis/DB für Pause/Resume-Flows
- Timeouts über `run_until`-Parameter steuern

## Weiterführende Dokumentation

- **Tool-Contracts**: [docs/agents/tool-contracts.md](../../docs/agents/tool-contracts.md)
- **RAG-Architektur**: [docs/rag/overview.md](../../docs/rag/overview.md)
- **Ingestion-Pipeline**: [docs/rag/ingestion.md](../../docs/rag/ingestion.md)
- **Observability**: [docs/observability/langfuse.md](../../docs/observability/langfuse.md)
- **LangGraph-Konzepte**: [ai_core/README.md](../README.md)

## Versionierung

Diese Dokumentation beschreibt den Stand vom **2025-11-13** und gilt für
NOESIS 2 auf Branch `main`.

**Änderungshistorie**:
- 2025-11-13: Vollständige Dokumentation aller Produktions-Graphen
- 2025-10: MVP-Release mit `retrieval_augmented_generation` als Breaking
  Contract v2
