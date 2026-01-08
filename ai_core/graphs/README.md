# Graph-Orchestrierung im AI Core

Die Graphen im `ai_core` kapseln Business- und technische Orchestrierung.
Business-Graphs (z.B. `framework_analysis_graph`) nutzen die GraphNode/GraphTransition-DSL,
waehrend technische Graphen ueber LangGraph gebaut sind und typisierte Outputs liefern.
Nicht jeder Graph verwendet `GraphTransition` oder die DSL, daher gilt die DSL-Beschreibung
nur fuer die entsprechenden Business-Graphs.

## Context & Identity (Pre-MVP ID Contract)

Graphs receive context via `normalize_meta()` in `ai_core/graph/schemas.py`. The
graph meta carries three structured contexts:

- **ScopeContext** (`scope_context`): `tenant_id`, `trace_id`, `invocation_id`,
  `run_id` and/or `ingestion_run_id`, `user_id` or `service_id`, `tenant_schema`,
  `idempotency_key`, `timestamp`
- **BusinessContext** (`business_context`): `case_id`, `collection_id`,
  `workflow_id`, `document_id`, `document_version_id`
- **ToolContext** (`tool_context`): Composition of scope + business + runtime metadata

Canonical runtime context injection pattern:

1. Boundary builds meta via `normalize_meta(request)`.
2. Graph entry parses `ToolContext` once via `tool_context_from_meta(meta)`.
3. Persist the validated context in state (e.g. `state["tool_context"] = context`).
4. Nodes read IDs from `context.scope.*` and `context.business.*`.

Do not re-derive IDs from headers or implicit state; context is the single
source of truth.

For entity persistence within graphs, use `audit_meta_from_scope()`:

```python
from ai_core.contracts.audit_meta import audit_meta_from_scope

# In a graph node that creates entities:
tool_context = state["tool_context"]
scope = tool_context.scope
audit_meta = audit_meta_from_scope(
    scope,
    created_by_user_id=scope.user_id,
    initiated_by_user_id=tool_context.metadata.get("initiated_by_user_id"),
)
entity.audit_meta = audit_meta.model_dump()
```

This ensures all persisted entities include traceability fields (`trace_id`, `invocation_id`, `created_by_user_id`, `last_hop_service_id`).

## Graph I/O contracts (mandatory)

All graphs declare versioned Pydantic input/output models and attach a `GraphIOSpec`
(`ai_core/graph/io.py`) to the compiled graph or graph class:

- Boundary payloads include `schema_id` + `schema_version`.
- Input/output are validated at the graph boundary.
- The `io_spec` is discoverable by registries and executors.

Legacy graphs without `io_spec` are tracked for migration in `roadmap/backlog.md`.


## DSL (business graphs)

Diese DSL wird aktuell vom `framework_analysis_graph` verwendet; technische Graphen nutzen LangGraph.

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

## Collection-Search-Graph

Der `collection_search`-Graph koordiniert Query-Expansion,
Websuche, Hybrid-Reranking, HITL-Review und Ingestion für beliebige Dokumentationstypen
innerhalb eines Tenants. Er nutzt den Worker-Graph
`hybrid_search_and_score`, um Web-Treffer und vorhandene RAG-Daten zu
verschmelzen und erzeugt dabei eine deterministische Telemetrie pro Knoten.

Die Knotenfolge lautet:

1. **strategy** - erzeugt Query-Expansions und Policies (LLM oder Fallback).
2. **search** - fuehrt Websuche ueber den konfigurierten Worker aus.
3. **embedding_rank** - berechnet heuristische Scores und bereitet Kandidaten vor.
4. **hybrid_score** - ruft `hybrid_search_and_score` fuer Re-Ranking auf.
5. **hitl** - baut HITL-Payload und Entscheidungspfad.
6. **build_plan** - erstellt den Plan (z.B. `CollectionSearchPlan`).
7. **delegate** - fuehrt den Plan optional ueber den Universal Ingestion Graph aus.

Der ehemalige Coverage-Verification-Schritt ist derzeit deaktiviert.

Die Tests in
`ai_core/tests/graphs/test_collection_search_graph.py`
validieren einen End-to-End-Lauf mit Mock-Komponenten sowie die HITL-Warte- und
Fehlerpfade.

## Framework-Analysis-Graph

Der `framework_analysis_graph` analysiert Rahmenvereinbarungen (KBV/GBV/BV/DV) im
Kontext der IT-Mitbestimmung und extrahiert automatisiert deren Struktur. Das
System identifiziert KI-first die vier Pflicht-Bausteine (Systembeschreibung,
Funktionsbeschreibung, Auswertungen, Zugriffsrechte) und kartiert ihre Position
in Hauptdokument oder Anlagen.

Der Graph unterstützt Multi-Versionierung: Bestehende Rahmenvereinbarungen
bleiben aktiv für laufende Fälle, während neue Versionen parallel erfasst werden
können. Jeder Fall bindet sich permanent an die bei Erstellung aktuelle
Framework-Version.

### Architektur-Highlights

* **AI-First-Ansatz**: Typ-Erkennung (KBV/GBV/BV/DV), Gremium-Extraktion und
  Komponenten-Lokalisierung erfolgen vollständig LLM-gestützt
* **Hybrid Search**: Nutzt `retrieve.run()` als Worker für semantische +
  lexikalische Suchen im bestehenden RAG-System
* **HITL-Trigger**: Niedrige Konfidenz oder fehlgeschlagene Validierung
  triggert automatisch Human-in-the-Loop-Review
* **Multi-Version-Support**: Incrementelle Versionierung mit `is_current`-Flag,
  alte Profile bleiben erhalten
* **Transaktionale Persistierung**: Atomare DB-Operationen für
  FrameworkProfile + FrameworkDocument

### Knotenfolge

Die sequentielle Ausführung umfasst 7 Knoten:

1. **detect_type_and_gremium** — Fetcht Dokumentanfang (2000 Zeichen) via
   `retrieve.run()`, analysiert Typ (KBV/GBV/BV/DV) und Gremium-Identifier
   (z. B. „KBR", „GBR_MUENCHEN") mittels LLM. Normalisiert Identifier (Umlaute,
   Sonderzeichen → Unterstriche).

2. **extract_toc** — Aggregiert Parent-Metadaten aus bis zu 100 Chunks, baut
   hierarchisches Inhaltsverzeichnis (Headings + Sections), sortiert nach
   `level` und `order`.

3. **locate_components** — Führt 4 parallele semantische Suchen für die
   Pflicht-Bausteine durch. LLM kartiert Fundstellen auf Location-Typen: `main`
   (Hauptdokument), `annex` (einzelne Anlage), `annex_group` (Anlagen-Gruppe
   wie „Anlage 3 → 3.1, 3.2"), `not_found`. Liefert `outline_path`, `chunk_ids`,
   `page_numbers`, `confidence`.

4. **validate_components** — Konfidenz-basierte Plausibilitätsprüfung: Prüft,
   ob gefundener Inhalt tatsächlich der erwarteten Semantik entspricht (z. B.
   Systembeschreibung = technische Spezifikationen). Setzt `validated`-Flag und
   `validation_notes`.

5. **assemble_profile** — Merged Located + Validated Components, berechnet
   `completeness_score` (0.0–1.0, 4 Komponenten = 1.0). Triggert HITL wenn:
   Konfidenz < Threshold ODER Validierung fehlgeschlagen. Loggt Warning mit
   `hitl_reasons`.

6. **finish** - Bundelt finales `FrameworkAnalysisDraft` (nur Orchestrierung).
   Persistierung erfolgt nachgelagert ueber die Service Boundary.

### Datenmodelle

**FrameworkProfile** (Django ORM):

```python
id: UUID (PK)
tenant: FK → customers.Tenant
gremium_identifier: str  # Normalisiert: "KBR", "GBR_MUENCHEN"
gremium_name_raw: str    # Original: "Konzernbetriebsrat der Telefónica"
version: int
is_current: bool
agreement_type: str      # kbv|gbv|bv|dv
structure: JSONField     # 4 Komponenten mit location/path/chunks/pages
analysis_metadata: JSONField
```

**Unique Constraint**: `(tenant, gremium_identifier, version)`

**FrameworkDocument** (M2M-Link):

```python
profile: FK → FrameworkProfile
document_collection: FK → DocumentCollection
document_id: UUID
document_type: main|annex|protocol|amendment
position: int
```

### Contracts & Validierung

Alle Inputs/Outputs nutzen Pydantic mit `frozen=True`, `extra="forbid"`:

* **FrameworkAnalysisInput**: `document_collection_id`, optional `document_id`,
  `force_reanalysis`, `confidence_threshold` (default 0.70)
* **FrameworkAnalysisOutput**: `profile_id`, `version`, `gremium_identifier`,
  `structure: FrameworkStructure`, `completeness_score`, `missing_components`,
  `hitl_required`, `hitl_reasons`, `analysis_metadata`
* **ComponentLocation**: `location` (Literal), `outline_path`, `chunk_ids`,
  `page_numbers`, `confidence` (0.0–1.0)
* **AssembledComponentLocation**: Erweitert ComponentLocation um `validated`,
  `validation_notes`, `annex_root`, `subannexes`

Confidenz-Bounds werden validiert, Location-Typen sind typsicher.

### API-Endpunkt

**POST** `/v1/ai/frameworks/analyze/`

Header:

* `X-Tenant-ID` (required)
* `X-Trace-ID` (optional, auto-generated)

Request Body:

```json
{
  "document_collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_id": "650e8400-e29b-41d4-a716-446655440001",
  "force_reanalysis": false,
  "confidence_threshold": 0.70
}
```

Response (200):

```json
{
  "profile_id": "750e8400-e29b-41d4-a716-446655440002",
  "version": 1,
  "gremium_identifier": "KBR",
  "completeness_score": 0.75,
  "missing_components": ["zugriffsrechte"],
  "hitl_required": false,
  "hitl_reasons": [],
  "structure": {
    "systembeschreibung": {
      "location": "main",
      "outline_path": "2",
      "heading": "§ 2 Systembeschreibung",
      "chunk_ids": ["chunk1"],
      "page_numbers": [2, 3],
      "confidence": 0.92,
      "validated": true,
      "validation_notes": "Plausible"
    }
  },
  "analysis_metadata": {
    "detected_type": "kbv",
    "type_confidence": 0.95,
    "gremium_name_raw": "Konzernbetriebsrat",
    "completeness_score": 0.75,
    "missing_components": ["zugriffsrechte"],
    "analysis_timestamp": "2025-01-15T10:30:00Z",
    "model_version": "framework_analysis_v1"
  }
}
```

Error-Codes:

* **400**: Invalid input (Validierung fehlgeschlagen)
* **403**: Tenant nicht gefunden
* **404**: DocumentCollection nicht gefunden
* **409**: Profil existiert bereits (force_reanalysis=true nötig)
* **500**: Analyse fehlgeschlagen (LLM-Fehler, DB-Fehler)

### Observability

**Langfuse-Spans** (via `@observe_span`):

* `framework.detect_type_and_gremium`
* `framework.extract_toc`
* `framework.locate_components`
* `framework.validate_components`
* `framework.assemble_profile`

**Strukturierte Logs**:

* `framework_graph_starting` (info): Start mit Input-Parametern
* `framework_hitl_required` (warning): HITL-Trigger mit Gründen
* `framework_profile_persisted` (info): Erfolgreiche Persistierung
* `framework_graph_completed` (info): Abschluss mit Metriken

Alle Logs enthalten `tenant_id`, `trace_id`, `gremium_identifier` für
korrelierte Auswertung in ELK + Langfuse.

### Tests

**Unit-Tests** (`ai_core/tests/test_framework_utils.py`):

* Gremium-Normalisierung (Umlaute, Sonderzeichen, Leerzeichen)
* ToC-Extraktion aus Parent-Metadaten (Hierarchie, Deduplication, Sorting)

**Contract-Tests** (`ai_core/tests/tools/test_framework_contracts.py`):

* Pydantic-Validierung für alle Input/Output-Modelle
* Confidenz-Bounds (0.0–1.0)
* Literal-Constraints für `agreement_type`, `location`
* Immutability (frozen Models)

**Integrations-Tests** (`ai_core/tests/graphs/test_framework_analysis_graph.py`):

* Graph-Builder (Knoten-Reihenfolge, Namen)
* Assemble-Logik (completeness_score, HITL-Trigger)
* End-to-End mit LLM-Mocks (retrieve.run, llm_client.call)
* Error-Handling (Retrieve-Fehler, Validierungs-Fehler)

### Deployment-Hinweise

**Django-Migrationen**: Nach Merge müssen Devs im Docker-Container ausführen:

```bash
docker compose exec web python manage.py makemigrations documents
docker compose exec web python manage.py migrate
```

**Prompt-Versions-Tracking**: Prompts liegen in `ai_core/prompts/framework/`:

* `detect_type_gremium.v1.md`
* `locate_components.v1.md`
* `validate_component.v1.md`

Bei Prompt-Updates neue Version anlegen (v2, v3, ...) und `load_prompt()`-Call
in Graph aktualisieren.

**LLM-Model-Version**: Aktuell fest `"framework_analysis_v1"` in
`analysis_metadata`. Bei Modell-Upgrades Version inkrementieren für A/B-Tests.

**HITL-Integration**: UI-Team muss `hitl_required` Flag auswerten und
Review-Flow bereitstellen. `hitl_reasons` liefert Begründungen für Benutzer.

### Universal Technical Graph (Pre-MVP)

Der `UniversalIngestionGraph` (`universal_ingestion_graph.py`) implementiert die konsolidierte Ingestion-Logik für `upload` und `crawler` Quellen. Er ersetzt die bisherigen Upload- und Crawler-Graphen.

**Zweck**:
Einheitlicher technischer Zugangspunkt für die Dokumentenverarbeitung unabhängig von der Quelle. In Phase 2/3 unterstützt er `ingest_only` für manuelle Uploads und Crawler-Runs.

### Contract

Boundary models live in `ai_core/graphs/technical/universal_ingestion_graph.py`:

**Input**: `UniversalIngestionGraphInput` (Pydantic)

* `schema_id`: `noesis.graphs.universal_ingestion`
* `schema_version`: `1.0.0`
* `input.normalized_document`: required `NormalizedDocument` (dict or object)
* `context`: serialized `ToolContext` (scope + business)

**Output**: `UniversalIngestionGraphOutput` (Pydantic)

* `schema_id`: `noesis.graphs.universal_ingestion`
* `schema_version`: `1.0.0`
* `decision`: `processed` | `skipped` | `failed`
* `reason_code`: `DUPLICATE` | `VALIDATION_ERROR` | `PERSISTENCE_ERROR` | `PROCESSING_ERROR` | null
* `reason`: Mensch-lesbarer Grund
* `document_id`: UUID des persistierten Dokuments
* `ingestion_run_id`: Verknuepfung zum Run
* `telemetry`: Trace- und Tenant-Kontext
* `formatted_status`: Legacy kompatibles Feld

### Knotenfolge

1. **validate_input**
   Prueft graph input, ToolContext und `normalized_document`. Validiert `collection_id` im BusinessContext.

2. **dedup**
   Platzhalter fuer Dokument-Deduplication (derzeit immer `new`).

3. **persist**
   Speichert das normalisierte Dokument initial im `DocumentRepository` (Upsert). Dies sichert die Daten vor der Verarbeitung.

4. **process**
   Delegiert die inhaltliche Verarbeitung (Parsing, Chunking, Embedding) an den shared `document_processing_graph`.
   * Injiziert `DocumentProcessingContext` und `DocumentPipelineConfig`.
   * Injiziert `Storage`-Service fuer Blob-Zugriffe.

5. **finalize**
   Mappt das Ergebnis auf `UniversalIngestionGraphOutput` und sammelt Telemetrie.

### Migration Status

In Phase 3 sind alle Call-Sites (Upload UI, Crawler Services) auf diesen Graphen migriert. Die Legacy-Graphen (`upload_ingestion_graph`, `crawler_ingestion_graph`) wurden entfernt.
