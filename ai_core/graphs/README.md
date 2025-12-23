# Graph-Orchestrierung im AI Core

Die Graphen im `ai_core` kapseln unsere Business-Orchestrierung. Jeder Graph
besteht aus benannten Knoten (`GraphNode`), die eine State-Mutation ausführen und
ein standardisiertes Übergabeobjekt (`GraphTransition`) zurückgeben. Die
Transitions enthalten `decision`, `reason` und ein `attributes`-Mapping und
ermöglichen damit konsistente Auswertung in Tests, Telemetrie und im
Observability-Layer.

## Context & Identity (Pre-MVP ID Contract)

Graphs receive context via `scope_context` in the graph meta. The `ScopeContext` includes:

- **Correlation IDs**: `tenant_id`, `trace_id`, `invocation_id`, `case_id`, `workflow_id`
- **Runtime IDs**: `run_id` and/or `ingestion_run_id` (may co-exist when workflow triggers ingestion)
- **Identity IDs**: `user_id` (User Request Hop) or `service_id` (S2S Hop) - mutually exclusive

For entity persistence within graphs, use `audit_meta_from_scope()`:

```python
from ai_core.contracts.audit_meta import audit_meta_from_scope

# In a graph node that creates entities:
audit_meta = audit_meta_from_scope(scope, created_by_user_id=scope.user_id)
entity.audit_meta = audit_meta.model_dump()
```

This ensures all persisted entities include traceability fields (`trace_id`, `invocation_id`, `created_by_user_id`, `last_hop_service_id`).

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

## Collection-Search-Graph

Der `collection_search`-Graph koordiniert Query-Expansion,
Websuche, Hybrid-Reranking, HITL-Review und Ingestion für beliebige Dokumentationstypen
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

6. **persist_profile** — Atomare Transaktion:
   * Prüft ob Profil existiert (Konflikt wenn `force_reanalysis=false`)
   * Setzt altes Profil `is_current=false`, inkrementiert Version
   * Erstellt `FrameworkProfile` mit Struktur-JSON + Metadata
   * Verknüpft `FrameworkDocument` für Hauptdokument
   * Loggt Persistierung mit `profile_id`, Version, Typ

7. **finish** — Bündelt finales `FrameworkAnalysisOutput` mit Transition-History

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
* `framework.persist_profile`

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

**Input**: `UniversalIngestionInput` (TypedDict)

* `source`: `upload` | `crawler` | `search`
* `mode`: `ingest_only` | `acquire_only` | `acquire_and_ingest`
* `collection_id`: UUID (Pflicht)
* `upload_blob`: Dictionary (für Source `upload`, enthält URI, Mimetype, Checksum)
* `normalized_document`: Dictionary (für Source `crawler`, pre-normalized)
* `search_query`: String (für Source `search`)
* `search_config`: Dict (für Source `search`, z.B. `top_n`, `min_snippet_length`)
* `metadata_obj`: Optionales Metadaten-Dict

**Output**: `UniversalIngestionOutput` (TypedDict)

* `decision`: `ingested` | `skipped` | `error` | `acquired`
* `reason`: Mensch-lesbarer Grund
* `document_id`: UUID des persistierten Dokuments
* `ingestion_run_id`: Verknüpfung zum Run
* `telemetry`: Trace- und Tenant-Kontext
* `transitions`: Liste durchlaufener Phasen
* `review_payload`: Daten für HITL (signaling only)
* `hitl_required`: Boolean
* `hitl_reasons`: Liste von Gründen

### Knotenfolge

1. **validate_input**
   Prüft, ob alle Pflichtfelder für die gewählte `source` und den `mode` vorhanden sind. Validiert Context-IDs (Tenant, Trace, Case).

2. **normalize_document**
   Erzeugt ein standardisiertes `NormalizedDocument`-Objekt.
   * Bei **Upload**: Baut Dokument aus `upload_blob` und Metadaten.
   * Bei **Crawler**: Validiert und übernimmt das vor-normalisierte Payload.

3. **persist**
   Speichert das normalisierte Dokument initial im `DocumentRepository` (Upsert). Dies sichert die Daten vor der Verarbeitung.

4. **process**
   Delegiert die inhaltliche Verarbeitung (Parsing, Chunking, Embedding) an den shared `document_processing_graph`.
   * Injiziert `DocumentProcessingContext` und `DocumentPipelineConfig`.
   * Injiziert `Storage`-Service für Blob-Zugriffe.

5. **finalize**
   Mappt das Ergebnis auf den `UniversalIngestionOutput`.
   * Setzt `hitl_`-Flags (aktuell immer False/Empty in Phase 2).
   * Sammelt Telemetrie.

### Migration Status

In Phase 3 sind alle Call-Sites (Upload UI, Crawler Services) auf diesen Graphen migriert. Die Legacy-Graphen (`upload_ingestion_graph`, `crawler_ingestion_graph`) wurden entfernt.
