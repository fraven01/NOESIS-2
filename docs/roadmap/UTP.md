# Roadmap: Universal Technical Graph (Pre‑MVP)

## Zielsetzung

Diese Roadmap definiert einen umsetzbaren, inkrementellen Plan zur Konsolidierung der bestehenden technischen Graphen in NOESIS 2 zu einem **Universal Technical Graph**, ohne technische Schulden aufzubauen und mit voller Offenheit für spätere Post‑MVP‑Erweiterungen.

Der Fokus liegt auf:

* klaren Architekturentscheidungen
* expliziten Nicht‑Zielen
* schrittweiser Umsetzung mit klaren Abbruchkriterien
* direkter Ableitbarkeit von Implementierungsplänen

---

## Leitplanken (verbindlich)

1. **Pre‑MVP‑Annahmen**

   * Breaking Changes sind erlaubt
   * UI kann angepasst werden
   * Datenbank kann zurückgesetzt werden

2. **Keine technische Schuld im Kern**

   * Keine Wrapper‑Graphen
   * Keine parallelen Legacy‑Pfad‑Pflege
   * Ein technischer Einstiegspunkt pro Fähigkeit

3. **Klare Trennung der Ebenen**

   * Business Graphen orchestrieren
   * Technical Graphen führen aus
   * Kein Rückruf von Technical → Business

4. **HITL ist signaling, nicht blocking**

   * Kein Graph‑Resume
   * Keine LangGraph Interrupts
   * Vollständige Review‑Payload wird erzeugt

---

## Zielarchitektur (Pre‑MVP)

### Graphen nach Konsolidierung

1. **Universal Technical Graph**

   * Konsolidiert:

     * Upload Ingestion
     * Crawler Ingestion
     * External Knowledge (Quick Search)
   * Konfigurierbar über Input und Mode

2. **Collection Search Graph**

   * Bleibt separat
   * Funktion: Such‑ und Ingest‑Planung
   * Orchestriert den Universal Technical Graph

3. **Framework Analysis Graph**

   * Business Graph
   * Fachliche Semantik
   * Wird auf LangGraph migriert

---

## Funktionsumfang des Universal Technical Graph

### Unterstützte Quellen (source)

* upload
* crawler
* search

### Unterstützte Modi (mode)

* acquire_only
* acquire_and_ingest
* ingest_only

### Persistenzmodell

* Alle Flows persistieren
* Zwei Stufen:

  * staged: Suchergebnisse, Kandidaten, Metadaten
  * ingested: NormalizedDocument + Processing

### HITL‑Semantik

* signaling only
* Output:

  * hitl_required
  * hitl_reasons
  * review_payload

Keine Blockierung, kein Resume.

---

## Roadmap‑Phasen

### Phase 0 – Architekturfixierung

**Ziel:** Entscheidungsraum schließen

Deliverables:

* Diese Roadmap verabschiedet
* Zielgraphen festgelegt
* Explizite Nicht‑Ziele dokumentiert

Abbruchkriterium:

* Offene Grundsatzfragen zu HITL oder Persistenz

---

### Phase 1 – Framework Analysis auf LangGraph migrieren

**Ziel:** Einheitliches Orchestrierungs‑Framework

Maßnahmen:

* Migration des Framework Analysis Graphen auf LangGraph
* Keine fachlichen Änderungen
* State explizit modellieren

Ergebnis:

* Alle Graphen nutzen LangGraph
* Einheitliche Observability

Abbruchkriterium:

* Fachlogik ändert sich

---

### Phase 2 – Universal Ingestion Kern aufbauen

**Ziel:** Gemeinsame technische Basis schaffen

Maßnahmen:

* [x] Neuer Universal Technical Graph
* [x] Implementierung:

  * [x] validate_input (source‑spezifisch)
  * [x] normalize_document
  * [x] run_document_processing
  * [x] map_results

Explizit enthalten:

* [x] Upload‑Pfad
* [x] Crawler‑Pfad

Explizit nicht enthalten:

* External Knowledge
* Collection Search

Abbruchkriterium:

* Upload oder Crawler benötigen Sondergraph

---

### Phase 3 – Upload und Crawler vollständig migrieren

**Ziel:** Legacy eliminieren

Maßnahmen:

* [x] UI und Services auf neuen Graph umstellen
* [x] Alte Upload‑ und Crawler‑Graphen löschen
* [x] Tests anpassen

Ergebnis:

* Einziger Ingestion‑Graph
* Kein doppelter Codepfad

Abbruchkriterium:

* Notwendigkeit von Wrappern

---

### Phase 4 – External Knowledge integrieren

**Ziel:** Ein technischer Einstiegspunkt für Wissensakquise

Maßnahmen:

* [x] Quick Search als source=search
* [x] mode=acquire_only oder acquire_and_ingest
* [x] Persistenz als staged

Ergebnis:

* External Knowledge Graph entfällt
* Universalgraph deckt alle technischen Akquise‑Flows ab

Abbruchkriterium:

* Vermischung mit Collection‑Planungslogik

---

### Phase 5 – Collection Search neu verorten

**Ziel:** Saubere Orchestrierung

Maßnahmen:

* [x] Collection Search ruft Universalgraph explizit auf
* [x] Keine direkte Ingestion mehr
* [x] Klare Trennung Planung vs Ausführung

Ergebnis:

* Verständliches mentales Modell

Abbruchkriterium:

* Collection Search enthält technische Ingestion‑Details

---

## Post‑MVP‑Themen (explizit außerhalb des Scopes)

* Blocking HITL mit Graph Resume
* Temporäre oder TTL‑Collections
* GraphOrchestrator Abstraktion
* Echte Parallelität
* Run‑ID‑Reform

Diese Themen werden **nicht vorbereitet**, sondern bewusst vertagt.

---

## Ableitung zu Implementierungsplänen

Diese Roadmap ist bewusst so formuliert, dass sie:

* in einzelne Implementierungspläne zerlegt werden kann
* pro Phase klare Deliverables und Abbruchkriterien besitzt
* von LLMs als Navigations‑ und Entscheidungsgrundlage nutzbar ist

Nächster Schritt:

* Erstellen eines **Implementierungsplans für Phase 2** auf Basis dieser Roadmap

---

# Implementierungsplan: Phase 2 – Universal Ingestion Kern

## Ziel von Phase 2

Ein neuer Universal Technical Graph existiert und kann die beiden Quellen upload und crawler über einen gemeinsamen Verarbeitungskern verarbeiten. External Knowledge und Collection Search sind in dieser Phase ausdrücklich nicht enthalten.

Ergebnis der Phase ist ein lauffähiger Graph, der von Code aus invokiert werden kann. UI Umstellung und Löschen der Altgraphen erfolgt erst in Phase 3.

## Scope

### In Scope

1. Neuer Graph unter ai_core/graphs/technical mit eigener State Definition
2. Source spezifische Validierung für upload und crawler
3. Normalisierung zu NormalizedDocument Payload oder Objekt, je nach bestehendem Contract
4. Delegation an den bestehenden Document Processing Graph
5. Einheitliches Output Mapping inklusive Entscheidung, Reason, Telemetrie, Transitions
6. Signaling HITL Payload im Output, ohne Blocking

### Out of Scope

1. External Knowledge Flow
2. Collection Search Orchestrierung
3. LangGraph Interrupts und Resume
4. Temporäre Collections
5. Graph Orchestrator Abstraktion

## Architekturentscheidungen in Phase 2

1. Source und mode steuern die Pfade. In Phase 2 werden nur source upload und crawler unterstützt.
2. HITL bleibt signaling only. Der Graph endet deterministisch.
3. Persistenz: Dokumente werden wie bisher über Repository Upsert materialisiert.
4. Collection ist Pflicht. In Phase 2 wird collection_id als Input verlangt. Fallback Mechanismen sind Phase 3 Thema.

## Geplanter Graph Contract

### Input

Ein einheitliches Input Modell, das beide Quellen abdeckt. Als Pydantic Model oder TypedDict, konsistent zu euren Konventionen.

Pflichtfelder

1. tenant_id
2. trace_id
3. invocation_id
4. collection_id
5. source
6. mode

Quelle upload

1. upload_blob oder upload_file_ref, abhängig vom bestehenden Upload Service
2. metadata_raw oder metadata_obj

Quelle crawler

1. url oder fetched_content payload oder bereits normalisierte crawler build payload
2. crawler_meta, falls vorhanden

Mode

1. ingest_only in Phase 2 ist der primäre Mode
2. acquire_only und acquire_and_ingest sind reserviert, aber noch nicht implementiert

### Output

Ein einheitliches Output Schema, das Upload und Crawler gleichermaßen bedienen kann.

Pflichtfelder

1. decision
2. reason
3. telemetry
4. transitions
5. hitl_required
6. hitl_reasons
7. review_payload
8. normalized_document_ref oder document_id
9. ingestion_run_id

## Node Design

### State

Definiere einen UniversalIngestionState mit folgenden Gruppen

1. input
2. context
3. artifacts
4. processing
5. output
6. error

### Nodes

1. validate_input

   * routing auf validate_upload oder validate_crawler
   * outcome typed, damit conditional edges robust sind

2. normalize_document

   * erzeugt NormalizedDocument
   * setzt source spezifische Metadaten

3. persist_or_upsert

   * ruft Repository Upsert auf
   * stellt sicher, dass Blobs materialisiert sind

4. build_processing_config

   * setzt Processing Flags
   * erlaubt source spezifische Defaults

5. run_document_processing

   * delegiert an build_document_processing_graph

6. map_results

   * normalisiert Output auf das gemeinsame Schema

7. finalize

   * setzt decision, reason, telemetry
   * setzt HITL signaling Felder, falls erforderlich

### Conditional Edges

1. validate_input entscheidet über continue oder early exit
2. mode ist in Phase 2 auf ingest_only beschränkt. Falls andere modes gesetzt sind, fail fast mit typed error

## Dateien und Touchpoints

### Neue Datei

1. ai_core/graphs/technical/universal_ingestion_graph.py

### Wiederverwendete Komponenten

1. documents/processing_graph.py
2. bestehende Upload Normalisierung helper in ai_core.services, soweit sinnvoll
3. bestehende Crawler Normalisierung in crawler_state_builder, soweit sinnvoll
4. DocumentRepository Upsert und Materialization

### Tests

1. Unit Tests für validate_upload und validate_crawler
2. Unit Test für normalize_document pro source
3. Integration Test, der einen minimalen NormalizedDocument durch den Document Processing Graph laufen lässt
4. Contract Test für Output Schema, unabhängig vom source

## Akzeptanzkriterien

1. Ein einzelner Graph kann upload ingest_only erfolgreich ausführen
2. Ein einzelner Graph kann crawler ingest_only erfolgreich ausführen
3. Output Schema ist für beide Pfade identisch strukturiert
4. Fehler sind typed und propagieren reproduzierbar
5. Observability ist konsistent, mindestens trace_id wird in allen Knoten weitergereicht
6. Kein alter Graph wird in Phase 2 verändert

## Risiken und Gegenmaßnahmen

1. Unterschiedliche Normalisierung erzeugt unterschiedliche Meta Felder

   * Gegenmaßnahme: normalize_document kapselt gemeinsame Felder und erlaubt source spezifische Erweiterung

2. Tests werden flakey durch externe Abhängigkeiten

   * Gegenmaßnahme: in Phase 2 nur ingest_only mit lokalem Input, keine Web Calls

3. Ingestion Run Semantik inkonsistent

   * Gegenmaßnahme: ingestion_run_id wird im Processing Startpunkt eindeutig erzeugt

## Definition of Done

* [x] universal_ingestion_graph kann lokal invokiert werden
* [x] Test Suite enthält mindestens einen grünen Integration Test je source
* [x] Dokumentation der Graph Inputs und Outputs ist in ai_core/graph/README.md ergänzt

---

# Implementierungsplan: Phase 3 – Migration und Entfernung der Altgraphen

## Ziel von Phase 3

Alle produktiven Call Sites nutzen ausschließlich den Universal Ingestion Graph für upload und crawler. Die bisherigen Upload Ingestion und Crawler Ingestion Graphen sind entfernt. UI und Services sind auf den neuen Contract umgestellt.

## Scope

### In Scope

1. Umstellung aller Call Sites für Upload auf den Universal Ingestion Graph
2. Umstellung aller Call Sites für Crawler Ingestion auf den Universal Ingestion Graph
3. Anpassung der UI Views und Service Entry Points an den neuen Input und Output Contract
4. Aktualisierung von Tests, Fixtures und Mocks
5. Entfernen der Altgraphen und ihrer direkten Adapter
6. Aktualisierung der Dokumentation und Graph Inventory

### Out of Scope

1. External Knowledge Integration
2. Collection Search Orchestrierung
3. Blocking HITL
4. Graph Orchestrator

## Migrationsstrategie

### Prinzip

Kein Parallelbetrieb. Jede umgestellte Call Site darf keine Abhängigkeit auf alte Graphen behalten. Nach vollständiger Migration werden die Altgraphen gelöscht.

### Reihenfolge

1. Upload Pfad zuerst, weil Input lokal und deterministisch ist
2. Crawler Pfad danach, weil Koordination und Runner häufiger Seiteneffekte hat

## Konkrete Arbeitspakete

### Paket 3.1 – Upload Call Sites migrieren

Ziel: Alle Upload Wege invoken universal_ingestion_graph mit source=upload und mode=ingest_only.

Checkliste

1. UI Entry Point identifizieren

   * theme.views oder entsprechende Views, die Upload auslösen
2. Service Entry Point identifizieren

   * handle_document_upload oder äquivalenter Service
3. Input Mapping

   * metadata_raw zu metadata_obj
   * file bytes zu blob Payload
   * collection_id als Pflichtfeld durchreichen
4. Invocation

   * universal_ingestion_graph.invoke mit context tenant_id, trace_id, invocation_id
5. Output Consumption

   * UI erwartet decision, reason, ingestion_run_id, document_id oder ref
   * HITL Felder optional anzeigen oder ignorieren

Akzeptanzkriterien

* Upload funktioniert Ende zu Ende über Universalgraph
* Output Contract wird korrekt angezeigt oder verarbeitet

### Paket 3.2 – Crawler Ingestion Call Sites migrieren

Ziel: Crawler Runner oder Koordinator ruft universal_ingestion_graph pro build auf oder nutzt eine Batch Strategie, ohne alten crawler_ingestion_graph.

Checkliste

1. Coordinator Entry Point identifizieren

   * run_crawler_runner und build_crawler_state Builder
2. Entscheiden, wo Normalisierung stattfindet

   * Option A: build_crawler_state erzeugt weiterhin NormalizedDocument und Universalgraph startet bei persist_or_upsert
   * Option B: build_crawler_state liefert raw fetched payload und Universalgraph macht normalize_document
   * Für Phase 3 bevorzugt Option A, um Risiko zu reduzieren
3. Invocation Pattern

   * pro URL ein invoke
   * oder kleine Batches, wenn bereits vorhanden
4. Output Mapping

   * build state updates
   * ingestion metrics
   * Fehler pro URL

Akzeptanzkriterien

* Crawler Ingestion läuft ohne alten Graph
* Fehlerfälle sind deterministisch und werden pro URL nachvollziehbar gemeldet

### Paket 3.3 – Tests und Regression Absicherung

Ziel: Stabilität bei Breaking Changes.

Pflichttests

1. Upload Integration Test
2. Crawler Integration Test
3. Contract Tests für Universalgraph Output
4. Negative Tests

   * invalid mime type
   * missing collection_id
   * unsupported mode

Optional

* Snapshot Tests für UI payload

Akzeptanzkriterien

* CI grün
* Keine flakey externen Abhängigkeiten

### Paket 3.4 – Altgraphen entfernen

Ziel: Keine technische Schuld.

Checkliste

1. Entfernen von

   * ai_core/graphs/technical/upload_ingestion_graph.py
   * ai_core/graphs/technical/crawler_ingestion_graph.py
2. Entfernen von Adaptern, die nur diese Graphen aufrufen
3. Entfernen oder Umbau von Dokumentation, die diese Graphen referenziert
4. Aktualisierung von Graph Registry und Inventory

Akzeptanzkriterien

* Keine Imports auf entfernte Module
* Graph Inventory listet den Universalgraph und keine Altgraphen

## Risiken und Gegenmaßnahmen

1. Crawler Koordination ist komplexer als Upload

   * Gegenmaßnahme: zuerst Upload migrieren, dann Crawler

2. Unterschiedliche Output Expectations in UI

   * Gegenmaßnahme: Contract Test plus kleine UI Anpassung, keine Wrapper

3. Builder versus Normalizer Unklarheit im Crawler

   * Gegenmaßnahme: in Phase 3 Option A wählen, Option B erst in Refactoring Phase

## Definition of Done

1. Upload und Crawler laufen ausschließlich über Universalgraph
2. Altgraphen sind gelöscht
3. Dokumentation ist aktualisiert
4. Test Suite deckt beide Pfade ab

---

# Implementierungsplan: Phase 4 – External Knowledge Integration

## Ziel von Phase 4

External Knowledge wird als eigener Graph entfernt. Der Universal Technical Graph unterstützt source=search und deckt Quick Search sowie Search plus optionaler Ingestion ab. Persistenz erfolgt als staged Artefakte. HITL bleibt signaling only.

## Scope

### In Scope

1. [x] Erweiterung des Universalgraph Inputs um source=search
2. [x] Implementierung der Pfade acquire_only und acquire_and_ingest
3. [x] Persistenz von Search Session (als staged Artifact in UniversalState)
4. [x] Generierung einer vollständigen review_payload für HITL
5. [x] Entfernung oder Migration des bisherigen external_knowledge_graph
6. [x] Anpassung der UI Endpoints für web_search und ingest_selected

### Out of Scope

1. Collection Search Planung und Hybrid Scoring
2. Parallelität für Multi-Query
3. Blocking HITL und Resume
4. Temporäre Collections

## Architekturentscheidungen in Phase 4

1. Quick Search ist eine Mode Konfiguration, kein eigener Graph.
2. Staged Persistenz ist Standard: Search Ergebnisse sind auditierbar.
3. Content Fetch und Blob Speicherung können in Phase 4 optional sein.

   * Minimal: persistiere nur Metadaten und Snippets
   * Optional: fetch und normalisiere bereits vor Approval
4. acquire_and_ingest ist ein expliziter zweiter Schritt oder ein Pfad, der direkt ingest auslöst. Kein Resume.

## Input Contract Erweiterung

### Neue Felder für source=search

1. search_query
2. search_provider oder worker config, falls mehrere Provider
3. search_limits

   * max_results
   * max_per_domain
   * freshness_window optional
4. selection_strategy

   * first
   * top_k
   * manual_selected
5. selected_urls optional für ingest_selected

### Mode Semantik

1. acquire_only

   * search, dedupe, score leichtgewichtig
   * persist staged search session
   * output candidate list

2. acquire_and_ingest

   * search wie oben
   * selection
   * ingest auslösen für selected

## Output Contract Erweiterung

Zusätzlich zu Phase 2 und 3 Output Feldern.

1. search_session_id
2. candidates

   * url
   * title
   * snippet
   * score
   * provenance
3. selected_urls
4. staged_artifacts_ref

HITL Felder

* hitl_required
* hitl_reasons
* review_payload

## Node Design Erweiterung

### Neue Nodes

1. search_web

   * nutzt WebSearchWorker

2. dedupe_candidates

   * URL Normalisierung, tracking params entfernen
   * domain limits

3. score_candidates_quick

   * einfache Heuristiken
   * optional Embedding Score später

4. persist_search_session

   * staged Persistenz
   * schreibt search_session und candidates

5. select_candidates

   * based on selection_strategy
   * manual_selected nutzt selected_urls

6. optionally_fetch_and_normalize

   * optional in Phase 4
   * falls aktiviert: erzeugt NormalizedDocument pro URL

7. trigger_ingest

   * ruft Universalgraph ingest_only pro NormalizedDocument auf
   * oder nutzt denselben Graph Pfad intern, aber ohne Orchestrator Abstraktion

### Conditional Edges

1. route on mode

   * acquire_only endet nach persist_search_session
   * acquire_and_ingest geht weiter zu select und ingest

2. route on selection_strategy

3. route on optional fetch before ingest

## Datenmodell und Persistenz

### Minimal Persistenz

1. Search Session Tabelle oder Dokument

   * tenant_id
   * collection_id
   * query
   * provider
   * created_at

2. Candidate Items

   * search_session_id
   * url
   * title
   * snippet
   * score
   * provenance

Optional

* store fetched headers and content hash

## Migrationsarbeit

### Paket 4.1 – Universalgraph source=search implementieren

Akzeptanz

* acquire_only liefert candidates und search_session_id
* staged Persistenz erfolgt

### Paket 4.2 – UI Endpoints umstellen

Akzeptanz

* web_search nutzt Universalgraph
* ingest_selected nutzt Universalgraph mit selected_urls

### Paket 4.3 – External Knowledge Graph entfernen

Akzeptanz

* kein Import und keine Invocation mehr
* Dokumentation aktualisiert

## Tests

1. Unit Test dedupe_candidates
2. Integration Test acquire_only
3. Integration Test acquire_and_ingest mit manual_selected
4. Contract Test candidates output

## Risiken und Gegenmaßnahmen

1. Persistenzmodell für Search Sessions noch nicht vorhanden

   * Gegenmaßnahme: minimaler DB Write mit eigener Tabelle oder Nutzung bestehender Modelle, falls passend

2. Ingestion Triggering erzeugt viele invocations

   * Gegenmaßnahme: top_k begrenzen, später Batch Pattern

3. UI Erwartungen an alte Payload

   * Gegenmaßnahme: UI Breaking Change akzeptieren, aber klarer neuer Contract

## Definition of Done

1. Universalgraph kann search acquire_only und acquire_and_ingest
2. UI nutzt ausschließlich Universalgraph
3. External Knowledge Graph ist entfernt
4. Tests grün und Output Contract stabil

---

# Implementierungsplan: Phase 5 – Collection Search als Planungsgraph verorten

## Ziel von Phase 5

Collection Search bleibt als eigener Graph bestehen, wird aber klar als **Planungsgraph** positioniert. Er erzeugt Suchstrategie, aggregiert Suchergebnisse, führt Hybrid Scoring aus und delegiert jede technische Ausführung an den Universal Technical Graph. Collection Search enthält danach keine eigene Ingestion Logik mehr.

## Scope

### In Scope

1. Refactoring von Collection Search, sodass Ingestion ausschließlich über Universalgraph erfolgt
2. Klare Trennung von Planungs Output versus Ausführungs Invocation
3. Definition eines stabilen Plan Output Contracts
4. Optional: Vereinheitlichung der HITL Review Payload zwischen Collection Search und Universalgraph
5. Tests für Plan Output und Delegation

### Out of Scope

1. Hybrid Search and Score Worker Architekturänderungen, sofern nicht notwendig
2. Graph Orchestrator Abstraktion
3. Parallelität
4. Blocking HITL

## Architekturentscheidungen

1. Collection Search liefert einen **Implementierungsplan** als Output Objekt.
2. Der Implementierungsplan ist ausführbar durch:

   * Business Graphen
   * UI
   * oder direkt durch Collection Search selbst, aber nur durch Invocation des Universalgraph
3. Collection Search darf optional einen execute Flag unterstützen, um Plan und Ausführung in einem Flow zu kombinieren. Die Ausführung erfolgt jedoch ausschließlich über Universalgraph.

## Plan Output Contract

### Core Felder

1. plan_id
2. tenant_id
3. collection_id
4. created_at
5. strategy

   * expanded_queries
   * provider
   * limits
6. candidates

   * url
   * title
   * snippet
   * provenance
7. scored_candidates

   * url
   * hybrid_score
   * rationale
8. selection

   * selected_urls
   * selection_reason
9. hitl

   * hitl_required
   * hitl_reasons
   * review_payload

### Execution Hints

1. execution_mode

   * acquire_and_ingest
2. ingest_policy

   * top_k
   * domain_limits
   * dedupe_policy

## Node Design Anpassung

### Bestehende Nodes bleiben

1. strategy
2. web_search
3. hybrid_score
4. hitl_gateway

### Neue oder geänderte Nodes

1. build_plan

   * erstellt Plan Output Contract
   * persistiert optional plan_id

2. optionally_execute_plan

   * wenn execute Flag gesetzt
   * ruft Universalgraph mit source=search und selected_urls
   * nutzt mode acquire_and_ingest oder ingest_only, abhängig davon ob Kandidaten bereits gefetchte NormalizedDocuments sind

### Conditional Edges

1. Wenn hitl_required true und execute true

   * Flow endet nach build_plan
   * UI oder Business Graph triggert später execution

2. Wenn hitl_required false und execute true

   * Flow geht in optionally_execute_plan

## Persistenz

Minimal

* Plan kann rein als Output geliefert werden

Optional

* Plan wird persistiert für Audit Trail
* Plan Execution kann referenziert werden

Empfehlung Pre-MVP

* Persistiere Plan, wenn das UI mehrere Review Schritte benötigt
* Sonst Output-only ist akzeptabel

## Refactoring Arbeitspakete

### Paket 5.1 – Plan Contract implementieren

Akzeptanz

* Graph liefert plan Output unabhängig von Ausführung

### Paket 5.2 – Delegation an Universalgraph

Akzeptanz

* Ingestion Triggering im Collection Search Code ist entfernt
* Universalgraph wird invokiert mit klaren Inputs

### Paket 5.3 – HITL Vereinheitlichung

Akzeptanz

* Review Payload Struktur ist konsistent mit Universalgraph review_payload

### Paket 5.4 – Tests

Pflicht

1. Unit Test Plan Builder
2. Integration Test Collection Search produce plan
3. Integration Test execute plan invokes Universalgraph
4. Negative Tests für missing collection_id und invalid strategy

## Risiken und Gegenmaßnahmen

1. Hybrid Scoring Worker liefert Output, der schwer in Plan Contract passt

   * Gegenmaßnahme: Adapter Layer im build_plan Node, nicht im UI

2. execute Flag vermischt Plan und Ausführung

   * Gegenmaßnahme: execute standardmäßig false, UI entscheidet

3. Persistenz von Plan erzeugt neues Datenmodell

   * Gegenmaßnahme: zunächst Output-only, Persistenz erst bei Bedarf

## Definition of Done

1. Collection Search ist als Planungsgraph klar dokumentiert
2. Ingestion erfolgt ausschließlich über Universalgraph
3. Plan Output Contract ist stabil
4. Tests grün
