# Graph-Orchestrierung im AI Core

Die Graphen unter `ai_core/graphs` bündeln Geschäftslogik in deterministischen
Sequenzen. Jeder Graph besteht aus benannten Knoten, die denselben State
weiterreichen, Mutationen durchführen und anschließend eine Transition mit
telemetrierelevanten Metadaten ausgeben.【F:ai_core/graphs/crawler_ingestion_graph.py†L41-L115】

## DSL & gemeinsame Konzepte

```python
@dataclass(frozen=True)
class GraphNode:
    name: str
    runner: Callable[[dict[str, Any]], tuple[GraphTransition, bool]]
```

- `GraphTransition` validiert `decision` und `reason`, erzwingt mindestens die
  Severity `info` und liefert ein Mapping für Observability und Tests zurück.【F:ai_core/graphs/crawler_ingestion_graph.py†L49-L74】
- Ein `GraphNode` kapselt die ausführbare Funktion und liefert zusammen mit der
  Transition ein `bool`, das den Kontrollfluss (`continue/stop`) steuert.【F:ai_core/graphs/crawler_ingestion_graph.py†L76-L94】
- Jeder Graph verwaltet ein State-Dictionary (z. B. `meta`, `artifacts`,
  `doc`), das zwischen den Knoten mutiert wird. Abschlüsse sammeln Transitionen
  für spätere Auswertungen, Telemetrie (`observe_span`) und Tests.【F:ai_core/graphs/crawler_ingestion_graph.py†L254-L338】【F:ai_core/graphs/crawler_ingestion_graph.py†L1002-L1016】

## Crawler-Ingestion-Graph

`CrawlerIngestionGraph` verarbeitet normalisierte Crawler-Dokumente in der
Sequenz `update_status_normalized → enforce_guardrails → document_pipeline →
ingest_decision → ingest → finish`. Alle Knoten sind als `_run_*`-Methoden
implementiert und werden aus der `GraphNode`-Registry abgerufen.【F:ai_core/graphs/crawler_ingestion_graph.py†L208-L250】【F:ai_core/graphs/crawler_ingestion_graph.py†L518-L1016】

1. **update_status_normalized** – setzt den Lifecycle per
   `DocumentsApiLifecycleService.update_lifecycle_status` auf `normalized`,
   bevor weitere Schritte laufen.【F:ai_core/graphs/crawler_ingestion_graph.py†L552-L611】
2. **enforce_guardrails** – kombiniert Frontier-/Tenant-Signale und ruft
   `ai_core.api.enforce_guardrails` auf. Policy-Denies erzeugen Events,
   Metriken und frühzeitige Stops.【F:ai_core/graphs/crawler_ingestion_graph.py†L672-L741】
3. **document_pipeline** – baut den Dokumenten-Prozessgraphen auf,
   orchestriert Parser, Asset-Extraktion sowie Chunking und persistiert
   Artefakte über das Repository.【F:ai_core/graphs/crawler_ingestion_graph.py†L760-L864】
4. **ingest_decision** – lädt Repository-Baselines, bewertet Deltas und legt
   `ingest_action` (`upsert/skip`) fest.【F:ai_core/graphs/crawler_ingestion_graph.py†L866-L930】
5. **ingest** – triggert Embeddings (`ai_core.api.trigger_embedding`), wertet
   optionale Upsert-Handler aus und sammelt Ergebnisse im State.【F:ai_core/graphs/crawler_ingestion_graph.py†L932-L1016】
6. **finish** – fasst Guardrail-, Delta-, Pipeline- und Embedding-Daten zu einem
   Abschluss-Payload zusammen.【F:ai_core/graphs/crawler_ingestion_graph.py†L940-L1016】

Der Graph erwartet zu Beginn ein `NormalizedDocument` unter
`state["normalized_document_input"]`; fehlt dieser Schlüssel, endet der Lauf mit
einem Fehler, da alle Folgeschritte den Payload benötigen.【F:ai_core/graphs/crawler_ingestion_graph.py†L424-L500】
Langfuse-Spans (`crawler.ingestion.*`) und strukturierte Transitionen werden in
Tests (`ai_core/tests/graphs/test_crawler_ingestion_graph.py`,
`ai_core/tests/test_crawler_delta.py`) geprüft.

## Upload-Ingestion-Graph

`UploadIngestionGraph` kapselt die Verarbeitung direkter Datei-Uploads und
modelliert eine eigenständige Sequenz
`accept_upload → quarantine_scan → deduplicate → parse → normalize →
delta_and_guardrails → persist_document → chunk_and_embed → lifecycle_hook →
finalize`. Optional kann der Lauf über `run_until` an Marker wie
`"parse_complete"` oder `"vector_complete"` gebunden werden; die Zuordnung zu
Knoten erfolgt über `_RUN_UNTIL_TO_NODE`.【F:ai_core/graphs/upload_ingestion_graph.py†L60-L120】

- **accept_upload** validiert Tenant, Uploader, Sichtbarkeit und Bytes, setzt
  Limits über `UPLOAD_MAX_BYTES` bzw. `UPLOAD_ALLOWED_MIME_TYPES` und legt
  Metadaten (`tags`, `source_key`, `filename`) im State ab.【F:ai_core/graphs/upload_ingestion_graph.py†L122-L212】
- **quarantine_scan** ruft optional einen Scanner auf, wenn
  `UPLOAD_QUARANTINE_ENABLED` aktiv ist und ein Callback konfiguriert wurde.【F:ai_core/graphs/upload_ingestion_graph.py†L214-L236】
- **deduplicate** bildet SHA-256-Hashes, erkennt Dubletten innerhalb der
  Graph-Instanz und erhöht bei `source_key` eine Versionszählung pro
  `(tenant, visibility, source_key)`.【F:ai_core/graphs/upload_ingestion_graph.py†L238-L285】
- **parse** und **normalize** dekodieren den Textinhalt, schneiden Snippets zu
  und speichern eine whitespace-normalisierte Variante für nachfolgende
  Entscheidungen.【F:ai_core/graphs/upload_ingestion_graph.py†L287-L323】
- **delta_and_guardrails** erlaubt optionale Callbacks für Guardrails und Delta.
  Nicht-`allow` Entscheidungen stoppen den Lauf; andernfalls wird ein Upsert
  ausgelöst.【F:ai_core/graphs/upload_ingestion_graph.py†L325-L357】
- **persist_document** ruft entweder den injizierten Persistence-Handler auf
  oder erzeugt eine UUID/Version und aktualisiert den Dedup-Cache.【F:ai_core/graphs/upload_ingestion_graph.py†L359-L408】
- **chunk_and_embed** erstellt einen einfachen Chunk (erste 128 Wörter) und
  delegiert optional an einen Embedding-Handler, bevor Telemetrie und Status
  aktualisiert werden.【F:ai_core/graphs/upload_ingestion_graph.py†L410-L443】
- **lifecycle_hook** und **finalize** führen optionale Nachbearbeitung aus und
  entscheiden, ob der Lauf als `completed` oder `skipped` endet.【F:ai_core/graphs/upload_ingestion_graph.py†L445-L492】

Der Graph erzeugt für jeden Knoten Telemetrie (`started_at`, `ended_at`,
`took_ms`) und sammelt Transitionen inklusive Diagnosen. Das Ergebnis enthält
Document-ID, Version, Snippets, Warnungen und den gesamten Laufstatus. Die Tests
in `ai_core/tests/graphs/test_upload_ingestion_graph.py` prüfen Nominalpfad,
`run_until` sowie Dubletten-Erkennung.【F:ai_core/graphs/upload_ingestion_graph.py†L124-L206】【F:ai_core/graphs/upload_ingestion_graph.py†L146-L204】【F:ai_core/tests/graphs/test_upload_ingestion_graph.py†L12-L54】

