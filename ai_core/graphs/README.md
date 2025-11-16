# Graph-Orchestrierung im AI Core

Die Graphen im `ai_core` kapseln unsere Business-Orchestrierung. Jeder Graph
besteht aus benannten Knoten (`GraphNode`), die eine State-Mutation ausführen und
ein standardisiertes Übergabeobjekt (`GraphTransition`) zurückgeben. Die
Transitions enthalten `decision`, `reason` und ein `attributes`-Mapping und
ermöglichen damit konsistente Auswertung in Tests, Telemetrie und im
Observability-Layer.

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
