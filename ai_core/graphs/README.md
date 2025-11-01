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
