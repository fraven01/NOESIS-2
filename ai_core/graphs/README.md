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
    runner: Callable[[State, Artifacts, Control], tuple[GraphTransition, bool]]
```

* **State** enthält den vollständigen Crawl-/RAG-Kontext inklusive Ids.
* **Artifacts** sammeln Zwischenergebnisse und werden zwischen den Knoten
  weitergereicht.
* **Control** modelliert Feature-Flags und manuelle Overrides (Shadow Mode,
  Review, Retire).

Ein `GraphTransition` validiert seine Felder beim Erzeugen und erzwingt
mindestens die Severity `info`. Dadurch können nachgelagerte Komponenten die
Transitions ohne zusätzliche Guards interpretieren.

## Crawler-Ingestion-Graph

Der `crawler_ingestion_graph` orchestriert den Crawler-Workflow in folgender
Reihenfolge:

1. **Frontier → Fetch → Parse → Normalize → Delta → Guardrails**
   prüfen, ob und wie ein Dokument verarbeitet werden darf.
2. **Ingestion Decision** legt fest, ob ein Dokument gespeichert (`UPSERT`),
   übersprungen oder in den `RETIRE`-Pfad überführt wird.
3. **Store** ruft `documents.repository.DocumentsRepository.upsert()` auf und
   persistiert den normalisierten Datensatz inklusive Workflow- und
   Collection-Kontext.
4. **Upsert** baut aus den Artefakten einen `Chunk` und schiebt ihn über den
   Standard-Vector-Client (`ai_core.rag.vector_client.get_default_client()`) in
   den Zielspace (`upsert_chunks`).
5. **Retire** aktualisiert bei Bedarf den Lifecycle über
   `update_lifecycle_state` des Vector-Clients und protokolliert die Entscheidung.

   Die Tests in `ai_core/tests/graphs/test_crawler_ingestion_graph.py` decken
   Guardrail-, Lifecycle- und Embedding-Entscheidungen ab, während
   `ai_core/tests/test_crawler_delta.py` die Delta-Heuristiken prüft.

Jeder Schritt emittiert eine Transition mit deterministischer Struktur. Diese
Transitions landen im aggregierten Ergebnis des Graph-Laufs und bilden die
Grundlage für Unit-Tests sowie Langfuse-Spans.
