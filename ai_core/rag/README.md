Das RAG-Verzeichnis liefert die Kern-Capabilities für Retrieval und Scoring,
verwaltet Vektorspeicherzugriffe und stellt damit die diagnostischen Daten
bereit, die in den Graphen und Views weitergegeben werden.

## RagResponse Schema (Dev Workbench)

`ai_core/rag/schemas.py` defines the structured response used by the
Developer RAG Chat.

- `RagResponse`: answer payload (Markdown + reasoning + sources)
- `RagReasoning`: reasoning summary with `analysis` and `gaps`
- `SourceRef`: source reference with `id`, `label`, `relevance_score`


## End-to-End Tests (Crawler → RAG)

- Die Guardrail- und Lifecycle-Pfade laufen zentral im LangGraph und werden über
  `ai_core/tests/graphs/test_universal_ingestion_graph.py` abgedeckt. Zusätzliche
  Delta-Heuristiken validiert `ai_core/tests/test_crawler_delta.py`.
