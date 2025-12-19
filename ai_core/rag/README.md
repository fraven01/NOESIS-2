Das RAG-Verzeichnis liefert die Kern-Capabilities für Retrieval und Scoring,
verwaltet Vektorspeicherzugriffe und stellt damit die diagnostischen Daten
bereit, die in den Graphen und Views weitergegeben werden.

## End-to-End Tests (Crawler → RAG)

- Die Guardrail- und Lifecycle-Pfade laufen zentral im LangGraph und werden über
  `ai_core/tests/graphs/test_universal_ingestion_graph.py` abgedeckt. Zusätzliche
  Delta-Heuristiken validiert `ai_core/tests/test_crawler_delta.py`.
