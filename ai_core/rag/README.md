Das RAG-Verzeichnis liefert die Kern-Capabilities für Retrieval und Scoring,
verwaltet Vektorspeicherzugriffe und stellt damit die diagnostischen Daten
bereit, die in den Graphen und Views weitergegeben werden.

## End-to-End Tests (Crawler → RAG)

- Die Guardrail- und Lifecycle-Pfade des Crawlers werden jetzt in
  `ai_core/tests/test_crawler_guardrails.py` sowie
  `ai_core/tests/test_crawler_retire.py` validiert. Damit liegt der gesamte
  Nachweis für das Crawler → RAG Zusammenspiel in einer Suite.
- Weitere Übergaben – Delta, Ingestion Decision und Vector-Upserts – bleiben in
  `ai_core/tests/test_crawler_delta.py` und
  `ai_core/tests/test_crawler_ingestion.py` gebündelt und spiegeln den
  Ablauf aus [`docs/rag/ingestion.md`](../../docs/rag/ingestion.md#crawler--rag-end-to-end).
