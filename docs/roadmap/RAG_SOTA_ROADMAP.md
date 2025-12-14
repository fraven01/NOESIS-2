# RAG SOTA Ausbaupfad (Stufe 1–3)

## Zielbild
Streaming-first Data Plane, modal-aware Chunking/Retrieval, stabile Zitierfähigkeit, observability + Governance. Nachfolgend die gestuften Ausbauschritte mit Kernaufgaben.

## Stufe 1 – SOTA Baseline (2–4 Wochen)
- Storage/Streaming: Storage-Interface um `open(uri)`, `download_to_path(uri, target_path)`, `put_stream(stream)` erweitern; Parser/Staging auf Streaming umstellen (keine RAM-Kopien).
- Content Addressed + Idempotenz: Schlüsselableitung aus sha256 erzwingen; Dedup im Write-Path; Idempotenz-Key durch Upload→Ingestion→Vector-Upsert durchreichen.
- Chunking/Meta: Chunk-Meta um `kind`, `page_index`, `locator`, `parent_ref`, `table_meta`-Ref erweitern; Table-Header/Summary ins Embedding-Text übernehmen; Parent-IDs stabil halten.
- Artefakte: Parsed Blocks als JSONL/Parquet optional ablegen; normalised plain text als eigenes Artefakt; Artefaktpfade in Meta referenzieren.
- Eval Kickstart: Kleines Golden-Set + Recall@K/MRR Pipeline (pytest marker/CLI); erste Traces inkl. bytes read/written pro Stage.

## Stufe 2 – SOTA Retrieval (4–8 Wochen)
- Hybrid Retrieval: Sparse (BM25/SPLADE) + Dense; RRF in Query-Pipeline integrieren; Konfigurierbare Weights.
- Reranker: Cross-Encoder/Mini-LLM auf Top K mit Caching/Rate-Limits; Kosten-Budgets.
- Modal-aware Chunk/Merge: Regeln pro `kind` (Tabellen nicht mergen, page-aware für PDF, slide-as-unit, Code separiert mit Sprache).
- Citations: Stable `block_id`/locator/page|slide_index in Meta; Antwort-Formatter liefert zitierfähige Quellen (doc_ref + offsets/ids).
- Query Routing: Heuristiken für table/page/slide Queries; Ranking-Boost auf passende Chunks.

## Stufe 3 – Large Scale & Governance (6–12 Wochen)
- Full Streaming: OCR/Render/Parser/Chunker ohne Groß-RAM, Streaming-Pfade auch für Ableitungen.
- Artefakt-Pipeline: Versionierte Artefakte (OCR text, tables JSON, renders/thumbnails); deterministisches Reprocessing.
- Dedup E2E: CAS als Primärpfad, Upserts idempotent/retry-sicher; max chunk count/embedding tokens pro Dokument.
- Observability/Costs: End-to-end Spans (ingestion→embed→upsert), bytes/tokens per stage, Quotas pro Tenant.
- Security/Governance: Retention/Deletion inkl. Artefakte, per-Tenant Encryption (falls Storage unterstützt), Audit-Logs, feinere PII-Redaction.

## Sofortige PR-Schnitte (empfohlen)
1) Storage-Streaming + CAS-Härtung (Interface + Tests).
2) Chunk-Meta-Anreicherung (kind/page/locator/table_meta) + Table-Text-Representation.
3) Artefakt-Ablage für parsed blocks (JSONL) + normalized text; Meta-Refs.
4) Eval-Baseline (kleines Golden-Set + Recall@K/MRR) + Trace-Metriken (bytes read/written).
