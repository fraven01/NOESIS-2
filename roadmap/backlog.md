# Backlog (RAG prioritized)

This is a curated list of open work items for vibe-coding execution.
Top-to-bottom order within each section is priority order.
Prefer linking each item to concrete code paths (and optionally to an issue).

## Next up (highest leverage)

### L-Track (Agentic Typing)

- [x] **L-Track-1: TaskContext models + Celery meta validation**
  - **Details:** Add TaskContext/TaskScopeContext/TaskContextMetadata Pydantic models; use them in common/celery.py to validate meta and remove raw dict fallbacks. Map all kwargs into metadata.
  - **Pointers:** `common/celery.py:270-470`, `ai_core/tool_contracts/base.py` (integration), new module for TaskContext
  - **Acceptance:** Celery context extraction validates via TaskContext; no raw meta.get("key_alias") access; args[0]/args[1] fallback removed; tests updated for strict meta.
  - **Effort:** L (1.5-2 Sprints)
  - **Breaking:** Ja (meta fallback removal / strict typing)

- [x] **L-Track-2: Node return TypedDicts for Universal Ingestion**
  - **Details:** Introduce TypedDict outputs (ValidateInputNodeOutput, DeduplicationNodeOutput, PersistNodeOutput, ProcessNodeOutput) and apply to node returns + graph state.
  - **Pointers:** `ai_core/graphs/technical/universal_ingestion_graph.py:110-280`
  - **Acceptance:** Node returns and graph state use TypedDicts; no dict[str, Any] in node outputs for these nodes.
  - **Effort:** M (1 Sprint)

- [x] **L-Track-3: Retrieval/RAG state typing + FilterSpec**
  - **Details:** Replace MutableMapping state in RAG graph with typed model/TypedDict; introduce FilterSpec and migrate retrieval call sites; consider converting HybridParameters dataclass to Pydantic.
  - **Pointers:** `ai_core/graphs/technical/retrieval_augmented_generation.py:120+`, `ai_core/nodes/retrieve.py:16+`, `ai_core/nodes/_hybrid_params.py:12+`
  - **Acceptance:** RAG state is typed (no MutableMapping for core fields); filters validated by FilterSpec; retrieval meta stays schema-compatible.
  - **Effort:** L (2 Sprints)
  - **Breaking:** Ja (FilterSpec / state typing)

- [x] **L-Track-4: AX-Score lift (documentation-only)**
  - **Details:** Add Field descriptions for ChunkMeta, CrawlerIngestionPayload, RetrieveMeta, ComposeOutput.
  - **Pointers:** `ai_core/rag/ingestion_contracts.py:69`, `ai_core/rag/ingestion_contracts.py:109`, `ai_core/nodes/retrieve.py:100`, `ai_core/nodes/compose.py:38`
  - **Acceptance:** All fields documented; no schema shape change.
  - **Effort:** S (0.5 Sprint)

### Meta-Fragen Recall Fix (RAG Review 2026-01-23)

Problem: Bei "Welche Fragen muss ich für Anlage 1 beantworten?" werden nur Header/Chunk 1 zurückgegeben, nicht alle Fragen-Chunks. Intent wird erkannt, aber Retrieval-Verhalten ändert sich nicht.

- [x] **META-R1: Document-Scoped Retrieval bei Extract/Checklist Intent**
  - **Details:** Bei `intent in {"extract_questions","checklist"}` und explizitem `doc_ref`-Match + Anchor-Hit im Top-K nur die Chunks dieses Dokuments nachladen. Scope wird respektiert, wenn gesetzt (`case_id`/`collection_id`), sonst tenant-weit ok. Neue Methode `get_chunks_by_document(document_id)` im VectorClient; Expansion bleibt an doc_ref und scope gebunden.
  - **Pointers:** `ai_core/graphs/technical/retrieval_augmented_generation.py:1040+`, `ai_core/rag/vector_client.py`
  - **Acceptance:** Query "Welche Fragen fuer Anlage 1" liefert alle Dokument-Chunks (nicht nur 1); Expansion nur bei doc_ref Match + Anchor-Hit; Regressiontest in `ai_core/tests/rag/test_meta_question_recall.py`
  - **Effort:** M (1 Sprint)

- [x] **META-R2: Erweiterte Kontextpräfixe mit doc_type + document_ref**
  - **Details:** `build_chunk_prefix()` erweitern: `document_ref`, `doc_type`, `section_path`, `chunk_position` (z.B. "Fragen 5-10 von 20"). Format stabil definieren und in BEIDEN Chunkern nutzen (LateChunker + AgenticChunker).
  - **Pointers:** `ai_core/rag/chunking/utils.py:126-154`, `ai_core/rag/chunking/late_chunker.py:1144`, `ai_core/rag/chunking/agentic_chunker.py:628`
  - **Acceptance:** Prefix-Format stabil: "<document_ref> - <doc_type> | <section_path> | <chunk_position>"; beide Chunker nutzen es; Re-Ingestion erforderlich
  - **Effort:** S (0.5 Sprint)
  - **Breaking:** Ja (Chunk-Format ändert sich, Re-Ingestion nötig)

- [x] **META-R3: Strikt gegateter Document-wide Expand Modus (Extract/Checklist)**
  - **Details:** Document-wide Expansion nur als Retrieval-Modus (kein Scoring-Bias): aktiv bei `intent in {"extract_questions","checklist"}` + explizitem `doc_ref`-Match + Anchor-Hit. EvidenceGraph liefert `get_all_document_chunks(document_id)` für die Expansion. Coverage wird als Dokument-Level Metrik genutzt (Diagnose), nicht als Boost.
  - **Pointers:** `ai_core/rag/evidence_graph.py:195+`, `ai_core/rag/metrics.py`, `ai_core/graphs/technical/retrieval_augmented_generation.py`
  - **Acceptance:** Expansion nur bei doc_ref Match + Anchor-Hit; kein globaler Bias in RerankFeatures; Coverage als Metrik erfasst
  - **Effort:** M (1 Sprint)

- [ ] **META-R4: Meta-Chunk Generator bei Ingestion (flagged, Router-only)**
  - **Details:** Optional per Feature-Flag. Synthetischer Meta-Chunk pro Dokument als Router (nicht Evidenz), strikt getrennt mit `kind: "meta"`. Meta-Chunk steuert Expansion über `referenced_chunk_ids`, wird nie in finalen Kontext gerankt oder zitiert. Für Fragenkataloge deterministisch aus extrahierten Fragen (kein LLM Summary) um Halluzination/Veraltung zu vermeiden.
  - **Pointers:** `ai_core/rag/meta_chunk_generator.py` (neu), `ai_core/tasks/ingestion_tasks.py`, `ai_core/nodes/retrieve.py`
  - **Acceptance:** Flag an/aus; Meta-Chunk nur Router in Retrieval-Kaskade; nie in finalem Kontext; Fragenkatalog-Meta deterministisch
  - **Effort:** L (1.5 Sprint)
  - **Breaking:** Ja (neue Chunks, Re-Ingestion nötig)

- [x] **META-R5: chunk_count Metadaten + Coverage-Metriken**
  - **Details:** Chunks bekommen `chunk_count` (Gesamtzahl im Dokument) als Metadatum. Neue Coverage-Metrik in `ai_core/rag/metrics.py`: `calculate_coverage(retrieved, document_id, total)` -> `{coverage_ratio, all_covered}`. Coverage als Diagnose für Meta-Fragen, kein Scoring-Boost.
  - **Pointers:** `ai_core/rag/chunking/late_chunker.py`, `ai_core/rag/chunking/agentic_chunker.py`, `ai_core/rag/metrics.py`
  - **Acceptance:** Jeder Chunk hat `chunk_index` + `chunk_count`; Coverage-Metrik implementiert; Langfuse-Tag `rag.coverage_ratio` bei Meta-Fragen
  - **Effort:** S (0.5 Sprint)
  - **Breaking:** Ja (Metadaten-Schema erweitert)


## Prompt quality + safety (RAG)

- [ ] **Harden RAG answer prompt (language, injection, empty-context handling)**:
  - **Details:** Update answer prompt to enforce same-language responses, ignore instructions in context, and handle empty/insufficient snippets explicitly.
  - **Pointers:** `ai_core/prompts/retriever/answer.v2.md`, `ai_core/nodes/compose.py`
  - **Acceptance:** Answer prompt includes language + injection guard + empty-context rules; compose path aligns with prompt format and handles empty snippet lists without hallucination.

- [ ] **Add rerank scoring rubric and tie-break rules**:
  - **Details:** Extend rerank prompt with explicit relevance rubric, tie-break guidance, and optional brief reason field.
  - **Pointers:** `ai_core/prompts/retriever/rerank.v1.md`, `ai_core/rag/rerank.py`
  - **Acceptance:** Prompt specifies score bands and tie-breaks; LLM rerank output remains schema-valid; fallback path unchanged.

- [ ] **Improve query transform + standalone question prompts**:
  - **Details:** Add injection guard, empty-input handling, and output constraints for query transform and standalone rewriting.
  - **Pointers:** `ai_core/prompts/retriever/query_transform.v1.md`, `ai_core/prompts/retriever/standalone_question.v1.md`
  - **Acceptance:** Query transform returns empty list for empty/unintelligible input and enforces max length; standalone prompt outputs one line and preserves unresolved references.

- [ ] **Fix prompt interpolation and streaming tag leakage in compose**:
  - **Details:** Align prompt assembly with template placeholders, and ensure streaming does not leak raw tags before parsing.
  - **Pointers:** `ai_core/nodes/compose.py`
  - **Acceptance:** Prompt templates are properly interpolated; streaming responses avoid exposing internal tags; legacy fallback output remains consistent.

- [ ] **Standardize prompt guards across retriever prompts**:
  - **Details:** Add a shared guard/language/edge-case section to retriever prompt templates in `ai_core/prompts/retriever/**`.
  - **Pointers:** `ai_core/prompts/retriever/**`
  - **Acceptance:** All retriever prompts include injection guard, language directive, and empty/malformed input handling; consistent gaps[] usage.

- [ ] **Add CI prompt schema validation (retriever prompts)**:
  - **Details:** Implement automated checks that retriever prompt templates include required sections and schema constraints.
  - **Pointers:** `ai_core/infra/prompts.py`, `ai_core/tests/test_prompts.py` (new/extended), `scripts/` (if needed)
  - **Acceptance:** CI fails if required retriever prompt sections are missing; tests cover at least one positive and one negative case.

## SOTA Developer RAG Chat (Pre-MVP)

**Roadmap**: [rag-chat-sota.md](rag-chat-sota.md)
**Total Effort**: ~3-4 Sprints (Medium-High Complexity)

## SOTA Retrieval Architecture

**Roadmap**: [metadata-aware-retrieval.md](metadata-aware-retrieval.md)
**Total Effort**: ~4-5 Sprints
**Vision**: Metadata-first, Passage-first, Hybrid, Multi-stage

**Principles**:
1. **Candidates not only dense**: Dense + lexical + late-interaction (RRF fusion)
2. **Reranking is structure-aware**: Parent/Section/Confidence/Adjacency as features
3. **Output are passages**: Merge adjacent chunks with section boundaries
4. **Query is planned**: Doc-type routing + query expansion + constraints
5. **Evidence graph**: Parent/Child/Adjacency as graph; rerank over subgraphs

### Phase 1: Foundation (P0)

- [x] **SOTA-R1.1: Evidence Graph Data Model**: Represent chunk relationships (parent_of, child_of, adjacent_to) as traversable in-memory graph built from retrieved chunks + metadata.
  - **Pointers:** `ai_core/rag/evidence_graph.py` (new), `ai_core/rag/ingestion_contracts.py:69-95`
  - **Acceptance:** EvidenceGraph with nodes + edges; traversal methods (get_adjacent, get_parent, get_subgraph); unit tests for construction + traversal
  - **Effort:** M (1 Sprint)

- [x] **SOTA-R1.2: Passage Assembly**: Merge adjacent chunks into coherent passages respecting section boundaries and token limits.
  - **Pointers:** `ai_core/rag/passage_assembly.py` (new), `ai_core/nodes/retrieve.py:543-600`
  - **Acceptance:** Passage dataclass; assembly respects section boundaries; token limit enforced; anchor on highest-scoring chunk
  - **Effort:** M (1 Sprint)

- [x] **SOTA-R1.3: Structure-Aware Rerank Features**: Extract rerank features from metadata + graph (parent_relevance, section_match, confidence, adjacency_bonus, doc_type_match).
  - **Pointers:** `ai_core/rag/rerank_features.py` (new), `ai_core/rag/rerank.py:152-241`
  - **Acceptance:** RerankFeatures dataclass; feature extraction; configurable weights per quality_mode; telemetry logs feature values
  - **Effort:** M (1 Sprint)

### Phase 2: Hybrid Candidates (P1)

- [x] **SOTA-R2.1: Lexical Search Integration (BM25)**: Add pg_trgm-based lexical search alongside dense retrieval for exact term matching.
  - **Pointers:** `ai_core/rag/lexical_search.py` (new), `ai_core/rag/query_builder.py`
  - **Acceptance:** Lexical search with pg_trgm; same Chunk format as dense; configurable similarity threshold; <500ms for 100k chunks
  - **Effort:** S (0.5 Sprint)

- [x] **SOTA-R2.2: Hybrid Candidate Fusion (RRF)**: Merge dense + lexical candidates using Reciprocal Rank Fusion with parallel execution.
  - **Pointers:** `ai_core/rag/hybrid_fusion.py` (new), `ai_core/nodes/retrieve.py`
  - **Acceptance:** RRF implementation; parallel dense + lexical; configurable weights; telemetry captures source distribution
  - **Effort:** S (0.5 Sprint)

- [x] **SOTA-R2.3: Query Planner**: Analyze query to determine doc-type routing, query expansion, and constraints (must_include, date_range, collections).
  - **Pointers:** `ai_core/rag/query_planner.py` (new), `ai_core/graphs/technical/rag_retrieval.py`
  - **Acceptance:** QueryPlan + QueryConstraints models; rule-based planner (fast); optional LLM planner; expansion templates per doc_type
  - **Effort:** M (1 Sprint)

### Phase 3: Advanced (P2)

- [ ] **SOTA-R3.1: Late-Interaction Retrieval (ColBERT-style)**: Token-level matching for precision on exact terms. Deferred - use lexical as interim.
  - **Acceptance:** ColBERT model hosting; token-level index; significant infra investment
  - **Effort:** L (2+ Sprints) - DEFERRED

- [x] **SOTA-R3.2: Cross-Document Evidence Linking**: Detect citations/references during ingestion, add "references" edges to Evidence Graph.
  - **Pointers:** `ai_core/rag/evidence_graph.py`, `ai_core/tasks/ingestion_tasks.py`, `documents/parsers_markdown.py`, `ai_core/rag/metadata_handler.py`
  - **Acceptance:** Ingestion extracts references into chunk meta (`reference_ids` + optional labels); EvidenceGraph builds `references` edges; retrieval can expand candidates via references behind a feature flag
  - **Effort:** M (1 Sprint)

- [x] **SOTA-R3.3: Adaptive Weight Learning**: Learn optimal rerank weights from implicit feedback (clicks, answer sources).
  - **Pointers:** `ai_core/rag/rerank_features.py`, `ai_core/rag/rerank.py`, `ai_core/rag/metrics.py` (or new `ai_core/rag/feedback.py`)
  - **Acceptance:** Feedback events collected and stored; scheduled job updates weights per tenant/quality_mode; A/B test switch or config flag selects learned vs static weights
  - **Effort:** M (1 Sprint)

### Phase 4: Contextual Retrieval (P3) - Anthropic SOTA

**Reference**: [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
**Problem**: Chunks verlieren Dokumentkontext beim Embedding. Statische Präfixe (META-R2) sind strukturell, nicht semantisch.
**Impact**: 49-67% weniger Retrieval-Fehler laut Anthropic (je nach Reranking-Stufe).

**Architektur-Hinweis**: Contextual Enrichment ist ein **Post-Chunking-Schritt**, orthogonal zur Chunking-Strategie:
- Je nach Modus läuft **Late** *oder* **Agentic** Chunking
- Late Chunking → bestimmt *wo* Chunk-Boundaries liegen (Embedding-Similarity)
- Agentic Chunking → bestimmt *wo* Boundaries liegen (LLM-Boundary-Detection)
- **Contextual Enrichment → bestimmt *was* embedded wird (Chunk + LLM-generierter Kontext)**

Der jeweils aktive Chunker ruft nach dem Chunking den Contextual Enrichment Service auf.

- **Bestehende Baseline (nicht Anthropic-Style): Context Header**
  - **Details:** LLM/heuristische `context_header`-Generierung (Titel/Section/Preview), gespeichert als Chunk-Meta; wird u.a. für `text_context`/Lexical-Index genutzt. Kein Whole-Document-Kontext, kein Embedding-Prefix.
  - **Pointers:** `ai_core/tasks/ingestion_tasks.py:127-1072`, `ai_core/prompts/retriever/context_header.v1.md`, `ai_core/management/commands/rebuild_rag_index.py:271-344`
  - **Acceptance:** Kontext-Header bleibt bestehen; Anthropic-Style Enrichment ergänzt (nicht ersetzt) die bestehende Header-Logik.

- [x] **SOTA-R4.1: Contextual Chunk Enrichment Service**
  - **Details:** Neuer Service `contextual_enrichment.py` der nach dem Chunking aufgerufen wird. Für jeden Chunk wird ein LLM-Call gemacht, der semantischen Kontext generiert (50-100 Tokens). Prompt-Template:
    ```
    <document>{{WHOLE_DOCUMENT}}</document>
    <chunk>{{CHUNK_CONTENT}}</chunk>
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    ```
    Mit Prompt Caching: ~$1/Million Tokens.
  - **Pointers:** `ai_core/rag/contextual_enrichment.py` (neu), `ai_core/tasks/ingestion_tasks.py:559-1072`, `ai_core/rag/ingestion_contracts.py`, `noesis2/settings/base.py`, `docs/rag/configuration.md`
  - **Acceptance:** Feature-Flag `RAG_CONTEXTUAL_ENRICHMENT` (ENV + Settings + Docs); Service ist chunker-agnostisch; Kontext als `contextual_prefix` in Chunk-Meta gespeichert; Unit-Tests für Service; keine **kwargs in neuen APIs
  - **Effort:** M (1 Sprint)
  - **Breaking:** Ja (neues Chunk-Meta-Feld `contextual_prefix`)

- [x] **SOTA-R4.1a: Integration in Late Chunker**
  - **Details:** `LateChunker.chunk()` ruft nach Boundary-Detection optional `enrich_chunks()` auf. Embedding erfolgt über `contextual_prefix + content` wenn vorhanden.
  - **Pointers:** `ai_core/rag/chunking/late_chunker.py:161`, `ai_core/rag/contextual_enrichment.py`
  - **Acceptance:** Late Chunker unterstützt `enable_contextual_enrichment` Parameter; Fallback auf statischen Präfix wenn LLM-Call fehlschlägt
  - **Effort:** S (0.5 Sprint)

- [x] **SOTA-R4.1b: Integration in Agentic Chunker**
  - **Details:** `AgenticChunker.chunk()` ruft nach LLM-Boundary-Detection optional `enrich_chunks()` auf. Kann denselben LLM-Call nutzen oder separaten.
  - **Pointers:** `ai_core/rag/chunking/agentic_chunker.py:286`, `ai_core/rag/contextual_enrichment.py`
  - **Acceptance:** Agentic Chunker unterstützt `enable_contextual_enrichment` Parameter; Option für Combined-LLM-Call (Boundaries + Context in einem)
  - **Effort:** S (0.5 Sprint)

- [x] **SOTA-R4.1c: HybridChunker Orchestrierung**
  - **Details:** `HybridChunker` steuert Contextual Enrichment zentral über `ChunkerConfig.enable_contextual_enrichment`. Routing-Rules können Enrichment pro `doc_type` aktivieren.
  - **Pointers:** `ai_core/rag/chunking/hybrid_chunker.py:38`, `config/rag_routing_rules.yaml`
  - **Acceptance:** Enrichment kann per Routing-Rule aktiviert werden; Default: aus (opt-in); Metriken für Enrichment-Latenz
  - **Effort:** S (0.5 Sprint)
  - **Breaking:** Nein (opt-in Feature)

- [ ] **SOTA-R4.2: Contextual BM25 Index**
  - **Details:** Lexikalischer Index (`pg_trgm`) ebenfalls über kontextualisierte Chunks. Gleicher Kontext-Präfix wie bei Embeddings.
  - **Pointers:** `ai_core/rag/vector_client.py:1707-1714`, `ai_core/management/commands/rebuild_rag_index.py:271-344`, `noesis2/settings/base.py:95-96`
  - **Acceptance:** BM25-Suche nutzt kontextualisierten Text; kein separater Index nötig (selbe `content`-Spalte); Metriken zeigen Verbesserung bei exakten Term-Matches
  - **Effort:** S (0.5 Sprint)

- [ ] **SOTA-R4.3: Domain-Specific Context Prompts**
  - **Details:** Prompt-Templates pro `doc_type` oder Tenant-Konfiguration. Beispiel für Fragenkataloge: "Beschreibe welche Fragen in diesem Abschnitt gestellt werden und zu welchem Themenbereich sie gehören."
  - **Pointers:** `ai_core/rag/contextual_enrichment.py`, `config/contextual_prompts/` (neu), `ai_core/tasks/ingestion_tasks.py:185-240`
  - **Acceptance:** Prompt-Routing nach `doc_type`; Default-Prompt für unbekannte Typen; A/B-Test-fähig pro Tenant
  - **Effort:** S (0.5 Sprint)

- [ ] **SOTA-R4.4: Batch-Optimierung mit Prompt Caching**
  - **Details:** Dokument-Text wird einmal im Cache gehalten, alle Chunks eines Dokuments werden sequentiell verarbeitet. Anthropic API Prompt Caching oder äquivalent für andere Provider.
  - **Pointers:** `ai_core/llm/client.py:278-694`, `ai_core/rag/contextual_enrichment.py`
  - **Acceptance:** Prompt-Cache-Hit-Rate >90% pro Dokument; Kosten <$1.50/Million Tokens; Latenz <500ms/Chunk bei Cache-Hit
  - **Effort:** M (1 Sprint)

- [ ] **SOTA-R4.5: Migration-Tooling für Re-Ingestion**
  - **Details:** Management-Command `python manage.py enrich_existing_chunks --tenant=X [--doc-type=Y]` für schrittweise Migration. Idempotent, resumable, mit Progress-Tracking.
  - **Pointers:** `ai_core/management/commands/enrich_existing_chunks.py` (neu), `ai_core/management/commands/reembed_documents.py:1-120`
  - **Acceptance:** Bestehende Chunks können nachträglich angereichert werden; keine Doppel-Anreicherung; Fortschritt in DB gespeichert; Abbruch/Resume möglich
  - **Effort:** M (1 Sprint)

### Quick Fixes (RAG Review 2026-01-23)

- [x] **QF-6: Preserve full context for primary document**
  - **Details:** Identify a primary document in the retrieval pool (highest top-score with >=2 chunks in top-k) and prevent its chunks from being down-scored by rerank. Budget selection should prioritize all chunks from this document before others.
  - **Pointers:** `ai_core/graphs/technical/retrieval_augmented_generation.py:1110-1360`, `ai_core/rag/rerank.py:210-330`
  - **Acceptance:** When a primary document is detected, its chunks remain at or above retrieval scores after rerank and are selected first for context until budget is exhausted.
  - **Effort:** S (0.5 Sprint)
  - **Breaking:** Ja (rerank + context selection semantics)

- [ ] **QF-5: Always include neighbor chunks in rerank pool**
  - **Details:** Mark adjacent neighbor chunks fetched in retrieval and ensure rerank pool includes them even when their scores are low/zero. This prevents adjacency expansion from being dropped before rerank.
  - **Pointers:** `ai_core/nodes/retrieve.py:460-520`, `ai_core/rag/rerank.py:210-320`
  - **Acceptance:** Neighbor chunks are present in the rerank candidate pool regardless of score; LLM/heuristic rerank can surface them when relevant.
  - **Effort:** XS (1h)
  - **Breaking:** Ja (rerank pool semantics)

- [x] **QF-4: Relax BM25 lexical match strictness**
  - **Details:** Replace strict `plainto_tsquery` (AND across all tokens) with a more permissive query (e.g. `websearch_to_tsquery` or OR-based terms) to avoid zero lexical hits on natural-language questions. Keep index usage (`text_context_tsv`) unchanged.
  - **Pointers:** `ai_core/rag/vector_client.py:1708-1723`, `ai_core/rag/lexical_search.py:73-112`
  - **Acceptance:** Lexical search returns matches for single-term queries embedded in longer questions (e.g., ?Schneemann? in a full sentence) without requiring all tokens; `rag.hybrid.sql_counts` shows `lex_rows > 0` for such cases.
  - **Effort:** XS (1h)
  - **Breaking:** Ja (lexical matching semantics)

- [ ] **QF-3: Disable MMR diversification in retrieval flow (keep implementation)**
  - **Details:** Keep `_apply_diversification()` implementation but remove it from the retrieval flow for pre-MVP accuracy-focused testing. Add docstrings documenting why it is bypassed and when to re-enable.
  - **Pointers:** `ai_core/nodes/retrieve.py:679-760`, `ai_core/nodes/retrieve.py:1010-1045`
  - **Acceptance:** Retrieval output order matches deduplicated relevance order; diversification is not executed; `_apply_diversification()` remains available with docstrings explaining current status.
  - **Effort:** XS (1h)
  - **Breaking:** Ja (ranking semantics change)

- [ ] **QF-1: Title-Anchor Bypass für Extract-Intent**
  - **Details:** Bei `intent in {"extract_questions","checklist"}` + doc_ref Match die `_is_title_anchor` Prüfung überspringen. Jeder Chunk mit doc_ref Match kann Document-Expansion triggern.
  - **Pointers:** `ai_core/graphs/technical/retrieval_augmented_generation.py:426-450`, `ai_core/graphs/technical/retrieval_augmented_generation.py:592-611`
  - **Acceptance:** Query "Welche Fragen für Anlage 1" liefert alle 3 Chunks auch wenn Chunk 2/3 (mit `?`) der Top-Hit ist; Test in `test_meta_question_recall.py` erweitert
  - **Effort:** XS (2h)

- [ ] **QF-2: relevance_score Fallback aus Retrieval-Score**
  - **Details:** Wenn LLM `relevance_score` nicht zurückgibt oder 0 ist, Fallback auf den originalen Retrieval-Score aus `snippets`. UI zeigt dann sinnvollen Wert statt 0%.
  - **Pointers:** `theme/chat_utils.py:320-330`, `ai_core/nodes/compose.py:232-241`
  - **Acceptance:** UI zeigt nie 0% wenn Chunk tatsächlich relevant war; Fallback-Logik dokumentiert; kein Breaking Change
  - **Effort:** XS (1h)

Future: Systeme nutzen oft eine Hybrid-Suche. Zuerst wird mit BM25 nach relevanten Dokumenten gesucht, und Trigramme helfen dabei, die Suchbegriffe des Nutzers vorab zu korrigieren oder zu vervollständigen.