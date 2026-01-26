# Backlog (RAG prioritized)

This is a curated list of open work items for vibe-coding execution.
Top-to-bottom order within each section is priority order.
Prefer linking each item to concrete code paths (and optionally to an issue).

## Next up (highest leverage)

- [ ] **LG-1: Explicit input/output schemas for LangGraph StateGraph**
  - **Details:** Add `input_schema`/`output_schema` to each `StateGraph(...)` initialization to avoid defaulting to full internal state.
  - **Pointers:** `ai_core/graphs/web_acquisition_graph.py:341`, `ai_core/graphs/technical/universal_ingestion_graph.py:520`, `ai_core/graphs/technical/retrieval_augmented_generation.py:1687`, `ai_core/graphs/technical/collection_search.py:1531`, `ai_core/graphs/business/framework_analysis/graph.py:44`, `documents/processing_graph.py:229`
  - **Acceptance:** Each StateGraph call defines explicit input/output schema; no graph defaults to full internal state; tests updated if needed.
  - **Effort:** S (0.5-1 Sprint)

- [ ] **TR-1: Ingestion task tracing + typed task context payloads (M Track)**
  - **Details:** Replace ad-hoc `**_task_context_payload(...)` and entry/exit dict merges with a typed TaskContext payload model. Ensure trace identifiers are present on task logs/observability for ingestion paths.
  - **Pointers:** `ai_core/tasks/ingestion_tasks.py:1276-1593`, `ai_core/tasks/helpers/embedding.py:144`, `ai_core/tasks/helpers/caching.py:121-140`, `ai_core/tasks/helpers/task_utils.py:182-238`, `ai_core/infra/observability.py:392-623`
  - **Acceptance:** Task context payload is built via a Pydantic model (no `**payload` in task helpers); trace_id and invocation_id appear in ingestion task logs/events; tests updated to assert tracing fields on ingestion task observability.
  - **Effort:** M (1 Sprint)

- [ ] **TR-2: Pipeline config/result kwargs hardening (M Track)**
  - **Details:** Replace `cls(**kwargs)`/dict merges in ingestion pipeline config/result/state paths with explicit Pydantic models, and remove untyped `**config_kwargs`/`**result` usage.
  - **Pointers:** `ai_core/ingestion.py:399-736`, `documents/pipeline.py:249-1110`, `documents/processing_graph.py:1146`
  - **Acceptance:** Pipeline config/result/state use typed constructors only; no `**config_kwargs`/`**result` merges in ingestion pipeline paths; tests updated for new schema validation behavior.
  - **Effort:** M (1 Sprint)

- [ ] **TR-L1: Full Pydantic TaskContext across ingestion execution (L Track)**
  - **Details:** Introduce canonical Pydantic models for all ingestion task payloads (entry/exit/update, retry metadata, chunk stats). Replace all `**payload` dict merges in ingestion task helpers and task bodies.
  - **Pointers:** `ai_core/tasks/ingestion_tasks.py:1276-1593`, `ai_core/tasks/helpers/task_utils.py:182-238`, `ai_core/tasks/helpers/embedding.py:144`, `ai_core/tasks/helpers/caching.py:121-140`
  - **Acceptance:** Every ingestion task uses typed payloads; no `**entry`/`**payload` merges in ingestion tasks; serialization via `.model_dump()` only; tests updated for strict validation.
  - **Effort:** L (2-3 Sprints)

- [ ] **TR-L2: Ingestion pipeline graph schema enforcement end-to-end (L Track)**
  - **Details:** Define explicit Pydantic input/output/state schemas for ingestion graph boundaries and enforce validation at graph edges (no raw dict state mutation).
  - **Pointers:** `ai_core/graphs/technical/universal_ingestion_graph.py:1-1200`, `documents/processing_graph.py:1-1300`, `ai_core/graphs/transition_contracts.py:1-220`
  - **Acceptance:** Graph entry/exit states validated by schemas; internal state updates use typed models; tests updated for boundary validation errors.
  - **Effort:** L (2-3 Sprints)

- [ ] **TR-L3: Vector client/store config hard-typing (L Track)**
  - **Details:** Replace free-form kwargs for vector client/store creation and search options with typed config models; remove pass-through `**kwargs` in constructors and factory helpers.
  - **Pointers:** `ai_core/rag/vector_store.py:900-1600`, `ai_core/rag/vector_client.py:580-4000`
  - **Acceptance:** All vector client/store constructors accept typed configs; no `**kwargs` pass-through in vector creation/search; tests updated for config validation.
  - **Effort:** L (2 Sprints)




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

### Quick Fixes



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
