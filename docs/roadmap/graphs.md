# Graph Inventory

This document provides a comprehensive inventory of all graphs in the `ai_core/graphs/` directory, classifying them by type, recording their inputs/outputs, and identifying key capability dependencies.

## Graph Classification

### Business Graphs

#### 1. Framework Analysis Graph

- **File**: `ai_core/graphs/business/framework_analysis_graph.py`
- **Type**: Custom (Sequential node execution)
- **Purpose**: Framework agreement analysis for AI-first structure detection
- **Implementation**: `FrameworkAnalysisGraph` class with sequential node execution

**Inputs:**

- `FrameworkAnalysisInput` (tenant_id, document_collection_id, document_id, force_reanalysis, confidence_threshold)
- `tenant_id: str`
- `tenant_schema: str`
- `trace_id: str`

**Outputs:**

- `FrameworkAnalysisOutput` (profile_id, version, gremium_identifier, structure, completeness_score, missing_components, hitl_required, hitl_reasons, idempotent, analysis_metadata)

**Key Capability Dependencies:**

- `ai_core.nodes.retrieve` - Document retrieval
- `ai_core.llm.client` - LLM calls for analysis
- `documents.services.framework_service.persist_profile` - Profile persistence
- `ai_core.infra.observability` - Observability and tracing
- `ai_core.infra.prompts` - Prompt loading

**Nodes:**

1. `detect_type_and_gremium` - Detect framework type and gremium
2. `extract_toc` - Extract table of contents from document
3. `locate_components` - Locate framework components
4. `validate_components` - Validate component locations
5. `assemble_profile` - Assemble final profile
6. `persist_profile` - Persist profile to database
7. `finish` - Finalize execution

### Technical Graphs

#### 2. Collection Search Graph

- **File**: `ai_core/graphs/technical/collection_search.py`
- **Type**: LangGraph (StateGraph-based)
- **Purpose**: Business graph orchestrating collection search and ingestion flows
- **Implementation**: `StateGraph` with compiled execution

**Inputs:**

- `GraphInput` (question, collection_scope, quality_mode, max_candidates, purpose, auto_ingest, auto_ingest_top_k, auto_ingest_min_score)
- Runtime context with tenant_id, workflow_id, case_id, trace_id, run_id, ingestion_run_id

**Outputs:**

- Search results with hybrid scoring
- Ingestion metadata
- Telemetry and transitions

**Key Capability Dependencies:**

- `langgraph.graph.StateGraph` - Graph orchestration framework
- `ai_core.tools.web_search.WebSearchWorker` - Web search execution
- `ai_core.llm.client` - LLM for strategy generation
- `llm_worker.graphs.hybrid_search_and_score` - Hybrid scoring subgraph
- `documents.services` - Ingestion triggering

**Nodes:**

1. `strategy` - Generate search strategy
2. `search` - Execute parallel web search
3. `embedding_rank` - Rank results using embeddings
4. `hybrid_score` - Execute hybrid scoring
5. `hitl` - Human-in-the-loop gateway
6. `ingestion` - Trigger document ingestion

#### 3. Crawler Ingestion Graph

- **File**: `ai_core/graphs/technical/crawler_ingestion_graph.py`
- **Type**: Custom (Sequential with inner graph delegation)
- **Purpose**: Minimal orchestration for crawler ingestion
- **Implementation**: `CrawlerIngestionGraph` class with document processing delegation

**Inputs:**

- Raw document data or normalized document input
- Runtime context (tenant_id, case_id, trace_id, workflow_id)
- Control flags (dry_run, review)

**Outputs:**

- Processed document with chunks
- Delta and guardrail decisions
- Completion payload
- Ingestion result

**Key Capability Dependencies:**

- `documents.processing_graph.build_document_processing_graph` - Inner document processing
- `documents.api.NormalizedDocumentPayload` - Document normalization
- `documents.repository.DocumentsRepository` - Document persistence
- `ai_core.api` - Guardrails, delta decisions, embeddings
- `documents.parsers.ParserDispatcher` - Document parsing

**Nodes:**

1. Input normalization and validation
2. Document processing graph execution
3. Result mapping and completion

#### 4. Upload Ingestion Graph

- **File**: `ai_core/graphs/technical/upload_ingestion_graph.py`
- **Type**: LangGraph (StateGraph-based)
- **Purpose**: Document processing for uploads
- **Implementation**: `StateGraph` with compiled execution

**Inputs:**

- `normalized_document_input: dict` - Normalized document data
- `run_until: str` (optional) - Processing phase limit
- Runtime context (tenant_id, workflow_id, case_id, trace_id)
- Runtime dependencies (repository, storage, embedder, etc.)

**Outputs:**

- `decision: str` - Processing outcome
- `reason: str` - Decision rationale
- `severity: str` - Outcome severity
- `document_id: str` - Processed document ID
- `version: str` - Document version
- `telemetry: dict` - Processing metrics
- `transitions: dict` - State transitions

**Key Capability Dependencies:**

- `langgraph.graph.StateGraph` - Graph orchestration
- `documents.processing_graph.build_document_processing_graph` - Inner processing
- `documents.contracts.NormalizedDocument` - Document contracts
- `ai_core.api` - Guardrails, delta decisions
- `documents.parsers` - Document parsing

**Nodes:**

1. `validate_input` - Validate and hydrate input
2. `build_config` - Build pipeline configuration
3. `run_processing` - Execute document processing
4. `map_results` - Map results to output format

### Feldvergleich: Upload vs. Crawler (NormalizedDocument)

**Kurz:** Beide Pfade erzeugen ein `NormalizedDocument`-Objekt und nutzen danach denselben DocumentProcessing-Graph. Die Normalisierungsschritte unterscheiden sich in Herkunft und Detailfeldern; die folgende Tabelle zeigt Unterschiede und Gemeinsamkeiten.

| Feld | Upload (`handle_document_upload`) | Crawler (`build_crawler_state` / `run_crawler_runner`) | Hinweis |
|---|---:|---|---|
| **ref.document_id** | UUID generiert (falls nicht geliefert) | UUID aus origin/document_id oder neu generiert | `ai_core.services.handle_document_upload` / `crawler_state_builder.py` |
| **ref.collection_id** | Aus `metadata_obj["collection_id"]` oder manuelle Collection | Aus `request_data.collection_id` / resolved collection UUID | Upload: `services.__init__` ; Crawler: `crawler_state_builder.py` |
| **ref.workflow_id** | `workflow_id` abgeleitet via `_derive_workflow_id` (z.B. `upload` oder Case) | `workflow_id` aus Crawler-Request / `resolve_workflow_id` | `services.__init__` / `crawler_runner.py` |
| **meta.tenant_id** | gesetzt (aus Request-Header) | gesetzt (aus Meta) | beide (Pflicht) |
| **meta.title** | `metadata_obj["title"]` oder `filename` fallback | `origin.title` oder extrahierter `<title>` aus HTML | Upload `_build_document_meta` / Crawler `build_crawler_state` |
| **meta.language** | optional aus Metadata | Herkunft: `origin.language` oder request | beide |
| **meta.tags** | optional aus Metadata | `origin.tags` + request tags | beide |
| **meta.origin_uri** | optional (wenn angegeben) | canonical URL des Origins (immer) | Upload optional / Crawler immer (`source.canonical_source`) |
| **meta.crawl_timestamp** | typischerweise nicht gesetzt | `fetched_at` (UTC) | Crawler only |
| **meta.external_ref** | `{ external_id, media_type }` (external_id generiert) | `{ provider, external_id, crawler_document_id, etag? }` | Upload `_build_document_meta` / Crawler `build_crawler_state` |
| **blob** (`InlineBlob`) | base64 encoded file, `media_type` via `_infer_media_type`, `sha256`, `size` | base64 encoded fetched body, `media_type` effective (headers/guess), `sha256`, `size` | Upload: `handle_document_upload` / Crawler: `crawler_state_builder` |
| **checksum / content_hash** | `sha256(file_bytes)` → `content_hash` / checksum | `blob.sha256` (fetch bytes) | beide |
| **source** | `"upload"` | `"crawler"` | `NormalizedDocument.source` |
| **created_at** | now() | fetched_at | unterschiedliche Timestamps |
| **Persistierung** | `repository.upsert(normalized_document, scope=...)` (via upload graph) → then queue ingestion | `crawler_runner` -> executes crawler graph; ingest specs -> Document lifecycle service / bulk ingest; updates build.state and persisted via crawler graph | `services.__init__` / `crawler_runner.py` / `crawler_ingestion_graph.py` |
| **Nachgelagerte Verarbeitung** | ruft `upload_ingestion_graph` → **DocumentProcessingGraph** (parse, chunk, embedding, delta, guardrails) | Crawler -> `crawler_ingestion_graph` → **DocumentProcessingGraph** (gleich) | Beide konvergieren auf `documents.processing_graph` |

**Empfehlung:** Extrahiere gemeinsame Normalisierungs-Helper (z. B. `normalize_external_ref()`, `normalize_blob_meta()`, `normalize_common_meta()`) und verwende diese in `handle_document_upload` und `build_crawler_state`; füge Tests, die `NormalizedDocument`-Shape validieren, hinzu.

#### 5. External Knowledge Graph

- **File**: `ai_core/graphs/technical/external_knowledge_graph.py`
- **Type**: LangGraph (StateGraph-based)
- **Purpose**: External knowledge acquisition and ingestion
- **Implementation**: `StateGraph` with compiled execution

**Inputs:**

- `query: str` - Search query
- `collection_id: str` - Target collection
- `enable_hitl: bool` - Human-in-the-loop flag
- `auto_ingest: bool` - Automatic ingestion flag
- `context: dict` - Runtime context (tenant_id, trace_id, etc.)

**Outputs:**

- `search_results: list` - Web search results
- `filtered_results: list` - Filtered and ranked results
- `selected_result: dict` - Selected result for ingestion
- `ingestion_result: dict` - Ingestion outcome
- `error: str` - Error information

**Key Capability Dependencies:**

- `langgraph.graph.StateGraph` - Graph orchestration
- `ai_core.tools.web_search.WebSearchWorker` - Web search
- `IngestionTrigger` protocol - Document ingestion

**Nodes:**

1. `search` - Execute web search
2. `select` - Filter and select best candidate
3. `ingest` - Trigger document ingestion

#### 6. Retrieval Augmented Generation Graph

- **File**: `ai_core/graphs/technical/retrieval_augmented_generation.py`
- **Type**: Custom (Sequential node execution)
- **Purpose**: Production RAG workflow
- **Implementation**: `RetrievalAugmentedGenerationGraph` class

**Inputs:**

- State with retrieval parameters
- Meta with context (tenant_id, case_id, trace_id, etc.)

**Outputs:**

- `answer: str` - Generated answer
- `prompt_version: str` - Prompt version used
- `retrieval: dict` - Retrieval metadata
- `snippets: list` - Retrieved snippets

**Key Capability Dependencies:**

- `ai_core.nodes.retrieve` - Document retrieval
- `ai_core.nodes.compose` - Answer composition
- `ai_core.tool_contracts.ToolContext` - Context management

**Nodes:**

1. Retrieve - Execute document retrieval
2. Compose - Generate answer from retrieved documents

#### 7. Info Intake Graph

- **File**: `ai_core/graphs/technical/info_intake.py`
- **Type**: Custom (Simple function)
- **Purpose**: Record incoming meta information
- **Implementation**: Simple function

**Inputs:**

- `state: Dict` - Workflow state
- `meta: Dict` - Context (tenant_id, case_id, trace_id)

**Outputs:**

- Updated state with meta information
- Result with received confirmation

**Key Capability Dependencies:**

- None (simple state management)

#### 8. RAG Demo Graph (Deprecated)

- **File**: `ai_core/graphs/technical/rag_demo.py` (REMOVED)
- **Type**: Deprecated
- **Purpose**: Legacy RAG demo (removed)
- **Implementation**: Empty module with deprecation warning

## Summary Statistics

### By Graph Type

- **LangGraph**: 3 graphs (Collection Search, Upload Ingestion, External Knowledge)
- **Custom**: 4 graphs (Framework Analysis, Crawler Ingestion, RAG, Info Intake)
- **Deprecated**: 1 graph (RAG Demo)

### By Domain

- **Business**: 1 graph (Framework Analysis)
- **Technical**: 7 graphs (Collection Search, Crawler Ingestion, Upload Ingestion, External Knowledge, RAG, Info Intake, RAG Demo)

### Key Capability Usage

- **Document Retrieval**: Framework Analysis, RAG
- **Web Search**: Collection Search, External Knowledge
- **Document Processing**: Crawler Ingestion, Upload Ingestion
- **LLM Integration**: Framework Analysis, Collection Search, RAG
- **Graph Orchestration**: Collection Search, Upload Ingestion, External Knowledge (LangGraph)
- **Persistence**: Framework Analysis, Crawler Ingestion
