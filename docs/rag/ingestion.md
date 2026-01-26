# RAG Ingestion (Code Map)

## Entry Points

upload:
- HTTP: RagUploadView.post (ai_core/views.py)
- Service: ai_core/services/document_upload.py:handle_document_upload
- Task: documents.tasks.upload_document_task
- Side effects: object_store metadata write, duplicate hash check (documents)

ingestion_run:
- HTTP: RagIngestionRunView.post (ai_core/views.py)
- Service: ai_core/services/ingestion.py:start_ingestion_run
- Task: ai_core/ingestion.py:run_ingestion (queue=ingestion)
- State inputs: tenant_id, case_id (business), document_ids, embedding_profile, trace_id, run_id, tenant_schema

crawler_runner:
- HTTP: CrawlerIngestionRunnerView.post (ai_core/views.py)
- Service: ai_core/services/crawler_runner.py:run_crawler_runner
- Graph: universal_ingestion_graph (ai_core/graphs/technical/universal_ingestion_graph.py)

## Universal Ingestion Graph (NormalizedDocument)

boundary:
- schema_id: noesis.graphs.universal_ingestion
- version: 1.0.0
- input: UniversalIngestionGraphInput { input.normalized_document, context }
- context: ToolContext dict (ToolContext.model_validate)

flow:
- bind_context -> validate_input -> dedup -> persist -> process -> finalize

requirements:
- ToolContext required
- BusinessContext.collection_id required

dependencies:
- documents/contracts.py:NormalizedDocument
- documents/processing_graph.py:build_document_processing_graph
- ai_core/services/document_processing_factory.py:build_document_processing_bundle
- ai_core/api.py:trigger_embedding

## Document Processing Pipeline

core:
- documents/pipeline.py (DocumentProcessingContext, DocumentPipelineConfig)
- documents/processing_graph.py (parse -> chunk -> embed -> persist)

chunking:
- configured by DocumentPipelineConfig
- chunker implementations live under ai_core/rag/chunking/*

## Ingestion Profile + Vector Space

resolution:
- ai_core/rag/ingestion_contracts.py:resolve_ingestion_profile
- ai_core/rag/vector_schema.py:ensure_vector_space_schema

state:
- embedding_profile resolved to vector_space_id + schema + dimension

## Persisted Artifacts

object_store:
- upload metadata: {tenant}/{workflow|case}/uploads/{document_id}.meta.json
- ingestion state: {tenant}/{case}/uploads/{document_id}.status.json
- parsed text: {tenant}/{case}/text/{document_id}.parsed.(txt|json)

## Queues

celery:
- ingestion (run_ingestion, process_document)
- rag_delete (rag.hard_delete)
- dead_letter (ai_core.ingestion.dead_letter)

## Failure Modes (code-level)

run_ingestion:
- missing_document_ids, missing_embedding_profile, missing_tenant_id, missing_run_id, missing_trace_id
- dead letter dispatch on group failure (queue=dead_letter)

universal_ingestion_graph:
- validation_error -> decision=failed, reason_code=VALIDATION_ERROR
- persistence_error / processing_error -> decision=failed
