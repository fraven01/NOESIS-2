# RAG Overview (Code Map)

## Scope

source_of_truth:
- ai_core/graphs/technical/retrieval_augmented_generation.py
- ai_core/graphs/technical/rag_retrieval.py
- ai_core/graphs/technical/universal_ingestion_graph.py
- ai_core/nodes/retrieve.py
- ai_core/nodes/compose.py
- ai_core/rag/*
- documents/*
- ai_core/views.py (RagQueryView, RagUploadView, RagIngestionRunView, RagIngestionStatusView, RagHardDeleteAdminView, CrawlerIngestionRunnerView)

## Entry Points (HTTP -> Service -> Graph/Task)

| Entry | Handler | Downstream |
| --- | --- | --- |
| RagQueryView.post | ai_core/services/rag_query.py | retrieval_augmented_generation graph |
| RagUploadView.post | ai_core/services/document_upload.py:handle_document_upload | documents.tasks.upload_document_task |
| RagIngestionRunView.post | ai_core/services/ingestion.py:start_ingestion_run | ai_core/ingestion.py:run_ingestion (queue=ingestion) |
| RagIngestionStatusView.get | ai_core/ingestion_status.py | status lookup only |
| RagHardDeleteAdminView.post | ai_core/rag/hard_delete.py:hard_delete | queue=rag_delete |
| CrawlerIngestionRunnerView.post | ai_core/services/crawler_runner.py:run_crawler_runner | universal_ingestion_graph |

## Contracts

graph_io:
- retrieval_augmented_generation: schema_id=noesis.graphs.retrieval_augmented_generation, version=1.1.0
- rag_retrieval: schema_id=noesis.graphs.rag_retrieval, version=1.0.0
- universal_ingestion: schema_id=noesis.graphs.universal_ingestion, version=1.0.0

tool_context:
- required for all graphs and tools
- ScopeContext holds infra ids (tenant_id, trace_id, invocation_id, run_id/ingestion_run_id)
- BusinessContext holds business ids (case_id, collection_id, workflow_id, thread_id, document_id, document_version_id)

## RAG Answer Flow (retrieval_augmented_generation)

nodes:
- contextualize: standalone question (ai_core/rag/standalone_question.py)
- cache_lookup: semantic cache (ai_core/rag/semantic_cache.py)
- cache_finalize: return cached payload
- transform: query variants + intent (ai_core/rag/strategy.py)
- retrieve: retrieve.run (ai_core/nodes/retrieve.py)
- rerank: rag.rerank (ai_core/rag/rerank.py)
- confidence: retry gating
- compose: compose.run or compose.run_extract_questions (ai_core/nodes/compose.py)

side_effects:
- ThreadAwareCheckpointer (chat history)
- semantic_cache lookup/store
- emit_event / update_observation
- enqueue_used_source_feedback

## Data Stores

documents:
- repository writes: documents/domain_service.py + documents/repository.py
- object_store metadata: ai_core/infra/object_store.py

vector_store:
- router: ai_core/rag/vector_store.py (pgvector backend)
- client: ai_core/rag/vector_client.py
- schemas: docs/rag/schema.sql (storage layout only)

## Observability

spans:
- rag.contextualize, rag.cache_lookup, rag.cache_finalize, rag.transform, rag.retrieve, rag.rerank, rag.confidence, rag.compose
- node.validate_input, node.dedup, node.persist, node.process, node.finalize

events:
- rag.cache.hit, rag.cache.miss, rag.cache.store, rag.confidence.retry, rag.drift.top1

## Related Docs

- docs/rag/rag-doc-index.json (dense map)
- docs/rag/ingestion.md
- docs/rag/retrieval-contracts.md
- docs/rag/lifecycle.md
- docs/rag/configuration.md
