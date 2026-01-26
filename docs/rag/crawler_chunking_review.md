# Crawler Ingestion + Chunking (Current Path)

## Status

legacy_graphs_removed:
- upload_ingestion_graph
- crawler_ingestion_graph
source: ai_core/graphs/README.md

## Current Entry Point

- HTTP: CrawlerIngestionRunnerView.post (ai_core/views.py)
- Service: ai_core/services/crawler_runner.py:run_crawler_runner
- Graph: universal_ingestion_graph (ai_core/graphs/technical/universal_ingestion_graph.py)

## Chunking Path

pipeline:
- documents/processing_graph.py (document processing graph)
- documents/pipeline.py (DocumentProcessingContext + DocumentPipelineConfig)
- ai_core/rag/chunking/* (chunker implementations)

notes:
- crawler runner builds NormalizedDocument via documents normalization and passes it to universal ingestion
- chunking is executed inside the document processing pipeline, not in a crawler-specific graph
- chunking config is driven by DocumentPipelineConfig and settings listed in docs/rag/configuration.md

## Audit Pointers (for regressions)

code_paths:
- ai_core/services/crawler_runner.py
- documents/processing_graph.py
- documents/pipeline.py
- ai_core/rag/chunking/README.md
