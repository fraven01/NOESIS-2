# Gemini Agent Context

Dieses Dokument enthält wesentlichen Kontext für den Gemini-Agenten, um in der NOESIS 2-Umgebung zu arbeiten.
This document provides essential context for the Gemini agent to operate within the NOESIS 2 environment.

## ID Contracts and Semantics

The following IDs are used throughout the NOESIS 2 system. Consistent use is critical for tracing, tenancy, and stability.

- **`tenant_id`**: Identifies a tenant. **MUST** be present in all operations involving tenant-specific data.
- **`case_id`**: Identifies a business case within a tenant (e.g., a ticket or request). Used for end-to-end tracking.
- **`trace_id`**: Correlates a single request across all services. Essential for observability.
- **`span_id`**: Identifies a single operation within a trace.
- **`workflow_id`**: Identifies a specific business workflow or graph.
- **`run_id`**: A unique ID for a single execution of a LangGraph agent.
- **`ingestion_run_id`**: A specialized ID for data ingestion processes.
- **`idempotency_key`**: A client-provided key for `POST` requests to prevent duplicate execution.
- **`invocation_id`**: Unique ID for each individual tool invocation within a workflow.
- **`document_id`**: Uniquely identifies a document.
- **`collection_id`**: Identifies a logical collection of documents.
- **`document_version_id`**: Unique ID for a specific version of a document.

### Usage

- Incoming HTTP requests require `X-Tenant-ID` and `X-Case-ID` headers.
- Tool calls within agents receive a `ToolContext` containing `tenant_id`, `trace_id`, and either `run_id` or `ingestion_run_id`.
- The `require_ids` function in `ai_core.ids.contracts` is used for validation.

For more details, see the full documentation: [ID-Verträge und Semantik](docs/ids.md).