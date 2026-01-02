# Logging Standards

This document codifies the logging conventions used across NOESIS-2. It is
intended to keep logs structured, searchable, and consistent across services.

## Principles

- Use structured logging with `extra={}` for all contextual data.
- Include runtime context when available: `tenant_id`, `trace_id`, `invocation_id`.
- Use event-style messages (`domain.action`) over free-form text.
- Never use `print()` in production code paths.
- Use `logger.exception()` for unexpected failures to capture stack traces.

## Required Context Fields

Include these keys when the data exists in the current scope:

- `tenant_id`
- `trace_id`
- `invocation_id`

Additional recommended fields (when relevant):

- `case_id`, `workflow_id`, `run_id`, `ingestion_run_id`
- `document_id`, `collection_id`
- `duration_ms`, `status_code`

## Examples

Structured info log:

```python
logger.info(
    "graph.execute",
    extra={
        "tenant_id": context.scope.tenant_id,
        "trace_id": context.scope.trace_id,
        "invocation_id": context.scope.invocation_id,
        "graph": context.graph_name,
        "duration_ms": duration_ms,
    },
)
```

Structured error log:

```python
logger.exception(
    "upload.failed",
    extra={
        "tenant_id": scope_context.get("tenant_id"),
        "trace_id": scope_context.get("trace_id"),
        "invocation_id": scope_context.get("invocation_id"),
        "document_id": document_id,
    },
)
```

## Anti-Patterns

- `print()` in production code
- Logging without context fields
- Free-form log messages that cannot be queried reliably

## Migration Guidance

If you find a `print()` or an unstructured log:

1. Replace it with `logger.debug/info/warning/exception`.
2. Add `extra` context from the nearest scope object (`ToolContext`, metadata, etc.).
3. Prefer consistent event names (e.g., `crawler.run.start`, `rag.query.failed`).
