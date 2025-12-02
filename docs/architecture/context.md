# ScopeContext Architecture

## Single Source of Truth

The `ScopeContext` is the definitive source of truth for all request-scoped identifiers and context information within the NOESIS 2 platform. It replaces ad-hoc context dictionaries and scattered ID handling.

### Core Principles

1. **Centralization**: All IDs (`tenant_id`, `trace_id`, `run_id`, etc.) are aggregated in one Pydantic model.
2. **Immutability**: Once established for a request or execution scope, the core IDs (especially `trace_id`) should not change.
3. **Validation**: The `ScopeContext` model enforces strict validation rules, such as the XOR constraint between `run_id` and `ingestion_run_id`.
4. **Propagation**: The context is propagated explicitly through function calls and implicitly via `contextvars` for logging and observability.

## Context Resolution (`normalize_request`)

The `normalize_request` function in `ai_core.ids.http_scope` is the entry point for HTTP requests.

1. **Header Extraction**: Reads standard headers (`X-Tenant-ID`, `X-Trace-ID`, etc.).
2. **Tenant Resolution**: Uses `TenantContext` to resolve the tenant from the request (domain, URL, or header). **Crucially, `TenantContext` is the authority; headers are only used if allowed by policy.**
3. **ID Coercion**: Generates missing `trace_id` or `invocation_id`.
4. **Scope Creation**: Instantiates `ScopeContext`, performing all necessary validations.

## Usage in Components

### Middleware

The `RequestContextMiddleware` calls `normalize_request` and attaches the resulting `ScopeContext` to `request.scope_context`. It also handles `ValidationError`s by returning appropriate 400 responses.

### Graphs

Graphs should initialize their state from a `ScopeContext`. When calling tools, they must derive a `ToolContext` from the current `ScopeContext`.

### Workers

Celery tasks receive context information (serialized `ScopeContext`) in their arguments. They must deserialize this into a `ScopeContext` object to ensure validation rules are respected before proceeding.

### Crawlers

Crawlers, often running as background tasks, must establish a `ScopeContext` at the start of their execution, typically generating a new `ingestion_run_id` and ensuring a valid `tenant_id` is present.
