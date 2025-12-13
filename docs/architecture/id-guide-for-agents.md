# ID Handling Guide for Coding Agents

This guide provides clear, actionable rules for handling IDs within the NOESIS-2 codebase. Follow these rules to ensure compliance with the architecture and avoid common bugs.

## Core ID Concepts

| ID | Variable Name | Purpose | Mandatory? | Source |
|---|---|---|---|---|
| **Tenant ID** | `tenant_id` | Identifies the customer/tenant. | **YES** | Header `X-Tenant-ID`, Token, or `TenantContext`. |
| **Case ID** | `case_id` | Identifies the business case. | **YES** (for business logic) | Header `X-Case-ID` or Dispatcher. |
| **Workflow ID** | `workflow_id` | Identifies the type of workflow. | **YES** (in Graphs) | Dispatcher. |
| **Run ID** | `run_id` | Single execution of a Graph. | **XOR** with `ingestion_run_id` | Generated per run. |
| **Ingestion Run ID** | `ingestion_run_id` | Single ingestion job. | **XOR** with `run_id` | Entry point (Worker/API). |
| **Trace ID** | `trace_id` | Distributed tracing. | **YES** | Header `X-Trace-ID` or generated. |

## Rules for Coding Agents

### 1. Always Use `ScopeContext` or `ToolContext`

Do not pass IDs around as loose arguments if possible. Use the standardized context objects.

- **Web/API Layer:** Use `ai_core.ids.normalize_request(request)` to get a `ScopeContext`.
- **Workers:** Receive `state` and `meta`. If you need to call a tool or graph, construct a `ScopeContext` first.
- **Tools:** Always accept `ToolContext` as the first argument.

### 2. Enforce Mutual Exclusion (XOR)

You must provide **exactly one** of `run_id` or `ingestion_run_id`.

- **Standard Graphs:** Use `run_id`.
- **Ingestion Graphs:** Use `ingestion_run_id`.

**Bad:**

```python
# ERROR: Missing runtime ID
context = ToolContext(tenant_id="t1", ...) 

# ERROR: Both IDs provided
context = ToolContext(tenant_id="t1", run_id="r1", ingestion_run_id="i1", ...)
```

**Good:**

```python
# Standard execution
context = ToolContext(tenant_id="t1", run_id="r1", ...)

# Ingestion execution
context = ToolContext(tenant_id="t1", ingestion_run_id="i1", ...)
```

### 3. Handling IDs in Tests

Tests are the most common source of ID errors. Use the following patterns to simplify testing.

#### A. Use `normalize_request` in View Tests

Don't manually construct contexts in view tests. Mock the headers and let `normalize_request` do the work.

```python
request = HttpRequest()
request.META = {
    "HTTP_X_TENANT_ID": "test-tenant",
    "HTTP_X_CASE_ID": "case-1",
}
scope = normalize_request(request) # Handles generation and validation
```

#### B. Mock `TenantContext`

If your code relies on `TenantContext.from_request`, you MUST mock it in your test `setUp`.

```python
from unittest.mock import patch

def setUp(self):
    self.patcher = patch("customers.tenant_context.TenantContext")
    self.mock_tenant = self.patcher.start()
    self.mock_tenant.from_request.return_value.schema_name = "test-tenant"

def tearDown(self):
    self.patcher.stop()
```

#### C. Use `ScopeContext` for Graph Tests

When testing graphs, pass a valid `ScopeContext` (or a dict that matches it) in the `meta` field.

```python
meta = {
    "tenant_id": "t1",
    "case_id": "c1",
    "trace_id": "tr1",
    "run_id": "r1", # OR ingestion_run_id
}
```

### 4. Common Pitfalls & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ValueError: Exactly one of run_id or ingestion_run_id...` | You provided neither or both in `ToolContext`/`ScopeContext`. | Ensure exactly one is set. If starting a new run, generate a UUID for `run_id`. |
| `TenantRequiredError` | `tenant_id` missing in headers and `TenantContext` could not resolve it. | Add `X-Tenant-ID` header or mock `TenantContext`. |
| `AttributeError: 'NoneType' object has no attribute 'schema_name'` | `TenantContext.from_request` returned `None`. | Mock `TenantContext` to return a mock object with `schema_name`. |

## Reference Implementation

- **ID Definitions:** `ai_core/contracts/scope.py`
- **Normalization Logic:** `ai_core/ids/http_scope.py`
- **Tool Contract:** `ai_core/tool_contracts/base.py`

### 5. Entity IDs & Idempotency (Collection/Document)

When handling database entities like **Document Collections**, we must distinguish between the **UUID** and the **Logical Key**.

- **UUID (`collection_id`):** The technical primary identifier (usually from the client/UI).
- **Logical Key (`key`):** The human-readable or source-derived identifier (e.g., "fiscal-2024").

#### The "Lookup-Before-Create" Pattern

Workers must be idempotent. A common error is strictly using a provided UUID as the *key* for creation, which fails if the logical entity already exists with a different key.

**Correct Pattern:**

1. Check if the entity exists by **UUID**.
2. If yes, use its **existing Key** for any `ensure_collection` or update calls.
3. If no, use the UUID (or provided Key) to create it.

```python
# Bad: Blindly using UUID as Key
service.ensure_collection(key=str(uuid), collection_id=uuid, ...)

# Good: Resolve existing Key first
existing = Collection.objects.filter(collection_id=uuid).first()
key_to_use = existing.key if existing else str(uuid)
service.ensure_collection(key=key_to_use, collection_id=uuid, ...)
```

This prevents `IntegrityError` (UniqueConstraint violations) when retrying tasks or re-crawling.
