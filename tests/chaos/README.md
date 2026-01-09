# Chaos Test Suite

Fault-injection tests for Redis, SQL, rate limits, and network issues in NOESIS 2.

## Purpose

Chaos tests validate system resilience and observability under failure conditions:
- **Fault injection**: Redis downtime, SQL outages, rate limits, network latency
- **Retry behavior**: Exponential backoff, dead-letter handling, idempotency
- **Degradation**: Graceful fallback, error propagation, user-facing messages
- **Observability**: Langfuse span tagging, event emission, trace propagation

## Running Chaos Tests

```bash
# All chaos tests
npm run test:py -- tests/chaos/ -v

# Specific test file
npm run test:py:single -- tests/chaos/test_tool_context_contracts.py

# With fault injection (environment variables)
REDIS_DOWN=1 npm run test:py -- tests/chaos/redis_faults.py -v
SQL_DOWN=1 npm run test:py -- tests/chaos/sql_faults.py -v
```

## Test Files

### Legacy Fault Injection Tests
- **`ingestion_faults.py`** - Rate limits, deduplication, dead letter handling
- **`sql_faults.py`** - SQL downtime, idempotency under failures
- **`redis_faults.py`** - Broker downtime, task backoff, Langfuse tags

### Contract Validation Tests (2026-01-05 Migration)
- **`test_tool_context_contracts.py`** - ToolContext parsing, ScopeContext/BusinessContext validation
- **`test_graph_io_contracts.py`** - Graph I/O Specs with `schema_id`/`schema_version`
- **`test_chunker_routing.py`** - Chunker mode routing, configuration validation
- **`test_observability_contracts.py`** - Langfuse tag propagation (scope/business context)

### Supporting Files
- **`conftest.py`** - Shared helpers (`_build_chaos_meta()`)
- **`fixtures.py`** - `ChaosEnvironment`, `MockLangfuseClient` fixtures
- **`reporting.py`** - Pytest plugin for JSON artifact generation
- **`runbook_contracts.py`** - Incident runbook validation

## Contract Migration (2026-01-05)

### New Meta Structure

**OLD (v1) - Flat dict:**
```python
meta = {
    "tenant_id": "tenant-001",
    "case_id": "case-001",
    "trace_id": "trace-001",
}
```

**NEW (v2) - Compositional (ScopeContext + BusinessContext):**
```python
from tests.chaos.conftest import _build_chaos_meta

meta = _build_chaos_meta(
    tenant_id="tenant-001",
    trace_id="trace-001",
    case_id="case-001",
    run_id="run-001",
)

# Result:
# {
#     "scope_context": {
#         "tenant_id": "tenant-001",
#         "trace_id": "trace-001",
#         "invocation_id": "chaos-invocation",
#         "run_id": "run-001",
#         "service_id": "chaos-test-runner",
#     },
#     "business_context": {
#         "case_id": "case-001",
#     },
# }
```

### Key Contracts

**ScopeContext** ([ai_core/contracts/scope.py](../../ai_core/contracts/scope.py)):
- **WHO, WHEN**: tenant_id, trace_id, invocation_id, run_id, service_id
- **Identity rules**: `service_id` for S2S Hops (Chaos tests), `user_id` for User Request Hops
- **Runtime IDs**: At least one of `run_id` OR `ingestion_run_id` required
- **Validation**: Rejects business IDs (case_id, collection_id, etc.)

**BusinessContext** ([ai_core/contracts/business.py](../../ai_core/contracts/business.py)):
- **WHAT**: case_id, collection_id, workflow_id, document_id, document_version_id
- All fields optional (tools/graphs validate required fields)

**ToolContext** ([ai_core/tool_contracts/base.py](../../ai_core/tool_contracts/base.py)):
- **Composition**: `scope: ScopeContext` + `business: BusinessContext` + runtime metadata
- **Parsing**: `tool_context_from_meta(meta)` supports both old and new formats
- **Backward compatibility**: Deprecated properties (`.tenant_id`, `.case_id`) still work

## Writing New Chaos Tests

### 1. Use `_build_chaos_meta()` Helper

```python
from tests.chaos.conftest import _build_chaos_meta

def test_my_chaos_scenario(chaos_env, langfuse_mock):
    """Test description."""
    meta = _build_chaos_meta(
        tenant_id="test-tenant",
        trace_id="test-trace",
        case_id="test-case",  # Optional business context
        run_id="test-run",    # Or ingestion_run_id
    )

    # Use meta in your test...
```

### 2. Enable Fault Injection

```python
def test_with_redis_down(chaos_env):
    """Test behavior when Redis is unavailable."""
    chaos_env.set_redis_down(True)

    # Your test logic...
    # Verify graceful degradation or error handling
```

### 3. Validate Observability

```python
def test_span_metadata(langfuse_mock, chaos_env):
    """Verify Langfuse span tagging."""
    import ai_core.infra.observability as observability

    meta = _build_chaos_meta(tenant_id="t", trace_id="tr")

    observability.record_span(
        "my.operation",
        trace_id=meta["scope_context"]["trace_id"],
        attributes={"tenant_id": meta["scope_context"]["tenant_id"]},
    )

    assert len(langfuse_mock.spans) == 1
    span = langfuse_mock.spans[0]
    assert span.trace_id == "tr"
    assert span.metadata["tenant_id"] == "t"
```

### 4. Test Contract Validation

```python
from ai_core.tool_contracts.base import tool_context_from_meta
from ai_core.contracts.scope import ScopeContext

def test_contract_validation():
    """Verify contract enforcement."""
    meta = _build_chaos_meta(tenant_id="t", trace_id="tr")

    # Valid parsing
    context = tool_context_from_meta(meta)
    assert context.scope.tenant_id == "t"

    # Invalid: business ID in scope
    with pytest.raises(ValueError, match="cannot include business IDs"):
        ScopeContext(
            tenant_id="t",
            trace_id="tr",
            invocation_id="inv",
            run_id="run",
            case_id="c",  # FORBIDDEN
        )
```

## Fixtures

### `chaos_env`
Controls fault-injection toggles:
```python
def test_example(chaos_env):
    chaos_env.set_redis_down(True)  # Enable Redis failures
    chaos_env.set_sql_down(True)    # Enable SQL failures
    chaos_env.reset()                # Reset all flags
```

### `langfuse_mock`
Captures Langfuse spans and events:
```python
def test_example(langfuse_mock):
    # Trigger code that emits spans...

    assert len(langfuse_mock.spans) == 2
    assert langfuse_mock.spans[0].trace_id == "my-trace"
    assert langfuse_mock.spans[0].metadata["tenant_id"] == "my-tenant"

    assert len(langfuse_mock.events) == 1
    assert langfuse_mock.events[0]["event"] == "my.event"
```

## Troubleshooting

### Test fails with "tenant_id not found"
**Cause**: Using old flat meta dict instead of new compositional structure.
**Fix**: Use `_build_chaos_meta()` helper or construct proper scope_context/business_context.

### Test fails with "At least one of run_id or ingestion_run_id"
**Cause**: ScopeContext requires runtime ID.
**Fix**: Pass `run_id` or `ingestion_run_id` to `_build_chaos_meta()`:
```python
meta = _build_chaos_meta(
    tenant_id="t",
    trace_id="tr",
    run_id="r",  # Add this
)
```

### Test fails with "cannot include business IDs"
**Cause**: Trying to pass case_id/collection_id into ScopeContext directly.
**Fix**: Use separate business_context or BusinessContext model:
```python
# WRONG (case_id belongs to BusinessContext, ScopeContext will reject it):
scope = ScopeContext(tenant_id="t", trace_id="tr", run_id="r", case_id="c")

# RIGHT:
meta = _build_chaos_meta(tenant_id="t", trace_id="tr", case_id="c")
# OR
scope = ScopeContext(tenant_id="t", trace_id="tr", run_id="r")
business = BusinessContext(case_id="c")
```

### Langfuse spans not captured
**Cause**: `langfuse_mock` fixture not included in test parameters.
**Fix**: Add `langfuse_mock` to test function signature.

## References

- [CLAUDE.md](../../CLAUDE.md) - Development workflow
- [AGENTS.md](../../AGENTS.md) - Architecture & contracts
- [ai_core/tool_contracts/base.py](../../ai_core/tool_contracts/base.py) - ToolContext
- [ai_core/contracts/scope.py](../../ai_core/contracts/scope.py) - ScopeContext
- [ai_core/contracts/business.py](../../ai_core/contracts/business.py) - BusinessContext
- [docs/architecture/id-guide-for-agents.md](../../docs/architecture/id-guide-for-agents.md) - ID propagation guide

---

**Last Updated**: 2026-01-05
**Migration**: Contract upgrade to ScopeContext/BusinessContext composition
