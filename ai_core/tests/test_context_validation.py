import pytest

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts import ContextError
from ai_core.tool_contracts.validation import require_business_field, require_runtime_id


def test_require_business_field_missing() -> None:
    business = BusinessContext(case_id=None)
    with pytest.raises(ContextError):
        require_business_field(business, "case_id", operation="Test")


def test_require_business_field_empty_string() -> None:
    business = BusinessContext(case_id="")
    with pytest.raises(ContextError):
        require_business_field(business, "case_id", operation="Test")


def test_require_business_field_success() -> None:
    business = BusinessContext(case_id="case-1")
    assert require_business_field(business, "case_id", operation="Test") == "case-1"


def test_require_runtime_id_prefers_primary() -> None:
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        run_id="run-1",
        ingestion_run_id="ingest-1",
    )
    assert require_runtime_id(scope, prefer="run_id") == "run-1"
    assert require_runtime_id(scope, prefer="ingestion_run_id") == "ingest-1"


def test_require_runtime_id_falls_back() -> None:
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        run_id=None,
        ingestion_run_id="ingest-1",
    )
    assert require_runtime_id(scope, prefer="run_id") == "ingest-1"


def test_require_runtime_id_errors_without_ids() -> None:
    scope = ScopeContext.model_construct(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        run_id=None,
        ingestion_run_id=None,
    )
    with pytest.raises(ContextError):
        require_runtime_id(scope, prefer="run_id")
