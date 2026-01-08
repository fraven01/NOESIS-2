import pytest

from ai_core.rag.limits import get_limit_setting, normalize_max_candidates
from ai_core.rag.router_validation import (
    RouterInputError,
    RouterInputErrorCode,
    emit_router_validation_failure,
    map_router_error_to_status,
    validate_search_inputs,
)
from ai_core.rag.visibility import Visibility
from common.logging import log_context


def test_validate_requires_tenant() -> None:
    with pytest.raises(RouterInputError) as excinfo:
        validate_search_inputs(tenant_id="  ")

    assert excinfo.value.code == RouterInputErrorCode.TENANT_REQUIRED
    assert excinfo.value.field == "tenant_id"


def test_validate_accepts_trimmed_values() -> None:
    result = validate_search_inputs(
        tenant_id="tenant-1 ",
        top_k="7",
        process=" draft ",
        doc_class=" legal ",
    )

    assert result.tenant_id == "tenant-1"
    assert result.process == "draft"
    assert result.doc_class == "legal"
    assert result.top_k == 7
    assert result.max_candidates is None
    assert result.effective_top_k == 7
    assert result.top_k_source == "from_state"
    expected_cap = int(get_limit_setting("RAG_MAX_CANDIDATES", 200))
    assert result.effective_max_candidates == normalize_max_candidates(
        7, None, expected_cap
    )
    assert result.visibility is Visibility.ACTIVE
    assert result.visibility_source == "from_default"
    assert result.max_candidates_source == "from_default"


def test_validate_normalizes_selector_case() -> None:
    result = validate_search_inputs(
        tenant_id="tenant-1",
        process="Review",
        doc_class="Manual",
    )

    assert result.process == "review"
    assert result.doc_class == "manual"


def test_validate_treats_blank_top_k_as_default() -> None:
    result = validate_search_inputs(tenant_id="tenant-1", top_k="  ")

    assert result.top_k is None
    assert result.process is None
    assert result.doc_class is None
    assert result.effective_top_k == 5
    assert result.top_k_source == "from_default"
    expected_cap = int(get_limit_setting("RAG_MAX_CANDIDATES", 200))
    assert result.effective_max_candidates == normalize_max_candidates(
        5, None, expected_cap
    )
    assert result.max_candidates_source == "from_default"
    assert result.visibility is Visibility.ACTIVE
    assert result.visibility_source == "from_default"


def test_validate_context_includes_sanitized_limits() -> None:
    result = validate_search_inputs(
        tenant_id="tenant-1",
        top_k="5",
        max_candidates="15",
        visibility="all",
    )

    assert result.top_k == 5
    assert result.max_candidates == 15
    assert result.context["top_k"] == 5
    assert result.context["max_candidates"] == 15
    assert result.effective_top_k == 5
    assert result.effective_max_candidates == 15
    assert result.visibility is Visibility.ALL
    assert result.visibility_source == "from_state"
    assert result.context["visibility"] == "all"
    assert result.context["visibility_source"] == "from_state"


@pytest.mark.parametrize("value", [0, -1, "not-a-number", 2.5])
def test_validate_rejects_invalid_top_k(value) -> None:
    with pytest.raises(RouterInputError) as excinfo:
        validate_search_inputs(tenant_id="tenant", top_k=value)

    assert excinfo.value.code == RouterInputErrorCode.TOP_K_INVALID


@pytest.mark.parametrize("value", [0, -3, "invalid", 1.7])
def test_validate_rejects_invalid_max_candidates(value) -> None:
    with pytest.raises(RouterInputError) as excinfo:
        validate_search_inputs(tenant_id="tenant", max_candidates=value)

    assert excinfo.value.code == RouterInputErrorCode.MAX_CANDIDATES_INVALID


def test_validate_guards_against_conflicting_limits(monkeypatch) -> None:
    monkeypatch.setenv("RAG_CANDIDATE_POLICY", "error")
    with pytest.raises(RouterInputError) as excinfo:
        validate_search_inputs(tenant_id="tenant", top_k=10, max_candidates=5)

    assert excinfo.value.code == RouterInputErrorCode.MAX_CANDIDATES_LT_TOP_K


def test_validate_conflicting_limits_can_be_normalized(monkeypatch) -> None:
    monkeypatch.setenv("RAG_CANDIDATE_POLICY", "normalize")

    result = validate_search_inputs(tenant_id="tenant", top_k=10, max_candidates=5)

    assert result.max_candidates == 10
    assert result.context["candidate_policy_action"] == "normalized_to_top_k"
    assert result.effective_top_k == 10
    assert result.effective_max_candidates == 10
    assert result.max_candidates_source == "from_state"
    assert result.visibility is Visibility.ACTIVE


def test_validate_rejects_invalid_visibility() -> None:
    with pytest.raises(RouterInputError) as excinfo:
        validate_search_inputs(tenant_id="tenant", visibility="unknown")

    assert excinfo.value.code == RouterInputErrorCode.VISIBILITY_INVALID


def test_map_router_error_to_status_returns_client_error() -> None:
    for code in (
        RouterInputErrorCode.TENANT_REQUIRED,
        RouterInputErrorCode.TOP_K_INVALID,
        RouterInputErrorCode.MAX_CANDIDATES_INVALID,
        RouterInputErrorCode.MAX_CANDIDATES_LT_TOP_K,
        "ROUTER_FUTURE_CODE",
    ):
        assert map_router_error_to_status(code) == 400


def test_emit_router_validation_failure_emits_span(monkeypatch) -> None:
    error = RouterInputError(
        RouterInputErrorCode.TOP_K_INVALID,
        "invalid",
        field="top_k",
        context={
            "tenant_id": "tenant-1",
            "process": "draft",
            "doc_class": "legal",
            "top_k": 99,
            "max_candidates": None,
        },
    )

    spans: list[tuple[str, dict[str, object] | None, str | None]] = []
    from ai_core.rag import router_validation as router_validation_module

    monkeypatch.setattr(
        router_validation_module,
        "record_span",
        lambda name, *, attributes=None: spans.append(
            (name, attributes, (attributes or {}).get("trace_id"))
        ),
    )

    with log_context(trace_id="trace-router"):
        emit_router_validation_failure(error)

    assert spans, "expected router validation failure to emit span"
    name, metadata, trace_id = spans[0]
    assert trace_id == "trace-router"
    assert name == "rag.router.validation_failed"
    assert metadata is not None
    assert metadata["error_code"] == RouterInputErrorCode.TOP_K_INVALID
    assert metadata["tenant_id"] == "tenant-1"
    assert metadata["process"] == "draft"
