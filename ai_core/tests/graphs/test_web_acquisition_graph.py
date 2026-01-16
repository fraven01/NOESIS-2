import pytest
from unittest.mock import MagicMock
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext

# Import the module under test
import ai_core.graphs.web_acquisition_graph as wag


def _create_context(
    tenant_id="t1",
    trace_id="tr1",
    invocation_id="inv1",
):
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=invocation_id,
        run_id="run-def",
    )
    business = BusinessContext(
        workflow_id="web-acq-test",
    )
    return scope.to_tool_context(business=business)


@pytest.fixture
def mock_search_worker():
    """Mock search worker returning standard results."""
    worker = MagicMock()
    from ai_core.tools.web_search import SearchResult, WebSearchResponse, ToolOutcome

    def run_fn(query, context):
        if query == "fail":
            return WebSearchResponse(
                results=[],
                outcome=ToolOutcome(
                    decision="error",
                    rationale="Search failed",
                    meta={"error": {"message": "Search failed"}},
                ),
            )

        results = [
            SearchResult(
                url="https://example.com/1",
                title="Result 1",
                snippet="Valid snippet length" * 5,
                source="test",
            ),
            SearchResult(
                url="https://example.com/short",
                title="Short Result",
                snippet="Short",
                source="test",
            ),
        ]
        return WebSearchResponse(
            results=results,
            outcome=ToolOutcome(decision="ok", rationale="Found", meta={}),
        )

    worker.run.side_effect = run_fn
    return worker


@pytest.fixture
def web_graph():
    return wag.build_web_acquisition_graph()


def test_wag_validation_error(web_graph):
    """Test missing query triggers validation error."""
    state = {
        "input": {"query": ""},
        "tool_context": _create_context(),
    }
    result = web_graph.invoke(state)
    output = result["output"]
    assert output["decision"] == "error"
    assert "query" in output["error"].lower() or "missing" in output["error"].lower()


def test_wag_search_success(web_graph, mock_search_worker):
    """Test successful search execution."""
    state = {
        "input": {
            "query": "test",
            "search_config": {"top_n": 5},
        },
        "tool_context": _create_context(),
    }

    # Inject worker via configurable
    config = {"configurable": {"search_worker": mock_search_worker}}

    result = web_graph.invoke(state, config=config)
    output = result["output"]

    assert output["decision"] == "acquired"
    assert len(output["search_results"]) == 2
    assert output["error"] is None


def test_wag_search_failure(web_graph, mock_search_worker):
    """Test search worker failure handling."""
    state = {
        "input": {"query": "fail"},
        "tool_context": _create_context(),
    }
    config = {"configurable": {"search_worker": mock_search_worker}}

    result = web_graph.invoke(state, config=config)
    output = result["output"]

    assert output["decision"] == "error"
    assert "Search failed" in output["error"]


def test_wag_accepts_legacy_context(web_graph, mock_search_worker):
    """Test legacy context dict parsing."""
    context_dict = _create_context().model_dump(mode="json")
    state = {
        "input": {"query": "test"},
        "context": context_dict,
    }
    config = {"configurable": {"search_worker": mock_search_worker}}

    result = web_graph.invoke(state, config=config)
    output = result["output"]

    assert output["decision"] == "acquired"
    assert output["error"] is None


def test_wag_missing_context_error(web_graph):
    """Test missing context returns validation error."""
    state = {
        "input": {"query": "test"},
    }

    result = web_graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "error"
    assert "tool_context" in output["error"]


def test_wag_preselected_results_skip_search(web_graph):
    """Test preselected results bypass search worker."""
    state = {
        "input": {
            "preselected_results": [
                {"url": "https://example.com/1", "snippet": "ok", "title": "One"}
            ]
        },
        "tool_context": _create_context(),
    }

    result = web_graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "acquired"
    assert len(output["search_results"]) == 1
    assert output["selected_result"] is None


def test_wag_rate_limit_denied(web_graph, monkeypatch):
    """Test rate limit short-circuits the search."""
    monkeypatch.setattr(wag, "_check_search_rate_limit", lambda tenant_id, query: False)
    state = {
        "input": {"query": "test"},
        "tool_context": _create_context(),
    }

    result = web_graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "error"
    assert "rate limit" in output["error"].lower()
    assert output["search_results"] == []


def test_wag_timeout_error(web_graph, monkeypatch):
    """Test timeout handling from the worker."""

    def raise_timeout(*_args, **_kwargs):
        raise TimeoutError("Search worker exceeded timeout")

    monkeypatch.setattr(wag, "_run_search_with_timeout", raise_timeout)
    state = {
        "input": {"query": "test"},
        "tool_context": _create_context(),
    }

    result = web_graph.invoke(
        state, config={"configurable": {"search_worker": MagicMock()}}
    )
    output = result["output"]

    assert output["decision"] == "error"
    assert "timeout" in output["error"].lower()


def test_wag_search_provider_error(web_graph):
    """Test search provider error handling."""
    from ai_core.tools.web_search import SearchProviderError

    worker = MagicMock()
    worker.run.side_effect = SearchProviderError("provider down")

    state = {
        "input": {"query": "test"},
        "tool_context": _create_context(),
    }
    config = {"configurable": {"search_worker": worker}}

    result = web_graph.invoke(state, config=config)
    output = result["output"]

    assert output["decision"] == "error"
    assert "provider down" in output["error"]
