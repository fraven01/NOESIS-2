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


def test_wag_select_best_mode(web_graph, mock_search_worker):
    """Test 'select_best' mode filters and selects a candidate."""
    state = {
        "input": {
            "query": "test",
            "mode": "select_best",
            # min_snippet_length 10 to include the 'valid' one but exclude 'short' one if we set logical limit
            "search_config": {"min_snippet_length": 20},
        },
        "tool_context": _create_context(),
    }

    config = {"configurable": {"search_worker": mock_search_worker}}

    result = web_graph.invoke(state, config=config)

    # Check selection logic in output
    # The 'select' node updates 'search_results' and 'selected_result'
    # Finalize puts them in output

    output = result["output"]
    results = output["search_results"]
    selected = output["selected_result"]

    # Only the long snippet should pass validation
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/1"

    assert selected is not None
    assert selected["url"] == "https://example.com/1"


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
