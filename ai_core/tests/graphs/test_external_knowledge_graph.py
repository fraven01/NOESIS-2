"""Tests for the ExternalKnowledgeGraph LangGraph implementation."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


from ai_core.graphs.external_knowledge_graph import (
    ExternalKnowledgeState,
    build_external_knowledge_graph,
)
from ai_core.tools.web_search import (
    BaseSearchAdapter,
    ProviderSearchResult,
    SearchAdapterResponse,
    SearchProviderError,
    SearchProviderTimeout,
    WebSearchWorker,
)


# --------------------------------------------------------------------- Stubs
class StubSearchAdapter(BaseSearchAdapter):
    """Search adapter returning a pre-seeded list of results."""

    provider_name = "stub"

    provider_name = "stub"

    def __init__(self, items: list[ProviderSearchResult]) -> None:
        self.provider_name = "stub"
        self.items = items

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        return SearchAdapterResponse(results=self.items[:max_results], status_code=200)


class RaisingSearchAdapter(BaseSearchAdapter):
    """Search adapter that raises an error when called."""

    provider_name = "raising"

    def __init__(self, error: SearchProviderError) -> None:
        self.provider_name = "raising"
        self.error = error

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        raise self.error


@dataclasses.dataclass
class StubIngestionTrigger:
    """Track calls to ingestion trigger."""

    outcome: Mapping[str, Any]
    call_count: int = 0
    last_call: dict[str, Any] | None = None

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self.call_count += 1
        self.last_call = {
            "url": url,
            "collection_id": collection_id,
            "context": context,
        }
        return self.outcome


@dataclasses.dataclass
class RaisingIngestionTrigger:
    """Ingestion trigger that raises an exception."""

    error: Exception
    call_count: int = 0

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self.call_count += 1
        raise self.error


# --------------------------------------------------------------------- Factories
def _make_worker(results: list[ProviderSearchResult]) -> WebSearchWorker:
    adapter = StubSearchAdapter(results)
    return WebSearchWorker(adapter, max_results=5, oversample_factor=1)


def _base_context() -> dict[str, str]:
    return {
        "tenant_id": "tenant-A",
        "trace_id": "trace-123",
        "workflow_id": "test-workflow",
        "run_id": "test-run-123",
    }


# --------------------------------------------------------------------- Tests
def test_external_knowledge_graph_success_flow() -> None:
    """Test full flow from search to ingestion."""
    # Setup
    pdf_result = ProviderSearchResult(
        url="https://example.org/report.pdf",
        title="Report",
        snippet="Good pdf content " * 5,
        source="stub",
        score=0.9,
        content_type="application/pdf",
    )
    worker = _make_worker([pdf_result])
    trigger = StubIngestionTrigger(outcome={"status": "queued", "task_ids": ["task-1"]})

    context = _base_context()
    context.update(
        {
            "runtime_worker": worker,
            "runtime_trigger": trigger,
            "top_n": 5,
            "prefer_pdf": True,
        }
    )

    input_state: ExternalKnowledgeState = {
        "query": "test query",
        "collection_id": "col-1",
        "enable_hitl": False,
        "auto_ingest": True,
        "context": context,
        "search_results": [],
        "selected_result": None,
        "ingestion_result": None,
        "error": None,
    }

    # Execute
    external_knowledge_graph = build_external_knowledge_graph()
    final_state = external_knowledge_graph.invoke(input_state)

    # Verify
    assert (
        final_state["error"] is None
    ), f"Graph execution failed with error: {final_state['error']}"
    assert (
        len(final_state["search_results"]) == 1
    ), f"Expected 1 search result, got {len(final_state['search_results'])}. State: {final_state}"
    assert final_state["selected_result"] is not None
    assert final_state["selected_result"]["url"] == "https://example.org/report.pdf"

    assert final_state["ingestion_result"] == {
        "status": "queued",
        "task_ids": ["task-1"],
    }

    assert trigger.call_count == 1
    assert trigger.last_call["url"] == "https://example.org/report.pdf"
    assert trigger.last_call["collection_id"] == "col-1"
    assert trigger.last_call["context"]["tenant_id"] == "tenant-A"


def test_external_knowledge_graph_no_results() -> None:
    """Test flow when search returns nothing."""
    worker = _make_worker([])
    trigger = StubIngestionTrigger(outcome={})

    context = _base_context()
    context.update(
        {
            "runtime_worker": worker,
            "runtime_trigger": trigger,
        }
    )

    input_state: ExternalKnowledgeState = {
        "query": "nothing",
        "collection_id": "col-1",
        "enable_hitl": False,
        "auto_ingest": True,
        "context": context,
        "search_results": [],
        "selected_result": None,
        "ingestion_result": None,
        "error": None,
    }

    external_knowledge_graph = build_external_knowledge_graph()
    final_state = external_knowledge_graph.invoke(input_state)

    assert final_state["search_results"] == []
    assert final_state["selected_result"] is None
    assert trigger.call_count == 0  # Should not trigger ingestion


def test_external_knowledge_graph_selection_filtering() -> None:
    """Test filtering logic (length, blocked domains)."""
    short_result = ProviderSearchResult(
        url="https://short.com", title="Short", snippet="tiny", source="stub", score=0.1
    )
    blocked_result = ProviderSearchResult(
        url="https://blocked.com/bad",
        title="Bad",
        snippet="Long enough snippet " * 5,
        source="stub",
        score=0.8,
    )
    good_result = ProviderSearchResult(
        url="https://good.com/ok",
        title="Good",
        snippet="Long enough snippet " * 5,
        source="stub",
        score=0.5,
    )

    worker = _make_worker([short_result, blocked_result, good_result])
    trigger = StubIngestionTrigger(outcome={"status": "ok"})

    context = _base_context()
    context.update(
        {
            "runtime_worker": worker,
            "runtime_trigger": trigger,
            "blocked_domains": ["blocked.com"],
            "min_snippet_length": 20,
        }
    )

    input_state: ExternalKnowledgeState = {
        "query": "filter test",
        "collection_id": "col-1",
        "enable_hitl": False,
        "auto_ingest": True,
        "context": context,
        "search_results": [],
        "selected_result": None,
        "ingestion_result": None,
        "error": None,
    }

    external_knowledge_graph = build_external_knowledge_graph()
    final_state = external_knowledge_graph.invoke(input_state)

    assert final_state["error"] is None, f"Graph error: {final_state['error']}"
    assert (
        len(final_state["search_results"]) == 3
    ), f"Expected 3 search results, got {len(final_state['search_results'])}"
    assert final_state["selected_result"] is not None
    # Short rejected, Blocked rejected. Good selected.
    assert final_state["selected_result"]["url"] == "https://good.com/ok"


def test_search_worker_error() -> None:
    """Test that search errors are handled gracefully."""
    # Setup: Create a worker that raises SearchProviderTimeout
    error = SearchProviderTimeout(
        message="Search timed out", retry_in_ms=5000, http_status=504
    )
    adapter = RaisingSearchAdapter(error)
    worker = WebSearchWorker(adapter, max_results=5, oversample_factor=1)
    trigger = StubIngestionTrigger(outcome={})

    context = _base_context()
    context.update(
        {
            "runtime_worker": worker,
            "runtime_trigger": trigger,
        }
    )

    input_state: ExternalKnowledgeState = {
        "query": "timeout query",
        "collection_id": "col-1",
        "enable_hitl": False,
        "auto_ingest": True,
        "context": context,
        "search_results": [],
        "selected_result": None,
        "ingestion_result": None,
        "error": None,
    }

    # Execute
    external_knowledge_graph = build_external_knowledge_graph()
    final_state = external_knowledge_graph.invoke(input_state)

    # Verify: Error should be captured, no ingestion triggered
    assert final_state["error"] is not None
    assert (
        "Search timed out" in final_state["error"]
        or "SearchProviderTimeout" in final_state["error"]
    )
    assert final_state["search_results"] == []
    assert final_state["selected_result"] is None
    assert trigger.call_count == 0  # No ingestion should happen


def test_ingestion_trigger_exception() -> None:
    """Test ingestion errors are caught and returned in state."""
    # Setup: Normal search, but ingestion trigger raises exception
    pdf_result = ProviderSearchResult(
        url="https://example.org/doc.pdf",
        title="Document",
        snippet="Valid content " * 10,
        source="stub",
        score=0.9,
        content_type="application/pdf",
    )
    worker = _make_worker([pdf_result])

    # Trigger that raises exception
    trigger = RaisingIngestionTrigger(
        error=RuntimeError("Ingestion service unavailable")
    )

    context = _base_context()
    context.update(
        {
            "runtime_worker": worker,
            "runtime_trigger": trigger,
            "prefer_pdf": True,
        }
    )

    input_state: ExternalKnowledgeState = {
        "query": "test query",
        "collection_id": "col-1",
        "enable_hitl": False,
        "auto_ingest": True,
        "context": context,
        "search_results": [],
        "selected_result": None,
        "ingestion_result": None,
        "error": None,
    }

    # Execute
    external_knowledge_graph = build_external_knowledge_graph()
    final_state = external_knowledge_graph.invoke(input_state)

    # Verify: Search and selection succeed, but ingestion fails gracefully
    assert final_state["error"] is None  # Graph completes successfully
    assert len(final_state["search_results"]) == 1
    assert final_state["selected_result"] is not None
    assert final_state["selected_result"]["url"] == "https://example.org/doc.pdf"

    # Ingestion result should contain error
    assert final_state["ingestion_result"] is not None
    assert final_state["ingestion_result"]["status"] == "error"
    assert "Ingestion service unavailable" in final_state["ingestion_result"]["reason"]
    assert trigger.call_count == 1  # Trigger was attempted


def test_auto_ingest_false_skips_ingestion() -> None:
    """Test that auto_ingest=False stops before ingestion."""
    # Setup: Normal search with valid result
    result = ProviderSearchResult(
        url="https://example.org/article.html",
        title="Article",
        snippet="Interesting article content " * 5,
        source="stub",
        score=0.8,
    )
    worker = _make_worker([result])
    trigger = StubIngestionTrigger(outcome={"status": "queued"})

    context = _base_context()
    context.update(
        {
            "runtime_worker": worker,
            "runtime_trigger": trigger,
        }
    )

    input_state: ExternalKnowledgeState = {
        "query": "test query",
        "collection_id": "col-1",
        "enable_hitl": False,
        "auto_ingest": False,  # KEY: auto_ingest is False
        "context": context,
        "search_results": [],
        "selected_result": None,
        "ingestion_result": None,
        "error": None,
    }

    # Execute
    external_knowledge_graph = build_external_knowledge_graph()
    final_state = external_knowledge_graph.invoke(input_state)

    # Verify: Search and selection succeed, but ingestion is skipped
    assert final_state["error"] is None
    assert len(final_state["search_results"]) == 1
    assert final_state["selected_result"] is not None
    assert final_state["selected_result"]["url"] == "https://example.org/article.html"

    # Ingestion should NOT be triggered
    assert trigger.call_count == 0
    assert final_state["ingestion_result"] is None  # Never reached ingestion node


def test_missing_runtime_worker() -> None:
    """Test error when runtime_worker not in context."""
    # Setup: Context WITHOUT runtime_worker
    trigger = StubIngestionTrigger(outcome={})

    context = _base_context()
    context.update(
        {
            # NO runtime_worker!
            "runtime_trigger": trigger,
        }
    )

    input_state: ExternalKnowledgeState = {
        "query": "test query",
        "collection_id": "col-1",
        "enable_hitl": False,
        "auto_ingest": True,
        "context": context,
        "search_results": [],
        "selected_result": None,
        "ingestion_result": None,
        "error": None,
    }

    # Execute
    external_knowledge_graph = build_external_knowledge_graph()
    final_state = external_knowledge_graph.invoke(input_state)

    # Verify: Error is set because worker is missing
    assert final_state["error"] is not None
    assert "No search worker" in final_state["error"]
    assert final_state["search_results"] == []
    assert final_state["selected_result"] is None
    assert trigger.call_count == 0
