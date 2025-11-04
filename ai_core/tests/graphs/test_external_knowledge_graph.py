"""Tests for the ExternalKnowledgeGraph orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from ai_core.graphs import external_knowledge_graph as ekg
from ai_core.graphs.external_knowledge_graph import (
    CrawlerIngestionOutcome,
    ExternalKnowledgeGraphConfig,
)
from ai_core.tools.web_search import (
    BaseSearchAdapter,
    ProviderSearchResult,
    SearchAdapterResponse,
    WebSearchWorker,
)


class StubSearchAdapter(BaseSearchAdapter):
    """Search adapter returning a pre-seeded list of results."""

    def __init__(self, provider: str, results: list[ProviderSearchResult]) -> None:
        self.provider_name = provider
        self._results = results

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        return SearchAdapterResponse(
            results=self._results[:max_results], status_code=200
        )


@dataclass
class StubIngestionAdapter:
    """Track calls to the ingestion adapter for assertions."""

    outcome: CrawlerIngestionOutcome
    last_payload: dict[str, Any] | None = None
    call_count: int = 0

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> CrawlerIngestionOutcome:
        self.call_count += 1
        self.last_payload = {
            "url": url,
            "collection_id": collection_id,
            "context": dict(context),
        }
        return self.outcome


class StubReviewEmitter:
    """Collect pending review payloads."""

    def __init__(self) -> None:
        self.emitted: list[Mapping[str, Any]] = []

    def emit(self, payload: Mapping[str, Any]) -> None:
        self.emitted.append(dict(payload))


class FailingReviewEmitter:
    """Emitter that always raises an exception."""

    def emit(self, payload: Mapping[str, Any]) -> None:  # pragma: no cover - simple
        raise RuntimeError("emitter offline")


@pytest.fixture
def observation_collector(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def _capture(**fields: Any) -> None:
        metadata = fields.get("metadata")
        if isinstance(metadata, Mapping):
            captured.append(dict(metadata))

    monkeypatch.setattr(ekg, "update_observation", _capture)
    return captured


def _make_worker(results: list[ProviderSearchResult]) -> WebSearchWorker:
    adapter = StubSearchAdapter("stub", results)
    return WebSearchWorker(adapter, max_results=5, oversample_factor=1)


def _base_meta() -> dict[str, str]:
    return {
        "tenant_id": "tenant-A",
        "workflow_id": "workflow-42",
        "case_id": "case-7",
    }


def test_external_knowledge_graph_ingests_pdf_without_hitl(
    observation_collector: list[dict[str, Any]],
) -> None:
    pdf_result = ProviderSearchResult(
        url="https://example.org/report.pdf",
        title="Climate Report",
        snippet="Comprehensive report covering emissions and climate change impacts.",
        source="example",
        score=0.8,
        content_type="application/pdf",
    )
    html_result = ProviderSearchResult(
        url="https://example.org/blog",
        title="Blog",
        snippet="Long-form analysis describing mitigation strategies in depth.",
        source="example",
        score=0.9,
        content_type="text/html",
    )
    worker = _make_worker([html_result, pdf_result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="ingested",
            crawler_decision="ingested",
            document_id="doc-123",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
    )
    graph._search_worker = worker

    state, result = graph.run(
        {"query": "climate report", "collection_id": "collection-1"},
        meta=_base_meta(),
    )

    assert result["outcome"] == "ingested"
    assert result["document_id"] == "doc-123"
    assert result["selected_url"] == "https://example.org/report.pdf"
    assert ingestion_adapter.call_count == 1
    assert ingestion_adapter.last_payload is not None
    assert (
        ingestion_adapter.last_payload["context"]["graph_name"] == "external_knowledge"
    )
    assert "ingestion_run_id" in ingestion_adapter.last_payload["context"]

    telemetry = result["telemetry"]
    assert telemetry["ids"]["tenant_id"] == "tenant-A"
    assert telemetry["nodes"]["k_search"]["provider"] == "stub"
    assert telemetry["nodes"]["k_search"]["graph_name"] == "external_knowledge"
    assert state["transitions"][-1]["decision"] == "ingested"

    for metadata in observation_collector:
        assert metadata["tenant_id"] == "tenant-A"
        assert metadata["collection_id"] == "collection-1"
        assert metadata["workflow_id"] == "workflow-42"
        assert metadata["case_id"] == "case-7"
        assert metadata["run_id"]


def test_external_knowledge_graph_handles_no_suitable_candidate(
    observation_collector: list[dict[str, Any]],
) -> None:
    short_result = ProviderSearchResult(
        url="https://spam.example.com/page",
        title="Spam",
        snippet="Too short.",
        source="spam",
        score=0.1,
        content_type="text/html",
    )
    worker = _make_worker([short_result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="skipped",
            crawler_decision="skipped",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(
            blocked_domains=frozenset({"spam.example.com"})
        ),
    )
    graph._search_worker = worker

    state, result = graph.run(
        {"query": "irrelevant", "collection_id": "collection-2"},
        meta=_base_meta(),
    )

    assert result["outcome"] == "nothing_suitable"
    assert result["document_id"] is None
    assert ingestion_adapter.call_count == 0
    assert state["transitions"][-1]["decision"] == "nothing_suitable"

    for metadata in observation_collector:
        assert metadata["tenant_id"] == "tenant-A"
        assert metadata["collection_id"] == "collection-2"


def test_external_knowledge_graph_hitl_rejection_skips_ingestion(
    observation_collector: list[dict[str, Any]],
) -> None:
    result = ProviderSearchResult(
        url="https://example.org/insight",
        title="Insight",
        snippet="Detailed research article covering governance policies at length.",
        source="example",
        score=0.7,
        content_type="text/html",
    )
    worker = _make_worker([result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="ingested",
            crawler_decision="ingested",
            document_id="doc-999",
        )
    )
    emitter = StubReviewEmitter()
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
        review_emitter=emitter,
    )
    graph._search_worker = worker

    initial_state, pending = graph.run(
        {
            "query": "research",
            "collection_id": "collection-3",
            "enable_hitl": True,
        },
        meta=_base_meta(),
    )

    assert pending["outcome"] == "error"
    assert pending["telemetry"]["review"]["status"] == "pending"
    assert emitter.emitted and emitter.emitted[0]["status"] == "PENDING_REVIEW"
    assert ingestion_adapter.call_count == 0

    initial_state["review_response"] = {"approved": False}
    resumed_state, final_result = graph.run(initial_state, meta=_base_meta())

    assert final_result["outcome"] == "rejected"
    assert final_result["document_id"] is None
    assert ingestion_adapter.call_count == 0
    assert final_result["telemetry"]["review"]["status"] == "rejected"
    assert resumed_state["review"]["status"] == "rejected"

    for metadata in observation_collector:
        assert metadata["tenant_id"] == "tenant-A"
        assert metadata["collection_id"] in {"collection-3"}


def test_external_knowledge_graph_run_until_after_search() -> None:
    result = ProviderSearchResult(
        url="https://example.org/resource",
        title="Resource",
        snippet="""Detailed analysis describing technology trends across the industry landscape.""",
        source="Example",
        score=0.6,
        content_type="text/html",
    )
    worker = _make_worker([result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="skipped",
            crawler_decision="skipped",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
    )
    graph._search_worker = worker

    state, payload = graph.run(
        {
            "query": "technology trends",
            "collection_id": "collection-5",
            "run_until": "after_search",
        },
        meta=_base_meta(),
    )

    assert payload["outcome"] == "stopped_after_search"
    assert payload["telemetry"]["stop_reason"] == "after_search"
    assert [entry["node"] for entry in state["transitions"]] == ["k_search"]
    assert ingestion_adapter.call_count == 0


def test_external_knowledge_graph_run_until_after_selection() -> None:
    result = ProviderSearchResult(
        url="https://example.org/resource",
        title="Resource",
        snippet="""Detailed analysis describing technology trends across the industry landscape.""",
        source="Example",
        score=0.6,
        content_type="text/html",
    )
    worker = _make_worker([result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="ingested",
            crawler_decision="ingested",
            document_id="doc-900",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
    )
    graph._search_worker = worker

    state, payload = graph.run(
        {
            "query": "technology trends",
            "collection_id": "collection-6",
            "run_until": "after_selection",
        },
        meta=_base_meta(),
    )

    assert payload["outcome"] == "stopped_after_selection"
    assert payload["telemetry"]["stop_reason"] == "after_selection"
    nodes = {entry["node"] for entry in state["transitions"]}
    assert nodes == {"k_search", "k_filter_and_select"}
    assert ingestion_adapter.call_count == 0


def test_external_knowledge_graph_run_until_review_complete() -> None:
    result = ProviderSearchResult(
        url="https://example.org/resource",
        title="Resource",
        snippet="""Detailed analysis describing technology trends across the industry landscape.""",
        source="Example",
        score=0.6,
        content_type="text/html",
    )
    worker = _make_worker([result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="ingested",
            crawler_decision="ingested",
            document_id="doc-901",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
    )
    graph._search_worker = worker

    state, pending = graph.run(
        {
            "query": "technology trends",
            "collection_id": "collection-7",
            "enable_hitl": True,
            "run_until": "review_complete",
        },
        meta=_base_meta(),
    )

    assert pending["telemetry"]["review"]["status"] == "pending"
    assert ingestion_adapter.call_count == 0

    state["review_response"] = {"approved": True}
    resumed_state, final_result = graph.run(state, meta=_base_meta())

    assert final_result["outcome"] == "stopped_after_review"
    assert final_result["telemetry"]["stop_reason"] == "review_complete"
    assert resumed_state["review"]["status"] == "approved"
    assert ingestion_adapter.call_count == 0


def test_external_knowledge_graph_rejected_count_accounts_for_topn_cut() -> None:
    long_snippet = (
        "Extensive discussion of industry developments and analytical observations "
        "covering multiple quarters and stakeholder perspectives."
    )
    results = [
        ProviderSearchResult(
            url=f"https://example.org/resource-{idx}",
            title=f"Resource {idx}",
            snippet=long_snippet,
            source="Example",
            score=1.0 - (idx * 0.1),
            content_type="text/html",
        )
        for idx in range(3)
    ]
    results.append(
        ProviderSearchResult(
            url="https://example.net/short",
            title="Short",
            snippet="tiny",
            source="Example",
            score=0.1,
            content_type="text/html",
        )
    )
    worker = _make_worker(results)
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="skipped",
            crawler_decision="skipped",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(top_n=1),
    )
    graph._search_worker = worker

    state, payload = graph.run(
        {
            "query": "market analysis",
            "collection_id": "collection-topn",
            "run_until": "after_selection",
        },
        meta=_base_meta(),
    )

    assert payload["outcome"] == "stopped_after_selection"
    selection_state = state["selection"]
    transition_meta = selection_state["transition"]["meta"]
    assert transition_meta["rejected_count"] == 3
    assert len(selection_state["shortlisted"]) == 1


def test_external_knowledge_graph_blocks_blocked_subdomains() -> None:
    result = ProviderSearchResult(
        url="https://www.spam.example.com/resource",
        title="Spam",
        snippet="""Detailed yet untrusted article with plenty of misleading statements in a long body.""",
        source="Spam",
        score=0.2,
        content_type="text/html",
    )
    worker = _make_worker([result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="skipped",
            crawler_decision="skipped",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(blocked_domains=frozenset({"example.com"})),
    )
    graph._search_worker = worker

    state, payload = graph.run(
        {"query": "spam", "collection_id": "collection-blocked"},
        meta=_base_meta(),
    )

    assert payload["outcome"] == "nothing_suitable"
    assert state["selection"]["transition"]["meta"]["rejected_count"] == 1
    assert ingestion_adapter.call_count == 0


def test_external_knowledge_graph_hitl_override_allows_valid_url() -> None:
    result = ProviderSearchResult(
        url="https://example.org/report",
        title="Report",
        snippet="""Detailed overview covering regulations and compliance frameworks across sectors.""",
        source="Example",
        score=0.9,
        content_type="text/html",
    )
    worker = _make_worker([result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="ingested",
            crawler_decision="ingested",
            document_id="doc-override",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
    )
    graph._search_worker = worker

    state, pending = graph.run(
        {
            "query": "regulations",
            "collection_id": "collection-override",
            "enable_hitl": True,
        },
        meta=_base_meta(),
    )

    assert pending["telemetry"]["review"]["status"] == "pending"

    override_url = "https://alt.example.net/curated"
    state["review_response"] = {"approved": True, "override_url": override_url}
    resumed_state, final_result = graph.run(state, meta=_base_meta())

    assert final_result["outcome"] == "ingested"
    assert resumed_state["review"]["transition"]["rationale"] == "hitl_approved"
    assert ingestion_adapter.last_payload is not None
    assert ingestion_adapter.last_payload["url"] == override_url


def test_external_knowledge_graph_hitl_override_blocklist_rejection() -> None:
    result = ProviderSearchResult(
        url="https://example.org/report",
        title="Report",
        snippet="""Detailed overview covering regulations and compliance frameworks across sectors.""",
        source="Example",
        score=0.9,
        content_type="text/html",
    )
    worker = _make_worker([result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="ingested",
            crawler_decision="ingested",
            document_id="doc-override",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(
            blocked_domains=frozenset({"blocked.example"})
        ),
    )
    graph._search_worker = worker

    state, pending = graph.run(
        {
            "query": "regulations",
            "collection_id": "collection-override",
            "enable_hitl": True,
        },
        meta=_base_meta(),
    )

    assert pending["telemetry"]["review"]["status"] == "pending"

    state["review_response"] = {
        "approved": True,
        "override_url": "https://sub.blocked.example/resource",
    }
    resumed_state, final_result = graph.run(state, meta=_base_meta())

    assert final_result["outcome"] == "rejected"
    assert (
        resumed_state["review"]["transition"]["rationale"]
        == "override_url_blocked_or_invalid"
    )
    assert ingestion_adapter.call_count == 0


def test_external_knowledge_graph_hitl_emitter_failure_records_error() -> None:
    result = ProviderSearchResult(
        url="https://example.org/resource",
        title="Resource",
        snippet="""Detailed analysis describing technology trends across the industry landscape.""",
        source="Example",
        score=0.6,
        content_type="text/html",
    )
    worker = _make_worker([result])
    ingestion_adapter = StubIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="ingested",
            crawler_decision="ingested",
            document_id="doc-111",
        )
    )
    graph = ekg.build_graph(
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
        review_emitter=FailingReviewEmitter(),
    )
    graph._search_worker = worker

    state, payload = graph.run(
        {
            "query": "technology trends",
            "collection_id": "collection-7",
            "enable_hitl": True,
        },
        meta=_base_meta(),
    )

    assert payload["outcome"] == "error"
    transitions = {entry["node"]: entry for entry in state["transitions"]}
    assert "k_hitl_gate_emit" in transitions
    emit = transitions["k_hitl_gate_emit"]
    assert emit["decision"] == "error"
    assert emit["meta"]["error"]["kind"] == "EmitterError"
    assert ingestion_adapter.call_count == 0
