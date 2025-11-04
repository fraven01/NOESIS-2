"""Acceptance and telemetry tests for ExternalKnowledgeGraph."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pytest

from ai_core.graphs.external_knowledge_graph import (
    CrawlerIngestionOutcome,
    ExternalKnowledgeGraph,
    ExternalKnowledgeGraphConfig,
)
from ai_core.tools.web_search import (
    BaseSearchAdapter,
    ProviderSearchResult,
    SearchAdapterResponse,
    SearchProviderError,
    SearchProviderQuotaExceeded,
    SearchProviderTimeout,
    WebSearchWorker,
)


class StaticSearchAdapter(BaseSearchAdapter):
    """Search adapter returning a static sequence of results."""

    def __init__(self, provider: str, results: list[ProviderSearchResult], status_code: int = 200) -> None:
        self.provider_name = provider
        self._results = results
        self._status_code = status_code

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        return SearchAdapterResponse(results=self._results[:max_results], status_code=self._status_code)


class RaisingSearchAdapter(BaseSearchAdapter):
    """Adapter that raises a configured provider error when searched."""

    def __init__(self, provider: str, error: SearchProviderError) -> None:
        self.provider_name = provider
        self._error = error

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:  # pragma: no cover - never returns
        raise self._error


@dataclass
class RecordingIngestionAdapter:
    """Ingestion adapter that records invocations."""

    outcome: CrawlerIngestionOutcome
    calls: list[dict[str, Any]] = field(default_factory=list)

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> CrawlerIngestionOutcome:
        self.calls.append({
            "url": url,
            "collection_id": collection_id,
            "context": dict(context),
        })
        return self.outcome


class RecordingReviewEmitter:
    """Collect review payloads for assertions."""

    def __init__(self) -> None:
        self.emitted: list[Mapping[str, Any]] = []

    def emit(self, payload: Mapping[str, Any]) -> None:
        self.emitted.append(dict(payload))


def _base_meta() -> dict[str, str]:
    return {
        "tenant_id": "tenant-1",
        "workflow_id": "workflow-7",
        "case_id": "case-99",
    }


@pytest.fixture
def capture_observations(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from ai_core.graphs import external_knowledge_graph as ekg
    from ai_core.infra import observability

    captured: list[dict[str, Any]] = []

    def _capture(**fields: Any) -> None:
        metadata = fields.get("metadata")
        if isinstance(metadata, Mapping):
            captured.append(dict(metadata))

    monkeypatch.setattr(ekg, "update_observation", _capture)
    monkeypatch.setattr(observability, "tracing_enabled", lambda: False)
    return captured


def _find_by_key(entries: Iterable[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    for entry in entries:
        if key in entry:
            return entry
    raise AssertionError(f"No entry contains key {key!r}")


def test_id_propagation_and_span_metadata(capture_observations: list[dict[str, Any]]) -> None:
    pdf_result = ProviderSearchResult(
        url="https://example.com/files/report.pdf",
        title="Quarterly Report",
        snippet="""This PDF document details quarterly financial and operational metrics across regions.""",
        source="example",
        score=0.3,
        content_type="application/pdf",
    )
    duplicate_with_tracking = ProviderSearchResult(
        url="https://example.com/articles/insight?utm_source=newsletter",
        title="Insight Article",
        snippet="""Comprehensive insight article covering regulatory changes affecting the energy sector.""",
        source="example",
        score=0.7,
        content_type="text/html",
    )
    duplicate_canonical = ProviderSearchResult(
        url="https://example.com/articles/insight",
        title="Insight Article",
        snippet="""Comprehensive insight article covering regulatory changes affecting the energy sector.""",
        source="example",
        score=0.8,
        content_type="text/html",
    )
    spam_result = ProviderSearchResult(
        url="https://spam.example.com/doorway",
        title="Doorway",
        snippet="Doorway page",
        source="spam",
        score=0.1,
        content_type="text/html",
    )
    adapter = StaticSearchAdapter(
        "static",
        [duplicate_with_tracking, duplicate_canonical, pdf_result, spam_result],
    )
    worker = WebSearchWorker(adapter, max_results=5, oversample_factor=1)
    ingestion_adapter = RecordingIngestionAdapter(
        outcome=CrawlerIngestionOutcome(
            decision="ingested",
            crawler_decision="ingested",
            document_id="doc-777",
        )
    )
    graph = ExternalKnowledgeGraph(
        search_worker=worker,
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
    )

    state, result = graph.run({"query": "industry outlook", "collection_id": "collection-42"}, meta=_base_meta())

    trace_id = state["meta"]["context"]["trace_id"]
    run_id = state["meta"]["run_id"]
    assert trace_id
    assert run_id
    assert result["telemetry"]["ids"]["trace_id"] == trace_id
    assert result["telemetry"]["ids"]["run_id"] == run_id

    transitions = {entry["node"]: entry for entry in state["transitions"]}
    assert set(transitions) == {"k_search", "k_filter_and_select", "k_trigger_ingestion"}

    search_meta = transitions["k_search"]["meta"]
    filter_meta = transitions["k_filter_and_select"]["meta"]
    ingestion_meta = transitions["k_trigger_ingestion"]["meta"]

    assert search_meta["graph_name"] == "external_knowledge"
    assert filter_meta["graph_name"] == "external_knowledge"
    assert ingestion_meta["graph_name"] == "external_knowledge"
    assert search_meta["trace_id"] == trace_id
    assert search_meta["run_id"] == run_id
    assert search_meta["worker_call_id"]
    assert "ingestion_run_id" not in search_meta

    assert filter_meta["trace_id"] == trace_id
    assert filter_meta["run_id"] == run_id
    assert filter_meta["rejected_count"] >= 1
    assert filter_meta.get("ingestion_run_id") is None

    ingestion_run_id = ingestion_meta["ingestion_run_id"]
    assert ingestion_run_id
    assert result["telemetry"]["ids"]["ingestion_run_id"] == ingestion_run_id
    assert ingestion_meta["collection_id"] == "collection-42"
    assert ingestion_meta["trace_id"] == trace_id
    assert ingestion_meta["run_id"] == run_id

    search_results = state["search"]["results"]
    assert search_results and len(search_results) == 3
    assert sum(1 for item in search_results if item["url"] == "https://example.com/articles/insight") == 1
    selected = state["selection"]["selected"]
    assert selected and selected["is_pdf"] is True
    assert result["selected_url"].endswith(".pdf")
    assert result["document_id"] == "doc-777"
    assert result["telemetry"]["ids"]["ingestion_run_id"] == ingestion_run_id

    assert ingestion_adapter.calls
    ingestion_call = ingestion_adapter.calls[0]
    assert ingestion_call["collection_id"] == "collection-42"
    assert ingestion_call["context"]["collection_id"] == "collection-42"

    assert capture_observations, "expected span metadata to be captured"
    search_span = _find_by_key(capture_observations, "provider")
    assert search_span["tenant_id"] == "tenant-1"
    assert search_span["collection_id"] == "collection-42"
    assert search_span["worker_call_id"] == search_meta["worker_call_id"]
    assert "ingestion_run_id" not in search_span
    assert search_span["graph_name"] == "external_knowledge"

    filter_span = _find_by_key(capture_observations, "rejected_count")
    assert filter_span["collection_id"] == "collection-42"
    assert filter_span["run_id"] == run_id
    assert filter_span["graph_name"] == "external_knowledge"

    trigger_span = _find_by_key(capture_observations, "crawler_decision")
    assert trigger_span["ingestion_run_id"] == ingestion_run_id
    assert trigger_span["collection_id"] == "collection-42"
    assert trigger_span["graph_name"] == "external_knowledge"

    composed = json.dumps({
        "state": state,
        "result": result,
        "spans": capture_observations,
    })
    assert "request_id" not in composed

    first_worker_call_id = search_meta["worker_call_id"]
    first_ingestion_run_id = ingestion_run_id

    capture_observations.clear()
    state_two, result_two = graph.run({"query": "industry outlook", "collection_id": "collection-42"}, meta=_base_meta())
    transitions_two = {entry["node"]: entry for entry in state_two["transitions"]}
    assert state_two["meta"]["run_id"] != run_id
    assert transitions_two["k_search"]["meta"]["worker_call_id"] != first_worker_call_id
    assert state_two["meta"].get("ingestion_run_id") != first_ingestion_run_id
    assert result_two["telemetry"]["ids"]["run_id"] == state_two["meta"]["run_id"]


def test_hitl_pause_and_resume(capture_observations: list[dict[str, Any]]) -> None:
    result = ProviderSearchResult(
        url="https://example.com/insight",
        title="Insight",
        snippet="""Thorough write-up discussing compliance and governance adjustments introduced this quarter.""",
        source="example",
        score=0.6,
        content_type="text/html",
    )
    worker = WebSearchWorker(StaticSearchAdapter("static", [result]), max_results=3, oversample_factor=1)
    ingestion_adapter = RecordingIngestionAdapter(
        outcome=CrawlerIngestionOutcome(decision="ingested", crawler_decision="ingested", document_id="doc-hitl"),
    )
    emitter = RecordingReviewEmitter()
    graph = ExternalKnowledgeGraph(
        search_worker=worker,
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
        review_emitter=emitter,
    )

    state, pending_result = graph.run(
        {"query": "governance update", "collection_id": "collection-hitl", "enable_hitl": True},
        meta=_base_meta(),
    )

    assert pending_result["telemetry"]["review"]["status"] == "pending"
    assert state["review"]["status"] == "pending"
    assert emitter.emitted and emitter.emitted[0]["status"] == "PENDING_REVIEW"

    review_payload = emitter.emitted[0]
    trace_id = state["meta"]["context"]["trace_id"]
    run_id = state["meta"]["run_id"]
    assert review_payload["trace_id"] == trace_id
    assert review_payload["run_id"] == run_id
    assert review_payload["collection_id"] == "collection-hitl"
    assert review_payload["review_token"] == state["review"]["payload"]["review_token"]

    state["review_response"] = {"approved": False}
    resumed_state, final_result = graph.run(state, meta=_base_meta())

    assert final_result["outcome"] == "rejected"
    assert resumed_state["review"]["status"] == "rejected"
    assert final_result["telemetry"]["review"]["status"] == "rejected"
    assert ingestion_adapter.calls == []

    transitions = {entry["node"]: entry for entry in resumed_state["transitions"]}
    hitl_meta = transitions["k_hitl_gate"]["meta"]
    assert hitl_meta["review_token"] == review_payload["review_token"]
    assert hitl_meta["approved"] is False
    assert hitl_meta["run_id"] == run_id
    assert final_result["telemetry"]["ids"]["run_id"] == run_id

    capture = _find_by_key(capture_observations, "review_token")
    assert capture["review_token"] == review_payload["review_token"]
    assert capture["trace_id"] == trace_id


@pytest.mark.parametrize(
    "error",
    [
        SearchProviderTimeout("timeout", retry_in_ms=1200, http_status=504),
        SearchProviderQuotaExceeded("quota", retry_in_ms=2500, http_status=429),
    ],
)
def test_provider_error_propagates_with_telemetry(
    error: SearchProviderError,
    capture_observations: list[dict[str, Any]],
) -> None:
    worker = WebSearchWorker(
        RaisingSearchAdapter("failing", error),
        max_attempts=1,
        max_results=3,
        oversample_factor=1,
        sleep=lambda _: None,
    )
    ingestion_adapter = RecordingIngestionAdapter(
        outcome=CrawlerIngestionOutcome(decision="skipped", crawler_decision="skipped"),
    )
    graph = ExternalKnowledgeGraph(
        search_worker=worker,
        ingestion_adapter=ingestion_adapter,
        config=ExternalKnowledgeGraphConfig(),
    )

    state, result = graph.run({"query": "resilience" , "collection_id": "collection-error"}, meta=_base_meta())

    assert result["outcome"] == "error"
    transitions = {entry["node"]: entry for entry in state["transitions"]}
    assert transitions["k_search"]["decision"] == "error"
    assert transitions["k_search"]["meta"]["error"]["kind"] == type(error).__name__
    assert "ingestion_run_id" not in result["telemetry"]["ids"]
    assert ingestion_adapter.calls == []

    telemetry_error = result["telemetry"]["nodes"]["k_search"]["error"]
    assert telemetry_error["kind"] == type(error).__name__

    error_span = _find_by_key(capture_observations, "error.kind")
    assert error_span["error.kind"] == type(error).__name__
    assert error_span["trace_id"] == result["telemetry"]["ids"]["trace_id"]
