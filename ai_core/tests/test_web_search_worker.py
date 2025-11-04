from __future__ import annotations

from typing import Sequence

import pytest

from ai_core.tools.web_search import (
    ProviderSearchResult,
    SearchAdapter,
    SearchAdapterResponse,
    SearchProviderBadResponse,
    SearchProviderQuotaExceeded,
    SearchProviderTimeout,
    WebSearchContext,
    WebSearchWorker,
)


class _FakeAdapter(SearchAdapter):
    def __init__(
        self,
        provider: str = "serp",
        *,
        responses: Sequence[SearchAdapterResponse] | None = None,
        side_effects: Sequence[Exception] | None = None,
    ) -> None:
        self._provider = provider
        self.responses = list(responses or [])
        self.side_effects = list(side_effects or [])

    def search(self, query: str, *, limit: int) -> SearchAdapterResponse:  # type: ignore[override]
        if self.side_effects:
            exc = self.side_effects.pop(0)
            raise exc
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("no response configured")

    @property
    def provider(self) -> str:  # type: ignore[override]
        return self._provider


@pytest.fixture
def context() -> WebSearchContext:
    return WebSearchContext(
        tenant_id="tenant-x",
        trace_id="trace-123",
        workflow_id="wf-456",
        case_id="case-789",
        run_id="run-abc",
        worker_call_id="",
    )


def _result(url: str, *, score: float | None = None, content_type: str | None = None) -> ProviderSearchResult:
    return ProviderSearchResult(
        url=url,
        title="Example Result",
        snippet="Snippet text",
        source="example",
        score=score,
        content_type=content_type,
    )


def test_successful_search(monkeypatch: pytest.MonkeyPatch, context: WebSearchContext) -> None:
    captured_span: dict[str, object] = {}

    def _capture_span(name: str, *, attributes: dict[str, object], trace_id: str | None = None) -> None:
        assert name == "tool.web_search"
        captured_span.update(attributes)
        assert trace_id == context.trace_id

    monkeypatch.setattr("ai_core.tools.web_search.record_span", _capture_span)

    adapter = _FakeAdapter(
        responses=[
            SearchAdapterResponse(
                results=[
                    _result("https://example.com/path?utm_source=test"),
                    _result("https://example.com/path"),
                    _result("https://files.example.com/doc.pdf?gclid=abc", content_type="application/pdf"),
                ],
                status_code=200,
            )
        ]
    )
    worker = WebSearchWorker(adapter, max_results=5, sleep=lambda _: None, timer=lambda: 100.0)

    response = worker.run(query=" hello world ", context=context)

    assert response.outcome.decision == "ok"
    assert response.outcome.rationale == "search_completed"
    assert response.outcome.meta["tenant_id"] == context.tenant_id
    assert response.outcome.meta["worker_call_id"]
    assert response.outcome.meta["latency_ms"] == 0
    assert response.outcome.meta["result_count"] == 2
    assert response.outcome.meta["normalized_result_count"] == 2
    assert response.outcome.meta["raw_result_count"] == 3
    assert len(response.results) == 2
    assert str(response.results[0].url) == "https://example.com/path"
    assert response.results[1].is_pdf is True
    assert captured_span["result.count"] == 2
    assert captured_span["normalized_result_count"] == 2
    assert captured_span["raw_result_count"] == 3
    assert captured_span["http.status"] == 200
    assert captured_span["provider"] == adapter.provider


def test_timeout_error(monkeypatch: pytest.MonkeyPatch, context: WebSearchContext) -> None:
    captured_span: dict[str, object] = {}

    def _capture_span(name: str, *, attributes: dict[str, object], trace_id: str | None = None) -> None:
        captured_span.update(attributes)

    monkeypatch.setattr("ai_core.tools.web_search.record_span", _capture_span)

    adapter = _FakeAdapter(side_effects=[SearchProviderTimeout("timeout", retry_in_ms=1200)])
    worker = WebSearchWorker(adapter, timer=lambda: 1.0, sleep=lambda _: None, max_attempts=1)

    response = worker.run(query="timeout", context=context)

    assert response.outcome.decision == "error"
    assert response.outcome.rationale == "provider_timeout"
    assert response.results == []
    error = response.outcome.meta["error"]
    assert error["kind"] == "SearchProviderTimeout"
    assert error["retry_in_ms"] == 1200
    assert response.outcome.meta["http_status"] == 504
    assert captured_span["http.status"] == 504
    assert captured_span["error.kind"] == "SearchProviderTimeout"


def test_quota_exceeded(monkeypatch: pytest.MonkeyPatch, context: WebSearchContext) -> None:
    captured_span: dict[str, object] = {}

    def _capture_span(name: str, *, attributes: dict[str, object], trace_id: str | None = None) -> None:
        captured_span.update(attributes)

    monkeypatch.setattr("ai_core.tools.web_search.record_span", _capture_span)

    adapter = _FakeAdapter(side_effects=[SearchProviderQuotaExceeded("quota", retry_in_ms=2500)])
    worker = WebSearchWorker(adapter, timer=lambda: 2.0, sleep=lambda _: None, max_attempts=1)

    response = worker.run(query="quota", context=context)

    assert response.outcome.decision == "error"
    assert response.outcome.rationale == "provider_rate_limited"
    assert response.outcome.meta["http_status"] == 429
    assert response.outcome.meta["error"]["retry_in_ms"] == 2500
    assert response.results == []
    assert captured_span["http.status"] == 429


def test_deduplication_and_validation(monkeypatch: pytest.MonkeyPatch, context: WebSearchContext) -> None:
    monkeypatch.setattr("ai_core.tools.web_search.record_span", lambda *_, **__: None)

    adapter = _FakeAdapter(
        responses=[
            SearchAdapterResponse(
                results=[
                    _result("https://example.com/path?utm_campaign=x"),
                    _result("https://example.com/path?utm_medium=y"),
                    _result("invalid-url"),
                ],
                status_code=200,
            )
        ]
    )
    worker = WebSearchWorker(adapter, max_results=3, sleep=lambda _: None, timer=lambda: 10.0)

    response = worker.run(query="dedupe", context=context)

    assert len(response.results) == 1
    assert str(response.results[0].url) == "https://example.com/path"
    assert response.outcome.meta["result_count"] == 1
    assert response.outcome.meta["normalized_result_count"] == 1
    assert response.outcome.meta["raw_result_count"] == 3


def test_invalid_query_only_operators(monkeypatch: pytest.MonkeyPatch, context: WebSearchContext) -> None:
    captured_span: dict[str, object] = {}

    def _capture_span(name: str, *, attributes: dict[str, object], trace_id: str | None = None) -> None:
        assert name == "tool.web_search"
        captured_span.update(attributes)
        assert trace_id == context.trace_id

    monkeypatch.setattr("ai_core.tools.web_search.record_span", _capture_span)

    adapter = _FakeAdapter(
        responses=[
            SearchAdapterResponse(
                results=[_result("https://example.com")],
                status_code=200,
            )
        ]
    )
    worker = WebSearchWorker(adapter, max_results=3, sleep=lambda _: None, timer=lambda: 42.0)

    response = worker.run(query="\u200b  site:  \u200c", context=context)

    assert response.outcome.decision == "error"
    assert response.outcome.rationale == "invalid_query"
    assert response.outcome.meta["error"]["kind"] == "ValidationError"
    assert response.outcome.meta["normalized_result_count"] == 0
    assert response.outcome.meta["raw_result_count"] == 0
    assert response.results == []
    assert captured_span["error.kind"] == "ValidationError"
    assert captured_span["normalized_result_count"] == 0
    assert captured_span["raw_result_count"] == 0


def test_bad_response_error(monkeypatch: pytest.MonkeyPatch, context: WebSearchContext) -> None:
    captured_span: dict[str, object] = {}

    def _capture_span(name: str, *, attributes: dict[str, object], trace_id: str | None = None) -> None:
        captured_span.update(attributes)

    monkeypatch.setattr("ai_core.tools.web_search.record_span", _capture_span)

    adapter = _FakeAdapter(side_effects=[SearchProviderBadResponse("bad")])
    worker = WebSearchWorker(adapter, timer=lambda: 3.0, sleep=lambda _: None)

    response = worker.run(query="bad", context=context)

    assert response.outcome.decision == "error"
    assert response.outcome.rationale == "provider_bad_response"
    assert response.outcome.meta["http_status"] == 502
    assert captured_span["http.status"] == 502
    assert response.results == []
