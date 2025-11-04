from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest
from pydantic import SecretStr
from requests.exceptions import Timeout as RequestsTimeout

from ai_core.tools.search_adapters.google import GoogleSearchAdapter
from ai_core.tools.web_search import (
    SearchProviderQuotaExceeded,
    SearchProviderTimeout,
)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = dict(headers or {})

    def json(self) -> Mapping[str, Any]:
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class _FakeSession:
    def __init__(
        self,
        responses: Sequence[_FakeResponse] | None = None,
        exceptions: Sequence[Exception] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._exceptions = list(exceptions or [])
        self.calls: list[dict[str, Any]] = []

    def get(
        self, url: str, *, params: Mapping[str, Any], timeout: float
    ) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        if self._exceptions:
            raise self._exceptions.pop(0)
        if not self._responses:
            raise AssertionError("no response configured")
        return self._responses.pop(0)


def _make_adapter(session: _FakeSession) -> GoogleSearchAdapter:
    return GoogleSearchAdapter(
        api_key=SecretStr("test-api-key"),
        search_engine_id="engine-1",
        session=session,
        timeout=5.0,
    )


def test_search_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_spans: list[dict[str, Any]] = []

    def _capture_span(
        name: str, *, attributes: dict[str, Any], trace_id: str | None = None
    ) -> None:
        assert name == "tool.web_search.provider"
        captured_spans.append(dict(attributes))

    monkeypatch.setattr(
        "ai_core.tools.search_adapters.google.record_span", _capture_span
    )

    response = _FakeResponse(
        status_code=200,
        headers={"X-RateLimit-Remaining": "37"},
        json_data={
            "items": [
                {
                    "title": "Quarterly Report",
                    "snippet": "Detailed PDF report for regulators.",
                    "link": "https://example.com/report.pdf?utm_source=newsletter&ref=42",
                    "mime": "application/pdf",
                    "displayLink": "example.com",
                },
                {
                    "title": "Industry Insight",
                    "snippet": "Long-form analysis of market trends.",
                    "link": "https://example.com/insight?utm_medium=email&ref=1",
                    "displayLink": "Example.com",
                },
            ]
        },
    )
    session = _FakeSession(responses=[response])
    adapter = _make_adapter(session)

    result = adapter.search("regulatory update", max_results=5)

    assert result.status_code == 200
    assert result.quota_remaining == 37
    assert len(result.results) == 2
    assert tuple(result.raw_results)[0].display_link == "example.com"
    assert result.results[0].url == "https://example.com/report.pdf"
    assert result.results[0].content_type == "application/pdf"
    assert result.results[1].url == "https://example.com/insight"
    assert result.results[1].source == "example.com"

    call = session.calls[0]
    assert call["params"]["safe"] == "active"
    assert call["params"]["num"] == "5"

    assert captured_spans
    span = captured_spans[-1]
    assert span["provider"] == "google"
    assert span["http.status"] == 200
    assert span["result.count"] == 2
    assert span["quota.remaining"] == 37


def test_quota_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ai_core.tools.search_adapters.google.record_span", lambda *_, **__: None
    )

    response = _FakeResponse(status_code=403, headers={"Retry-After": "120"})
    adapter = _make_adapter(_FakeSession(responses=[response]))

    with pytest.raises(SearchProviderQuotaExceeded) as exc_info:
        adapter.search("compliance", max_results=4)

    error = exc_info.value
    assert error.http_status == 403
    assert error.retry_in_ms == 120_000


def test_empty_items(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_spans: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "ai_core.tools.search_adapters.google.record_span",
        lambda name, *, attributes, trace_id=None: captured_spans.append(
            dict(attributes)
        ),
    )

    response = _FakeResponse(status_code=200, json_data={})
    adapter = _make_adapter(_FakeSession(responses=[response]))

    result = adapter.search("missing results", max_results=3)

    assert result.status_code == 200
    assert result.results == ()
    assert result.raw_results == ()
    assert captured_spans[-1]["result.count"] == 0


def test_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_spans: list[dict[str, Any]] = []

    def _capture_span(
        name: str, *, attributes: dict[str, Any], trace_id: str | None = None
    ) -> None:
        captured_spans.append(dict(attributes))

    monkeypatch.setattr(
        "ai_core.tools.search_adapters.google.record_span", _capture_span
    )

    session = _FakeSession(exceptions=[RequestsTimeout("timeout")])
    adapter = _make_adapter(session)

    with pytest.raises(SearchProviderTimeout):
        adapter.search("delayed", max_results=2)

    assert captured_spans
    assert captured_spans[0]["error.kind"] == "Timeout"
