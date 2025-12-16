from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping

import pytest

from ai_core.graphs.technical.crawler_ingestion_graph import GraphTransition
from ai_core.graphs.technical.transition_contracts import StandardTransitionResult
from ai_core.infra import object_store
from common.assets import perceptual_hash, sha256_bytes
from crawler.errors import CrawlerError, ErrorClass
from crawler.fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from crawler.worker import CrawlerWorker


def _build_fetch_result(
    *,
    status: FetchStatus = FetchStatus.FETCHED,
    payload: bytes | None = b"payload",
    error: CrawlerError | None = None,
    request_metadata: Mapping[str, object] | None = None,
    content_type: str = "text/plain",
    canonical_source: str = "https://example.com/docs",
) -> FetchResult:
    metadata = FetchMetadata(
        status_code=200,
        content_type=content_type,
        etag="abc",
        last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
        content_length=len(payload or b"") if payload is not None else None,
    )
    telemetry = FetchTelemetry(latency=0.42, bytes_downloaded=len(payload or b""))
    request = FetchRequest(
        canonical_source=canonical_source,
        politeness=PolitenessContext(host="example.com"),
        metadata=request_metadata or {},
    )
    return FetchResult(
        status=status,
        request=request,
        payload=payload,
        metadata=metadata,
        telemetry=telemetry,
        error=error,
    )


class _StubFetcher:
    def __init__(self, result: FetchResult) -> None:
        self.result = result
        self.requests: list[FetchRequest] = []

    def fetch(
        self, request: FetchRequest
    ) -> FetchResult:  # pragma: no cover - simple passthrough
        self.requests.append(request)
        return self.result


class _MappingFetcher:
    def __init__(self, mapping: Mapping[str, FetchResult]) -> None:
        self.mapping = mapping
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResult:
        self.requests.append(request)
        return self.mapping[request.canonical_source]


_SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _StubTask:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    def delay(self, state: dict[str, object], meta: dict[str, object]):  # type: ignore[no-untyped-def]
        self.calls.append((state, meta))
        return SimpleNamespace(id="task-123")


@pytest.mark.django_db
def test_worker_triggers_guardrail_event(tmp_path, monkeypatch) -> None:
    fetch_result = _build_fetch_result(payload=b"payload")
    fetcher = _StubFetcher(fetch_result)

    events: list[tuple[str, dict[str, object]]] = []

    def event_callback(name: str, payload: dict[str, object]) -> None:
        events.append((name, payload))

    # Mock the ingestion graph to emit guardrail event
    def _mock_run_ingestion_graph(state: dict[str, object], meta: dict[str, object]):  # type: ignore[no-untyped-def]
        run_id = state.get("graph_run_id", "test-run")
        transition = GraphTransition(
            StandardTransitionResult(
                phase="guardrails",
                decision="denied",
                reason="policy_denied",
                guardrail=None,
                severity="error",
                context={"policy_events": ("max_document_bytes",)},
            )
        )
        # Emit guardrail denied event
        ingestion_event_emitter = meta.get("ingestion_event_emitter")
        if callable(ingestion_event_emitter):
            payload = {
                "transition": transition.to_dict(),
                "run_id": run_id,
                "document_id": state.get("raw_document", {}).get("document_id"),
                "reason": transition.reason,
                "policy_events": ["max_document_bytes"],
            }
            ingestion_event_emitter("guardrail_denied", payload)
        return state, {
            "decision": transition.decision,
            "reason": transition.reason,
            "severity": transition.severity,
            "phase": transition.phase,
            "graph_run_id": run_id,
            "transitions": {"enforce_guardrails": transition.result},
        }

    class _ExecutingTask:
        def delay(self, state: dict[str, object], meta: dict[str, object]):  # type: ignore[no-untyped-def]
            _mock_run_ingestion_graph(state, meta)
            return SimpleNamespace(id="task-456")

    # No need to mock build_graph anymore
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    worker = CrawlerWorker(
        fetcher,
        ingestion_task=_ExecutingTask(),
        ingestion_event_emitter=event_callback,
    )

    worker.process(
        fetch_result.request,
        tenant_id="tenant-a",
        case_id="case-b",
        document_id="doc-1",
        document_metadata={"source": "crawler"},
    )

    guardrail_events = [
        payload for name, payload in events if name == "guardrail_denied"
    ]
    assert guardrail_events, "expected guardrail_denied event"
    payload = guardrail_events[0]
    assert payload["document_id"] == "doc-1"
    assert payload["reason"] == "policy_denied"
    assert payload["policy_events"] == ["max_document_bytes"]


@pytest.mark.django_db
def test_worker_publishes_ingestion_task(tmp_path, monkeypatch) -> None:
    fetch_result = _build_fetch_result(payload=b"hello world")
    fetcher = _StubFetcher(fetch_result)
    task = _StubTask()
    worker = CrawlerWorker(fetcher, ingestion_task=task)

    request = fetch_result.request
    overrides = {"guardrails": {"max_document_bytes": 1024}}
    metadata = {"provider": "docs", "tags": ["hr"], "source": "integration"}
    meta_overrides = {"trace_id": "trace-1"}

    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    publish_result = worker.process(
        request,
        tenant_id="tenant-a",
        case_id="case-b",
        crawl_id="crawl-1",
        idempotency_key="idemp-1",
        frontier_state={"slot": "default"},
        document_id="doc-1",
        document_metadata=metadata,
        ingestion_overrides=overrides,
        meta_overrides=meta_overrides,
        trace_id="trace-1",
    )

    assert publish_result.status == "published"
    assert publish_result.task_id == "task-123"
    assert len(task.calls) == 1

    state_payload, meta_payload = task.calls[0]
    assert state_payload["tenant_id"] == "tenant-a"
    assert state_payload["case_id"] == "case-b"
    assert state_payload["frontier"] == {"slot": "default"}
    raw_document = state_payload["raw_document"]
    assert raw_document["document_id"] == "doc-1"
    # New external_ref structure
    assert raw_document["metadata"]["external_ref"]["provider"] == "docs"
    assert raw_document["metadata"]["source"] == "integration"
    assert raw_document["metadata"]["origin_uri"] == request.canonical_source
    payload_uri = raw_document["payload_path"]
    assert payload_uri.startswith("objectstore://")
    assert payload_uri.endswith(".bin")

    # Strip scheme to get relative path for file system check
    relative_path = payload_uri.replace("objectstore://", "")
    stored_payload = (object_store.BASE_PATH / relative_path).read_bytes()
    assert stored_payload == fetch_result.payload
    assert state_payload["raw_payload_path"] == payload_uri
    assert state_payload["guardrails"] == overrides["guardrails"]
    assert meta_payload["trace_id"] == "trace-1"
    assert meta_payload["idempotency_key"] == "idemp-1"
    assert meta_payload["crawl_id"] == "crawl-1"


def test_worker_propagates_trace_id_from_request_metadata(
    tmp_path, monkeypatch
) -> None:
    fetch_result = _build_fetch_result(
        payload=b"payload", request_metadata={"trace_id": "trace-from-request"}
    )
    fetcher = _StubFetcher(fetch_result)
    task = _StubTask()
    worker = CrawlerWorker(fetcher, ingestion_task=task)

    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    publish_result = worker.process(
        fetch_result.request,
        tenant_id="tenant-a",
        document_metadata={"source": "crawler"},
    )

    assert publish_result.published
    assert len(task.calls) == 1
    _, meta_payload = task.calls[0]
    assert meta_payload["tenant_id"] == "tenant-a"
    assert meta_payload["trace_id"] == "trace-from-request"
    assert "request_id" not in meta_payload


def test_worker_returns_failure_without_publishing() -> None:
    fetch_error = CrawlerError(
        ErrorClass.TIMEOUT, "timeout", source="https://example.com"
    )
    fetch_result = _build_fetch_result(
        status=FetchStatus.TEMPORARY_ERROR,
        payload=None,
        error=fetch_error,
    )
    fetcher = _StubFetcher(fetch_result)
    task = _StubTask()
    worker = CrawlerWorker(fetcher, ingestion_task=task)

    publish_result = worker.process(
        fetch_result.request,
        tenant_id="tenant-a",
    )

    assert publish_result.status == FetchStatus.TEMPORARY_ERROR.value
    assert publish_result.error is fetch_error
    assert task.calls == []


def test_worker_sets_default_provider_from_source(tmp_path, monkeypatch) -> None:
    fetch_result = _build_fetch_result(payload=b"payload")
    fetcher = _StubFetcher(fetch_result)
    task = _StubTask()
    worker = CrawlerWorker(fetcher, ingestion_task=task)

    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    publish_result = worker.process(
        fetch_result.request,
        tenant_id="tenant-a",
        document_metadata={"source": "crawler"},
    )

    assert publish_result.published
    state_payload, _ = task.calls[0]
    metadata = state_payload["raw_document"]["metadata"]
    # New external_ref structure
    assert metadata["external_ref"]["provider"] == "web"
    assert metadata["source"] == "crawler"


def test_worker_raises_without_source_metadata(tmp_path, monkeypatch) -> None:
    fetch_result = _build_fetch_result(payload=b"payload")
    fetcher = _StubFetcher(fetch_result)
    task = _StubTask()
    worker = CrawlerWorker(fetcher, ingestion_task=task)

    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    with pytest.raises(ValueError) as excinfo:
        worker.process(
            fetch_result.request,
            tenant_id="tenant-a",
        )

    assert str(excinfo.value) == "document_metadata.source_required"


def test_worker_extracts_image_assets(tmp_path, monkeypatch) -> None:
    page_url = "https://example.com/docs"
    image_url = "https://example.com/assets/test.png"
    html_body = (
        b'<html><body><img src="/assets/test.png" alt="Example image"></body></html>'
    )

    page_result = _build_fetch_result(
        payload=html_body, content_type="text/html", canonical_source=page_url
    )
    image_result = _build_fetch_result(
        payload=_SAMPLE_PNG, content_type="image/png", canonical_source=image_url
    )
    fetcher = _MappingFetcher({page_url: page_result, image_url: image_result})
    task = _StubTask()

    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    worker = CrawlerWorker(fetcher, ingestion_task=task)

    publish_result = worker.process(
        page_result.request,
        tenant_id="tenant-a",
        case_id="case-b",
        crawl_id="crawl-1",
        document_id="doc-1",
        document_metadata={"source": "crawler"},
    )

    assert publish_result.published
    state_payload, _ = task.calls[0]
    assets = state_payload.get("assets")
    assert assets and len(assets) == 1

    asset = assets[0]
    assert isinstance(asset, dict)
    assert asset["media_type"] == "image/png"
    assert asset.get("content") is None

    meta = dict(asset["metadata"])
    assert meta["origin_uri"] == image_url
    assert meta["sha256"] == sha256_bytes(_SAMPLE_PNG)
    assert meta.get("perceptual_hash") == perceptual_hash(_SAMPLE_PNG)
    assert meta.get("caption_candidates") == [("alt_text", "Example image")]

    assert asset.get("file_uri") is not None
    file_uri = asset["file_uri"]
    assert file_uri.startswith("objectstore://")
    relative_path = file_uri.replace("objectstore://", "")
    stored_bytes = (tmp_path / relative_path).read_bytes()
    assert stored_bytes == _SAMPLE_PNG
