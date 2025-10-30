from __future__ import annotations

from types import SimpleNamespace

from ai_core.infra import object_store
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
) -> FetchResult:
    metadata = FetchMetadata(
        status_code=200,
        content_type="text/plain",
        etag="abc",
        last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
        content_length=len(payload or b"") if payload is not None else None,
    )
    telemetry = FetchTelemetry(latency=0.42, bytes_downloaded=len(payload or b""))
    request = FetchRequest(
        canonical_source="https://example.com/docs",
        politeness=PolitenessContext(host="example.com"),
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


class _StubTask:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    def delay(self, state: dict[str, object], meta: dict[str, object]):  # type: ignore[no-untyped-def]
        self.calls.append((state, meta))
        return SimpleNamespace(id="task-123")


def test_worker_publishes_ingestion_task(tmp_path, monkeypatch) -> None:
    fetch_result = _build_fetch_result(payload=b"hello world")
    fetcher = _StubFetcher(fetch_result)
    task = _StubTask()
    worker = CrawlerWorker(fetcher, ingestion_task=task)

    request = fetch_result.request
    overrides = {"guardrails": {"max_document_bytes": 1024}}
    metadata = {"provider": "docs", "tags": ["hr"]}
    meta_overrides = {"trace_id": "trace-1"}

    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    publish_result = worker.process(
        request,
        tenant_id="tenant-a",
        case_id="case-b",
        crawl_id="crawl-1",
        idempotency_key="idemp-1",
        request_id="req-1",
        frontier_state={"slot": "default"},
        document_id="doc-1",
        document_metadata=metadata,
        ingestion_overrides=overrides,
        meta_overrides=meta_overrides,
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
    assert raw_document["metadata"]["provider"] == "docs"
    assert raw_document["metadata"]["source"] == "crawler"
    assert raw_document["metadata"]["origin_uri"] == request.canonical_source
    payload_path = raw_document["payload_path"]
    assert payload_path.endswith(".bin")
    stored_payload = (object_store.BASE_PATH / payload_path).read_bytes()
    assert stored_payload == fetch_result.payload
    assert state_payload["guardrails"] == overrides["guardrails"]
    assert meta_payload["trace_id"] == "trace-1"
    assert meta_payload["idempotency_key"] == "idemp-1"
    assert meta_payload["crawl_id"] == "crawl-1"


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


def test_worker_sets_default_provider_and_source(tmp_path, monkeypatch) -> None:
    fetch_result = _build_fetch_result(payload=b"payload")
    fetcher = _StubFetcher(fetch_result)
    task = _StubTask()
    worker = CrawlerWorker(fetcher, ingestion_task=task)

    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    publish_result = worker.process(
        fetch_result.request,
        tenant_id="tenant-a",
    )

    assert publish_result.published
    state_payload, _ = task.calls[0]
    metadata = state_payload["raw_document"]["metadata"]
    assert metadata["provider"] == "web"
    assert metadata["source"] == "crawler"
