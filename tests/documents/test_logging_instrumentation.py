import ast
import base64
import hashlib
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import (
    ProxyTracerProvider,
    StatusCode,
    format_span_id,
    format_trace_id,
)

from common.logging import configure_logging
from documents.captioning import AssetExtractionPipeline, DeterministicCaptioner
from documents.contracts import (
    Asset,
    AssetRef,
    DocumentMeta,
    DocumentRef,
    FileBlob,
    InlineBlob,
    NormalizedDocument,
)
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage
from documents.cli import CLIContext, main as cli_main


@pytest.fixture(scope="module", autouse=True)
def _configure_logging() -> None:
    configure_logging()


@pytest.fixture(scope="module")
def span_exporter() -> InMemorySpanExporter:
    provider = trace.get_tracer_provider()
    if isinstance(provider, ProxyTracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    elif not isinstance(provider, TracerProvider):  # pragma: no cover - defensive
        raise AssertionError("unsupported tracer provider type")

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    try:
        yield exporter
    finally:
        processor.shutdown()
        exporter.clear()


def _make_document(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    document_id: UUID | None = None,
    collection_id: UUID | None = None,
    version: str | None = None,
    checksum: str = "a" * 64,
    blob: FileBlob | None = None,
    assets: list[Asset] | None = None,
) -> NormalizedDocument:
    doc_uuid = document_id or uuid4()
    ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=doc_uuid,
        collection_id=collection_id,
        version=version,
    )
    meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        title="Log Sample",
        tags=["alpha"],
    )
    blob = blob or FileBlob(type="file", uri="memory://doc", sha256=checksum, size=16)
    return NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=blob,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source="upload",
        assets=list(assets or []),
    )


def _make_asset(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    document_id: UUID,
    asset_id: UUID | None = None,
    checksum: str = "b" * 64,
    blob: FileBlob | None = None,
    caption_method: str = "none",
) -> Asset:
    asset_uuid = asset_id or uuid4()
    ref = AssetRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        asset_id=asset_uuid,
        document_id=document_id,
    )
    blob = blob or FileBlob(type="file", uri="memory://asset", sha256=checksum, size=8)
    return Asset(
        ref=ref,
        media_type="image/png",
        blob=blob,
        caption_method=caption_method,
        created_at=datetime.now(timezone.utc),
        checksum=checksum,
    )


def _inline_asset(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    document_id: UUID,
    payload: bytes,
) -> Asset:
    encoded_payload = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    encoded = InlineBlob(
        type="inline",
        media_type="image/png",
        base64=encoded_payload,
        sha256=digest,
        size=len(payload),
    )
    return Asset(
        ref=AssetRef(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            asset_id=uuid4(),
            document_id=document_id,
        ),
        media_type="image/png",
        blob=encoded,
        caption_method="none",
        created_at=datetime.now(timezone.utc),
        checksum=digest,
    )


def _json_events(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in caplog.records:
        message = record.getMessage()
        payload: dict[str, object] | None = None
        try:
            parsed = json.loads(message)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(message)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed
        if payload and payload.get("event") == event:
            records.append(payload)
    return records


def _assert_no_base64(entries: list[dict[str, object]]) -> None:
    for entry in entries:
        payload = json.dumps(entry)
        assert "base64" not in payload


def test_repository_upsert_emits_structured_logs(
    caplog: pytest.LogCaptureFixture, span_exporter: InMemorySpanExporter
) -> None:
    span_exporter.clear()
    caplog.clear()
    caplog.set_level("INFO")
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    doc = _make_document(tenant_id="tenant-log")

    repo.upsert(doc)

    events = _json_events(caplog, "docs.upsert")
    assert any(event.get("phase") == "start" for event in events)
    exit_events = [event for event in events if event.get("status") == "ok"]
    assert exit_events, "expected exit log with status ok"
    exit_event = exit_events[-1]
    assert exit_event["tenant_id"] == "tenant-log"
    doc_id = str(doc.ref.document_id)
    logged_id = exit_event["document_id"]
    assert doc_id[:8] in logged_id and doc_id[-12:] in logged_id
    assert exit_event["sha256_prefix"] == doc.blob.sha256[:8]
    assert "duration_ms" in exit_event
    _assert_no_base64(events)

    spans = span_exporter.get_finished_spans()
    upsert_span = next(span for span in spans if span.name == "docs.upsert")
    assert upsert_span.status.status_code is StatusCode.OK
    assert upsert_span.attributes["noesis.tenant_id"] == "tenant-log"
    assert upsert_span.attributes["noesis.document_id"] == str(doc.ref.document_id)
    assert upsert_span.attributes["noesis.size_bytes"] == doc.blob.size
    span_context = upsert_span.context
    assert exit_event["trace_id"] == format_trace_id(span_context.trace_id)
    assert exit_event["span_id"] == format_span_id(span_context.span_id)
    span_exporter.clear()


def test_repository_add_asset_logs_sha_prefix(
    caplog: pytest.LogCaptureFixture, span_exporter: InMemorySpanExporter
) -> None:
    span_exporter.clear()
    caplog.clear()
    caplog.set_level("INFO")
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    doc = _make_document(tenant_id="tenant-asset")
    repo.upsert(doc)
    asset = _make_asset(tenant_id="tenant-asset", document_id=doc.ref.document_id)

    repo.add_asset(asset)

    events = _json_events(caplog, "assets.add")
    exit_event = [event for event in events if event.get("status") == "ok"][-1]
    assert exit_event["asset_id"] == str(asset.ref.asset_id)
    assert exit_event["sha256_prefix"] == asset.blob.sha256[:8]
    assert exit_event["uri_kind"] == "memory"
    _assert_no_base64(events)

    spans = span_exporter.get_finished_spans()
    asset_span = next(span for span in spans if span.name == "assets.add")
    assert asset_span.status.status_code is StatusCode.OK
    assert asset_span.attributes["noesis.asset_id"] == str(asset.ref.asset_id)
    assert asset_span.attributes["noesis.uri_kind"] == "memory"
    assert asset_span.attributes["noesis.size_bytes"] == asset.blob.size
    span_exporter.clear()


def test_storage_put_get_log_metadata(
    caplog: pytest.LogCaptureFixture, span_exporter: InMemorySpanExporter
) -> None:
    span_exporter.clear()
    caplog.clear()
    caplog.set_level("INFO")
    storage = InMemoryStorage()

    uri, checksum, size = storage.put(b"payload-bytes")
    payload = storage.get(uri)

    put_events = _json_events(caplog, "storage.put")
    put_exit = [event for event in put_events if event.get("status") == "ok"][-1]
    assert put_exit["sha256_prefix"] == checksum[:8]
    assert put_exit["size_bytes"] == size
    assert put_exit["uri_kind"] == "memory"

    get_events = _json_events(caplog, "storage.get")
    get_exit = [event for event in get_events if event.get("status") == "ok"][-1]
    assert get_exit["uri_kind"] == "memory"
    assert get_exit["size_bytes"] == len(payload)

    spans = span_exporter.get_finished_spans()
    put_span = next(span for span in spans if span.name == "storage.put")
    assert put_span.status.status_code is StatusCode.OK
    assert put_span.attributes["noesis.uri_kind"] == "memory"
    assert put_span.attributes["noesis.size_bytes"] == size
    get_span = next(span for span in spans if span.name == "storage.get")
    assert get_span.status.status_code is StatusCode.OK
    assert get_span.attributes["noesis.uri_kind"] == "memory"
    assert get_span.attributes["noesis.size_bytes"] == len(payload)
    span_exporter.clear()


def test_storage_get_error_marks_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    span_exporter.clear()
    storage = InMemoryStorage()

    with pytest.raises(ValueError, match="storage_uri_unsupported"):
        storage.get("file://unsupported")

    spans = span_exporter.get_finished_spans()
    error_span = next(span for span in spans if span.name == "storage.get")
    assert error_span.status.status_code is StatusCode.ERROR
    assert error_span.attributes["error.type"] == "ValueError"
    assert error_span.attributes["error.message"] == "storage_uri_unsupported"
    span_exporter.clear()


def test_cli_command_logs_context(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    caplog.set_level("INFO")
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    context = CLIContext(repository=repo, storage=storage)
    inline_payload = "aGVsbG8="
    workflow_id = "workflow-1"

    exit_code = cli_main(
        [
            "docs",
            "add",
            "--tenant",
            "cli-tenant",
            "--workflow",
            workflow_id,
            "--collection",
            str(uuid4()),
            "--inline",
            inline_payload,
            "--media-type",
            "text/plain",
            "--source",
            "upload",
        ],
        context=context,
    )
    assert exit_code == 0

    events = _json_events(caplog, "cli.docs.add")
    exit_event = [event for event in events if event.get("status") == "ok"][-1]
    assert exit_event["tenant_id"] == "cli-tenant"
    assert exit_event["status"] == "ok"
    assert "duration_ms" in exit_event
    _assert_no_base64(events)


def test_caption_pipeline_emits_caption_run(
    caplog: pytest.LogCaptureFixture, span_exporter: InMemorySpanExporter
) -> None:
    span_exporter.clear()
    caplog.clear()
    caplog.set_level("INFO")
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = DeterministicCaptioner()
    pipeline = AssetExtractionPipeline(
        repository=repo,
        storage=storage,
        captioner=captioner,
    )
    inline_bytes = b"caption-image"
    doc_id = uuid4()
    asset = _inline_asset(tenant_id="cap-tenant", document_id=doc_id, payload=inline_bytes)
    doc = _make_document(tenant_id="cap-tenant", document_id=doc_id, assets=[asset])

    pipeline.process_document(doc)

    events = _json_events(caplog, "pipeline.assets_caption.item")
    exit_event = [event for event in events if event.get("status") == "ok"][-1]
    assert exit_event["tenant_id"] == "cap-tenant"
    assert exit_event["model"]
    assert 0.0 <= exit_event["caption_confidence"] <= 1.0
    _assert_no_base64(events)

    spans = span_exporter.get_finished_spans()
    doc_span = next(span for span in spans if span.name == "pipeline.assets_caption")
    assert doc_span.status.status_code is StatusCode.OK
    assert doc_span.attributes["noesis.tenant_id"] == "cap-tenant"

    item_span = next(
        span for span in spans if span.name == "pipeline.assets_caption.item"
    )
    assert item_span.status.status_code is StatusCode.OK
    assert item_span.attributes["noesis.caption.method"] == "vlm_caption"
    assert item_span.attributes["noesis.caption.model"]
    assert 0.0 <= item_span.attributes["noesis.caption.confidence"] <= 1.0
    span_exporter.clear()
