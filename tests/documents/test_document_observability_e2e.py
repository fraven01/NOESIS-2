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
from opentelemetry.trace import ProxyTracerProvider, StatusCode

from common.logging import configure_logging
from documents import metrics
from documents.captioning import AssetExtractionPipeline, DeterministicCaptioner
from documents.cli import CLIContext, main as cli_main
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


@pytest.fixture(autouse=True)
def _reset_metrics_state() -> None:
    metrics.reset_metrics()
    yield
    metrics.reset_metrics()


def _json_events(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
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
            events.append(payload)
    return events


def _assert_no_base64(entries: list[dict[str, object]]) -> None:
    for entry in entries:
        serialized = json.dumps(entry)
        assert "base64" not in serialized


def _inline_blob(payload: bytes, media_type: str) -> InlineBlob:
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    return InlineBlob(
        type="inline",
        media_type=media_type,
        base64=encoded,
        sha256=digest,
        size=len(payload),
    )


def _make_document(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    collection_id: UUID | None = None,
    document_id: UUID | None = None,
    blob: InlineBlob | FileBlob,
    checksum: str,
) -> NormalizedDocument:
    ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id or uuid4(),
        collection_id=collection_id,
    )
    meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        title="E2E",
        tags=["obs"],
    )
    return NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=blob,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source="upload",
        assets=[],
    )


def _make_inline_asset(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    document_id: UUID,
    payload: bytes,
) -> Asset:
    blob = _inline_blob(payload, "image/png")
    return Asset(
        ref=AssetRef(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            asset_id=uuid4(),
            document_id=document_id,
        ),
        media_type="image/png",
        blob=blob,
        caption_method="none",
        created_at=datetime.now(timezone.utc),
        checksum=blob.sha256,
    )


def test_document_repository_observability(
    caplog: pytest.LogCaptureFixture, span_exporter: InMemorySpanExporter
) -> None:
    caplog.clear()
    caplog.set_level("INFO")
    span_exporter.clear()

    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)

    tenant_id = "tenant-e2e"
    collection_id = uuid4()
    payload = b"inline-document"
    blob = _inline_blob(payload, "text/plain")
    checksum = hashlib.sha256(payload).hexdigest()
    document = _make_document(
        tenant_id=tenant_id,
        collection_id=collection_id,
        blob=blob,
        checksum=checksum,
    )
    workflow_id = document.ref.workflow_id

    stored = repo.upsert(document)
    assert isinstance(stored.blob, FileBlob)

    fetched = repo.get(
        tenant_id, stored.ref.document_id, workflow_id=stored.ref.workflow_id
    )
    assert fetched is not None
    assert isinstance(fetched.blob, FileBlob)

    deleted = repo.delete(
        tenant_id, stored.ref.document_id, workflow_id=stored.ref.workflow_id
    )
    assert deleted is True

    upsert_events = _json_events(caplog, "docs.upsert")
    assert any(event.get("phase") == "start" for event in upsert_events)
    upsert_exit = [event for event in upsert_events if event.get("status") == "ok"][-1]
    assert upsert_exit["tenant_id"] == tenant_id
    assert upsert_exit["workflow_id"] == workflow_id
    assert "duration_ms" in upsert_exit
    _assert_no_base64(upsert_events)

    get_exit = [event for event in _json_events(caplog, "docs.get") if event.get("status") == "ok"][-1]
    assert get_exit["document_id"] == str(stored.ref.document_id)
    assert get_exit["workflow_id"] == workflow_id

    delete_exit = [event for event in _json_events(caplog, "docs.delete") if event.get("status") == "ok"][-1]
    assert delete_exit["deleted"] is True
    assert delete_exit["workflow_id"] == workflow_id

    storage_events = _json_events(caplog, "storage.put")
    storage_exit = [event for event in storage_events if event.get("status") == "ok"][-1]
    assert storage_exit["sha256_prefix"] == checksum[:8]

    spans = span_exporter.get_finished_spans()
    span_names = {span.name: span for span in spans}
    assert span_names["docs.upsert"].status.status_code is StatusCode.OK
    assert span_names["docs.upsert"].attributes["noesis.workflow_id"] == workflow_id
    assert span_names["docs.get"].status.status_code is StatusCode.OK
    assert span_names["docs.get"].attributes["noesis.workflow_id"] == workflow_id
    assert span_names["docs.delete"].status.status_code is StatusCode.OK
    assert span_names["docs.delete"].attributes["noesis.workflow_id"] == workflow_id
    assert span_names["storage.put"].status.status_code is StatusCode.OK

    assert (
        metrics.counter_value(
            metrics.DOCUMENT_OPERATION_TOTAL,
            event="docs.upsert",
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.counter_value(
            metrics.DOCUMENT_OPERATION_TOTAL,
            event="docs.get",
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.counter_value(
            metrics.DOCUMENT_OPERATION_TOTAL,
            event="docs.delete",
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.counter_value(
            metrics.STORAGE_OPERATION_TOTAL,
            event="storage.put",
            status="ok",
            workflow_id="unknown",
        )
        or 0.0
    ) >= 1.0

    assert (
        metrics.histogram_count(
            metrics.DOCUMENT_OPERATION_DURATION_MS,
            event="docs.upsert",
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.histogram_count(
            metrics.DOCUMENT_OPERATION_DURATION_MS,
            event="docs.get",
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.histogram_count(
            metrics.DOCUMENT_OPERATION_DURATION_MS,
            event="docs.delete",
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.histogram_count(
            metrics.STORAGE_OPERATION_DURATION_MS,
            event="storage.put",
            status="ok",
            workflow_id="unknown",
        )
        or 0.0
    ) >= 1.0

    span_exporter.clear()


def test_caption_pipeline_observability(
    caplog: pytest.LogCaptureFixture, span_exporter: InMemorySpanExporter
) -> None:
    caplog.clear()
    caplog.set_level("INFO")
    span_exporter.clear()

    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = DeterministicCaptioner()
    pipeline = AssetExtractionPipeline(repository=repo, storage=storage, captioner=captioner)

    tenant_id = "tenant-cap"
    doc_id = uuid4()
    doc_payload = b"caption-doc"
    doc_blob = _inline_blob(doc_payload, "application/pdf")
    checksum = hashlib.sha256(doc_payload).hexdigest()
    document = _make_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        blob=doc_blob,
        checksum=checksum,
    )
    workflow_id = document.ref.workflow_id

    repo.upsert(document)
    metrics.reset_metrics()

    asset_payload = b"image-bytes"
    asset = _make_inline_asset(tenant_id=tenant_id, document_id=doc_id, payload=asset_payload)
    stored_asset = repo.add_asset(asset)

    stored_document = repo.get(tenant_id, doc_id, workflow_id=document.ref.workflow_id)
    assert stored_document is not None
    pipeline.process_document(stored_document)

    updated_asset = repo.get_asset(
        tenant_id, stored_asset.ref.asset_id, workflow_id=stored_asset.ref.workflow_id
    )
    assert updated_asset is not None
    assert updated_asset.text_description
    assert updated_asset.caption_method == "vlm_caption"

    asset_events = _json_events(caplog, "assets.add")
    asset_exit = [event for event in asset_events if event.get("status") == "ok"][-1]
    assert asset_exit["asset_id"] == str(stored_asset.ref.asset_id)
    assert asset_exit["workflow_id"] == stored_asset.ref.workflow_id

    pipeline_events = _json_events(caplog, "pipeline.assets_caption")
    pipeline_exit = [event for event in pipeline_events if event.get("status") == "ok"][-1]
    assert pipeline_exit["processed_assets"] >= 1
    assert pipeline_exit["workflow_id"] == workflow_id
    _assert_no_base64(asset_events + pipeline_events)

    spans = span_exporter.get_finished_spans()
    span_names = {span.name: span for span in spans}
    assert span_names["assets.add"].status.status_code is StatusCode.OK
    assert span_names["assets.add"].attributes["noesis.workflow_id"] == stored_asset.ref.workflow_id
    assert span_names["pipeline.assets_caption"].status.status_code is StatusCode.OK
    assert span_names["pipeline.assets_caption"].attributes["noesis.workflow_id"] == workflow_id
    assert span_names["pipeline.assets_caption.item"].status.status_code is StatusCode.OK
    assert (
        span_names["pipeline.assets_caption.item"].attributes["noesis.workflow_id"]
        == stored_asset.ref.workflow_id
    )
    assert span_names["assets.caption.run"].status.status_code is StatusCode.OK

    assert (
        metrics.counter_value(
            metrics.ASSET_OPERATION_TOTAL,
            event="assets.add",
            status="ok",
            workflow_id=stored_asset.ref.workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.counter_value(
            metrics.PIPELINE_OPERATION_TOTAL,
            event="pipeline.assets_caption",
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.counter_value(
            metrics.CAPTION_RUNS_TOTAL,
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0

    assert (
        metrics.histogram_count(
            metrics.ASSET_OPERATION_DURATION_MS,
            event="assets.add",
            status="ok",
            workflow_id=stored_asset.ref.workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.histogram_count(
            metrics.PIPELINE_OPERATION_DURATION_MS,
            event="pipeline.assets_caption",
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.histogram_count(
            metrics.CAPTION_DURATION_MS,
            status="ok",
            workflow_id=workflow_id,
        )
        or 0.0
    ) >= 1.0

    span_exporter.clear()


def test_storage_error_observability(
    caplog: pytest.LogCaptureFixture, span_exporter: InMemorySpanExporter
) -> None:
    caplog.clear()
    caplog.set_level("INFO")
    span_exporter.clear()

    storage = InMemoryStorage()

    with pytest.raises(KeyError, match="storage_uri_missing"):
        storage.get("memory://does-not-exist")

    events = _json_events(caplog, "storage.get")
    error_event = [event for event in events if event.get("status") == "error"][-1]
    assert error_event["error_kind"] == "KeyError"
    assert "storage_uri_missing" in str(error_event.get("error_msg"))
    _assert_no_base64(events)

    spans = span_exporter.get_finished_spans()
    error_span = next(span for span in spans if span.name == "storage.get")
    assert error_span.status.status_code is StatusCode.ERROR

    assert (
        metrics.counter_value(
            metrics.STORAGE_OPERATION_TOTAL,
            event="storage.get",
            status="error",
            workflow_id="unknown",
        )
        or 0.0
    ) >= 1.0
    assert (
        metrics.histogram_count(
            metrics.STORAGE_OPERATION_DURATION_MS,
            event="storage.get",
            status="error",
            workflow_id="unknown",
        )
        or 0.0
    ) >= 1.0

    span_exporter.clear()


def test_cli_observability(
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    span_exporter: InMemorySpanExporter,
) -> None:
    caplog.clear()
    caplog.set_level("INFO")
    span_exporter.clear()

    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    context = CLIContext(repository=repo, storage=storage)

    tenant_id = "tenant-cli"
    collection_id = uuid4()
    inline_payload = base64.b64encode(b"cli-document").decode("ascii")
    workflow_id = "workflow-1"

    def _counter_total(metric_obj, event: str, status: str = "ok") -> float:
        return sum(
            metrics.counter_value(
                metric_obj,
                event=event,
                status=status,
                workflow_id=label,
            )
            or 0.0
            for label in {workflow_id, "unknown"}
        )

    def _hist_total(metric_obj, event: str, status: str = "ok") -> float:
        return sum(
            metrics.histogram_count(
                metric_obj,
                event=event,
                status=status,
                workflow_id=label,
            )
            or 0.0
            for label in {workflow_id, "unknown"}
        )

    exit_code = cli_main(
        [
            "--json",
            "docs",
            "add",
            "--tenant",
            tenant_id,
            "--workflow-id",
            workflow_id,
            "--collection",
            str(collection_id),
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
    add_output = json.loads(capfd.readouterr().out)
    document_id = add_output["ref"]["document_id"]

    exit_code = cli_main(
        [
            "--json",
            "docs",
            "get",
            "--tenant",
            tenant_id,
            "--doc-id",
            document_id,
        ],
        context=context,
    )
    assert exit_code == 0
    capfd.readouterr()

    exit_code = cli_main(
        [
            "--json",
            "docs",
            "list",
            "--tenant",
            tenant_id,
            "--collection",
            str(collection_id),
            "--latest-only",
        ],
        context=context,
    )
    assert exit_code == 0
    capfd.readouterr()

    exit_code = cli_main(
        [
            "--json",
            "docs",
            "delete",
            "--tenant",
            tenant_id,
            "--doc-id",
            document_id,
        ],
        context=context,
    )
    assert exit_code == 0
    capfd.readouterr()

    for event in ("cli.docs.add", "cli.docs.get", "cli.docs.list", "cli.docs.delete"):
        events = _json_events(caplog, event)
        exit_event = [entry for entry in events if entry.get("status") == "ok"][-1]
        tenant_value = exit_event.get("tenant_id")
        if tenant_value is not None:
            assert tenant_value == tenant_id
        workflow_value = exit_event.get("workflow_id")
        if workflow_value is not None:
            assert workflow_value == workflow_id
        assert "duration_ms" in exit_event
        _assert_no_base64(events)

    spans = span_exporter.get_finished_spans()
    span_names = {span.name: span for span in spans}
    assert span_names["cli.docs.add"].status.status_code is StatusCode.OK
    assert span_names["cli.docs.get"].status.status_code is StatusCode.OK
    assert span_names["cli.docs.list"].status.status_code is StatusCode.OK
    assert span_names["cli.docs.delete"].status.status_code is StatusCode.OK

    assert _counter_total(metrics.CLI_OPERATION_TOTAL, event="cli.docs.add") >= 1.0
    assert _counter_total(metrics.CLI_OPERATION_TOTAL, event="cli.docs.get") >= 1.0
    assert _counter_total(metrics.CLI_OPERATION_TOTAL, event="cli.docs.list") >= 1.0
    assert _counter_total(metrics.CLI_OPERATION_TOTAL, event="cli.docs.delete") >= 1.0

    assert _hist_total(metrics.CLI_OPERATION_DURATION_MS, event="cli.docs.add") >= 1.0
    assert _hist_total(metrics.CLI_OPERATION_DURATION_MS, event="cli.docs.get") >= 1.0
    assert _hist_total(metrics.CLI_OPERATION_DURATION_MS, event="cli.docs.list") >= 1.0
    assert _hist_total(metrics.CLI_OPERATION_DURATION_MS, event="cli.docs.delete") >= 1.0

    assert _counter_total(metrics.DOCUMENT_OPERATION_TOTAL, event="docs.upsert") >= 1.0
    assert _counter_total(metrics.DOCUMENT_OPERATION_TOTAL, event="docs.get") >= 1.0
    assert _counter_total(metrics.DOCUMENT_OPERATION_TOTAL, event="docs.list") >= 1.0
    assert _counter_total(metrics.DOCUMENT_OPERATION_TOTAL, event="docs.delete") >= 1.0

    span_exporter.clear()

