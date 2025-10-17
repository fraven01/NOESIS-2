from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path("docs/observability/documents-telemetry.md")


def _load_doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_documentation_exists() -> None:
    assert DOC_PATH.exists(), "observability guide is missing"


def test_logging_fields_are_documented() -> None:
    text = _load_doc()
    fields = [
        "event",
        "status",
        "duration_ms",
        "tenant_id",
        "document_id",
        "collection_id",
        "version",
        "asset_id",
        "source",
        "size_bytes",
        "uri_kind",
        "sha256_prefix",
        "model",
        "caption_method",
        "caption_confidence",
        "trace_id",
        "span_id",
    ]
    for field in fields:
        assert re.search(rf"`{re.escape(field)}`", text), f"field `{field}` missing"


def test_metric_names_are_listed() -> None:
    text = _load_doc()
    metrics = [
        "documents_operation_total",
        "documents_operation_duration_ms",
        "documents_asset_operation_total",
        "documents_asset_operation_duration_ms",
        "documents_storage_operation_total",
        "documents_storage_operation_duration_ms",
        "documents_pipeline_operation_total",
        "documents_pipeline_operation_duration_ms",
        "documents_cli_operation_total",
        "documents_cli_operation_duration_ms",
        "documents_other_operation_total",
        "documents_other_operation_duration_ms",
        "documents_caption_runs_total",
        "documents_caption_duration_ms",
    ]
    for metric in metrics:
        assert f"`{metric}`" in text, f"metric `{metric}` missing"


def test_event_span_mapping_mentions_all_events() -> None:
    text = _load_doc()
    events = [
        "docs.upsert",
        "docs.get",
        "docs.list",
        "docs.list_latest",
        "docs.delete",
        "assets.add",
        "assets.get",
        "assets.list",
        "assets.delete",
        "storage.put",
        "storage.get",
        "pipeline.assets_caption",
        "pipeline.assets_caption.item",
        "assets.caption.run",
        "assets.caption.process_assets",
        "assets.caption.process_collection",
        "assets.caption.load_payload",
        "cli.schema.print",
        "cli.docs.add",
        "cli.docs.get",
        "cli.docs.list",
        "cli.docs.delete",
        "cli.assets.add",
        "cli.assets.get",
        "cli.assets.list",
        "cli.assets.delete",
        "cli.main",
    ]
    for event in events:
        assert f"`{event}`" in text, f"event `{event}` missing"


def test_runbook_sections_cover_required_scenarios() -> None:
    text = _load_doc()
    assert "Storage nicht erreichbar" in text
    assert "Captioner-Timeout" in text

