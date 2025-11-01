from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ai_core import api as ai_core_api
import ai_core.graphs.upload_ingestion_graph as upload_ingestion_graph
from documents.api import NormalizedDocumentPayload


def _scanner(
    _: bytes, __: dict[str, object]
) -> upload_ingestion_graph.GraphTransition:
    return upload_ingestion_graph.GraphTransition(decision="proceed", reason="clean")


def test_nominal_run() -> None:
    graph = upload_ingestion_graph.UploadIngestionGraph(quarantine_scanner=_scanner)
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "trace_id": "trace-1",
        "file_bytes": b"Hello world",
        "filename": "example.txt",
    }

    result = graph.run(payload)

    assert result["decision"] == "completed"
    transitions = result["transitions"]
    assert transitions["accept_upload"]["decision"] == "accepted"
    assert transitions["deduplicate"]["decision"] == "proceed"
    assert transitions["parse"]["decision"] == "parse_complete"
    assert transitions["normalize"]["decision"] == "normalize_complete"
    assert transitions["persist_document"]["decision"] == "persist_complete"
    assert result["document_id"]


def test_run_until_stops_after_marker() -> None:
    graph = upload_ingestion_graph.UploadIngestionGraph()
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "trace_id": "trace-1",
        "file_bytes": b"Hello world",
    }

    result = graph.run(payload, run_until="parse_complete")

    assert "chunk_and_embed" not in result["transitions"]
    assert result["transitions"]["parse"]["decision"] == "parse_complete"


def test_duplicate_upload_skipped() -> None:
    graph = upload_ingestion_graph.UploadIngestionGraph()
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "trace_id": "trace-1",
        "file_bytes": b"Hello world",
    }

    first = graph.run(payload)
    second = graph.run(payload)

    assert first["decision"] == "completed"
    assert second["decision"] == "skip_duplicate"
    assert second["transitions"]["deduplicate"]["decision"] == "skip_duplicate"


def test_guardrail_allow_and_delta_merge_policy_events() -> None:
    guardrail_calls: list[NormalizedDocumentPayload] = []
    baseline_calls: list[Mapping[str, object]] = []

    def guardrail_stub(
        *, normalized_document: NormalizedDocumentPayload, **_: object
    ) -> ai_core_api.GuardrailDecision:
        guardrail_calls.append(normalized_document)
        return ai_core_api.GuardrailDecision(
            "allow",
            "guardrail_ok",
            {"policy_events": ("guardrail_allow",)},
        )

    def delta_stub(
        *,
        normalized_document: NormalizedDocumentPayload,
        baseline: Mapping[str, object],
        **_: object,
    ) -> ai_core_api.DeltaDecision:
        assert guardrail_calls and normalized_document is guardrail_calls[0]
        baseline_calls.append(baseline)
        return ai_core_api.DeltaDecision(
            "new",
            "delta_new",
            {"policy_events": ("delta_new",), "version": 1},
        )

    graph = upload_ingestion_graph.UploadIngestionGraph(
        guardrail_enforcer=guardrail_stub,
        delta_decider=delta_stub,
    )
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "trace_id": "trace-1",
        "file_bytes": b"Hello world",
        "filename": "example.txt",
    }

    result = graph.run(payload)

    assert guardrail_calls
    assert isinstance(guardrail_calls[0], NormalizedDocumentPayload)
    assert baseline_calls and isinstance(baseline_calls[0], Mapping)
    transition = result["transitions"]["delta_and_guardrails"]
    assert transition["decision"] == "upsert"
    diagnostics = transition["diagnostics"]
    assert set(diagnostics["policy_events"]) == {"guardrail_allow", "delta_new"}
    assert diagnostics["guardrail"]["decision"] == "guardrail_allow"
    assert diagnostics["version"] == 1


def test_guardrail_deny_short_circuits_delta() -> None:
    guardrail_calls: list[NormalizedDocumentPayload] = []
    delta_called = False

    def guardrail_stub(
        *, normalized_document: NormalizedDocumentPayload, **_: object
    ) -> ai_core_api.GuardrailDecision:
        guardrail_calls.append(normalized_document)
        return ai_core_api.GuardrailDecision(
            "deny",
            "blocked",
            {"policy_events": ("upload_blocked",), "severity": "error"},
        )

    def delta_stub(**_: object) -> ai_core_api.DeltaDecision:  # pragma: no cover - should not run
        nonlocal delta_called
        delta_called = True
        return ai_core_api.DeltaDecision("new", "should_not_happen", {})

    graph = upload_ingestion_graph.UploadIngestionGraph(
        guardrail_enforcer=guardrail_stub,
        delta_decider=delta_stub,
    )
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "trace_id": "trace-1",
        "file_bytes": b"Hello world",
        "filename": "example.txt",
    }

    result = graph.run(payload)

    assert guardrail_calls and isinstance(guardrail_calls[0], NormalizedDocumentPayload)
    assert not delta_called
    assert result["decision"] == "skip_guardrail"
    transition = result["transitions"]["delta_and_guardrails"]
    assert transition["decision"] == "skip_guardrail"
    diagnostics = transition["diagnostics"]
    assert diagnostics["policy_events"] == ("upload_blocked",)
    assert diagnostics["allowed"] is False


def test_upload_ingestion_spans(monkeypatch) -> None:
    import ai_core.infra.observability as observability

    recorded: list[str] = []
    snapshots: list[dict[str, Any]] = []

    def fake_observe_span(name: str, auto_annotate: bool = False, **_kwargs: Any):
        def decorator(func):
            def wrapped(*args, **kwargs):
                result = func(*args, **kwargs)
                recorded.append(name)
                if (
                    name.startswith("upload.ingestion.")
                    and name != "upload.ingestion.run"
                    and len(args) >= 2
                    and isinstance(args[1], Mapping)
                ):
                    self = args[0]
                    state = args[1]
                    metadata = {"phase": name}
                    if hasattr(self, "_collect_observability_metadata"):
                        metadata.update(self._collect_observability_metadata(state))
                    snapshots.append(metadata)
                return result

            return wrapped

        return decorator

    monkeypatch.setattr(observability, "observe_span", fake_observe_span)
    module = importlib.reload(upload_ingestion_graph)
    try:
        graph = module.UploadIngestionGraph(quarantine_scanner=_scanner)
        payload = {
            "tenant_id": "tenant-a",
            "uploader_id": "user-1",
            "trace_id": "trace-1",
            "file_bytes": b"Hello world",
            "filename": "example.txt",
        }

        result = graph.run(payload)

        expected = [
            "upload.ingestion.run",
            "upload.ingestion.accept_upload",
            "upload.ingestion.quarantine_scan",
            "upload.ingestion.deduplicate",
            "upload.ingestion.parse",
            "upload.ingestion.normalize",
            "upload.ingestion.delta_and_guardrails",
            "upload.ingestion.persist_document",
            "upload.ingestion.chunk_and_embed",
            "upload.ingestion.lifecycle_hook",
            "upload.ingestion.finalize",
        ]

        node_expected = expected[1:]
        phases = [entry.get("phase") for entry in snapshots]
        assert recorded[: len(node_expected)] == node_expected
        assert phases == node_expected
        assert all(entry.get("trace_id") == payload["trace_id"] for entry in snapshots)
        workflow_expected = payload.get("workflow_id", "upload")
        assert all(entry.get("workflow_id") == workflow_expected for entry in snapshots)
        doc_entries = [entry for entry in snapshots if entry.get("document_id")]
        assert doc_entries
        assert all(isinstance(entry.get("document_id"), str) and entry["document_id"] for entry in doc_entries)
        assert recorded.count("upload.ingestion.run") == 1
        assert recorded[-1] == "upload.ingestion.run"
    finally:
        importlib.reload(upload_ingestion_graph)


def test_upload_transition_metadata_contains_ids() -> None:
    graph = upload_ingestion_graph.UploadIngestionGraph(quarantine_scanner=_scanner)
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "trace_id": "trace-meta",
        "workflow_id": "flow-meta",
        "file_bytes": b"Hello world",
        "filename": "example.txt",
    }

    result = graph.run(payload)

    transitions = result["transitions"]
    for diagnostics in (
        transition["diagnostics"] for transition in transitions.values()
    ):
        assert diagnostics["trace_id"] == "trace-meta"
        assert diagnostics["workflow_id"] == "flow-meta"

    document_id = str(result["document_id"])
    assert document_id
    doc_ids = {
        diagnostics["document_id"]
        for diagnostics in (
            transition["diagnostics"] for transition in transitions.values()
        )
        if diagnostics.get("document_id")
    }
    assert document_id in doc_ids
