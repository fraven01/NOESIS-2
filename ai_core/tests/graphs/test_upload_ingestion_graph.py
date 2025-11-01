from __future__ import annotations

from ai_core.graphs.upload_ingestion_graph import (
    GraphTransition,
    UploadIngestionGraph,
)


def _scanner(_: bytes, __: dict[str, object]) -> GraphTransition:
    return GraphTransition(decision="proceed", reason="clean")


def test_nominal_run() -> None:
    graph = UploadIngestionGraph(quarantine_scanner=_scanner)
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
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
    graph = UploadIngestionGraph()
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "file_bytes": b"Hello world",
    }

    result = graph.run(payload, run_until="parse_complete")

    assert "chunk_and_embed" not in result["transitions"]
    assert result["transitions"]["parse"]["decision"] == "parse_complete"


def test_duplicate_upload_skipped() -> None:
    graph = UploadIngestionGraph()
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "file_bytes": b"Hello world",
    }

    first = graph.run(payload)
    second = graph.run(payload)

    assert first["decision"] == "completed"
    assert second["decision"] == "skip_duplicate"
    assert second["transitions"]["deduplicate"]["decision"] == "skip_duplicate"

