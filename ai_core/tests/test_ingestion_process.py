import json
from types import SimpleNamespace

import pytest

from ai_core import ingestion
from ai_core.ingestion import process_document
from ai_core.infra import object_store


class DummyRetryError(Exception):
    pass


def test_process_document_retry_and_resume(monkeypatch, tmp_path):
    tenant = "tenant-retry"
    case = "case-retry"
    document_id = "doc-retry"
    tenant_schema = "tenant-schema"

    store_path = tmp_path / "store"
    monkeypatch.setattr(object_store, "BASE_PATH", store_path)

    upload_dir = ingestion._upload_dir(tenant, case)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_bytes = b"payload for fallback external id"
    (upload_dir / f"{document_id}_source.txt").write_bytes(file_bytes)
    (upload_dir / f"{document_id}.meta.json").write_text(
        json.dumps({"tenant": tenant, "case": case, "label": "example"})
    )

    tenant_key = object_store.sanitize_identifier(tenant)
    case_key = object_store.sanitize_identifier(case)

    counts = {
        step: 0
        for step in [
            "ingest_raw",
            "extract_text",
            "pii_mask",
            "chunk",
            "embed",
        ]
    }

    artifact_paths = {
        step: "/".join([tenant_key, case_key, "artifacts", f"{step}.txt"])
        for step in counts
    }

    content_hash_value = "hash-abc123"

    def make_step_stub(step_name):
        def _stub(meta, *args, **kwargs):
            counts[step_name] += 1
            relative_path = artifact_paths[step_name]
            object_store.write_bytes(
                relative_path,
                f"{step_name}:{counts[step_name]}".encode("utf-8"),
            )
            result = {"path": relative_path}
            if step_name == "ingest_raw":
                result["content_hash"] = content_hash_value
            return result

        return _stub

    for step_name in counts:
        monkeypatch.setattr(ingestion.pipe, step_name, make_step_stub(step_name))

    def failing_upsert(meta, path, tenant_schema=None):
        raise RuntimeError("upsert failed")

    monkeypatch.setattr(ingestion.pipe, "upsert", failing_upsert)

    def fake_retry(*args, **kwargs):
        raise DummyRetryError("retry triggered")

    monkeypatch.setattr(process_document, "retry", fake_retry, raising=False)
    process_document.request = SimpleNamespace(retries=0)

    with pytest.raises(DummyRetryError):
        process_document(tenant, case, document_id, tenant_schema=tenant_schema)

    assert all(counts[step] == 1 for step in counts)

    status_path = ingestion._status_store_path(tenant, case, document_id)
    status_state = object_store.read_json(status_path)

    assert status_state["last_error"]["step"] == "upsert"
    assert "upsert failed" in status_state["last_error"]["message"]
    assert status_state["last_error"]["retry"] == 0

    for step_name, path in artifact_paths.items():
        step_state = status_state["steps"][step_name]
        assert step_state["path"] == path
        assert step_state["cleaned"] is False
        assert (object_store.BASE_PATH / path).exists()

    meta_path = ingestion._meta_store_path(tenant, case, document_id)
    meta_state = object_store.read_json(meta_path)
    assert "external_id" in meta_state and meta_state["external_id"]
    assert "tenant" not in meta_state
    assert "case" not in meta_state

    prior_counts = counts.copy()

    class SuccessfulUpsertResult:
        def __init__(self):
            self.documents = [{"action": "inserted", "chunk_count": 1}]

        def __int__(self):
            return 1

    def successful_upsert(meta, path, tenant_schema=None):
        return SuccessfulUpsertResult()

    monkeypatch.setattr(ingestion.pipe, "upsert", successful_upsert)
    process_document.request.retries = 1

    result = process_document(tenant, case, document_id, tenant_schema=tenant_schema)

    assert counts == prior_counts
    assert result["inserted"] == 1
    assert result["skipped"] == 0
    assert result["content_hash"] == content_hash_value
    assert result["external_id"] == meta_state["external_id"]

    for path in artifact_paths.values():
        assert not (object_store.BASE_PATH / path).exists()

    status_state = object_store.read_json(status_path)
    assert status_state["last_error"] is None
    for step_name in artifact_paths:
        step_state = status_state["steps"][step_name]
        assert step_state["cleaned"] is True
        assert step_state.get("path") is None
    assert status_state["steps"]["upsert"]["cleaned"] is True
