import base64
import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ai_core import ingestion
from ai_core.ingestion import process_document
from ai_core.infra import object_store
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)
from documents.repository import InMemoryDocumentsRepository


class DummyRetryError(Exception):
    pass


def test_process_document_retry_and_resume(monkeypatch, tmp_path):
    tenant = "tenant-retry"
    case = "case-retry"
    tenant_schema = "tenant-schema"
    document_uuid = uuid4()
    document_id = str(document_uuid)

    store_path = tmp_path / "store"
    monkeypatch.setattr(object_store, "BASE_PATH", store_path)

    repository = InMemoryDocumentsRepository()
    workflow_id = case.replace(":", "_")
    payload = b"payload for fallback external id"
    encoded = base64.b64encode(payload).decode("ascii")
    checksum = hashlib.sha256(payload).hexdigest()
    blob = InlineBlob(
        type="inline",
        media_type="text/plain",
        base64=encoded,
        sha256=checksum,
        size=len(payload),
    )
    document_ref = DocumentRef(
        tenant_id=tenant,
        workflow_id=workflow_id,
        document_id=document_uuid,
    )
    document_meta = DocumentMeta(
        tenant_id=tenant,
        workflow_id=workflow_id,
        title="Example",
        external_ref={"external_id": "external-123", "media_type": "text/plain"},
    )
    normalized_document = NormalizedDocument(
        ref=document_ref,
        meta=document_meta,
        blob=blob,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source="upload",
    )
    repository.upsert(normalized_document)

    import ai_core.services as services_module

    monkeypatch.setattr(
        services_module,
        "_get_documents_repository",
        lambda: repository,
        raising=False,
    )
    monkeypatch.setattr(
        services_module, "_DOCUMENTS_REPOSITORY", repository, raising=False
    )

    tenant_key = object_store.sanitize_identifier(tenant)
    case_key = object_store.sanitize_identifier(case)

    counts = {step: 0 for step in ["pii_mask", "chunk", "embed"]}

    profile_id = "standard"

    artifact_paths = {
        step: "/".join([tenant_key, case_key, "artifacts", f"{step}.txt"])
        for step in counts
    }

    content_hash_value = checksum

    def make_step_stub(step_name):
        def _stub(meta, *args, **kwargs):
            assert meta["embedding_profile"] == profile_id
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
        process_document(
            tenant,
            case,
            document_id,
            profile_id,
            tenant_schema=tenant_schema,
            trace_id="test-trace-id",
        )

    status_path = ingestion._status_store_path(tenant, case, document_id)
    status_state = object_store.read_json(status_path)

    assert status_state["last_error"]["step"] == "upsert"
    assert "upsert failed" in status_state["last_error"]["message"]
    assert status_state["last_error"]["retry"] == 0

    parse_state = status_state["steps"]["parse"]
    assert parse_state["cleaned"] is False
    assert (object_store.BASE_PATH / parse_state["path"]).exists()
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
    assert meta_state["embedding_profile"] == profile_id

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

    result = process_document(
        tenant,
        case,
        document_id,
        profile_id,
        tenant_schema=tenant_schema,
        trace_id="test-trace-id",
    )

    assert counts == prior_counts
    assert result["inserted"] == 1
    assert result["skipped"] == 0
    assert result["content_hash"] == content_hash_value
    assert result["external_id"] == meta_state["external_id"]
    assert result["embedding_profile"] == profile_id
    assert result["vector_space_id"] == "global"

    assert not (object_store.BASE_PATH / parse_state["path"]).exists()
    for path in artifact_paths.values():
        assert not (object_store.BASE_PATH / path).exists()

    status_state = object_store.read_json(status_path)
    assert status_state["last_error"] is None
    for step_name in ["parse", *artifact_paths.keys()]:
        step_state = status_state["steps"][step_name]
        assert step_state["cleaned"] is True
        assert step_state.get("path") is None
    assert status_state["steps"]["upsert"]["cleaned"] is True


def test_process_document_copies_pipeline_config_to_state(monkeypatch, tmp_path):
    tenant = "tenant-pipeline"
    case = "case-pipeline"
    tenant_schema = tenant
    document_uuid = uuid4()
    document_id = str(document_uuid)

    store_path = tmp_path / "store"
    monkeypatch.setattr(object_store, "BASE_PATH", store_path)

    repository = InMemoryDocumentsRepository()
    workflow_id = case.replace(":", "_")
    payload = b"payload for pipeline overrides"
    encoded = base64.b64encode(payload).decode("ascii")
    checksum = hashlib.sha256(payload).hexdigest()
    blob = InlineBlob(
        type="inline",
        media_type="text/plain",
        base64=encoded,
        sha256=checksum,
        size=len(payload),
    )
    document_ref = DocumentRef(
        tenant_id=tenant,
        workflow_id=workflow_id,
        document_id=document_uuid,
    )
    document_meta = DocumentMeta(
        tenant_id=tenant,
        workflow_id=workflow_id,
        pipeline_config={"enable_ocr": True},
    )
    normalized_document = NormalizedDocument(
        ref=document_ref,
        meta=document_meta,
        blob=blob,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source="upload",
    )
    repository.upsert(normalized_document)

    import ai_core.services as services_module

    monkeypatch.setattr(
        services_module,
        "_get_documents_repository",
        lambda: repository,
        raising=False,
    )
    monkeypatch.setattr(
        services_module, "_DOCUMENTS_REPOSITORY", repository, raising=False
    )

    vector_space = SimpleNamespace(
        id="global",
        schema="public",
        backend="pgvector",
        dimension=1536,
    )
    binding = SimpleNamespace(
        profile_id="standard",
        resolution=SimpleNamespace(vector_space=vector_space),
    )
    monkeypatch.setattr(
        ingestion,
        "resolve_ingestion_profile",
        lambda profile: binding,
        raising=False,
    )
    monkeypatch.setattr(
        ingestion,
        "ensure_vector_space_schema",
        lambda _: False,
        raising=False,
    )

    captured: dict[str, object] = {}

    class CaptureConfig(Exception):
        pass

    def capture_build_config(*, state=None, meta=None):
        captured["state"] = state
        captured["meta"] = meta
        raise CaptureConfig

    monkeypatch.setattr(
        ingestion,
        "_build_document_pipeline_config",
        capture_build_config,
        raising=False,
    )

    process_document.request = SimpleNamespace(retries=0, called_directly=True)

    def fake_retry(*args, **kwargs):
        raise CaptureConfig

    monkeypatch.setattr(process_document, "retry", fake_retry, raising=False)
    with pytest.raises(CaptureConfig):
        process_document(
            tenant,
            case,
            document_id,
            "standard",
            tenant_schema=tenant_schema,
            trace_id="test-trace-id",
        )
    assert captured["meta"]["pipeline_config"] == {"enable_ocr": True}
    assert captured["state"]["pipeline_config"] == {"enable_ocr": True}
    assert captured["state"]["meta"]["pipeline_config"] == {"enable_ocr": True}
