"""Unit tests for the UploadWorker."""

from unittest.mock import patch, MagicMock

import pytest
from documents.upload_worker import UploadWorker, WorkerPublishResult


class MockUploadedFile:
    def __init__(self, name: str, content: bytes, content_type: str):
        self.name = name
        self.content = content
        self.content_type = content_type
        self.size = len(content)

    def read(self):
        return self.content


@pytest.fixture
def mock_blob_writer():
    with patch("documents.upload_worker.ObjectStoreBlobWriter") as mock:
        writer_instance = mock.return_value
        writer_instance.put.return_value = (
            "s3://bucket/tenant/uploads/blob-id",
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            100,
        )
        yield mock


@pytest.fixture
def mock_run_graph():
    with patch("documents.upload_worker.run_ingestion_graph") as mock:
        mock.delay.return_value.id = "task-uuid"
        yield mock


@pytest.fixture
def mock_domain_service():
    with patch("documents.upload_worker.DocumentDomainService") as mock_cls:
        service = mock_cls.return_value
        service.ingest_document.return_value.document.id = (
            "00000000-0000-0000-0000-000000000001"
        )
        service.ingest_document.return_value.collection_ids = ["col-uuid"]
        service.ensure_collection.return_value.collection_id = "col-uuid"
        yield service


@pytest.fixture
def mock_tenants():
    with patch("customers.tenant_context.TenantContext") as mock_ctx, patch(
        "django_tenants.utils.tenant_context"
    ) as mock_utils:
        mock_ctx.resolve_identifier.return_value = MagicMock(schema_name="tenant-1")
        mock_utils.return_value.__enter__.return_value = None
        yield mock_ctx


def test_upload_worker_process_success(
    mock_blob_writer, mock_run_graph, mock_domain_service, mock_tenants
):
    worker = UploadWorker()
    file_content = b"test content"
    upload = MockUploadedFile("test.txt", file_content, "text/plain")

    result = worker.process(
        upload,
        tenant_id="tenant-1",
        case_id="case-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        document_metadata={"key": "value"},
    )

    assert isinstance(result, WorkerPublishResult)
    assert result.status == "published"
    assert result.document_id == "00000000-0000-0000-0000-000000000001"
    assert result.task_id == "task-uuid"

    # Verify State
    state = result.state
    assert state["tenant_id"] == "tenant-1"
    assert state["case_id"] == "case-1"
    assert state["trace_id"] == "trace-1"
    assert state["raw_payload_path"] == "s3://bucket/tenant/uploads/blob-id"

    # Verify Normalized Input
    norm_input = state["normalized_document_input"]
    assert norm_input["ref"]["tenant_id"] == "tenant-1"
    assert norm_input["blob"]["uri"] == "s3://bucket/tenant/uploads/blob-id"
    assert (
        norm_input["blob"]["sha256"]
        == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    )
    assert norm_input["source"] == "upload"

    # Verify Meta
    # We can't easily check the meta passed to delay() without capturing it from the mock call args
    call_args = mock_run_graph.delay.call_args
    meta_arg = call_args[0][1]
    assert meta_arg["tenant_id"] == "tenant-1"
    assert meta_arg["invocation_id"] == "inv-1"


def test_upload_worker_register_document_failure_continues(
    mock_blob_writer, mock_run_graph, mock_domain_service
):
    # Simulate domain service failure
    mock_domain_service.ingest_document.side_effect = Exception("DB Error")

    worker = UploadWorker()
    upload = MockUploadedFile("test.txt", b"data", "text/plain")

    result = worker.process(
        upload,
        tenant_id="tenant-1",
    )

    # Needs to handle failure gracefully and still publish if possible,
    # OR fail. The implementation currently logs and returns None for ID,
    # but still proceeds to dispatch graph (which might handle persistence itself or fail later).
    # Since we generate a fallback ID in compose_state, it should succeed.

    assert result.status == "published"
    assert result.document_id is not None  # Should be a generated UUID
    assert result.task_id == "task-uuid"


def test_upload_worker_sets_media_type_on_fileblob(
    mock_blob_writer, mock_run_graph, mock_domain_service, mock_tenants
):
    worker = UploadWorker()
    file_content = b"pdf content"
    # Upload with explicit content_type
    upload = MockUploadedFile("test.pdf", file_content, "application/pdf")

    result = worker.process(
        upload,
        tenant_id="tenant-1",
        document_metadata={},
    )

    state = result.state
    norm_input = state["normalized_document_input"]

    # 1. Verify FileBlob has media_type
    assert norm_input["blob"]["media_type"] == "application/pdf"

    # 2. Verify pipeline_config has media_type (for Fix 1.3)
    assert norm_input["meta"]["pipeline_config"]["media_type"] == "application/pdf"
