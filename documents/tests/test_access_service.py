"""Unit tests for ``DocumentAccessService`` business logic."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4


from documents.access_service import (
    AccessError,
    DocumentAccessResult,
    DocumentAccessService,
)


def _build_stub_document(tenant_id: str, workflow_id: str, document_id):
    """Return a lightweight document stub with the required identifiers."""
    ref = SimpleNamespace(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
    )
    return SimpleNamespace(ref=ref)


def test_get_document_for_download_success(tmp_path, monkeypatch):
    """Successful access resolves the file and returns metadata."""
    document_id = uuid4()
    repo = Mock()
    repo.get.return_value = _build_stub_document("tenant-a", "upload", document_id)

    blob_path = tmp_path / "test.bin"
    blob_path.write_bytes(b"content")

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path",
        lambda *args, **kwargs: blob_path,
    )

    service = DocumentAccessService(repo)
    result, error = service.get_document_for_download("tenant-a", document_id)

    assert error is None
    assert isinstance(result, DocumentAccessResult)
    assert result.blob_path == blob_path
    assert result.file_size == blob_path.stat().st_size
    assert result.document is repo.get.return_value
    repo.get.assert_called_once_with("tenant-a", document_id)


def test_get_document_for_download_not_found(monkeypatch):
    """Missing documents return a 404 access error."""
    document_id = uuid4()
    repo = Mock()
    repo.get.return_value = None

    # Monkeypatch upload resolver to ensure it is not reached
    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path",
        lambda *args, **kwargs: Path("/tmp/never-used"),
    )

    service = DocumentAccessService(repo)
    result, error = service.get_document_for_download("tenant-a", document_id)

    assert result is None
    assert isinstance(error, AccessError)
    assert error.error_code == "DocumentNotFound"
    repo.get.assert_called_once_with("tenant-a", document_id)


def test_get_document_for_download_tenant_mismatch(tmp_path, monkeypatch):
    """Requests for another tenant are rejected with 403."""
    document_id = uuid4()
    repo = Mock()
    repo.get.return_value = _build_stub_document("tenant-b", "upload", document_id)

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path",
        lambda *args, **kwargs: tmp_path / "unused.bin",
    )

    service = DocumentAccessService(repo)
    result, error = service.get_document_for_download("tenant-a", document_id)

    assert result is None
    assert isinstance(error, AccessError)
    assert error.error_code == "TenantMismatch"


def test_get_document_for_download_blob_missing(tmp_path, monkeypatch):
    """Missing files are reported as BlobNotFound."""
    document_id = uuid4()
    repo = Mock()
    repo.get.return_value = _build_stub_document("tenant-a", "upload", document_id)

    missing_blob = tmp_path / "does-not-exist.bin"
    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path",
        lambda *args, **kwargs: missing_blob,
    )

    service = DocumentAccessService(repo)
    result, error = service.get_document_for_download("tenant-a", document_id)

    assert result is None
    assert isinstance(error, AccessError)
    assert error.error_code == "BlobNotFound"
