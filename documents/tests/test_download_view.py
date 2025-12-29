"""Tests for document download view."""

import os
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

from django.test import RequestFactory
from django.http import FileResponse

from documents.contracts import (
    NormalizedDocument,
    DocumentRef,
    DocumentMeta,
    InlineBlob,
)
from documents.views import document_download

pytestmark = pytest.mark.django_db


@pytest.fixture
def mock_repo(monkeypatch):
    """Mock repository."""
    repo = Mock()
    monkeypatch.setattr("documents.views._get_documents_repository", lambda: repo)
    return repo


@pytest.fixture(autouse=True)
def mock_authenticated_user(monkeypatch):
    user = SimpleNamespace(is_authenticated=True, id=1)
    monkeypatch.setattr("documents.views._resolve_request_user", lambda request: user)
    return user


@pytest.fixture(autouse=True)
def allow_download_permission(monkeypatch):
    access = SimpleNamespace(allowed=True, reason=None)
    monkeypatch.setattr(
        "documents.views.DocumentAuthzService.user_can_access_document_id",
        lambda **kwargs: access,
    )
    return access


@pytest.fixture(autouse=True)
def mock_activity_logger(monkeypatch):
    mock_logger = Mock()
    monkeypatch.setattr("documents.views.ActivityTracker.log", mock_logger)
    return mock_logger


@pytest.fixture
def sample_document(tmp_path):
    """Create sample document with actual file."""
    # Create physical file
    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"PDF content here")

    # Create NormalizedDocument matching actual structure
    doc_ref = DocumentRef(
        tenant_id="tenant-test",
        workflow_id="upload",
        document_id=UUID("12345678-1234-1234-1234-123456789abc"),
        collection_id=None,
        version=None,
    )

    doc_meta = DocumentMeta(
        tenant_id="tenant-test",
        workflow_id="upload",
        title="document.pdf",
        external_ref={"provider": "upload"},
    )

    blob = InlineBlob(
        type="inline",
        media_type="application/pdf",
        base64="UERGIGNvbnRlbnQgaGVyZQ==",
        sha256="a" * 64,
        size=16,
    )

    doc = NormalizedDocument(
        ref=doc_ref,
        meta=doc_meta,
        blob=blob,
        checksum="a" * 64,
        created_at=datetime.now(timezone.utc),
        source="upload",
    )

    return doc, test_file


def test_document_download_success(mock_repo, sample_document, monkeypatch):
    """Test successful download via FileResponse."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    # Mock get_upload_file_path to return our test file
    # NOTE: After refactoring, this is now used in DocumentAccessService
    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(f"/documents/download/{doc.ref.document_id}/")
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    assert isinstance(response, FileResponse)
    assert response["Content-Type"] == "application/pdf"
    assert response["Content-Length"] == "16"
    assert "document.pdf" in response["Content-Disposition"]
    assert "ETag" in response
    assert "Last-Modified" in response
    assert response["Accept-Ranges"] == "bytes"

    # Verify repo was called correctly
    mock_repo.get.assert_called_once_with("tenant-test", doc.ref.document_id)


def test_document_download_logs_activity(
    mock_repo, sample_document, monkeypatch, mock_activity_logger
):
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(f"/documents/download/{doc.ref.document_id}/")
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    assert isinstance(response, FileResponse)
    mock_activity_logger.assert_called_once()
    call_kwargs = mock_activity_logger.call_args.kwargs
    assert call_kwargs["activity_type"] == "DOWNLOAD"
    assert call_kwargs["document_id"] == doc.ref.document_id


def test_document_download_not_found(mock_repo):
    """Test 404 when document not found."""
    mock_repo.get.return_value = None

    factory = RequestFactory()
    request = factory.get("/documents/download/12345678-1234-1234-1234-123456789abc/")
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, "12345678-1234-1234-1234-123456789abc")

    assert response.status_code == 404
    import json

    data = json.loads(response.content)
    assert data["error"]["code"] == "DocumentNotFound"


def test_document_download_tenant_mismatch(mock_repo, sample_document, monkeypatch):
    """Test 403 when tenant mismatch."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(f"/documents/download/{doc.ref.document_id}/")
    request.tenant = SimpleNamespace(
        tenant_id="tenant-other", schema_name="other"  # Different tenant
    )

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 403
    import json

    data = json.loads(response.content)
    assert data["error"]["code"] == "TenantMismatch"


def test_document_download_head_method(mock_repo, sample_document, monkeypatch):
    """Test HEAD request returns metadata without body."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.head(f"/documents/download/{doc.ref.document_id}/")
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 200
    assert response["Content-Type"] == "application/pdf"
    assert response["Content-Length"] == "16"
    assert "ETag" in response
    assert not response.content  # No body for HEAD


def test_document_download_etag_304(mock_repo, sample_document, monkeypatch):
    """Test 304 Not Modified when ETag matches."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    # Generate expected ETag
    st = os.stat(test_file)
    etag = f'W/"{st.st_size:x}-{int(st.st_mtime):x}"'

    factory = RequestFactory()
    request = factory.get(
        f"/documents/download/{doc.ref.document_id}/", HTTP_IF_NONE_MATCH=etag
    )
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 304
    assert response["ETag"] == etag


def test_document_download_crlf_protection(mock_repo, sample_document, monkeypatch):
    """Test CRLF injection in filename is sanitized."""
    doc, test_file = sample_document
    # Create document with malicious filename
    doc.meta.title = "evil\r\nSet-Cookie: pwned=true.pdf"
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(f"/documents/download/{doc.ref.document_id}/")
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    disposition = response["Content-Disposition"]
    # CRLF should be removed - this prevents header injection
    assert "\r" not in disposition
    assert "\n" not in disposition
    # Verify filename is properly quoted (Set-Cookie as text within quotes is safe)
    assert "filename=" in disposition
    assert "filename*=" in disposition


def test_document_download_range_request(mock_repo, sample_document, monkeypatch):
    """Test Range request returns 206 Partial Content."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(
        f"/documents/download/{doc.ref.document_id}/", HTTP_RANGE="bytes=0-4"
    )
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 206
    assert response["Content-Length"] == "5"
    assert response["Content-Range"] == "bytes 0-4/16"


def test_document_download_range_invalid(mock_repo, sample_document, monkeypatch):
    """Test invalid Range request returns 416."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(
        f"/documents/download/{doc.ref.document_id}/",
        HTTP_RANGE="bytes=100-200",  # Beyond file size
    )
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 416
    assert response["Content-Range"] == "bytes */16"
    # Verify 416 includes cache headers
    assert response["Cache-Control"] == "private, max-age=3600"
    assert response["Accept-Ranges"] == "bytes"


def test_document_download_invalid_uuid(mock_repo):
    """Test 400 when document_id is not a valid UUID."""
    mock_repo.get.return_value = None

    factory = RequestFactory()
    request = factory.get("/documents/download/not-a-uuid/")
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, "not-a-uuid")

    assert response.status_code == 400
    import json

    data = json.loads(response.content)
    assert data["error"]["code"] == "InvalidDocumentId"


def test_document_download_suffix_range(mock_repo, sample_document, monkeypatch):
    """Test suffix range (bytes=-N) returns last N bytes."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(
        f"/documents/download/{doc.ref.document_id}/", HTTP_RANGE="bytes=-4"
    )
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 206
    assert response["Content-Length"] == "4"
    assert response["Content-Range"] == "bytes 12-15/16"  # Last 4 bytes


def test_document_download_multiple_etags(mock_repo, sample_document, monkeypatch):
    """Test 304 when If-None-Match contains multiple ETags."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        "documents.access_service.get_upload_file_path", lambda *args: test_file
    )

    # Generate expected ETag
    st = os.stat(test_file)
    etag = f'W/"{st.st_size:x}-{int(st.st_mtime):x}"'

    factory = RequestFactory()
    request = factory.get(
        f"/documents/download/{doc.ref.document_id}/",
        HTTP_IF_NONE_MATCH=f'W/"wrong-tag", {etag}, W/"another-wrong"',
    )
    request.tenant = SimpleNamespace(tenant_id="tenant-test", schema_name="test")

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 304
    assert response["ETag"] == etag
    # Verify 304 includes cache headers
    assert response["Cache-Control"] == "private, max-age=3600"
    assert response["Accept-Ranges"] == "bytes"
