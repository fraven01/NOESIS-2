"""Tests for document download view."""
import os
import pytest
import email.utils as email_utils
from datetime import datetime, timezone
from pathlib import Path
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


@pytest.fixture
def mock_repo(monkeypatch):
    """Mock repository."""
    repo = Mock()
    monkeypatch.setattr('documents.views._get_documents_repository', lambda: repo)
    return repo


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
    monkeypatch.setattr(
        'documents.views.get_upload_file_path',
        lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(f'/documents/download/{doc.ref.document_id}/')
    request.tenant = SimpleNamespace(
        tenant_id='tenant-test',
        schema_name='test'
    )

    response = document_download(request, str(doc.ref.document_id))

    assert isinstance(response, FileResponse)
    assert response['Content-Type'] == 'application/pdf'
    assert response['Content-Length'] == '16'
    assert 'document.pdf' in response['Content-Disposition']
    assert 'ETag' in response
    assert 'Last-Modified' in response
    assert response['Accept-Ranges'] == 'bytes'

    # Verify repo was called correctly
    mock_repo.get.assert_called_once_with('tenant-test', doc.ref.document_id)


def test_document_download_not_found(mock_repo):
    """Test 404 when document not found."""
    mock_repo.get.return_value = None

    factory = RequestFactory()
    request = factory.get('/documents/download/12345678-1234-1234-1234-123456789abc/')
    request.tenant = SimpleNamespace(tenant_id='tenant-test', schema_name='test')

    response = document_download(request, '12345678-1234-1234-1234-123456789abc')

    assert response.status_code == 404
    import json
    data = json.loads(response.content)
    assert data['error']['code'] == 'DocumentNotFound'


def test_document_download_tenant_mismatch(mock_repo, sample_document, monkeypatch):
    """Test 403 when tenant mismatch."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        'documents.views.get_upload_file_path',
        lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(f'/documents/download/{doc.ref.document_id}/')
    request.tenant = SimpleNamespace(
        tenant_id='tenant-other',  # Different tenant
        schema_name='other'
    )

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 403
    import json
    data = json.loads(response.content)
    assert data['error']['code'] == 'TenantMismatch'


def test_document_download_head_method(mock_repo, sample_document, monkeypatch):
    """Test HEAD request returns metadata without body."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        'documents.views.get_upload_file_path',
        lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.head(f'/documents/download/{doc.ref.document_id}/')
    request.tenant = SimpleNamespace(tenant_id='tenant-test', schema_name='test')

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 200
    assert response['Content-Type'] == 'application/pdf'
    assert response['Content-Length'] == '16'
    assert 'ETag' in response
    assert not response.content  # No body for HEAD


def test_document_download_etag_304(mock_repo, sample_document, monkeypatch):
    """Test 304 Not Modified when ETag matches."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        'documents.views.get_upload_file_path',
        lambda *args: test_file
    )

    # Generate expected ETag
    st = os.stat(test_file)
    etag = f'W/"{st.st_size:x}-{int(st.st_mtime):x}"'

    factory = RequestFactory()
    request = factory.get(
        f'/documents/download/{doc.ref.document_id}/',
        HTTP_IF_NONE_MATCH=etag
    )
    request.tenant = SimpleNamespace(tenant_id='tenant-test', schema_name='test')

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 304
    assert response['ETag'] == etag


def test_document_download_crlf_protection(mock_repo, sample_document, monkeypatch):
    """Test CRLF injection in filename is sanitized."""
    doc, test_file = sample_document
    # Create document with malicious filename
    doc.meta.title = 'evil\r\nSet-Cookie: pwned=true.pdf'
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        'documents.views.get_upload_file_path',
        lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(f'/documents/download/{doc.ref.document_id}/')
    request.tenant = SimpleNamespace(tenant_id='tenant-test', schema_name='test')

    response = document_download(request, str(doc.ref.document_id))

    disposition = response['Content-Disposition']
    # CRLF should be removed - this prevents header injection
    assert '\r' not in disposition
    assert '\n' not in disposition
    # Verify filename is properly quoted (Set-Cookie as text within quotes is safe)
    assert 'filename=' in disposition
    assert 'filename*=' in disposition


def test_document_download_range_request(mock_repo, sample_document, monkeypatch):
    """Test Range request returns 206 Partial Content."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        'documents.views.get_upload_file_path',
        lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(
        f'/documents/download/{doc.ref.document_id}/',
        HTTP_RANGE='bytes=0-4'
    )
    request.tenant = SimpleNamespace(tenant_id='tenant-test', schema_name='test')

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 206
    assert response['Content-Length'] == '5'
    assert response['Content-Range'] == 'bytes 0-4/16'


def test_document_download_range_invalid(mock_repo, sample_document, monkeypatch):
    """Test invalid Range request returns 416."""
    doc, test_file = sample_document
    mock_repo.get.return_value = doc

    monkeypatch.setattr(
        'documents.views.get_upload_file_path',
        lambda *args: test_file
    )

    factory = RequestFactory()
    request = factory.get(
        f'/documents/download/{doc.ref.document_id}/',
        HTTP_RANGE='bytes=100-200'  # Beyond file size
    )
    request.tenant = SimpleNamespace(tenant_id='tenant-test', schema_name='test')

    response = document_download(request, str(doc.ref.document_id))

    assert response.status_code == 416
    assert response['Content-Range'] == 'bytes */16'
