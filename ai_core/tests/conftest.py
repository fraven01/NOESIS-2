import base64
from typing import Any

import pytest

from ai_core import ingestion_status
from ai_core.infra import object_store
from documents.contracts import FileBlob, InlineBlob, NormalizedDocument
from documents.repository import InMemoryDocumentsRepository

from tests.plugins.rag_db import *  # noqa: F401,F403


class CapturingInMemoryDocumentsRepository(InMemoryDocumentsRepository):
    """In-memory repository that also records persisted docs for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.saved: list[NormalizedDocument] = []

    def upsert(
        self,
        doc: NormalizedDocument,
        workflow_id: str | None = None,
        *,
        scope: object | None = None,
        audit_meta: object | None = None,
    ) -> NormalizedDocument:
        payload = _extract_payload(doc.blob)
        original_flag = getattr(self, "_inline_conversion_enabled", True)
        self._inline_conversion_enabled = False
        try:
            persisted = super().upsert(
                doc, workflow_id=workflow_id, scope=scope, audit_meta=audit_meta
            )
        finally:
            self._inline_conversion_enabled = original_flag
        source_media_type = getattr(doc.blob, "media_type", None)
        if source_media_type:
            try:
                object.__setattr__(persisted.blob, "media_type", source_media_type)
            except Exception:
                pass
        # Ensure a parsable media_type for ingestion (markdown default)
        if not getattr(persisted.blob, "media_type", None):
            try:
                object.__setattr__(persisted.blob, "media_type", "text/markdown")
            except Exception:
                pass
        self.saved.append(persisted)

        # Emit legacy-style artifacts for tests that assert on object_store paths.
        try:
            tenant_segment = object_store.sanitize_identifier(doc.ref.tenant_id)
            workflow_segment = object_store.sanitize_identifier("upload")
            storage_prefix = f"{tenant_segment}/{workflow_segment}/uploads"
            if payload:
                upload_filename = f"{doc.ref.document_id}_upload.bin"
                object_store.put_bytes(f"{storage_prefix}/{upload_filename}", payload)
            metadata: dict[str, object] = {}
            external_ref = getattr(doc.meta, "external_ref", {}) or {}
            external_id = external_ref.get("external_id")
            if external_id:
                metadata["external_id"] = external_id
            if doc.ref.collection_id is not None:
                metadata["collection_id"] = str(doc.ref.collection_id)
            if doc.ref.version is not None:
                metadata["version"] = doc.ref.version
            object_store.write_json(
                f"{storage_prefix}/{doc.ref.document_id}.meta.json", metadata
            )
        except Exception:
            # Tests should still pass even if artifact writing fails
            pass

        return persisted

    def get(  # type: ignore[override]
        self,
        tenant_id: str,
        document_id: Any,
        version: str | None = None,
        *,
        prefer_latest: bool = False,
        workflow_id: str | None = None,
    ) -> NormalizedDocument | None:
        doc = super().get(
            tenant_id,
            document_id,
            version,
            prefer_latest=prefer_latest,
            workflow_id=workflow_id,
        )
        if doc is None:
            return None
        if getattr(self, "_inline_conversion_enabled", True):
            payload = _extract_payload(doc.blob)
            if payload:
                media_type = getattr(doc.blob, "media_type", None) or "text/markdown"
                doc.blob = InlineBlob(
                    type="inline",
                    media_type=media_type,
                    base64=base64.b64encode(payload).decode("ascii"),
                    sha256=getattr(doc.blob, "sha256", doc.checksum),
                    size=len(payload),
                )
        return doc


def _extract_payload(blob: Any) -> bytes:
    if isinstance(blob, InlineBlob):
        return blob.decoded_payload()
    if isinstance(blob, FileBlob):
        try:
            return object_store.read_bytes(getattr(blob, "uri", ""))
        except Exception:
            return b""
    return b""


@pytest.fixture
def documents_repository_stub(monkeypatch) -> CapturingInMemoryDocumentsRepository:
    from ai_core import services, views

    repository = CapturingInMemoryDocumentsRepository()

    monkeypatch.setattr(views, "DOCUMENTS_REPOSITORY", repository, raising=False)
    monkeypatch.setattr(services, "_DOCUMENTS_REPOSITORY", None, raising=False)
    try:
        yield repository
    finally:
        monkeypatch.setattr(views, "DOCUMENTS_REPOSITORY", None, raising=False)


@pytest.fixture(autouse=True)
def _auto_documents_repository(documents_repository_stub):
    yield


@pytest.fixture(autouse=True)
def ingestion_status_store(monkeypatch):
    from ai_core.tests.doubles import MemoryDocumentLifecycleStore

    store = MemoryDocumentLifecycleStore()
    monkeypatch.setattr(ingestion_status, "_LIFECYCLE_STORE", store, raising=False)
    yield store


@pytest.fixture(autouse=True)
def disable_async_graphs(monkeypatch):
    """Disable async graph execution in tests to run graphs synchronously."""
    from ai_core import services

    monkeypatch.setattr(services, "_should_enqueue_graph", lambda graph_name: False)
    yield


@pytest.fixture
def tenant_client(client, monkeypatch):
    """Django test client with tenant middleware mocked for easy testing.

    This fixture bypasses the django-tenants middleware hostname lookup
    and directly sets the tenant on the request, allowing tests to work
    without needing to configure proper domain names.
    """
    from django_tenants.middleware import TenantMainMiddleware
    from customers.models import Tenant

    def _mock_process_request(self, request):
        """Mock middleware to set tenant directly without hostname lookup."""
        # Use hardcoded 'autotest' schema created by test_tenant_schema_name fixture
        request.tenant = Tenant.objects.get(schema_name="autotest")
        return None

    monkeypatch.setattr(TenantMainMiddleware, "process_request", _mock_process_request)

    return client


@pytest.fixture(autouse=True)
def cleanup_graph_cache():
    """Clear graph cache after each test to prevent memory leaks.

    This ensures:
    - Test isolation (no cache pollution between tests)
    - Memory management (graph is garbage collected)
    - Fresh graph state for each test
    """
    yield
    try:
        from ai_core.graphs.technical.universal_ingestion_graph import (
            _clear_cached_processing_graph,
        )

        _clear_cached_processing_graph()
    except ImportError:
        # Module not loaded, nothing to clear
        pass
