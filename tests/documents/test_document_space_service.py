from uuid import uuid4

import pytest
from django.utils import timezone
from django_tenants.utils import schema_context

from cases.models import Case
from documents.contracts import DocumentMeta, DocumentRef, FileBlob, NormalizedDocument
from documents.services.document_space_service import (
    DocumentSpaceRequest,
    DocumentSpaceService,
)
from documents.models import DocumentCollection
from documents.repository import DocumentsRepository


class _CollectionServiceStub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ensure_manual_collection(self, tenant: object, **_: object) -> str:
        tenant_str = str(tenant)
        self.calls.append(tenant_str)
        return tenant_str


class _RepositoryStub(DocumentsRepository):
    def __init__(self, doc: NormalizedDocument) -> None:
        self._doc = doc

    def list_by_collection(
        self,
        tenant_id: str,
        collection_id,
        limit: int = 100,
        cursor=None,
        latest_only: bool = False,
        *,
        workflow_id=None,
    ):
        return ([self._doc.ref], None)

    def list_latest_by_collection(
        self,
        tenant_id: str,
        collection_id,
        limit: int = 100,
        cursor=None,
        *,
        workflow_id=None,
    ):
        return self.list_by_collection(
            tenant_id=tenant_id,
            collection_id=collection_id,
            limit=limit,
            cursor=cursor,
            workflow_id=workflow_id,
        )

    def get(
        self,
        tenant_id: str,
        document_id,
        version: str | None = None,
        *,
        prefer_latest: bool = False,
        workflow_id: str | None = None,
        include_retired: bool = False,
    ):
        if document_id == self._doc.ref.document_id:
            return self._doc
        return None


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_document_space_service_builds_context(tenant_pool):
    tenant = tenant_pool["alpha"]
    with schema_context(tenant.schema_name):
        case = Case.objects.create(
            tenant=tenant, external_id="CASE-999", title="Service Case"
        )
        collection = DocumentCollection.objects.create(
            tenant=tenant,
            case=case,
            name="Service Docs",
            key="service-docs",
            collection_id=uuid4(),
            type="service",
        )

    document_id = uuid4()
    workflow_id = "service-workflow"
    checksum_value = "b" * 64
    normalized = NormalizedDocument(
        ref=DocumentRef(
            tenant_id=tenant.schema_name,
            workflow_id=workflow_id,
            document_id=document_id,
            collection_id=collection.collection_id,
            document_collection_id=collection.id,
            version="v9",
        ),
        meta=DocumentMeta(
            tenant_id=tenant.schema_name,
            workflow_id=workflow_id,
            title="Service Doc",
            language="de",
            tags=["demo"],
            document_collection_id=collection.id,
            origin_uri="https://example.com/demo.pdf",
        ),
        blob=FileBlob(
            type="file",
            uri="object://tenant/service/demo.pdf",
            sha256=checksum_value,
            size=42,
        ),
        checksum=checksum_value,
        created_at=timezone.now(),
        source="upload",
    )

    repository = _RepositoryStub(normalized)
    service = DocumentSpaceService(collection_service=_CollectionServiceStub())
    params = DocumentSpaceRequest(
        requested_collection=str(collection.id),
        limit=25,
        latest_only=True,
        cursor=None,
        workflow_filter=None,
        search_term="Service",
    )
    result = service.build_context(
        tenant_context=tenant.schema_name,
        tenant_obj=tenant,
        params=params,
        repository=repository,
    )

    assert result.documents
    doc_payload = result.documents[0]
    assert doc_payload["title"] == "Service Doc"
    assert result.selected_collection_identifier == str(collection.id)
    assert not result.collection_warning
    assert result.document_summary["displayed"] == 1


def test_inference_prioritizes_pipeline_config():
    service = DocumentSpaceService()

    # Mock doc with pipeline_config and metadata
    class MockMeta:
        pipeline_config = {"media_type": "application/pdf"}  # Expected
        metadata = {"content_type": "text/html"}  # Trap
        origin_uri = "http://example.com/file.html"  # Trap
        # title is missing/ignored in this case

    class MockDoc:
        meta = MockMeta()
        title = "file.html"

    # We expect application/pdf because pipeline_config has priority
    result = service._infer_media_type_from_doc(MockDoc(), None)
    assert result == "application/pdf"


def test_inference_fallbacks():
    service = DocumentSpaceService()

    # Case 1: External Ref has priority over metadata but after pipeline_config
    class MockDocEmpty:
        meta = type(
            "Meta", (), {"pipeline_config": {}, "metadata": {}, "origin_uri": None}
        )()
        title = None

    external_ref = {"media_type": "application/json"}
    result = service._infer_media_type_from_doc(MockDocEmpty(), external_ref)
    assert result == "application/json"

    # Case 2: Metadata fallback
    class MockDocMeta:
        meta = type(
            "Meta",
            (),
            {
                "pipeline_config": {},
                "metadata": {"mime_type": "text/markdown"},
                "origin_uri": None,
            },
        )()
        title = None

    result = service._infer_media_type_from_doc(MockDocMeta(), None)
    assert result == "text/markdown"

    # Case 3: URL/Extension fallback
    class MockDocUrl:
        meta = type(
            "Meta",
            (),
            {
                "pipeline_config": {},
                "metadata": {},
                "origin_uri": "http://example.com/data.csv",  # unknown extension in map
            },
        )()
        title = "report.pdf"  # .pdf is known

    result = service._infer_media_type_from_doc(MockDocUrl(), None)
    assert result == "application/pdf"


def test_finalize_blob_media_type_uses_default():
    service = DocumentSpaceService()

    # Blob with no media type
    blob_info = {"media_type": None}

    # Doc without any hints
    class MockDocNone:
        meta = type(
            "Meta", (), {"pipeline_config": {}, "metadata": {}, "origin_uri": None}
        )()
        title = "unknown_file"

    service._finalize_blob_media_type(blob_info, MockDocNone(), None)

    # Should fall back to DEFAULT_MEDIA_TYPE (application/octet-stream)
    assert blob_info["media_type"] == "application/octet-stream"
    assert blob_info["media_type_display"] == "application/octet-stream"
