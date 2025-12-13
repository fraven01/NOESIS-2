import base64
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import jsonschema
import pytest

from pydantic import TypeAdapter, ValidationError

from documents.contracts import (
    Asset,
    AssetRef,
    BlobLocator,
    DocumentMeta,
    DocumentRef,
    NormalizedDocument,
    asset_ref_schema,
    asset_schema,
    blob_locator_schema,
    document_meta_schema,
    document_ref_schema,
    normalized_document_schema,
)
from documents.contracts_context import (
    asset_media_guard,
    set_asset_media_guard,
    set_strict_checksums,
    strict_checksums,
)
from documents.contract_utils import (
    is_bcp47_like,
    is_image_mediatype,
    normalize_media_type,
    normalize_tags,
    truncate_text,
    validate_bbox,
)


BLOB_LOCATOR_ADAPTER = TypeAdapter(BlobLocator)
DEFAULT_WORKFLOW = "workflow-1"


@pytest.fixture(autouse=True)
def reset_contract_toggles():
    set_strict_checksums(False)
    set_asset_media_guard(None)
    yield
    set_strict_checksums(False)
    set_asset_media_guard(None)


def test_document_ref_normalization():
    doc_id = uuid4()
    ref = DocumentRef(
        tenant_id="  tenant\u200b ",
        workflow_id="  workflow-01  ",
        document_id=doc_id,
        version=" v1.2 ",
    )
    assert ref.tenant_id == "tenant"
    assert ref.workflow_id == "workflow-01"
    assert ref.version == "v1.2"
    assert ref.document_id is doc_id


def test_document_and_asset_ref_accept_string_ids():
    doc_id = uuid4()
    collection_id = uuid4()
    asset_id = uuid4()
    doc_ref = DocumentRef(
        tenant_id="tenant",
        workflow_id="flow-1",
        document_id=str(doc_id),
        collection_id=str(collection_id),
    )
    assert doc_ref.document_id == doc_id
    assert doc_ref.collection_id == collection_id

    asset_ref = AssetRef(
        tenant_id="tenant",
        workflow_id="flow-1",
        asset_id=str(asset_id),
        document_id=str(doc_id),
        collection_id=str(collection_id),
    )
    assert asset_ref.asset_id == asset_id
    assert asset_ref.document_id == doc_id
    assert asset_ref.collection_id == collection_id


@pytest.mark.parametrize(
    "tenant_id",
    ["", "   ", "\u200b"],
)
def test_document_ref_rejects_empty_tenant(tenant_id):
    with pytest.raises(ValueError):
        DocumentRef(tenant_id=tenant_id, workflow_id="wf", document_id=uuid4())


def test_document_ref_version_regex():
    ref = DocumentRef(
        tenant_id="acme", workflow_id="wf", document_id=uuid4(), version=""
    )
    assert ref.version is None
    with pytest.raises(ValueError):
        DocumentRef(
            tenant_id="acme",
            workflow_id="wf",
            document_id=uuid4(),
            version="bad version",
        )


def test_document_meta_tags_and_language():
    meta = DocumentMeta(
        tenant_id=" acme ",
        workflow_id=" Workflow-Id ",
        title="  Example Title  ",
        language="en-US",
        tags=[" alpha ", "beta"],
    )
    assert meta.tenant_id == "acme"
    assert meta.workflow_id == "Workflow-Id"
    assert meta.title == "Example Title"
    assert meta.tags == ["alpha", "beta"]
    assert meta.language == "en-US"


def test_document_meta_tags_deduplicate_and_strip_invisibles():
    meta = DocumentMeta(
        tenant_id="acme",
        workflow_id="wf",
        tags=["  foo  ", "foo", "bar\u200b"],
    )
    assert meta.tags == ["bar", "foo"]


@pytest.mark.parametrize("tags", [["bad tag"], ["a" * 65], ["value!"], ["white space"]])
def test_document_meta_invalid_tags(tags):
    with pytest.raises(ValueError):
        DocumentMeta(tenant_id="acme", workflow_id="wf", tags=tags)


def test_document_meta_empty_tags_are_removed():
    meta = DocumentMeta(tenant_id="acme", workflow_id="wf", tags=["", "  ", "value"])
    assert meta.tags == ["value"]


def test_document_meta_external_ref_entry_limit():
    valid = {f"key{i}": f"value{i}" for i in range(16)}
    meta = DocumentMeta(tenant_id="acme", workflow_id="wf", external_ref=valid)
    assert len(meta.external_ref or {}) == 16

    too_many = {f"key{i}": "value" for i in range(17)}
    with pytest.raises(ValueError):
        DocumentMeta(tenant_id="acme", workflow_id="wf", external_ref=too_many)


def test_contract_utils_tag_helper_sorting_and_deduplication():
    assert normalize_tags(["beta", "alpha", "alpha", "\u200b"]) == ["alpha", "beta"]


def test_contract_utils_language_checks():
    assert is_bcp47_like("de-DE")
    assert not is_bcp47_like("en--us")
    assert not is_bcp47_like("-en")


def test_contract_utils_media_type_guard():
    assert is_image_mediatype(" Image/PNG ")
    assert not is_image_mediatype("text/plain")
    assert not is_image_mediatype("invalid type")


def test_contract_utils_media_type_normalization():
    assert normalize_media_type(" Image/PNG ") == "image/png"
    assert normalize_media_type("image/unspecified") == "image/unspecified"
    with pytest.raises(ValueError):
        normalize_media_type("invalid type")


def test_contract_utils_media_type_strips_parameters():
    # normalize_media_type strips parameters per docstring:
    # "Parameterized values such as text/html; charset=utf-8 are stripped to their base type."
    assert normalize_media_type("text/html; charset=utf-8") == "text/html"


def test_contract_utils_bbox_validation():
    coords = [0.1, 0.2, 0.9, 0.95]
    assert validate_bbox(coords) == coords
    with pytest.raises(ValueError):
        validate_bbox([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        validate_bbox([0.0, 0.0, 1.1, 1.0])
    with pytest.raises(ValueError):
        validate_bbox([0.0, 0.0, 1.0])


def test_contract_utils_truncate_text_utf8_safe():
    euro_text = "€€€"
    truncated = truncate_text(euro_text, 4)
    assert truncated == "€"
    assert truncate_text(euro_text, 9) == euro_text
    assert truncate_text(None, 5) is None


@pytest.mark.parametrize(
    "language",
    ["not--valid", "toolong-123456789", "en_US", "en-", "-en", "123-456"],
)
def test_document_meta_invalid_language(language):
    with pytest.raises(ValueError):
        DocumentMeta(tenant_id="acme", workflow_id="wf", language=language)


def make_inline_blob(
    media_type: str = "image/png", payload: bytes = b"payload"
) -> BlobLocator:
    payload_bytes = payload
    payload = base64.b64encode(payload_bytes).decode()
    return {
        "type": "inline",
        "media_type": media_type,
        "base64": payload,
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "size": len(payload_bytes),
    }


def make_file_blob(uri: str, payload: bytes) -> dict:
    return {
        "type": "file",
        "uri": uri,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def test_blob_locator_variants():
    file_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        {
            "type": "file",
            "uri": "s3://bucket/object",
            "sha256": "b" * 64,
            "size": 0,
        }
    )
    assert file_blob.sha256 == "b" * 64

    inline_blob = BLOB_LOCATOR_ADAPTER.validate_python(make_inline_blob("Image/PNG"))
    assert inline_blob.media_type == "image/png"

    for kind in ("http", "https"):
        external_blob = BLOB_LOCATOR_ADAPTER.validate_python(
            {"type": "external", "kind": kind, "uri": "https://example.com/file"}
        )
        assert external_blob.kind == kind


def test_inline_blob_size_mismatch():
    payload = base64.b64encode(b"payload").decode()
    with pytest.raises(ValueError):
        BLOB_LOCATOR_ADAPTER.validate_python(
            {
                "type": "inline",
                "media_type": "image/png",
                "base64": payload,
                "sha256": hashlib.sha256(b"payload").hexdigest(),
                "size": len(b"payload") + 1,
            }
        )


def test_blob_locator_validation_errors():
    with pytest.raises(ValueError):
        BLOB_LOCATOR_ADAPTER.validate_python(
            {"type": "file", "uri": "a", "sha256": "z" * 64, "size": -1}
        )

    with pytest.raises(ValueError):
        BLOB_LOCATOR_ADAPTER.validate_python(
            {
                "type": "inline",
                "media_type": "image/png",
                "base64": "@@",
                "sha256": "a" * 64,
                "size": 1,
            }
        )

    with pytest.raises(ValueError) as exc:
        BLOB_LOCATOR_ADAPTER.validate_python({"type": "unknown"})
    assert "type" in str(exc.value)
    assert "file" in str(exc.value)


def build_asset(
    media_type: str = "image/png", workflow_id: str = DEFAULT_WORKFLOW
) -> Asset:
    blob = BLOB_LOCATOR_ADAPTER.validate_python(make_inline_blob(media_type))
    return Asset(
        ref=AssetRef(
            tenant_id="acme",
            workflow_id=workflow_id,
            asset_id=uuid4(),
            document_id=uuid4(),
        ),
        media_type=media_type,
        blob=blob,
        caption_method="manual",
        created_at=datetime.now(timezone.utc),
        checksum=blob.sha256,
    )


def test_asset_bbox_and_limits():
    asset = build_asset()
    asset.bbox = [0.1, 0.2, 0.9, 0.95]
    asset.context_before = "context"
    asset.context_after = "after"
    asset.ocr_text = "ocr"
    asset.text_description = "desc"
    assert asset.bbox == [0.1, 0.2, 0.9, 0.95]


def test_asset_workflow_id_property():
    asset = build_asset()
    assert asset.workflow_id == asset.ref.workflow_id


def test_asset_invalid_bbox():
    asset = build_asset()
    with pytest.raises(ValueError):
        data = asset.model_dump(mode="python")
        data["bbox"] = [0.0, 0.0, 0.0, 1.0]
        Asset.model_validate(data)
    with pytest.raises(ValueError):
        data = asset.model_dump(mode="python")
        data["bbox"] = [0.0, 0.0, 1.1, 1.0]
        Asset.model_validate(data)


def test_asset_media_type_consistency():
    with pytest.raises(ValueError):
        asset = build_asset(media_type="image/jpeg")
        data = asset.model_dump(mode="python")
        data["blob"] = BLOB_LOCATOR_ADAPTER.validate_python(
            make_inline_blob("application/pdf")
        )
        Asset.model_validate(data)


def test_asset_media_type_case_insensitive_acceptance():
    inline_blob = make_inline_blob("image/png")
    asset = Asset.model_validate(
        {
            "ref": {
                "tenant_id": "acme",
                "workflow_id": DEFAULT_WORKFLOW,
                "asset_id": str(uuid4()),
                "document_id": str(uuid4()),
            },
            "media_type": "IMAGE/PNG",
            "blob": inline_blob,
            "caption_method": "manual",
            "created_at": datetime.now(timezone.utc),
            "checksum": inline_blob["sha256"],
        }
    )
    assert asset.media_type == "image/png"
    assert asset.blob.media_type == "image/png"


def test_asset_media_guard_rejects_mismatched_prefix():
    file_blob = {
        "type": "file",
        "uri": "s3://bucket/asset.png",
        "sha256": "f" * 64,
        "size": 10,
    }
    with asset_media_guard(["image/"]):
        with pytest.raises(ValueError):
            Asset.model_validate(
                {
                    "ref": {
                        "tenant_id": "acme",
                        "workflow_id": DEFAULT_WORKFLOW,
                        "asset_id": str(uuid4()),
                        "document_id": str(uuid4()),
                    },
                    "media_type": "application/pdf",
                    "blob": file_blob,
                    "caption_method": "manual",
                    "created_at": datetime.now(timezone.utc),
                    "checksum": file_blob["sha256"],
                }
            )


def test_asset_media_guard_allows_matching_prefix():
    file_blob = {
        "type": "file",
        "uri": "s3://bucket/asset.png",
        "sha256": "f" * 64,
        "size": 10,
    }
    with asset_media_guard(["image/"]):
        asset = Asset.model_validate(
            {
                "ref": {
                    "tenant_id": "acme",
                    "workflow_id": DEFAULT_WORKFLOW,
                    "asset_id": str(uuid4()),
                    "document_id": str(uuid4()),
                },
                "media_type": "image/png",
                "blob": file_blob,
                "caption_method": "manual",
                "created_at": datetime.now(timezone.utc),
                "checksum": file_blob["sha256"],
            }
        )
    assert asset.media_type == "image/png"


def test_asset_media_guard_supports_multiple_prefixes():
    file_blob = {
        "type": "file",
        "uri": "s3://bucket/asset.mp4",
        "sha256": "e" * 64,
        "size": 42,
    }
    with asset_media_guard(["image/", "video/"]):
        asset = Asset.model_validate(
            {
                "ref": {
                    "tenant_id": "acme",
                    "workflow_id": DEFAULT_WORKFLOW,
                    "asset_id": str(uuid4()),
                    "document_id": str(uuid4()),
                },
                "media_type": "video/mp4",
                "blob": file_blob,
                "caption_method": "manual",
                "created_at": datetime.now(timezone.utc),
                "checksum": file_blob["sha256"],
            }
        )
    assert asset.media_type == "video/mp4"


def test_asset_caption_requirements():
    asset = build_asset()
    data = asset.model_dump(mode="python")
    data.update(
        {
            "caption_method": "vlm_caption",
            "caption_model": "  my-model  ",
            "caption_confidence": 0.9,
        }
    )
    asset = Asset.model_validate(data)
    assert asset.caption_model == "my-model"

    with pytest.raises(ValueError):
        data = asset.model_dump(mode="python")
        data["caption_confidence"] = None
        Asset.model_validate(data)

    with pytest.raises(ValueError):
        data = asset.model_dump(mode="python")
        data["caption_confidence"] = 2.0
        Asset.model_validate(data)

    with pytest.raises(ValueError):
        data = asset.model_dump(mode="python")
        data["caption_model"] = "   "
        data["caption_method"] = "vlm_caption"
        data["caption_confidence"] = 0.5
        Asset.model_validate(data)


def test_asset_page_index_limit():
    with pytest.raises(ValueError):
        data = build_asset().model_dump(mode="python")
        data["page_index"] = -1
        Asset.model_validate(data)


def test_asset_caption_confidence_boundaries_and_model_normalization():
    asset = build_asset()
    data = asset.model_dump(mode="python")
    data.update(
        {
            "caption_method": "vlm_caption",
            "caption_model": "  model-x  ",
            "caption_confidence": 0.0,
        }
    )
    validated = Asset.model_validate(data)
    assert validated.caption_confidence == 0.0
    assert validated.caption_model == "model-x"

    data["caption_confidence"] = 1.0
    validated = Asset.model_validate(data)
    assert validated.caption_confidence == 1.0


def test_asset_context_is_truncated_to_limit():
    large = "a" * 2050
    data = build_asset().model_dump(mode="python")
    data["context_before"] = large
    validated = Asset.model_validate(data)
    assert len(validated.context_before) == 2048
    assert validated.context_before == large[:2048]


def test_asset_context_byte_boundaries():
    exact_context = "a" * 2048
    exact_ocr = "b" * 8192
    data = build_asset().model_dump(mode="python")
    data["context_after"] = exact_context
    data["ocr_text"] = exact_ocr
    validated = Asset.model_validate(data)
    assert validated.context_after == exact_context
    assert validated.ocr_text == exact_ocr


def test_asset_text_fields_truncate_utf8_boundaries():
    data = build_asset().model_dump(mode="python")
    data["text_description"] = "ü" * 2050
    data["ocr_text"] = "ü" * 5000
    validated = Asset.model_validate(data)
    assert validated.text_description == "ü" * 1024
    assert len(validated.ocr_text.encode("utf-8")) <= 8192


def build_document(workflow_id: str = DEFAULT_WORKFLOW) -> NormalizedDocument:
    doc_id = uuid4()
    blob = BLOB_LOCATOR_ADAPTER.validate_python(make_inline_blob())
    asset = Asset(
        ref=AssetRef(
            tenant_id="acme",
            workflow_id=workflow_id,
            asset_id=uuid4(),
            document_id=doc_id,
        ),
        media_type="image/png",
        blob=blob,
        caption_method="none",
        created_at=datetime.now(timezone.utc),
        checksum=blob.sha256,
    )
    doc_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        {
            "type": "file",
            "uri": "s3://bucket/document.pdf",
            "sha256": "e" * 64,
            "size": 123,
        }
    )
    return NormalizedDocument(
        ref=DocumentRef(tenant_id="acme", workflow_id=workflow_id, document_id=doc_id),
        meta=DocumentMeta(tenant_id="acme", workflow_id=workflow_id),
        blob=doc_blob,
        checksum=doc_blob.sha256,
        created_at=datetime.now(timezone.utc),
        assets=[asset],
    )


def test_normalized_document_relationships():
    doc = build_document()
    assert doc.assets[0].ref.document_id == doc.ref.document_id


def test_normalized_document_meta_workflow_mismatch():
    doc = build_document()
    data = doc.model_dump(mode="python")
    data["meta"]["workflow_id"] = "other-flow"
    with pytest.raises(ValidationError) as exc:
        NormalizedDocument.model_validate(data)
    assert any(
        "meta_workflow_mismatch" in err.get("msg", "") for err in exc.value.errors()
    )


def test_normalized_document_asset_workflow_mismatch():
    doc = build_document()
    data = doc.model_dump(mode="python")
    data["assets"][0]["ref"]["workflow_id"] = "other-flow"
    with pytest.raises(ValidationError) as exc:
        NormalizedDocument.model_validate(data)
    assert any(
        "asset_workflow_mismatch" in err.get("msg", "") for err in exc.value.errors()
    )


def test_normalized_document_invalid_tenant():
    doc_id = uuid4()
    blob = BLOB_LOCATOR_ADAPTER.validate_python(make_inline_blob())
    asset = Asset(
        ref=AssetRef(
            tenant_id="other",
            workflow_id=DEFAULT_WORKFLOW,
            asset_id=uuid4(),
            document_id=doc_id,
        ),
        media_type="image/png",
        blob=blob,
        caption_method="none",
        created_at=datetime.now(timezone.utc),
        checksum=blob.sha256,
    )
    doc_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        {
            "type": "file",
            "uri": "s3://bucket/document.pdf",
            "sha256": "e" * 64,
            "size": 123,
        }
    )
    with pytest.raises(ValueError):
        NormalizedDocument(
            ref=DocumentRef(
                tenant_id="acme",
                workflow_id=DEFAULT_WORKFLOW,
                document_id=doc_id,
            ),
            meta=DocumentMeta(tenant_id="acme", workflow_id=DEFAULT_WORKFLOW),
            blob=doc_blob,
            checksum=doc_blob.sha256,
            created_at=datetime.now(timezone.utc),
            assets=[asset],
        )


def test_normalized_document_requires_utc():
    with pytest.raises(ValueError):
        doc_blob = BLOB_LOCATOR_ADAPTER.validate_python(
            {
                "type": "file",
                "uri": "s3://bucket/document.pdf",
                "sha256": "e" * 64,
                "size": 123,
            }
        )
        NormalizedDocument(
            ref=DocumentRef(
                tenant_id="acme", workflow_id=DEFAULT_WORKFLOW, document_id=uuid4()
            ),
            meta=DocumentMeta(tenant_id="acme", workflow_id=DEFAULT_WORKFLOW),
            blob=doc_blob,
            checksum=doc_blob.sha256,
            created_at=datetime.now(),
        )


def test_normalized_document_collection_mismatch():
    doc_id = uuid4()
    blob = BLOB_LOCATOR_ADAPTER.validate_python(make_inline_blob())
    asset = Asset(
        ref=AssetRef(
            tenant_id="acme",
            workflow_id=DEFAULT_WORKFLOW,
            asset_id=uuid4(),
            document_id=doc_id,
            collection_id=None,
        ),
        media_type="image/png",
        blob=blob,
        caption_method="none",
        created_at=datetime.now(timezone.utc),
        checksum=blob.sha256,
    )
    doc_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        {
            "type": "file",
            "uri": "s3://bucket/document.pdf",
            "sha256": "e" * 64,
            "size": 123,
        }
    )
    with pytest.raises(ValueError):
        NormalizedDocument(
            ref=DocumentRef(
                tenant_id="acme",
                workflow_id=DEFAULT_WORKFLOW,
                document_id=doc_id,
                collection_id=uuid4(),
            ),
            meta=DocumentMeta(tenant_id="acme", workflow_id=DEFAULT_WORKFLOW),
            blob=doc_blob,
            checksum=doc_blob.sha256,
            created_at=datetime.now(timezone.utc),
            assets=[asset],
        )


def test_document_asset_json_schema_roundtrip():
    doc = build_document()
    dumped = doc.model_dump(mode="python")
    restored = NormalizedDocument.model_validate(dumped)
    assert restored == doc

    assert "Document Reference" in document_ref_schema()["title"]
    assert "Blob" in blob_locator_schema()["title"]
    assert "Normalized Document" in normalized_document_schema()["title"]
    assert "Document Metadata" in document_meta_schema()["title"]
    assert "Asset Reference" in asset_ref_schema()["title"]
    assert "Document Asset" in asset_schema()["title"]


def test_strict_checksums_validate_matching_payloads():
    doc = build_document()
    with strict_checksums(True):
        validated = NormalizedDocument.model_validate(doc.model_dump(mode="python"))
    assert validated == doc


def test_strict_checksums_detect_mismatches():
    doc = build_document()
    data = doc.model_dump(mode="python")
    data["checksum"] = "0" * 64
    with strict_checksums(True):
        with pytest.raises(ValueError):
            NormalizedDocument.model_validate(data)

    data = doc.model_dump(mode="python")
    data["assets"][0]["checksum"] = "0" * 64
    with strict_checksums(True):
        with pytest.raises(ValueError):
            NormalizedDocument.model_validate(data)


def test_strict_checksums_inline_blob_digest_mismatch():
    inline = make_inline_blob()
    inline["sha256"] = "0" * 64
    with strict_checksums(True):
        with pytest.raises(ValueError):
            BLOB_LOCATOR_ADAPTER.validate_python(inline)


def assert_document_roundtrip(doc: NormalizedDocument) -> None:
    """Validate a document via model roundtrip and JSON schema."""

    python_dump = doc.model_dump(mode="python")
    restored = NormalizedDocument.model_validate(python_dump)
    assert restored == doc
    jsonschema.validate(
        instance=doc.model_dump(mode="json"),
        schema=normalized_document_schema(),
    )


def test_source_upload_examples_roundtrip():
    tenant = "tenant-upload"
    workflow_id = "upload-flow"
    doc_without_assets_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_file_blob("memory://upload/doc.pdf", b"upload-document")
    )
    doc_without_assets = NormalizedDocument(
        ref=DocumentRef(tenant_id=tenant, workflow_id=workflow_id, document_id=uuid4()),
        meta=DocumentMeta(
            tenant_id=tenant,
            workflow_id=workflow_id,
            title="Upload Document",
            tags=["upload"],
        ),
        blob=doc_without_assets_blob,
        checksum=doc_without_assets_blob.sha256,
        created_at=datetime(2024, 1, 1, 8, tzinfo=timezone.utc),
        source="upload",
        assets=[],
    )
    assert_document_roundtrip(doc_without_assets)

    doc_with_asset_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_file_blob("memory://upload/with-asset.pdf", b"upload-document-with-asset")
    )
    asset_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_inline_blob("image/png", payload=b"upload-asset")
    )
    doc_with_asset_id = uuid4()
    asset = Asset(
        ref=AssetRef(
            tenant_id=tenant,
            workflow_id=workflow_id,
            asset_id=uuid4(),
            document_id=doc_with_asset_id,
        ),
        media_type="image/png",
        blob=asset_blob,
        caption_method="manual",
        text_description="User supplied screenshot",
        created_at=datetime(2024, 1, 1, 8, tzinfo=timezone.utc),
        checksum=asset_blob.sha256,
    )
    doc_with_asset = NormalizedDocument(
        ref=DocumentRef(
            tenant_id=tenant,
            workflow_id=workflow_id,
            document_id=doc_with_asset_id,
        ),
        meta=DocumentMeta(
            tenant_id=tenant,
            workflow_id=workflow_id,
            title="Upload With Asset",
        ),
        blob=doc_with_asset_blob,
        checksum=doc_with_asset_blob.sha256,
        created_at=datetime(2024, 1, 1, 8, tzinfo=timezone.utc),
        source="upload",
        assets=[asset],
    )
    assert_document_roundtrip(doc_with_asset)


def test_source_crawler_example_roundtrip():
    tenant = "tenant-crawler"
    workflow_id = "crawler-flow"
    doc_id = uuid4()
    doc_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_file_blob("memory://crawler/doc.html", b"<html>Crawled Content</html>")
    )
    asset_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_inline_blob("image/png", payload=b"crawler-image")
    )
    asset = Asset(
        ref=AssetRef(
            tenant_id=tenant,
            workflow_id=workflow_id,
            asset_id=uuid4(),
            document_id=doc_id,
            collection_id=uuid4(),
        ),
        media_type="image/png",
        blob=asset_blob,
        origin_uri="https://example.com/docs/123#figure-1",
        page_index=0,
        bbox=[0.1, 0.1, 0.7, 0.8],
        context_before="Lead paragraph",
        context_after="Follow up details",
        ocr_text="Captured text from crawler",
        caption_method="ocr_only",
        created_at=datetime(2024, 5, 1, 12, tzinfo=timezone.utc),
        checksum=asset_blob.sha256,
    )
    crawler_doc = NormalizedDocument(
        ref=DocumentRef(
            tenant_id=tenant,
            workflow_id=workflow_id,
            document_id=doc_id,
            collection_id=asset.ref.collection_id,
        ),
        meta=DocumentMeta(
            tenant_id=tenant,
            workflow_id=workflow_id,
            tags=["crawler", "example"],
            origin_uri="https://example.com/docs/123",
            crawl_timestamp=datetime(2024, 5, 1, 11, tzinfo=timezone.utc),
        ),
        blob=doc_blob,
        checksum=doc_blob.sha256,
        created_at=datetime(2024, 5, 1, 12, 30, tzinfo=timezone.utc),
        source="crawler",
        assets=[asset],
    )
    assert_document_roundtrip(crawler_doc)


def test_source_integration_example_roundtrip():
    tenant = "tenant-integration"
    workflow_id = "integration-flow"
    doc_id = uuid4()
    doc_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_file_blob("memory://integration/page.html", b"Integration content")
    )
    asset_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_inline_blob("image/png", payload=b"integration-screenshot")
    )
    asset = Asset(
        ref=AssetRef(
            tenant_id=tenant,
            workflow_id=workflow_id,
            asset_id=uuid4(),
            document_id=doc_id,
        ),
        media_type="image/png",
        blob=asset_blob,
        origin_uri="https://confluence.example.com/page#attachment",
        caption_method="none",
        text_description="Integration screenshot",
        created_at=datetime(2024, 4, 10, 9, tzinfo=timezone.utc),
        checksum=asset_blob.sha256,
    )
    integration_doc = NormalizedDocument(
        ref=DocumentRef(tenant_id=tenant, workflow_id=workflow_id, document_id=doc_id),
        meta=DocumentMeta(
            tenant_id=tenant,
            workflow_id=workflow_id,
            tags=["integration"],
            origin_uri="https://confluence.example.com/page",
            external_ref={"provider": "confluence", "id": "XYZ"},
        ),
        blob=doc_blob,
        checksum=doc_blob.sha256,
        created_at=datetime(2024, 4, 10, 9, 30, tzinfo=timezone.utc),
        source="integration",
        assets=[asset],
    )
    assert_document_roundtrip(integration_doc)


def test_source_integration_llm_context_example_roundtrip():
    tenant = "tenant-llm"
    workflow_id = "llm-flow"
    doc_id = uuid4()
    payload = b"# Release Notes\nGenerated summary"
    inline_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_inline_blob("text/markdown", payload=payload)
    )
    llm_doc = NormalizedDocument(
        ref=DocumentRef(tenant_id=tenant, workflow_id=workflow_id, document_id=doc_id),
        meta=DocumentMeta(
            tenant_id=tenant,
            workflow_id=workflow_id,
            tags=["llm", "summary"],
            external_ref={
                "provider": "noesis-llm",
                "model": "gpt-4o-mini",
                "generated_at": "2024-06-01T12:00:00+00:00",
                "topic": "release-notes",
            },
        ),
        blob=inline_blob,
        checksum=inline_blob.sha256,
        created_at=datetime(2024, 6, 1, 12, 5, tzinfo=timezone.utc),
        source="integration",
        assets=[],
    )
    assert_document_roundtrip(llm_doc)
    assert llm_doc.meta.origin_uri is None
    assert llm_doc.meta.crawl_timestamp is None
    assert llm_doc.blob.media_type == "text/markdown"
    assert llm_doc.blob.base64 == base64.b64encode(payload).decode()
    assert llm_doc.blob.sha256 == hashlib.sha256(payload).hexdigest()
    assert llm_doc.checksum == llm_doc.blob.sha256
    assert llm_doc.blob.decoded_payload() == payload


@pytest.mark.parametrize("missing_key", ["provider", "model", "generated_at", "topic"])
def test_source_integration_llm_context_requires_external_ref_keys(missing_key: str):
    tenant = "tenant-llm"
    workflow_id = "llm-flow"
    doc_id = uuid4()
    payload = b"# Title\nBody"
    inline_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_inline_blob("text/markdown", payload=payload)
    )
    external_ref = {
        "provider": "noesis-llm",
        "model": "gpt-4o-mini",
        "generated_at": "2024-06-01T12:00:00+00:00",
        "topic": "release-notes",
    }
    external_ref.pop(missing_key)
    with pytest.raises(ValueError) as exc_info:
        NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant, workflow_id=workflow_id, document_id=doc_id
            ),
            meta=DocumentMeta(
                tenant_id=tenant,
                workflow_id=workflow_id,
                external_ref=external_ref,
            ),
            blob=inline_blob,
            checksum=inline_blob.sha256,
            created_at=datetime(2024, 6, 1, 12, 5, tzinfo=timezone.utc),
            source="integration",
        )
    assert f"llm_external_ref_missing_{missing_key}" in str(exc_info.value)


def test_source_integration_llm_context_rejects_origin_and_crawl_timestamp():
    tenant = "tenant-llm"
    workflow_id = "llm-flow"
    payload = b"Body"
    inline_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_inline_blob("text/markdown", payload=payload)
    )
    base_ref = {
        "provider": "noesis-llm",
        "model": "gpt-4o-mini",
        "generated_at": "2024-06-01T12:00:00+00:00",
        "topic": "release-notes",
    }
    with pytest.raises(ValueError) as exc_info:
        NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant, workflow_id=workflow_id, document_id=uuid4()
            ),
            meta=DocumentMeta(
                tenant_id=tenant,
                workflow_id=workflow_id,
                origin_uri="https://example.com/llm",
                external_ref=base_ref,
            ),
            blob=inline_blob,
            checksum=inline_blob.sha256,
            created_at=datetime(2024, 6, 1, 13, tzinfo=timezone.utc),
            source="integration",
        )
    assert "llm_origin_present" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant, workflow_id=workflow_id, document_id=uuid4()
            ),
            meta=DocumentMeta(
                tenant_id=tenant,
                workflow_id=workflow_id,
                crawl_timestamp=datetime(2024, 6, 1, 13, tzinfo=timezone.utc),
                external_ref=base_ref,
            ),
            blob=inline_blob,
            checksum=inline_blob.sha256,
            created_at=datetime(2024, 6, 1, 13, tzinfo=timezone.utc),
            source="integration",
        )
    assert "llm_crawl_timestamp_present" in str(exc_info.value)


def test_source_integration_llm_context_generated_at_must_be_iso8601():
    tenant = "tenant-llm"
    workflow_id = "llm-flow"
    inline_blob = BLOB_LOCATOR_ADAPTER.validate_python(
        make_inline_blob("text/markdown", payload=b"Body")
    )
    with pytest.raises(ValueError) as exc_info:
        NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant, workflow_id=workflow_id, document_id=uuid4()
            ),
            meta=DocumentMeta(
                tenant_id=tenant,
                workflow_id=workflow_id,
                external_ref={
                    "provider": "noesis-llm",
                    "model": "gpt-4o-mini",
                    "generated_at": "not-a-timestamp",
                    "topic": "release-notes",
                },
            ),
            blob=inline_blob,
            checksum=inline_blob.sha256,
            created_at=datetime(2024, 6, 1, 13, tzinfo=timezone.utc),
            source="integration",
        )
    assert "llm_generated_at_invalid" in str(exc_info.value)
