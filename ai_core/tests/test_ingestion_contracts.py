import base64
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from ai_core.rag.embedding_config import reset_embedding_configuration_cache
from ai_core.rag.ingestion_contracts import (
    IngestionAction,
    IngestionContractErrorCode,
    build_crawler_ingestion_payload,
    ensure_embedding_dimensions,
    log_embedding_quality_stats,
    resolve_ingestion_profile,
)
from ai_core.tools import InputError
from ai_core.rag.vector_space_resolver import (
    VectorSpaceResolverError,
    VectorSpaceResolverErrorCode,
)
from ai_core.rag.schemas import Chunk
from ai_core.rag.deduplication import DedupSignatures
from common.logging import log_context
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)


@pytest.fixture(autouse=True)
def _reset_embedding_cache() -> None:
    reset_embedding_configuration_cache()
    yield
    reset_embedding_configuration_cache()


def _configure_embeddings(settings) -> None:
    settings.RAG_VECTOR_STORES = {
        "global": {"backend": "pgvector", "schema": "rag", "dimension": 1536}
    }
    settings.RAG_EMBEDDING_PROFILES = {
        "standard": {
            "model": "oai-embed-large",
            "model_version": "v1",
            "dimension": 1536,
            "vector_space": "global",
            "chunk_hard_limit": 512,
        }
    }


def _make_normalized_document(
    *, source: str = "crawler", workflow_id: str = "wf-doc"
) -> NormalizedDocument:
    payload = b"dummy content"
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
        tenant_id="tenant-a",
        workflow_id=workflow_id,
        document_id=uuid4(),
    )
    document_meta = DocumentMeta(
        tenant_id="tenant-a",
        workflow_id=workflow_id,
        origin_uri="https://example.com/doc",
        external_ref={"provider": source, "external_id": "ext-1"},
    )
    return NormalizedDocument(
        ref=document_ref,
        meta=document_meta,
        blob=blob,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source=source,
    )


def test_resolve_ingestion_profile_success(settings) -> None:
    _configure_embeddings(settings)
    result = resolve_ingestion_profile(" standard ")
    assert result.profile_id == "standard"
    assert result.resolution.vector_space.id == "rag/standard@v1"


def test_resolve_ingestion_profile_requires_value(settings) -> None:
    _configure_embeddings(settings)
    with pytest.raises(InputError) as excinfo:
        resolve_ingestion_profile(None)
    assert excinfo.value.code == IngestionContractErrorCode.PROFILE_REQUIRED


def test_resolve_ingestion_profile_rejects_non_string(settings) -> None:
    _configure_embeddings(settings)
    with pytest.raises(InputError) as excinfo:
        resolve_ingestion_profile(123)
    assert excinfo.value.code == IngestionContractErrorCode.PROFILE_INVALID


def test_resolve_ingestion_profile_unknown_id(settings) -> None:
    _configure_embeddings(settings)
    with pytest.raises(InputError) as excinfo:
        resolve_ingestion_profile("unknown")
    assert excinfo.value.code == IngestionContractErrorCode.PROFILE_UNKNOWN


def test_resolve_ingestion_profile_missing_space(settings, monkeypatch) -> None:
    _configure_embeddings(settings)

    from ai_core.rag import ingestion_contracts as contracts_module

    def _raise_missing_space(profile_id: str):
        raise VectorSpaceResolverError(
            VectorSpaceResolverErrorCode.VECTOR_SPACE_UNKNOWN,
            "vector space missing",
        )

    monkeypatch.setattr(
        contracts_module, "resolve_vector_space_full", _raise_missing_space
    )

    with pytest.raises(InputError) as excinfo:
        resolve_ingestion_profile("standard")
    assert excinfo.value.code == IngestionContractErrorCode.VECTOR_SPACE_UNKNOWN


def test_ensure_embedding_dimensions_allows_matching_vectors() -> None:
    chunks = [Chunk(content="c", meta={"external_id": "doc-1"}, embedding=[0.1, 0.2])]

    ensure_embedding_dimensions(
        chunks,
        2,
        tenant_id="tenant-a",
        process="review",
        workflow_id="flow-a",
        embedding_profile="standard",
        vector_space_id="rag/standard@v1",
    )


def test_ensure_embedding_dimensions_raises_on_mismatch() -> None:
    chunks = [Chunk(content="c", meta={"external_id": "doc-1"}, embedding=[0.1])]

    with pytest.raises(InputError) as excinfo:
        ensure_embedding_dimensions(
            chunks,
            2,
            tenant_id="tenant-a",
            process="review",
            workflow_id="flow-a",
            embedding_profile="standard",
            vector_space_id="rag/standard@v1",
        )

    error = excinfo.value
    assert error.code == IngestionContractErrorCode.VECTOR_DIMENSION_MISMATCH
    assert error.context["tenant"] == "tenant-a"
    assert error.context["process"] == "review"
    assert error.context["workflow_id"] == "flow-a"
    assert error.context["embedding_profile"] == "standard"
    assert error.context["vector_space_id"] == "rag/standard@v1"
    assert error.context["expected_dimension"] == 2
    assert error.context["observed_dimension"] == 1
    assert error.context["chunk_index"] == 0
    assert error.context["external_id"] == "doc-1"


def test_ensure_embedding_dimensions_raises_on_zero_vector() -> None:
    chunks = [Chunk(content="c", meta={"external_id": "doc-1"}, embedding=[0.0, 0.0])]

    with pytest.raises(InputError) as excinfo:
        ensure_embedding_dimensions(
            chunks,
            2,
            tenant_id="tenant-a",
            process="review",
            workflow_id="flow-a",
            embedding_profile="standard",
            vector_space_id="rag/standard@v1",
        )

    error = excinfo.value
    assert error.code == IngestionContractErrorCode.EMBEDDING_ZERO
    assert error.context["tenant"] == "tenant-a"
    assert error.context["chunk_index"] == 0


def test_ensure_embedding_dimensions_raises_on_non_finite() -> None:
    chunks = [
        Chunk(
            content="c",
            meta={"external_id": "doc-1"},
            embedding=[float("nan"), 0.2],
        )
    ]

    with pytest.raises(InputError) as excinfo:
        ensure_embedding_dimensions(
            chunks,
            2,
            tenant_id="tenant-a",
            process="review",
            workflow_id="flow-a",
            embedding_profile="standard",
            vector_space_id="rag/standard@v1",
        )

    error = excinfo.value
    assert error.code == IngestionContractErrorCode.EMBEDDING_INVALID
    assert error.context["invalid_reason"] == "non_finite"
    assert error.context["chunk_index"] == 0


def test_log_embedding_quality_stats_reports_outlier() -> None:
    chunks = [
        Chunk(content="c1", meta={"external_id": "doc-1"}, embedding=[1.0, 0.0]),
        Chunk(content="c2", meta={"external_id": "doc-2"}, embedding=[1.0, 0.0]),
        Chunk(content="c3", meta={"external_id": "doc-3"}, embedding=[-1.0, 0.0]),
    ]

    with log_context(trace_id="trace-1"):
        payload = log_embedding_quality_stats(
            chunks,
            sample_size=10,
            outlier_threshold=0.1,
        )

    assert payload is not None
    assert payload["sample_size"] == 3
    assert payload["outlier_count"] == 1


def test_build_crawler_ingestion_payload_uses_document_source(settings) -> None:
    _configure_embeddings(settings)
    document = _make_normalized_document(source="upload", workflow_id="wf-upload")
    signatures = DedupSignatures(content_hash=document.checksum)

    payload = build_crawler_ingestion_payload(
        document=document,
        signatures=signatures,
        case_id="case-1",
        action=IngestionAction.UPSERT,
        lifecycle_state="active",
        embedding_profile="standard",
    )

    assert payload.chunk_meta is not None
    assert payload.chunk_meta.source == "upload"
    assert payload.chunk_meta.process == "upload"
    assert payload.chunk_meta.workflow_id == "wf-upload"


def test_build_crawler_ingestion_payload_allows_source_overrides(settings) -> None:
    _configure_embeddings(settings)
    document = _make_normalized_document(source="crawler", workflow_id="wf-crawler")
    signatures = DedupSignatures(content_hash=document.checksum)

    payload = build_crawler_ingestion_payload(
        document=document,
        signatures=signatures,
        case_id="case-2",
        action=IngestionAction.UPSERT,
        lifecycle_state="active",
        embedding_profile="standard",
        source="integration",
        process="sync",
    )

    assert payload.chunk_meta is not None
    assert payload.chunk_meta.source == "integration"
    assert payload.chunk_meta.process == "sync"
