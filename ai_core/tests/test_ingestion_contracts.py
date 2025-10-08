import pytest

from ai_core.rag.embedding_config import reset_embedding_configuration_cache
from ai_core.rag.ingestion_contracts import (
    IngestionContractErrorCode,
    ensure_embedding_dimensions,
    resolve_ingestion_profile,
)
from ai_core.tools import InputError
from ai_core.rag.vector_space_resolver import (
    VectorSpaceResolverError,
    VectorSpaceResolverErrorCode,
)
from ai_core.rag.schemas import Chunk


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
            "dimension": 1536,
            "vector_space": "global",
        }
    }


def test_resolve_ingestion_profile_success(settings) -> None:
    _configure_embeddings(settings)
    result = resolve_ingestion_profile(" standard ")
    assert result.profile_id == "standard"
    assert result.resolution.vector_space.id == "global"


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
        doc_class="manual",
        embedding_profile="standard",
        vector_space_id="global",
    )


def test_ensure_embedding_dimensions_raises_on_mismatch() -> None:
    chunks = [Chunk(content="c", meta={"external_id": "doc-1"}, embedding=[0.1])]

    with pytest.raises(InputError) as excinfo:
        ensure_embedding_dimensions(
            chunks,
            2,
            tenant_id="tenant-a",
            process="review",
            doc_class="manual",
            embedding_profile="standard",
            vector_space_id="global",
        )

    error = excinfo.value
    assert error.code == IngestionContractErrorCode.VECTOR_DIMENSION_MISMATCH
    assert error.context["tenant"] == "tenant-a"
    assert error.context["process"] == "review"
    assert error.context["doc_class"] == "manual"
    assert error.context["embedding_profile"] == "standard"
    assert error.context["vector_space_id"] == "global"
    assert error.context["expected_dimension"] == 2
    assert error.context["observed_dimension"] == 1
    assert error.context["chunk_index"] == 0
    assert error.context["external_id"] == "doc-1"
