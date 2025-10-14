import pytest

from ai_core.rag.embedding_config import (
    EmbeddingConfiguration,
    reset_embedding_configuration_cache,
)
from ai_core.rag.vector_space_resolver import (
    VectorSpaceResolverError,
    VectorSpaceResolverErrorCode,
    resolve_vector_space_full,
    resolve_vector_space,
)
from common.logging import log_context


@pytest.fixture(autouse=True)
def _reset_embedding_cache() -> None:
    reset_embedding_configuration_cache()
    yield
    reset_embedding_configuration_cache()


def _configure_embeddings(settings) -> None:
    settings.RAG_VECTOR_STORES = {
        "global": {
            "backend": "pgvector",
            "schema": "rag",
            "dimension": 1536,
        },
        "legacy": {
            "backend": "pgvector",
            "schema": "rag_legacy",
            "dimension": 1024,
        },
    }
    settings.RAG_EMBEDDING_PROFILES = {
        "standard": {
            "model": "oai-embed-large",
            "dimension": 1536,
            "vector_space": "global",
            "chunk_hard_limit": 512,
        },
        "legacy": {
            "model": "oai-embed-small",
            "dimension": 1024,
            "vector_space": "legacy",
            "chunk_hard_limit": 400,
        },
    }


def test_resolve_vector_space_returns_config(settings) -> None:
    _configure_embeddings(settings)

    resolved = resolve_vector_space("standard")

    assert resolved.id == "global"
    assert resolved.dimension == 1536
    assert resolved.backend == "pgvector"
    assert resolved.schema == "rag"


def test_resolve_vector_space_full_returns_profile_and_space(settings) -> None:
    _configure_embeddings(settings)

    resolution = resolve_vector_space_full("legacy")

    assert resolution.profile.id == "legacy"
    assert resolution.profile.vector_space == "legacy"
    assert resolution.vector_space.id == "legacy"
    assert resolution.vector_space.dimension == 1024


@pytest.mark.parametrize("value", [None, "", "   "])
def test_requires_profile_identifier(settings, value) -> None:
    _configure_embeddings(settings)

    with pytest.raises(VectorSpaceResolverError) as excinfo:
        resolve_vector_space(value)

    assert excinfo.value.code == VectorSpaceResolverErrorCode.PROFILE_REQUIRED


def test_unknown_profile_raises(settings) -> None:
    _configure_embeddings(settings)

    with pytest.raises(VectorSpaceResolverError) as excinfo:
        resolve_vector_space("missing")

    assert excinfo.value.code == VectorSpaceResolverErrorCode.PROFILE_UNKNOWN
    assert "missing" in excinfo.value.message


def test_missing_vector_space_is_guarded(settings, monkeypatch) -> None:
    _configure_embeddings(settings)
    from ai_core.rag import vector_space_resolver as resolver_module

    original = resolver_module.get_embedding_configuration()
    broken = EmbeddingConfiguration(
        vector_spaces=dict(original.vector_spaces),
        embedding_profiles=dict(original.embedding_profiles),
    )
    broken.vector_spaces.pop("global")

    monkeypatch.setattr(
        resolver_module,
        "get_embedding_configuration",
        lambda: broken,
    )

    with pytest.raises(VectorSpaceResolverError) as excinfo:
        resolve_vector_space("standard")

    assert excinfo.value.code == VectorSpaceResolverErrorCode.VECTOR_SPACE_UNKNOWN
    assert "global" in excinfo.value.message


def test_vector_space_resolution_emits_trace_metadata(settings, monkeypatch) -> None:
    _configure_embeddings(settings)

    spans: list[dict[str, object]] = []
    from ai_core.rag import vector_space_resolver as resolver_module

    monkeypatch.setattr(
        resolver_module.tracing,
        "emit_span",
        lambda **kwargs: spans.append(kwargs),
    )

    with log_context(trace_id="trace-space", tenant="tenant-a"):
        resolution = resolve_vector_space_full("standard")

    assert resolution.vector_space.id == "global"
    assert spans, "expected vector space resolver to emit a Langfuse span"
    span = spans[0]
    assert span["trace_id"] == "trace-space"
    assert span["node_name"] == "rag.vector_space.resolve"
    metadata = span["metadata"]
    assert metadata["vector_space_id"] == "global"
    assert metadata["embedding_profile"] == "standard"
