"""Tests for embedding profile and vector space configuration validation."""

from __future__ import annotations

import pytest

from ai_core.rag.embedding_config import (
    EmbeddingConfigurationError,
    get_embedding_configuration,
    get_embedding_profile,
    get_vector_space,
    reset_embedding_configuration_cache,
    validate_embedding_configuration,
)


@pytest.fixture(autouse=True)
def _reset_embedding_config_cache() -> None:
    reset_embedding_configuration_cache()
    yield
    reset_embedding_configuration_cache()


def _base_vector_space() -> dict[str, object]:
    return {
        "backend": "pgvector",
        "schema": "rag",
        "dimension": 1536,
    }


def _base_profile() -> dict[str, object]:
    return {
        "model": "oai-embed-large",
        "dimension": 1536,
        "vector_space": "global",
        "chunk_hard_limit": 512,
    }


def test_get_embedding_configuration_returns_dataclasses(settings) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    settings.RAG_EMBEDDING_PROFILES = {"standard": _base_profile()}

    config = get_embedding_configuration()

    assert config.vector_spaces["global"].dimension == 1536
    assert config.vector_spaces["global"].schema == "rag"
    assert config.embedding_profiles["standard"].model == "oai-embed-large"
    assert (
        config.embedding_profiles["standard"].vector_space
        == config.vector_spaces["global"].id
    )
    assert config.embedding_profiles["standard"].chunk_hard_limit == 512


def test_getters_return_single_entries(settings) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    settings.RAG_EMBEDDING_PROFILES = {"standard": _base_profile()}

    vector_space = get_vector_space("global")
    profile = get_embedding_profile("standard")

    assert vector_space.schema == "rag"
    assert profile.model == "oai-embed-large"
    assert profile.vector_space == "global"
    assert profile.chunk_hard_limit == 512


def test_get_embedding_profile_falls_back_to_default(settings, caplog) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    settings.RAG_EMBEDDING_PROFILES = {
        "standard": _base_profile(),
        "extended": {
            "model": "oai-embed-xl",
            "dimension": 1536,
            "vector_space": "global",
            "chunk_hard_limit": 1024,
        },
    }
    settings.RAG_DEFAULT_EMBEDDING_PROFILE = "standard"

    with caplog.at_level("WARNING"):
        profile = get_embedding_profile("unknown")

    assert profile.id == "standard"
    assert profile.chunk_hard_limit == 512
    assert "Falling back to default profile" in caplog.text


def test_get_embedding_profile_uses_safe_limit_when_default_missing(
    settings, caplog
) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    settings.RAG_EMBEDDING_PROFILES = {"standard": _base_profile()}
    settings.RAG_DEFAULT_EMBEDDING_PROFILE = "missing"

    with caplog.at_level("WARNING"):
        profile = get_embedding_profile("unknown")

    assert profile.id == "standard"
    assert profile.chunk_hard_limit == 512
    assert "Default profile 'missing' unavailable" in caplog.text


def test_validate_configuration_raises_on_dimension_mismatch(settings) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    mismatched = _base_profile()
    mismatched["dimension"] = 1024
    settings.RAG_EMBEDDING_PROFILES = {"standard": mismatched}

    with pytest.raises(EmbeddingConfigurationError) as excinfo:
        validate_embedding_configuration()

    message = str(excinfo.value)
    assert "EMB_DIM_MISMATCH" in message
    assert "Dimension mismatch" in message


def test_validate_configuration_requires_known_vector_space(settings) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    profile = _base_profile()
    profile["vector_space"] = "unknown"
    settings.RAG_EMBEDDING_PROFILES = {"standard": profile}

    with pytest.raises(EmbeddingConfigurationError) as excinfo:
        get_embedding_configuration()

    message = str(excinfo.value)
    assert "EMB_UNKNOWN_SPACE" in message
    assert "unknown vector space" in message


def test_validate_configuration_requires_schema(settings) -> None:
    settings.RAG_VECTOR_STORES = {
        "global": {"backend": "pgvector", "dimension": 1536, "schema": ""}
    }
    settings.RAG_EMBEDDING_PROFILES = {"standard": _base_profile()}

    with pytest.raises(EmbeddingConfigurationError) as excinfo:
        get_embedding_configuration()

    assert "must define a schema" in str(excinfo.value)


def test_vector_space_dimension_must_be_positive(settings) -> None:
    settings.RAG_VECTOR_STORES = {
        "global": {"backend": "pgvector", "schema": "rag", "dimension": 0}
    }
    settings.RAG_EMBEDDING_PROFILES = {"standard": _base_profile()}

    with pytest.raises(EmbeddingConfigurationError) as excinfo:
        get_embedding_configuration()

    message = str(excinfo.value)
    assert "EMB_SPACE_DIM_INVALID" in message
    assert "must be positive" in message


def test_profile_dimension_must_be_positive_integer(settings) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    profile = _base_profile()
    profile["dimension"] = "invalid"
    settings.RAG_EMBEDDING_PROFILES = {"standard": profile}

    with pytest.raises(EmbeddingConfigurationError) as excinfo:
        get_embedding_configuration()

    message = str(excinfo.value)
    assert "EMB_PROFILE_DIM_INVALID" in message
    assert "must be an integer" in message


def test_profile_chunk_limit_defaults_when_missing(settings) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    profile = _base_profile()
    profile.pop("chunk_hard_limit")
    settings.RAG_EMBEDDING_PROFILES = {"standard": profile}

    config = get_embedding_configuration()

    assert config.embedding_profiles["standard"].chunk_hard_limit == 512


def test_profile_chunk_limit_requires_positive_integer(settings) -> None:
    settings.RAG_VECTOR_STORES = {"global": _base_vector_space()}
    profile = _base_profile()
    profile["chunk_hard_limit"] = 0
    settings.RAG_EMBEDDING_PROFILES = {"standard": profile}

    with pytest.raises(EmbeddingConfigurationError) as excinfo:
        get_embedding_configuration()

    message = str(excinfo.value)
    assert "EMB_PROFILE_CHUNK_LIMIT_INVALID" in message
    assert "chunk_hard_limit" in message

    profile["chunk_hard_limit"] = "invalid"
    settings.RAG_EMBEDDING_PROFILES = {"standard": profile}
    reset_embedding_configuration_cache()

    with pytest.raises(EmbeddingConfigurationError) as excinfo:
        get_embedding_configuration()

    assert "EMB_PROFILE_CHUNK_LIMIT_INVALID" in str(excinfo.value)
