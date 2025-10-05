"""Tests for the embedding profile resolver."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ai_core.rag.embedding_config import (
    EmbeddingConfiguration,
    reset_embedding_configuration_cache,
)
from ai_core.rag.profile_resolver import (
    ProfileResolverError,
    ProfileResolverErrorCode,
    resolve_embedding_profile,
)
from ai_core.rag.routing_rules import get_routing_table, reset_routing_rules_cache
from common.logging import log_context


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    reset_embedding_configuration_cache()
    reset_routing_rules_cache()
    yield
    reset_routing_rules_cache()
    reset_embedding_configuration_cache()


def _write_routing_rules(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


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
        },
        "legacy": {
            "model": "oai-embed-small",
            "dimension": 1024,
            "vector_space": "legacy",
        },
    }


def test_requires_tenant_id(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules: []
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with pytest.raises(ProfileResolverError) as excinfo:
        resolve_embedding_profile(tenant_id="  ")

    assert excinfo.value.code == ProfileResolverErrorCode.TENANT_REQUIRED
    assert "tenant_id is required" in str(excinfo.value)


def test_resolves_profile_with_defaults(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: legacy
            tenant: tenant-a
            process: review
            doc_class: manual
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    assert (
        resolve_embedding_profile(
            tenant_id="tenant-a", process="review", doc_class="manual"
        )
        == "legacy"
    )
    assert (
        resolve_embedding_profile(tenant_id="tenant-a", process="", doc_class="manual")
        == "standard"
    )
    assert (
        resolve_embedding_profile(tenant_id="tenant-b", process=None, doc_class=None)
        == "standard"
    )


def test_resolver_is_case_insensitive(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: legacy
            tenant: tenant-a
            process: Review
            doc_class: Manual
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    assert (
        resolve_embedding_profile(
            tenant_id="tenant-a", process="REVIEW", doc_class="manual"
        )
        == "legacy"
    )


def test_resolved_profile_must_exist(tmp_path, settings, monkeypatch) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules: []
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    # Materialise routing table before simulating configuration drift.
    get_routing_table()

    from ai_core.rag import embedding_config as embedding_config_module
    from ai_core.rag import profile_resolver as profile_resolver_module

    original_config = embedding_config_module.get_embedding_configuration()
    broken_config = EmbeddingConfiguration(
        vector_spaces=dict(original_config.vector_spaces),
        embedding_profiles=dict(original_config.embedding_profiles),
    )
    broken_config.embedding_profiles.pop("standard")

    monkeypatch.setattr(
        profile_resolver_module,
        "get_embedding_configuration",
        lambda: broken_config,
    )

    with pytest.raises(ProfileResolverError) as excinfo:
        resolve_embedding_profile(tenant_id="tenant-a")

    assert excinfo.value.code == ProfileResolverErrorCode.UNKNOWN_PROFILE
    assert "not configured" in str(excinfo.value)


def test_profile_resolution_emits_trace_metadata(
    tmp_path, settings, monkeypatch
) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: legacy
            tenant: tenant-a
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    spans: list[dict[str, object]] = []

    from ai_core.rag import profile_resolver as resolver_module

    monkeypatch.setattr(
        resolver_module.tracing,
        "emit_span",
        lambda **kwargs: spans.append(kwargs),
    )

    with log_context(trace_id="trace-profile", tenant="tenant-a"):
        assert (
            resolve_embedding_profile(
                tenant_id="tenant-a", process=None, doc_class=None
            )
            == "legacy"
        )

    assert spans, "expected resolver to emit a Langfuse span"
    span = spans[0]
    assert span["trace_id"] == "trace-profile"
    assert span["node_name"] == "rag.profile.resolve"
    metadata = span["metadata"]
    assert metadata["embedding_profile"] == "legacy"
    assert metadata["tenant"] == "tenant-a"
