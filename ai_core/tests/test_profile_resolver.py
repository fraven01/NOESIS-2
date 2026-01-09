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
def _reset_caches(settings) -> None:
    reset_embedding_configuration_cache()
    reset_routing_rules_cache()
    settings.RAG_ROUTING_FLAGS = {}
    yield
    reset_routing_rules_cache()
    reset_embedding_configuration_cache()
    settings.RAG_ROUTING_FLAGS = {}


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
            "chunk_hard_limit": 512,
        },
        "legacy": {
            "model": "oai-embed-small",
            "dimension": 1024,
            "vector_space": "legacy",
            "chunk_hard_limit": 400,
        },
    }


@pytest.fixture
def enable_collection_routing(settings):
    settings.RAG_ROUTING_FLAGS = {"rag.use_collection_routing": True}
    yield
    settings.RAG_ROUTING_FLAGS = {}


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


def test_resolver_maps_doc_class_to_collection_when_flag_enabled(
    tmp_path, settings, enable_collection_routing
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
            doc_class: Manual
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    assert (
        resolve_embedding_profile(
            tenant_id="tenant-a", process=None, doc_class="manual"
        )
        == "legacy"
    )
    assert (
        resolve_embedding_profile(
            tenant_id="tenant-a", process=None, collection_id="manual"
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

    spans: list[tuple[str, dict[str, object] | None, str | None]] = []

    from ai_core.rag import profile_resolver as resolver_module

    monkeypatch.setattr(
        resolver_module,
        "record_span",
        lambda name, *, attributes=None: spans.append(
            (name, attributes, (attributes or {}).get("trace_id"))
        ),
    )

    with log_context(trace_id="trace-profile", tenant="tenant-a"):
        assert (
            resolve_embedding_profile(
                tenant_id="tenant-a",
                process=None,
                doc_class=None,
                collection_id="Docs",
                workflow_id="FlowA",
                language="DE",
                size="Large",
            )
            == "legacy"
        )

    assert spans, "expected resolver to emit a span"
    name, metadata, trace_id = spans[0]
    assert name == "rag.profile.resolve"
    assert trace_id == "trace-profile"
    assert metadata is not None
    assert metadata["embedding_profile"] == "legacy"
    assert metadata["tenant_id"] == "tenant-a"
    assert metadata["collection_id"] == "docs"
    assert metadata["workflow_id"] == "flowa"
    assert metadata["language"] == "de"
    assert metadata["size"] == "large"
    assert metadata["resolver_path"] == "rules[0]"
    assert metadata["chosen_profile"] == "legacy"
    assert metadata["fallback_used"] is False


def test_profile_resolution_marks_default_fallback(
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
            tenant: other-tenant
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    spans: list[tuple[str, dict[str, object] | None, str | None]] = []

    from ai_core.rag import profile_resolver as resolver_module

    monkeypatch.setattr(
        resolver_module,
        "record_span",
        lambda name, *, attributes=None: spans.append(
            (name, attributes, (attributes or {}).get("trace_id"))
        ),
    )

    with log_context(trace_id="trace-default", tenant="tenant-a"):
        assert (
            resolve_embedding_profile(
                tenant_id="tenant-a",
                process=None,
                collection_id=None,
                workflow_id=None,
            )
            == "standard"
        )

    _, metadata, _ = spans[0]
    assert metadata is not None
    assert metadata["resolver_path"] == "default_profile"
    assert metadata["fallback_used"] is True
