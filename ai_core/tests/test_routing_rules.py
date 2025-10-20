"""Tests for embedding routing rules."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

from ai_core.rag.embedding_config import reset_embedding_configuration_cache
from ai_core.rag.routing_rules import (
    RoutingConfigurationError,
    RoutingErrorCode,
    RoutingRule,
    RoutingTable,
    get_routing_table,
    reset_routing_rules_cache,
)


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
            "chunk_hard_limit": 400,
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


def test_resolve_prefers_collection_specific_rule(
    tmp_path, settings, enable_collection_routing
) -> None:
    _configure_embeddings(settings)
    settings.RAG_EMBEDDING_PROFILES["premium"] = {
        "model": "oai-embed-large",
        "dimension": 1536,
        "vector_space": "global",
        "chunk_hard_limit": 640,
    }
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: legacy
            tenant: tenant-a
          - profile: premium
            tenant: tenant-a
            collection_id: policies
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    table = get_routing_table()
    premium_rule = next(rule for rule in table.rules if rule.profile == "premium")
    assert premium_rule.collection_id == "policies"
    assert premium_rule.workflow_id is None
    assert premium_rule.process is None

    assert (
        table.resolve(
            tenant="tenant-a",
            process="review",
            collection_id="policies",
            workflow_id=None,
            doc_class=None,
        )
        == "premium"
    )
    assert (
        table.resolve(
            tenant="tenant-a",
            process="review",
            collection_id=None,
            workflow_id=None,
            doc_class=None,
        )
        == "legacy"
    )
    assert (
        table.resolve(
            tenant="tenant-b",
            process="review",
            collection_id="policies",
            workflow_id=None,
            doc_class=None,
        )
        == "standard"
    )


def test_rules_are_case_insensitive(
    tmp_path, settings, enable_collection_routing
) -> None:
    _configure_embeddings(settings)
    settings.RAG_ROUTING_FLAGS["rag.use_collection_routing"] = True
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: legacy
            tenant: tenant-a
            process: Review
            workflow_id: Draft
            collection_id: Policies
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    table = get_routing_table()

    resolution = table.resolve_with_metadata(
        tenant="tenant-a",
        process="review",
        collection_id="policies",
        workflow_id="draft",
        doc_class=None,
    )
    assert resolution.rule is not None
    assert resolution.profile == "legacy"
    assert resolution.resolver_path == "rules[0]"


@pytest.mark.parametrize(
    "collection_id,workflow_id,process,expected",
    [
        ("policies", "review-flow", "review", "enterprise"),
        (None, "review-flow", "review", "premium"),
        (None, None, "review", "advanced"),
        (None, None, "draft", "legacy"),
    ],
)
def test_specificity_precedence(
    tmp_path,
    settings,
    enable_collection_routing,
    collection_id,
    workflow_id,
    process,
    expected,
) -> None:
    _configure_embeddings(settings)
    settings.RAG_EMBEDDING_PROFILES.update(
        {
            "advanced": {
                "model": "oai-embed-large",
                "dimension": 1536,
                "vector_space": "global",
                "chunk_hard_limit": 768,
            },
            "premium": {
                "model": "oai-embed-large",
                "dimension": 1536,
                "vector_space": "global",
                "chunk_hard_limit": 896,
            },
            "enterprise": {
                "model": "oai-embed-large",
                "dimension": 1536,
                "vector_space": "global",
                "chunk_hard_limit": 1024,
            },
        }
    )
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: legacy
            tenant: tenant-a
          - profile: advanced
            tenant: tenant-a
            process: review
          - profile: premium
            tenant: tenant-a
            workflow_id: review-flow
          - profile: enterprise
            tenant: tenant-a
            collection_id: policies
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    table = get_routing_table()

    assert (
        table.resolve(
            tenant="tenant-a",
            process=process,
            collection_id=collection_id,
            workflow_id=workflow_id,
            doc_class=None,
        )
        == expected
    )


def test_unknown_profile_raises(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: unknown
        rules: []
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with pytest.raises(RoutingConfigurationError) as excinfo:
        get_routing_table()

    message = str(excinfo.value)
    assert "ROUTE_UNKNOWN_PROFILE_DEFAULT" in message
    assert "Default routing profile" in message


def test_overlapping_rules_with_same_specificity_fail(
    tmp_path, settings, enable_collection_routing
) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: standard
            tenant: tenant-a
            workflow_id: shared
          - profile: legacy
            tenant: tenant-a
            workflow_id: shared
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with pytest.raises(RoutingConfigurationError) as excinfo:
        get_routing_table()

    message = str(excinfo.value)
    assert "ROUTE_DUP_SELECTOR" in message
    assert "Duplicate routing selector" in message


def test_duplicate_rule_same_target_is_tolerated(
    tmp_path, settings, caplog, enable_collection_routing
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
            collection_id: policies
          - profile: legacy
            tenant: tenant-a
            collection_id: policies
          - profile: legacy
            tenant: tenant-a
            collection_id: policies
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with caplog.at_level(logging.WARNING):
        table = get_routing_table()

    dup_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "ROUTE_DUP_SAME_TARGET" in record.getMessage()
    ]

    assert len(dup_warnings) == 1

    assert (
        table.resolve(
            tenant="tenant-a",
            process="review",
            collection_id="policies",
            workflow_id=None,
            doc_class=None,
        )
        == "legacy"
    )


def test_duplicate_rule_conflicting_targets_raises(
    tmp_path, settings, enable_collection_routing
) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: standard
            tenant: tenant-a
            workflow_id: shared
          - profile: legacy
            tenant: tenant-a
            workflow_id: shared
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with pytest.raises(RoutingConfigurationError) as excinfo:
        get_routing_table()

    message = str(excinfo.value)
    assert "ROUTE_DUP_SELECTOR" in message
    assert "Duplicate routing selector" in message


def test_doc_class_alias_when_collection_routing_enabled(
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
            doc_class: manual
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    table = get_routing_table()

    assert (
        table.resolve(
            tenant="tenant-a",
            process=None,
            collection_id="manual",
            workflow_id=None,
            doc_class="manual",
        )
        == "legacy"
    )


def test_doc_class_routing_when_feature_disabled(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: legacy
            tenant: tenant-a
            doc_class: manual
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    table = get_routing_table()

    assert (
        table.resolve(
            tenant="tenant-a",
            process=None,
            collection_id=None,
            workflow_id=None,
            doc_class="manual",
        )
        == "legacy"
    )


def test_resolve_raises_on_ambiguous_runtime_match() -> None:
    table = RoutingTable(
        default_profile="standard",
        rules=(
            RoutingRule(
                profile="legacy",
                index=0,
                tenant="tenant-a",
            ),
            RoutingRule(
                profile="premium",
                index=1,
                tenant="tenant-a",
            ),
        ),
    )

    with pytest.raises(RoutingConfigurationError) as excinfo:
        table.resolve(
            tenant="tenant-a",
            process="draft",
            collection_id=None,
            workflow_id=None,
            doc_class="manual",
        )

    message = str(excinfo.value)
    assert RoutingErrorCode.CONFLICT in message
    assert "Ambiguous routing rules" in message


def test_same_specificity_same_profile_emits_warning(
    tmp_path, settings, caplog, enable_collection_routing
) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: standard
            tenant: tenant-a
            workflow_id: shared
          - profile: standard
            tenant: tenant-a
            workflow_id: shared
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with caplog.at_level(logging.WARNING):
        table = get_routing_table()

    assert table.default_profile == "standard"
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "ROUTE_DUP_SAME_TARGET" in record.getMessage()
    ]
    assert warnings


def test_routing_table_without_default_raises() -> None:
    table = RoutingTable(default_profile="", rules=())

    with pytest.raises(RoutingConfigurationError) as excinfo:
        table.resolve(
            tenant="tenant-a",
            process=None,
            collection_id=None,
            workflow_id=None,
            doc_class=None,
        )

    assert RoutingErrorCode.NO_MATCH in str(excinfo.value)
