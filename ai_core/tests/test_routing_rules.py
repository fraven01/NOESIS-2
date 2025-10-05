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


def test_resolve_prefers_more_specific_rule(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: legacy
            tenant: tenant-a
          - profile: standard
            tenant: tenant-a
            doc_class: legal
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    table = get_routing_table()

    assert (
        table.resolve(tenant="tenant-a", process="review", doc_class="legal")
        == "standard"
    )
    assert (
        table.resolve(tenant="tenant-a", process="review", doc_class="manual")
        == "legacy"
    )
    assert (
        table.resolve(tenant="tenant-b", process="review", doc_class="legal")
        == "standard"
    )


def test_rules_are_case_insensitive(tmp_path, settings) -> None:
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

    table = get_routing_table()

    assert (
        table.resolve(tenant="tenant-a", process="review", doc_class="manual")
        == "legacy"
    )


@pytest.mark.parametrize(
    "process,doc_class,expected",
    [
        ("review", "legal", "enterprise"),
        ("review", "manual", "premium"),
        ("draft", "legal", "legacy"),
        ("draft", None, "legacy"),
        ("review", None, "premium"),
    ],
)
def test_specificity_precedence(tmp_path, settings, process, doc_class, expected) -> None:
    _configure_embeddings(settings)
    settings.RAG_EMBEDDING_PROFILES.update(
        {
            "premium": {
                "model": "oai-embed-large",
                "dimension": 1536,
                "vector_space": "global",
            },
            "enterprise": {
                "model": "oai-embed-large",
                "dimension": 1536,
                "vector_space": "global",
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
          - profile: premium
            tenant: tenant-a
            process: review
          - profile: enterprise
            tenant: tenant-a
            process: review
            doc_class: legal
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    table = get_routing_table()

    assert (
        table.resolve(tenant="tenant-a", process=process, doc_class=doc_class)
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


def test_overlapping_rules_with_same_specificity_fail(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: standard
            tenant: tenant-a
            process: review
          - profile: legacy
            tenant: tenant-a
            doc_class: legal
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with pytest.raises(RoutingConfigurationError) as excinfo:
        get_routing_table()

    message = str(excinfo.value)
    assert "ROUTE_CONFLICT" in message
    assert "Overlapping routing rules" in message


def test_duplicate_rule_same_target_is_tolerated(tmp_path, settings, caplog) -> None:
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
          - profile: legacy
            tenant: tenant-a
            process: review
          - profile: legacy
            tenant: tenant-a
            process: review
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
        table.resolve(tenant="tenant-a", process="review", doc_class="manual")
        == "legacy"
    )


def test_duplicate_rule_conflicting_targets_raises(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: standard
            tenant: tenant-a
            process: review
          - profile: legacy
            tenant: tenant-a
            process: review
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with pytest.raises(RoutingConfigurationError) as excinfo:
        get_routing_table()

    message = str(excinfo.value)
    assert "ROUTE_DUP_SELECTOR" in message
    assert "Duplicate routing selector" in message


def test_resolve_raises_on_ambiguous_runtime_match() -> None:
    table = RoutingTable(
        default_profile="standard",
        rules=(
            RoutingRule(profile="legacy", tenant="tenant-a", process=None, doc_class=None),
            RoutingRule(profile="premium", tenant="tenant-a", process=None, doc_class=None),
        ),
    )

    with pytest.raises(RoutingConfigurationError) as excinfo:
        table.resolve(tenant="tenant-a", process="draft", doc_class="manual")

    message = str(excinfo.value)
    assert RoutingErrorCode.CONFLICT in message
    assert "Ambiguous routing rules" in message


def test_same_specificity_same_profile_still_conflicts(tmp_path, settings) -> None:
    _configure_embeddings(settings)
    rules_file = tmp_path / "routing.yaml"
    _write_routing_rules(
        rules_file,
        """
        default_profile: standard
        rules:
          - profile: standard
            tenant: tenant-a
            process: review
          - profile: standard
            tenant: tenant-a
            doc_class: legal
        """,
    )
    settings.RAG_ROUTING_RULES_PATH = rules_file

    with pytest.raises(RoutingConfigurationError) as excinfo:
        get_routing_table()

    message = str(excinfo.value)
    assert "ROUTE_CONFLICT" in message
    assert "Overlapping routing rules" in message


def test_routing_table_without_default_raises() -> None:
    table = RoutingTable(default_profile="", rules=())

    with pytest.raises(RoutingConfigurationError) as excinfo:
        table.resolve(tenant="tenant-a", process=None, doc_class=None)

    assert RoutingErrorCode.NO_MATCH in str(excinfo.value)
