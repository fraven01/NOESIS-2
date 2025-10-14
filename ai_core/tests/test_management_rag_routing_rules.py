"""Tests for the rag_routing_rules management command."""

from __future__ import annotations

import io
import textwrap

import pytest
from django.core.management import CommandError, call_command

from ai_core.rag.embedding_config import reset_embedding_configuration_cache
from ai_core.rag.routing_rules import reset_routing_rules_cache


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_embedding_configuration_cache()
    reset_routing_rules_cache()
    yield
    reset_embedding_configuration_cache()
    reset_routing_rules_cache()


@pytest.fixture
def routing_config(tmp_path, settings):
    rules = tmp_path / "routing.yaml"
    rules.write_text(
        textwrap.dedent(
            """
            default_profile: standard
            rules:
              - tenant: acme
                process: Review
                profile: premium
            """
        ).strip()
    )
    settings.RAG_ROUTING_RULES_PATH = str(rules)
    settings.RAG_VECTOR_STORES = {
        "global": {
            "backend": "pgvector",
            "schema": "rag",
            "dimension": 1536,
        }
    }
    settings.RAG_EMBEDDING_PROFILES = {
        "standard": {
            "model": "oai-embed-large",
            "dimension": 1536,
            "vector_space": "global",
            "chunk_hard_limit": 512,
        },
        "premium": {
            "model": "vertex_ai/text-embedding-004",
            "dimension": 1536,
            "vector_space": "global",
            "chunk_hard_limit": 768,
        },
    }
    return rules


def test_command_lists_routing_rules(routing_config):
    buffer = io.StringIO()

    call_command("rag_routing_rules", stdout=buffer)

    output = buffer.getvalue()
    assert "Default profile: standard" in output
    assert "tenant=acme, process=review, doc_class=* -> premium" in output


def test_command_resolves_selector(routing_config):
    buffer = io.StringIO()

    call_command(
        "rag_routing_rules",
        tenant="acme",
        process="REVIEW",
        stdout=buffer,
    )

    output = buffer.getvalue()
    assert (
        "Resolved selector tenant=acme, process=review, doc_class=* -> premium"
        in output
    )


def test_command_requires_tenant_for_resolution(routing_config):
    with pytest.raises(CommandError) as exc:
        call_command("rag_routing_rules", process="review")

    assert "--tenant is required" in str(exc.value)


def test_command_rejects_empty_tenant(routing_config):
    with pytest.raises(CommandError) as exc:
        call_command("rag_routing_rules", tenant="  ")

    assert "non-empty" in str(exc.value)
