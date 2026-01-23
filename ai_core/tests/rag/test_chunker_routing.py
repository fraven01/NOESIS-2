"""Tests for chunker mode routing functionality."""

import pytest
from ai_core.rag.routing_rules import (
    resolve_chunker_mode,
    get_routing_table,
    reset_routing_rules_cache,
)
from ai_core.rag.chunking import get_chunker_config_from_routing, ChunkerMode


@pytest.fixture(autouse=True)
def reset_routing_cache():
    """Reset routing rules cache before and after each test."""
    reset_routing_rules_cache()
    yield
    reset_routing_rules_cache()


def test_resolve_chunker_mode_returns_default_for_unknown_tenant():
    """Test that unknown tenants get the default chunker mode."""
    mode = resolve_chunker_mode(tenant="unknown")
    assert mode == "late"  # Default mode from config


def test_resolve_chunker_mode_returns_configured_mode_for_known_tenant():
    """Test that known tenants get their configured chunker mode."""
    mode = resolve_chunker_mode(tenant="dev")
    assert mode == "agentic"  # Configured in rag_routing_rules.yaml


def test_resolve_chunker_mode_with_collection_id():
    """Test chunker mode resolution with collection_id."""
    mode = resolve_chunker_mode(tenant="demo", collection_id="support-faq")
    assert mode == "late"  # Configured in rag_routing_rules.yaml


def test_resolve_chunker_mode_with_workflow_id():
    """Test chunker mode resolution with workflow_id."""
    mode = resolve_chunker_mode(tenant="demo", workflow_id="onboarding")
    assert mode == "late"  # Configured in rag_routing_rules.yaml


def test_get_chunker_config_from_routing_returns_late_mode():
    """Test that get_chunker_config_from_routing returns correct mode."""
    config = get_chunker_config_from_routing(tenant_id="dev")
    assert config.mode == ChunkerMode.AGENTIC


def test_get_chunker_config_from_routing_respects_collection():
    """Test that routing respects collection_id parameter."""
    config = get_chunker_config_from_routing(
        tenant_id="demo", collection_id="support-faq"
    )
    assert config.mode == ChunkerMode.LATE


def test_routing_table_includes_default_chunker_mode():
    """Test that routing table loads default_chunker_mode from YAML."""
    table = get_routing_table()
    assert table.default_chunker_mode == "late"


def test_routing_table_rules_include_chunker_mode():
    """Test that routing rules include chunker_mode field."""
    table = get_routing_table()
    # Find rule for dev tenant
    dev_rule = next((r for r in table.rules if r.tenant == "dev"), None)
    assert dev_rule is not None, "dev tenant rule should exist"
    assert dev_rule.chunker_mode == "agentic"


def test_resolve_chunker_mode_falls_back_to_default():
    """Test that unmatched selectors fall back to default_chunker_mode."""
    mode = resolve_chunker_mode(
        tenant="unknown_tenant", collection_id="unknown_collection"
    )
    assert mode == "late"  # Should fall back to default_chunker_mode


def test_chunker_config_from_routing_uses_django_settings_for_other_fields():
    """Test that routing only affects mode, other fields come from settings."""
    config = get_chunker_config_from_routing(tenant_id="dev")

    # Mode should be from routing
    assert config.mode == ChunkerMode.AGENTIC

    # Other fields should be from Django settings defaults
    assert config.late_chunk_model == "embedding"  # From RAG_LATE_CHUNK_MODEL
    assert config.agentic_chunk_model == "agentic-chunk"  # From RAG_AGENTIC_CHUNK_MODEL
    assert config.quality_model == "quality-eval"  # From RAG_QUALITY_EVAL_MODEL
    assert config.max_chunk_tokens == 450  # From RAG_CHUNK_TARGET_TOKENS
    assert config.overlap_tokens == 80  # From RAG_CHUNK_OVERLAP_TOKENS
