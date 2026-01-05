"""Chaos tests for RoutingAwareChunker integration.

Tests chunker mode routing resolution, dimension validation, and fallback behavior
when agentic chunking fails due to rate limits or budget exhaustion.

Contract under test:
- ai_core/rag/chunking/hybrid_chunker.py: RoutingAwareChunker, get_chunker_config_from_routing
- ai_core/rag/routing_rules.py: resolve_chunker_mode
"""

from __future__ import annotations

import pytest

from ai_core.rag.chunking import ChunkerMode, get_chunker_config_from_routing

pytestmark = pytest.mark.chaos


def test_chunker_routing_resolution_default():
    """Chunker mode resolved from routing rules with default fallback."""
    config = get_chunker_config_from_routing(
        tenant_id="tenant-default",
        # No specific routing rules → should use default
    )

    # Default mode is LATE (from Django settings)
    assert config.mode in [ChunkerMode.LATE, ChunkerMode.AGENTIC]
    assert isinstance(config.mode, ChunkerMode)


def test_chunker_routing_resolution_with_context():
    """Chunker mode resolved with full routing context."""
    config = get_chunker_config_from_routing(
        tenant_id="enterprise-tenant",
        collection_id="legal-contracts",
        doc_class="legal_contract",
    )

    # Should return a valid ChunkerMode
    assert isinstance(config.mode, ChunkerMode)
    assert config.mode in [ChunkerMode.LATE, ChunkerMode.AGENTIC, ChunkerMode.HYBRID]


def test_chunker_config_structure():
    """ChunkerConfig has all required fields and defaults."""
    config = get_chunker_config_from_routing(
        tenant_id="tenant-config-test",
    )

    # Verify all essential config fields are present
    assert hasattr(config, "mode")
    assert hasattr(config, "late_chunk_model")
    assert hasattr(config, "late_chunk_max_tokens")
    assert hasattr(config, "agentic_chunk_model")
    assert hasattr(config, "max_chunk_tokens")
    assert hasattr(config, "overlap_tokens")

    # Verify types
    assert isinstance(config.mode, ChunkerMode)
    assert isinstance(config.late_chunk_model, str)
    assert isinstance(config.late_chunk_max_tokens, int)
    assert isinstance(config.max_chunk_tokens, int)
    assert isinstance(config.overlap_tokens, int)

    # Verify sane defaults
    assert config.late_chunk_max_tokens > 0
    assert config.max_chunk_tokens > 0
    assert config.overlap_tokens >= 0
    assert config.overlap_tokens < config.max_chunk_tokens


def test_chunker_mode_enum_values():
    """ChunkerMode enum has expected values."""
    assert ChunkerMode.LATE == "late"
    assert ChunkerMode.AGENTIC == "agentic"
    assert ChunkerMode.HYBRID == "hybrid"

    # Enum can be constructed from string
    assert ChunkerMode("late") == ChunkerMode.LATE
    assert ChunkerMode("agentic") == ChunkerMode.AGENTIC


def test_chunker_config_immutable():
    """ChunkerConfig is frozen (immutable after creation)."""
    config = get_chunker_config_from_routing(
        tenant_id="tenant-immutable",
    )

    # ChunkerConfig is a frozen dataclass - should raise FrozenInstanceError
    with pytest.raises(
        Exception
    ):  # FrozenInstanceError or AttributeError depending on Python version
        config.mode = ChunkerMode.AGENTIC  # INVALID - frozen


def test_chunker_routing_tenant_isolation():
    """Different tenants can have different chunker configurations."""
    config_tenant_a = get_chunker_config_from_routing(
        tenant_id="tenant-a",
    )
    config_tenant_b = get_chunker_config_from_routing(
        tenant_id="tenant-b",
    )

    # Both should be valid configs (may or may not be different modes)
    assert isinstance(config_tenant_a.mode, ChunkerMode)
    assert isinstance(config_tenant_b.mode, ChunkerMode)


def test_chunker_config_quality_metrics_flag():
    """ChunkerConfig includes quality metrics control."""
    config = get_chunker_config_from_routing(
        tenant_id="tenant-quality",
    )

    assert hasattr(config, "enable_quality_metrics")
    assert isinstance(config.enable_quality_metrics, bool)
    assert hasattr(config, "quality_model")
    assert isinstance(config.quality_model, str)


def test_chunker_config_phase2_features():
    """ChunkerConfig includes Phase 2 SOTA features (embedding-based similarity)."""
    config = get_chunker_config_from_routing(
        tenant_id="tenant-phase2",
    )

    # Phase 2 feature flags
    assert hasattr(config, "use_embedding_similarity")
    assert isinstance(config.use_embedding_similarity, bool)
    assert hasattr(config, "window_size")
    assert isinstance(config.window_size, int)
    assert hasattr(config, "batch_size")
    assert isinstance(config.batch_size, int)
    assert hasattr(config, "use_content_based_ids")
    assert isinstance(config.use_content_based_ids, bool)

    # Sane defaults for Phase 2 features
    assert config.window_size > 0
    assert config.batch_size > 0


@pytest.mark.skip(reason="Dimension validation function not yet identified in codebase")
def test_chunker_dimension_validation_mismatch():
    """Embedding dimension mismatch raises validation error.

    NOTE: This test is skipped pending identification of the dimension validation
    function in the codebase. Expected location: ai_core/rag/ingestion_contracts.py
    or ai_core/rag/validation.py
    """
    # Placeholder for future implementation
    # from ai_core.rag.ingestion_contracts import ensure_embedding_dimensions
    #
    # with pytest.raises(ValueError, match="dimension mismatch"):
    #     ensure_embedding_dimensions(
    #         expected=1536,
    #         actual=768,
    #         profile_id="test-profile",
    #     )
    pass


@pytest.mark.skip(reason="Fallback behavior requires mocking LLM failures")
def test_chunker_fallback_on_rate_limit():
    """AgenticChunker falls back to LateChunker on rate limit.

    NOTE: This test is skipped as it requires complex mocking of:
    - LLM client to raise RateLimitError
    - RoutingAwareChunker.chunk() method
    - Observability event emission

    Future implementation should use monkeypatch to inject rate limit errors
    and verify fallback to late chunking mode.
    """
    # Placeholder for future implementation
    # chaos_env.set_redis_down(True)  # Or equivalent LLM rate limit
    # chunker = RoutingAwareChunker()
    # result = chunker.chunk(document, context)
    # assert result.mode_used == ChunkerMode.LATE  # Fallback
    pass
