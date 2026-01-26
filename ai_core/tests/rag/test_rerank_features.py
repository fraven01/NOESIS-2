"""Tests for structure-aware rerank features."""

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.rag.rerank_features import extract_rerank_features, resolve_weight_profile
from ai_core.tool_contracts import tool_context_from_scope


def _context() -> object:
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invoke",
        run_id="run",
    )
    business = BusinessContext()
    return tool_context_from_scope(scope, business)


def test_extract_rerank_features_basic():
    matches = [
        {
            "id": "c1",
            "text": "alpha",
            "score": 0.9,
            "meta": {
                "chunk_id": "c1",
                "document_id": "doc-1",
                "section_path": ["A"],
                "chunk_index": 0,
            },
        },
        {
            "id": "c2",
            "text": "beta",
            "score": 0.5,
            "meta": {
                "chunk_id": "c2",
                "document_id": "doc-1",
                "section_path": ["A"],
                "chunk_index": 1,
                "parent_ids": ["c1"],
                "parents": [{"score": 0.7}],
            },
        },
        {
            "id": "c3",
            "text": "gamma",
            "score": 0.4,
            "meta": {
                "chunk_id": "c3",
                "document_id": "doc-1",
                "section_path": ["B"],
                "chunk_index": 2,
            },
        },
    ]
    features = extract_rerank_features(matches, context=_context())
    assert len(features) == 3
    anchor = next(feature for feature in features if feature.chunk_id == "c1")
    neighbor = next(feature for feature in features if feature.chunk_id == "c2")
    other = next(feature for feature in features if feature.chunk_id == "c3")

    assert anchor.section_match == 1.0
    assert neighbor.section_match == 1.0
    assert other.section_match == 0.0
    assert neighbor.parent_relevance == 0.7


def test_resolve_weight_profile_defaults():
    weights = resolve_weight_profile(None)
    assert weights["confidence"] == 0.3


def test_extract_rerank_features_question_density():
    matches = [
        {
            "id": "q1",
            "text": "Welche Fragen muessen beantwortet werden?",
            "score": 0.6,
            "meta": {
                "chunk_id": "q1",
                "document_id": "doc-1",
                "section_path": ["A"],
                "chunk_index": 0,
            },
        }
    ]
    features = extract_rerank_features(matches, context=_context())
    assert features[0].question_density > 0.0


def test_resolve_weight_profile_prefers_learned(monkeypatch, db):
    from ai_core.models import RagRerankWeight

    learned = {
        "parent_relevance": 0.4,
        "section_match": 0.1,
        "confidence": 0.2,
        "adjacency_bonus": 0.2,
        "doc_type_match": 0.1,
    }
    RagRerankWeight.objects.create(
        tenant_id="tenant",
        quality_mode="standard",
        weights=learned,
        sample_count=12,
    )
    monkeypatch.setenv("RAG_RERANK_WEIGHT_MODE", "learned")
    result = resolve_weight_profile("standard", context=_context())
    assert result == learned
