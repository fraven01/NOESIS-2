import pytest

from ai_core.nodes import retrieve
from ai_core.nodes._hybrid_params import TOPK_DEFAULT, TOPK_MAX, parse_hybrid_parameters
from ai_core.settings import RAG


def test_parse_defaults_and_clamps_values():
    state = {"hybrid": {"alpha": 1.2, "min_sim": -0.1, "top_k": 999}}

    params = parse_hybrid_parameters(state)

    assert params.alpha == pytest.approx(1.0)
    assert params.min_sim == pytest.approx(0.0)
    assert params.top_k == TOPK_MAX
    assert params.vec_limit == 50
    assert params.lex_limit == 50
    assert params.trgm_limit is None
    assert params.max_candidates == max(
        params.top_k, params.vec_limit, params.lex_limit
    )
    assert params.diversify_strength == pytest.approx(RAG.DIVERSIFY_STRENGTH_DEFAULT)
    assert state["hybrid"] == params.as_dict()


def test_parse_rejects_unknown_keys():
    state = {"hybrid": {"alpha": 0.5, "unexpected": 3}}

    with pytest.raises(ValueError, match=r"Unknown hybrid parameter\(s\): unexpected"):
        parse_hybrid_parameters(state)


def test_parse_raises_max_candidates_to_top_k():
    state = {
        "hybrid": {
            "top_k": 7,
            "vec_limit": 3,
            "lex_limit": 4,
            "max_candidates": 2,
        }
    }

    params = parse_hybrid_parameters(state)

    assert params.top_k == 7
    assert params.max_candidates == 7
    assert params.max_candidates >= params.top_k
    assert params.max_candidates >= max(params.vec_limit, params.lex_limit)
    assert params.diversify_strength == pytest.approx(RAG.DIVERSIFY_STRENGTH_DEFAULT)


def test_parse_supports_override_top_k_and_promotes_candidates():
    state = {
        "hybrid": {
            "top_k": 2,
            "vec_limit": 3,
            "lex_limit": 4,
            "max_candidates": 4,
        }
    }

    params = parse_hybrid_parameters(state, override_top_k=999)

    assert params.top_k == TOPK_MAX
    assert params.max_candidates >= params.top_k
    assert state["hybrid"]["top_k"] == TOPK_MAX
    assert params.diversify_strength == pytest.approx(RAG.DIVERSIFY_STRENGTH_DEFAULT)


def test_deduplicate_matches_prefers_best_score_and_is_stable():
    matches = [
        {"id": "doc-1", "hash": "chunk-1", "score": 0.5, "source": "vector"},
        {"id": "doc-1", "hash": "chunk-1", "score": 0.9, "source": "lexical"},
        {"id": "doc-2", "hash": "chunk-2", "score": 0.9, "source": "vector"},
    ]

    deduplicated = retrieve._deduplicate_matches(matches)

    assert [match["id"] for match in deduplicated] == ["doc-1", "doc-2"]
    assert deduplicated[0]["score"] == pytest.approx(0.9)
    assert deduplicated[0]["source"] == "lexical"
    assert deduplicated[1]["score"] == pytest.approx(0.9)
    assert deduplicated[1]["source"] == "vector"


def test_deduplicate_matches_are_trimmed_to_top_k_after_sorting():
    matches = [
        {"id": f"doc-{index:02d}", "score": 1.0 - (index * 0.01), "source": "vector"}
        for index in range(TOPK_MAX + 2)
    ]

    deduplicated = retrieve._deduplicate_matches(matches)
    final = deduplicated[:TOPK_MAX]

    assert len(final) == TOPK_MAX
    assert final[-1]["score"] >= deduplicated[TOPK_MAX]["score"]


def test_parse_rejects_boolean_for_integer_fields():
    state = {"hybrid": {"top_k": True}}

    with pytest.raises(ValueError, match=r"hybrid\.top_k must be an integer"):
        parse_hybrid_parameters(state)


def test_parse_requires_hybrid_block():
    with pytest.raises(ValueError, match="state must include a 'hybrid' configuration"):
        parse_hybrid_parameters({})


def test_parse_trgm_limit_accepts_strings_and_clamps_fraction():
    state = {"hybrid": {"trgm_limit": "1.5", "top_k": TOPK_DEFAULT}}

    params = parse_hybrid_parameters(state)

    assert params.trgm_limit == pytest.approx(1.0)
    assert state["hybrid"]["trgm_limit"] == params.trgm_limit


def test_parse_accepts_custom_diversify_strength():
    state = {"hybrid": {"diversify_strength": 0.8}}

    params = parse_hybrid_parameters(state)

    assert params.diversify_strength == pytest.approx(0.8)
    assert state["hybrid"]["diversify_strength"] == pytest.approx(0.8)


def test_diversification_promotes_dissimilar_matches():
    matches = [
        {
            "id": "doc-1",
            "text": "Alpha beta gamma",
            "score": 0.95,
        },
        {
            "id": "doc-2",
            "text": "Alpha beta gamma delta",
            "score": 0.93,
        },
        {
            "id": "doc-3",
            "text": "Completely unrelated content",
            "score": 0.9,
        },
    ]

    diversified = retrieve._apply_diversification(matches, top_k=2, strength=0.5)

    assert [match["id"] for match in diversified[:2]] == ["doc-1", "doc-3"]


def test_diversification_respects_zero_strength():
    matches = [
        {"id": "doc-1", "text": "One", "score": 0.9},
        {"id": "doc-2", "text": "Two", "score": 0.8},
        {"id": "doc-3", "text": "Three", "score": 0.7},
    ]

    diversified = retrieve._apply_diversification(matches, top_k=2, strength=0.0)

    assert [match["id"] for match in diversified] == [
        "doc-1",
        "doc-2",
        "doc-3",
    ]
