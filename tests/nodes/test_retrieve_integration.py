import pytest

from ai_core.nodes import retrieve
from ai_core.rag.schemas import Chunk


class _DummyProfile:
    def __init__(self, vector_space: str) -> None:
        self.vector_space = vector_space


class _DummyConfig:
    def __init__(self, profile: str, vector_space: str) -> None:
        self.embedding_profiles = {profile: _DummyProfile(vector_space)}


class _HybridSearchResult:
    def __init__(
        self,
        chunks,
        *,
        vector_candidates: int,
        lexical_candidates: int,
        alpha: float,
        min_sim: float,
    ) -> None:
        self.chunks = chunks
        self.vector_candidates = vector_candidates
        self.lexical_candidates = lexical_candidates
        self.alpha = alpha
        self.min_sim = min_sim


class _FakeRouter:
    def __init__(self, response):
        self._response = response
        self.for_tenant_calls = []
        self.hybrid_calls = []

    def for_tenant(self, tenant_id, tenant_schema=None):
        self.for_tenant_calls.append((tenant_id, tenant_schema))
        return self

    def hybrid_search(self, query, **kwargs):
        self.hybrid_calls.append((query, kwargs))
        return self._response


def _patch_routing(monkeypatch, profile: str = "standard", space: str = "rag/global"):
    monkeypatch.setattr(
        "ai_core.nodes.retrieve.resolve_embedding_profile",
        lambda *, tenant_id, process=None, doc_class=None: profile,
    )
    monkeypatch.setattr(
        "ai_core.nodes.retrieve.get_embedding_configuration",
        lambda: _DummyConfig(profile, space),
    )


@pytest.mark.parametrize("trgm_limit", [None, "0.4"])
def test_retrieve_happy_path(monkeypatch, trgm_limit):
    _patch_routing(monkeypatch)

    vector_chunks = [
        Chunk("Vector Match A", {"id": "doc-1", "score": 0.9, "source": "vector"}),
        Chunk("Vector Match B", {"id": "doc-2", "score": 0.4, "source": "vector"}),
    ]
    lexical_chunks = [
        Chunk("Lexical Match A", {"id": "doc-1", "score": 0.7, "source": "lexical"}),
        Chunk("Lexical Match B", {"id": "doc-3", "score": 0.8, "source": "lexical"}),
    ]

    response = _HybridSearchResult(
        vector_chunks + lexical_chunks,
        vector_candidates=len(vector_chunks),
        lexical_candidates=len(lexical_chunks),
        alpha=0.55,
        min_sim=0.35,
    )

    router = _FakeRouter(response)
    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: router)

    state = {
        "query": "find documents",
        "filters": {"project": "demo"},
        "process": "review",
        "doc_class": "policy",
        "hybrid": {
            "alpha": 0.55,
            "min_sim": 0.35,
            "top_k": 3,
            "vec_limit": 10,
            "lex_limit": 8,
            "trgm_limit": trgm_limit,
        },
    }
    meta = {
        "tenant_id": "tenant-123",
        "tenant_schema": "tenant-schema",
        "case_id": "case-7",
    }

    new_state, payload = retrieve.run(state, meta)

    assert router.for_tenant_calls == [("tenant-123", "tenant-schema")]
    assert len(router.hybrid_calls) == 1
    query, params = router.hybrid_calls[0]
    assert query == "find documents"
    assert params["case_id"] == "case-7"
    assert params["top_k"] == 3
    assert params["filters"] == {"project": "demo"}
    assert params["alpha"] == pytest.approx(0.55)
    assert params["min_sim"] == pytest.approx(0.35)
    assert params["vec_limit"] == 10
    assert params["lex_limit"] == 8
    if trgm_limit is None:
        assert params["trgm_limit"] is None
    else:
        assert params["trgm_limit"] == pytest.approx(0.4)
    assert params["max_candidates"] >= 3
    assert params["process"] == "review"
    assert params["doc_class"] == "policy"

    matches = payload["matches"]
    assert len(matches) == 3
    assert {match["id"] for match in matches} == {"doc-1", "doc-2", "doc-3"}
    assert matches[0]["id"] == "doc-1"
    assert matches[0]["score"] == pytest.approx(0.9)
    assert matches[0]["score"] >= matches[-1]["score"]
    assert all(0.0 <= match["score"] <= 1.0 for match in matches)
    assert new_state["matches"] == matches
    assert new_state["snippets"] == matches

    meta_payload = payload["meta"]
    assert meta_payload["vector_candidates"] == len(vector_chunks)
    assert meta_payload["lexical_candidates"] == len(lexical_chunks)
    assert meta_payload["top_k_effective"] == 3
    assert meta_payload["alpha"] == pytest.approx(0.55)
    assert meta_payload["min_sim"] == pytest.approx(0.35)
    assert meta_payload["routing"] == {
        "profile": "standard",
        "vector_space_id": "rag/global",
    }
