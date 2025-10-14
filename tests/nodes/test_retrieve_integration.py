import pytest

from ai_core.nodes import retrieve
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_store import VectorStoreRouter
from ai_core.tool_contracts import ToolContext


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
        visibility: str = "active",
    ) -> None:
        self.chunks = list(chunks)
        self.vector_candidates = vector_candidates
        self.lexical_candidates = lexical_candidates
        self.fused_candidates = len(self.chunks)
        self.duration_ms = 0.0
        self.alpha = alpha
        self.min_sim = min_sim
        self.visibility = visibility
        self.deleted_matches_blocked = 0


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


class _VisibilityStore:
    def __init__(self, active_chunk: Chunk, deleted_chunk: Chunk) -> None:
        self.name = "global"
        self._active = active_chunk
        self._deleted = deleted_chunk
        self.hybrid_calls: list[dict[str, object]] = []

    def hybrid_search(
        self,
        query,
        tenant_id,
        *,
        case_id=None,
        top_k=5,
        filters=None,
        alpha=None,
        min_sim=None,
        vec_limit=None,
        lex_limit=None,
        trgm_limit=None,
        max_candidates=None,
        visibility=None,
        visibility_override_allowed=False,
        **kwargs,
    ):
        effective = visibility or "active"
        if effective == "deleted":
            chunks = [self._deleted]
        elif effective == "all":
            chunks = [self._active, self._deleted]
        else:
            chunks = [self._active]

        result = _HybridSearchResult(
            chunks,
            vector_candidates=len(chunks),
            lexical_candidates=0,
            alpha=0.45,
            min_sim=0.2,
            visibility=effective,
        )
        if effective == "active":
            result.deleted_matches_blocked = 1
        self.hybrid_calls.append(
            {
                "query": query,
                "tenant_id": tenant_id,
                "case_id": case_id,
                "top_k": top_k,
                "filters": filters,
                "alpha": alpha,
                "min_sim": min_sim,
                "vec_limit": vec_limit,
                "lex_limit": lex_limit,
                "trgm_limit": trgm_limit,
                "max_candidates": max_candidates,
                "visibility": visibility,
                "visibility_override_allowed": visibility_override_allowed,
                "effective": effective,
            }
        )
        return result


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

    tenant = "tenant-123"
    case = "case-7"
    vector_chunks = [
        Chunk(
            "Vector Match A",
            {
                "id": "doc-1",
                "score": 0.9,
                "source": "vector",
                "tenant_id": tenant,
                "case_id": case,
            },
        ),
        Chunk(
            "Vector Match B",
            {
                "id": "doc-2",
                "score": 0.4,
                "source": "vector",
                "tenant_id": tenant,
                "case_id": case,
            },
        ),
    ]
    lexical_chunks = [
        Chunk(
            "Lexical Match A",
            {
                "id": "doc-1",
                "score": 0.7,
                "source": "lexical",
                "tenant_id": tenant,
                "case_id": case,
            },
        ),
        Chunk(
            "Lexical Match B",
            {
                "id": "doc-3",
                "score": 0.8,
                "source": "lexical",
                "tenant_id": tenant,
                "case_id": case,
            },
        ),
    ]

    response = _HybridSearchResult(
        vector_chunks + lexical_chunks,
        vector_candidates=len(vector_chunks),
        lexical_candidates=len(lexical_chunks),
        alpha=0.55,
        min_sim=0.35,
    )
    response.deleted_matches_blocked = 2

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
    params = retrieve.RetrieveInput.from_state(state)
    context = ToolContext(
        tenant_id=tenant,
        tenant_schema="tenant-schema",
        case_id=case,
    )

    result = retrieve.run(context, params)

    assert router.for_tenant_calls == [("tenant-123", "tenant-schema")]
    assert len(router.hybrid_calls) == 1
    query, call_params = router.hybrid_calls[0]
    assert query == "find documents"
    assert call_params["case_id"] == "case-7"
    assert call_params["top_k"] == 3
    assert call_params["filters"] == {"project": "demo"}
    assert call_params["alpha"] == pytest.approx(0.55)
    assert call_params["min_sim"] == pytest.approx(0.35)
    assert call_params["vec_limit"] == 10
    assert call_params["lex_limit"] == 8
    if trgm_limit is None:
        assert call_params["trgm_limit"] is None
    else:
        assert call_params["trgm_limit"] == pytest.approx(0.4)
    assert call_params["max_candidates"] >= 3
    assert call_params["process"] == "review"
    assert call_params["doc_class"] == "policy"
    assert call_params["visibility"] is None
    assert call_params["visibility_override_allowed"] is False

    matches = result.matches
    assert len(matches) == 3
    # Dechunking now returns top chunks, allowing multiple per document.
    ids = [match["id"] for match in matches]
    assert ids == ["doc-1", "doc-3", "doc-1"]
    assert ids.count("doc-1") == 2
    assert "doc-2" not in ids
    assert matches[0]["id"] == "doc-1"
    assert matches[0]["score"] == pytest.approx(0.9)
    assert matches[0]["score"] >= matches[-1]["score"]
    assert all(0.0 <= match["score"] <= 1.0 for match in matches)

    meta_payload = result.meta
    assert isinstance(meta_payload.took_ms, int)
    assert meta_payload.took_ms >= 0
    assert meta_payload.vector_candidates == len(vector_chunks)
    assert meta_payload.lexical_candidates == len(lexical_chunks)
    assert meta_payload.deleted_matches_blocked == 2
    assert meta_payload.top_k_effective == call_params["top_k"]
    assert meta_payload.matches_returned == len(matches)
    assert meta_payload.alpha == pytest.approx(0.55)
    assert meta_payload.min_sim == pytest.approx(0.35)
    assert meta_payload.routing.profile == "standard"
    assert meta_payload.routing.vector_space_id == "rag/global"
    assert meta_payload.visibility_effective == "active"


def _run_visibility_scenario(
    monkeypatch: pytest.MonkeyPatch,
    *,
    requested: str | None = None,
    override_allowed: bool = False,
):
    _patch_routing(monkeypatch)

    tenant = "tenant-visibility"
    case = "case-vis"
    active_chunk = Chunk(
        "Active",
        {
            "id": "doc-active",
            "score": 0.91,
            "source": "vector",
            "tenant_id": tenant,
            "case_id": case,
        },
    )
    deleted_chunk = Chunk(
        "Deleted",
        {
            "id": "doc-deleted",
            "score": 0.42,
            "source": "vector",
            "tenant_id": tenant,
            "case_id": case,
        },
    )

    store = _VisibilityStore(active_chunk, deleted_chunk)
    router = VectorStoreRouter({"global": store})
    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: router)

    state = {"query": "check visibility", "hybrid": {}}
    if requested is not None:
        state["visibility"] = requested

    params = retrieve.RetrieveInput.from_state(state)
    context = ToolContext(
        tenant_id=tenant,
        case_id=case,
        visibility_override_allowed=override_allowed,
    )

    result = retrieve.run(context, params)

    recorded = store.hybrid_calls[-1]
    return recorded, result


def test_retrieve_visibility_defaults_to_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded, result = _run_visibility_scenario(monkeypatch)

    assert recorded["effective"] == "active"
    assert recorded.get("visibility") == "active"
    assert not recorded.get("visibility_override_allowed")

    matches = result.matches
    assert [match["id"] for match in matches] == ["doc-active"]
    meta = result.meta
    assert meta.visibility_effective == "active"
    assert meta.deleted_matches_blocked == 1


def test_retrieve_visibility_all_requires_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded, result = _run_visibility_scenario(
        monkeypatch, requested="all", override_allowed=True
    )

    assert recorded["effective"] == "all"
    assert recorded.get("visibility") == "all"
    assert recorded.get("visibility_override_allowed") is True

    matches = result.matches
    assert [match["id"] for match in matches] == ["doc-active", "doc-deleted"]
    meta = result.meta
    assert meta.visibility_effective == "all"
    assert meta.deleted_matches_blocked == 0


def test_retrieve_visibility_deleted_only_for_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded, result = _run_visibility_scenario(
        monkeypatch, requested="deleted", override_allowed=True
    )

    assert recorded["effective"] == "deleted"
    assert recorded.get("visibility") == "deleted"
    assert recorded.get("visibility_override_allowed") is True

    matches = result.matches
    assert [match["id"] for match in matches] == ["doc-deleted"]
    meta = result.meta
    assert meta.visibility_effective == "deleted"
    assert meta.deleted_matches_blocked == 0


def test_retrieve_visibility_override_denied_without_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded, result = _run_visibility_scenario(
        monkeypatch, requested="all", override_allowed=False
    )

    assert recorded["effective"] == "active"
    assert recorded.get("visibility") == "active"
    assert recorded.get("visibility_override_allowed") is False

    matches = result.matches
    assert [match["id"] for match in matches] == ["doc-active"]
    meta = result.meta
    assert meta.visibility_effective == "active"
    assert meta.deleted_matches_blocked == 1
