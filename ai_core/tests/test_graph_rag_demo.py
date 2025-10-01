"""Tests for the rag_demo graph node and HTTP endpoint."""

from __future__ import annotations

import json
from typing import Iterable, List

import pytest

from ai_core.graph import registry
from ai_core.graph.adapters import module_runner
from ai_core.graphs import rag_demo
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_client import HybridSearchResult


pytestmark = pytest.mark.django_db


class _TenantRouter:
    """Fake tenant-scoped router returning deterministic matches."""

    def __init__(self) -> None:
        self._matches: List[tuple[str, str]] = [
            ("First tenant scoped match", "fake-1"),
            ("Second tenant scoped match", "fake-2"),
        ]

    def search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> Iterable[Chunk]:
        assert filters is not None, "filters must be provided"
        assert (
            filters.get("tenant_id") == tenant_id
        ), "tenant_id filter must match requested tenant"
        del case_id  # not used in the fake router
        del query  # the fake router does not evaluate query semantics
        chunks = []
        for idx, (text, match_id) in enumerate(self._matches):
            chunks.append(
                Chunk(
                    content=text,
                    meta={
                        "id": match_id,
                        "tenant_id": tenant_id,
                        "vscore": 0.6 - idx * 0.1,
                        "lscore": 0.4 - idx * 0.1,
                        "fused": 0.5 - idx * 0.1,
                        "score": 0.5 - idx * 0.1,
                    },
                )
            )
        return chunks[:top_k]


@pytest.fixture(autouse=True)
def _configure_rag_demo(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Ensure the rag_demo graph uses a fake router during tests."""

    class FakeRouter:
        def for_tenant(self) -> _TenantRouter:
            return _TenantRouter()

    store_path = tmp_path_factory.mktemp("rag-demo-store")
    monkeypatch.setattr("ai_core.infra.object_store.BASE_PATH", store_path)
    monkeypatch.setattr(
        rag_demo,
        "get_default_router",
        lambda: FakeRouter().for_tenant(),
    )
    monkeypatch.setattr("ai_core.infra.rate_limit.check", lambda *args, **kwargs: True)
    registry.register("rag_demo", module_runner(rag_demo))


def test_rag_demo_route_returns_matches(client) -> None:
    response = client.post(
        "/ai/v1/rag-demo/",
        data=json.dumps({"query": "Alpha"}),
        content_type="application/json",
        **{"HTTP_X_TENANT_ID": "dev", "HTTP_X_CASE_ID": "local"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["query"] == "Alpha"
    assert len(data["matches"]) >= 2
    assert "meta" in data
    assert data["meta"]["index_kind"]
    assert "db_latency_ms" in data["meta"]
    assert all("fused" in match for match in data["matches"])
    assert "warnings" not in data


def test_rag_demo_missing_query_returns_error(client) -> None:
    response = client.post(
        "/ai/v1/rag-demo/",
        data=json.dumps({}),
        content_type="application/json",
        **{"HTTP_X_TENANT_ID": "dev", "HTTP_X_CASE_ID": "missing"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["error"] == "missing_query"
    assert data["matches"] == []


def test_rag_demo_run_falls_back_to_demo_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_demo, "get_default_router", lambda: object())

    state = {"query": "Alpha"}
    meta = {"tenant_id": "dev"}

    new_state, result = rag_demo.run(state, meta)

    assert result["ok"] is True
    assert result["query"] == "Alpha"
    assert len(result["matches"]) == 2
    assert result["error"] == "router missing search"
    assert new_state["rag_demo"]["retrieved_count"] == 2


def test_rag_demo_run_handles_for_tenant_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Router:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def for_tenant(self, tenant_id: str) -> _TenantRouter:
            self.calls.append(tenant_id)
            return _TenantRouter()

    router = Router()
    monkeypatch.setattr(rag_demo, "get_default_router", lambda: router)

    state = {"query": "Alpha"}
    meta = {"tenant_id": "dev", "case_id": "case"}

    _, result = rag_demo.run(state, meta)

    assert result["ok"] is True
    assert len(result["matches"]) == 2
    assert router.calls == ["dev"]


def test_rag_demo_run_handles_for_tenant_with_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Router:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def for_tenant(self, tenant_id: str, tenant_schema: str) -> _TenantRouter:
            self.calls.append((tenant_id, tenant_schema))
            return _TenantRouter()

    router = Router()
    monkeypatch.setattr(rag_demo, "get_default_router", lambda: router)

    state = {"query": "Alpha"}
    meta = {"tenant_id": "dev", "tenant_schema": "public"}

    _, result = rag_demo.run(state, meta)

    assert result["ok"] is True
    assert len(result["matches"]) == 2
    assert router.calls == [("dev", "public")]


def test_rag_demo_hybrid_search_uses_trimmed_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Router:
        def __init__(self) -> None:
            self.received_query: str | None = None

        def for_tenant(self, tenant_id: str) -> Router:
            assert tenant_id == "dev"
            return self

        def hybrid_search(
            self,
            query: str,
            *,
            tenant_id: str,
            case_id: str | None = None,
            top_k: int,
            filters: dict[str, object] | None,
            alpha: float,
            min_sim: float,
            trgm_limit: float | None = None,
            trgm_threshold: float | None = None,
        ) -> HybridSearchResult:
            assert filters is not None
            assert filters.get("tenant_id") == tenant_id
            assert trgm_limit == 3.0
            self.received_query = query
            chunk = Chunk(
                content="lexical match",
                meta={
                    "id": "lex-1",
                    "tenant_id": tenant_id,
                    "vscore": 0.05,
                    "lscore": 0.42,
                    "fused": 0.21,
                    "score": 0.21,
                },
            )
            return HybridSearchResult(
                chunks=[chunk],
                vector_candidates=1,
                lexical_candidates=1,
                fused_candidates=1,
                duration_ms=1.5,
                alpha=alpha,
                min_sim=min_sim,
                vec_limit=top_k,
                lex_limit=top_k,
            )

    router = Router()
    monkeypatch.setattr(rag_demo, "get_default_router", lambda: router)

    state = {"query": "  zebragurke  ", "trgm_limit": 3, "top_k": 1}
    meta = {"tenant_id": "dev"}

    _, result = rag_demo.run(state, meta)

    assert router.received_query == "zebragurke"
    assert result["matches"], "expected at least one match"
    match = result["matches"][0]
    assert match["lscore"] > 0
    assert result["meta"]["lexical_candidates"] == 1


def test_rag_demo_response_contains_scores_and_meta() -> None:
    state = {"query": "Alpha", "top_k": 1}
    meta = {"tenant_id": "dev"}

    _, result = rag_demo.run(state, meta)

    assert result["ok"] is True
    assert result["meta"]["index_kind"]
    assert "latency_ms" in result["meta"]
    assert "db_latency_ms" in result["meta"]
    assert result["matches"]
    match = result["matches"][0]
    assert "vscore" in match and "lscore" in match and "fused" in match
    assert match["fused"] == match["score"]


def test_rag_demo_zero_hits_falls_back_to_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    class RouterZero:
        def for_tenant(self, tenant_id: str):
            class Client:
                def search(
                    self,
                    query: str,
                    tenant_id: str,
                    case_id: str | None = None,
                    top_k: int = 5,
                    filters: dict[str, object] | None = None,
                ) -> Iterable[Chunk]:
                    assert filters is not None
                    assert filters.get("tenant_id") == tenant_id
                    del query, tenant_id, case_id, top_k
                    return []  # zero hits

            return Client()

    monkeypatch.setattr(rag_demo, "get_default_router", lambda: RouterZero())

    state = {"query": "Alpha", "top_k": 2}
    meta = {"tenant_id": "demo", "case_id": "local"}
    new_state, result = rag_demo.run(state, meta)

    assert result["ok"] is True
    assert len(result["matches"]) > 0  # demo fallback kicked in
    assert (
        "warnings" in result and "no_vector_matches_demo_fallback" in result["warnings"]
    )
    assert new_state["rag_demo"]["retrieved_count"] == len(result["matches"])


def test_rag_demo_run_zero_matches_adds_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Router:
        def search(
            self,
            query: str,
            *,
            tenant_id: str,
            case_id: str | None = None,
            top_k: int = 5,
            filters: dict[str, object] | None = None,
        ) -> Iterable[Chunk]:
            del query, tenant_id, case_id, top_k, filters
            return []

    monkeypatch.setattr(rag_demo, "get_default_router", lambda: Router())

    state = {"query": "Alpha"}
    meta = {"tenant_id": "dev"}

    new_state, result = rag_demo.run(state, meta)

    assert result["ok"] is True
    assert result["warnings"] == ["no_vector_matches_demo_fallback"]
    assert len(result["matches"]) == 2
    assert new_state["rag_demo"]["retrieved_count"] == 2


def test_rag_demo_no_hit_above_threshold_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Router:
        def hybrid_search(
            self,
            query: str,
            tenant_id: str,
            *,
            case_id: str | None = None,
            top_k: int = 5,
            filters: dict[str, object] | None = None,
            alpha: float | None = None,
            min_sim: float | None = None,
        ) -> HybridSearchResult:
            del query, tenant_id, case_id, top_k, filters, alpha, min_sim
            return HybridSearchResult(
                chunks=[],
                vector_candidates=2,
                lexical_candidates=1,
                fused_candidates=2,
                duration_ms=5.0,
                alpha=0.7,
                min_sim=0.95,
                vec_limit=5,
                lex_limit=5,
                below_cutoff=2,
                returned_after_cutoff=0,
            )

    monkeypatch.setattr(rag_demo, "get_default_router", lambda: Router())

    state = {"query": "Alpha"}
    meta = {"tenant_id": "dev"}

    _, result = rag_demo.run(state, meta)

    assert result["ok"] is True
    assert result["matches"] == []
    assert result["warnings"] == ["no_hit_above_threshold"]
    assert result["meta"]["below_cutoff"] == 2
    assert result["meta"]["returned_after_cutoff"] == 0
