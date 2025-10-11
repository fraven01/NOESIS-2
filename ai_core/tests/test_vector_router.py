"""Unit tests for the vector store router abstraction."""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

import pytest

from ai_core.rag import metrics
from ai_core.rag.schemas import Chunk
from ai_core.rag.limits import get_limit_setting, normalize_max_candidates
from ai_core.rag.router_validation import (
    RouterInputError,
    RouterInputErrorCode,
)
from ai_core.rag.vector_client import HybridSearchResult
from ai_core.rag.vector_store import VectorStore, VectorStoreRouter
from common.logging import log_context


class FakeStore(VectorStore):
    """In-memory fake that records invocations for assertions."""

    def __init__(
        self, name: str, *, search_result: Iterable[Chunk] | None = None
    ) -> None:
        self.name = name
        self.search_calls: list[Mapping[str, object]] = []
        self.upsert_calls: list[list[Chunk]] = []
        self._search_result = list(search_result or [])
        self.health_checks = 0

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        self.upsert_calls.append(chunk_list)
        return len(chunk_list)

    def search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, str | None] | None = None,
    ) -> list[Chunk]:
        self.search_calls.append(
            {
                "query": query,
                "tenant_id": tenant_id,
                "case_id": case_id,
                "top_k": top_k,
                "filters": filters,
            }
        )
        return self._search_result

    def health_check(self) -> bool:
        self.health_checks += 1
        return True


class HybridEnabledStore(VectorStore):
    def __init__(self, name: str, result: HybridSearchResult) -> None:
        self.name = name
        self._result = result
        self.hybrid_calls: list[Mapping[str, object]] = []
        self.search_calls: list[Mapping[str, object]] = []
        self.upsert_calls: list[list[Chunk]] = []

    def upsert_chunks(
        self, chunks: Iterable[Chunk]
    ) -> int:  # pragma: no cover - unused
        chunk_list = list(chunks)
        self.upsert_calls.append(chunk_list)
        return len(chunk_list)

    def search(  # pragma: no cover - fallback path exercise separate fake
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
    ) -> list[Chunk]:
        self.search_calls.append(
            {
                "query": query,
                "tenant_id": tenant_id,
                "case_id": case_id,
                "top_k": top_k,
                "filters": filters,
            }
        )
        return list(self._result.chunks)

    def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        alpha: float | None = None,
        min_sim: float | None = None,
        vec_limit: int | None = None,
        lex_limit: int | None = None,
        trgm_limit: float | None = None,
        max_candidates: int | None = None,
        visibility: str | None = None,
        visibility_override_allowed: bool = False,
    ) -> HybridSearchResult:
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
            }
        )
        return self._result


@pytest.fixture
def router_and_stores() -> tuple[VectorStoreRouter, FakeStore, FakeStore]:
    global_store = FakeStore(
        "global",
        search_result=[
            Chunk(content="global", meta={"tenant_id": "t", "case_id": "case"})
        ],
    )
    silo_store = FakeStore(
        "silo",
        search_result=[
            Chunk(content="silo", meta={"tenant_id": "t", "case_id": "case"})
        ],
    )
    router = VectorStoreRouter({"global": global_store, "silo": silo_store})
    return router, global_store, silo_store


def test_router_requires_tenant_id(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, _, _ = router_and_stores
    spans: list[dict[str, object]] = []
    from ai_core.rag import router_validation as router_validation_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        router_validation_module.tracing,
        "emit_span",
        lambda **kwargs: spans.append(kwargs),
    )

    with log_context(trace_id="trace-router-test"):
        with pytest.raises(RouterInputError) as excinfo:
            router.search("query", tenant_id="")

    monkeypatch.undo()

    assert excinfo.value.code == RouterInputErrorCode.TENANT_REQUIRED
    assert spans, "expected router validation failure to emit span"
    metadata = spans[0]["metadata"]
    assert metadata["error_code"] == RouterInputErrorCode.TENANT_REQUIRED


def test_router_caps_top_k_to_10(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores
    router.search("query", tenant_id="tenant", top_k=25)
    assert global_store.search_calls[-1]["top_k"] == 10


def test_router_accepts_process_and_doc_class(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores
    router.search(
        "query",
        tenant_id="tenant",
        process="draft",
        doc_class="legal",
    )

    assert global_store.search_calls[-1]["tenant_id"] == "tenant"


def test_router_normalizes_empty_filters(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores
    router.search(
        "query",
        tenant_id="tenant",
        filters={"case_id": "", "source": "doc", "extra": ""},
    )
    filters = global_store.search_calls[-1]["filters"]
    assert filters == {
        "case_id": None,
        "source": "doc",
        "extra": None,
        "visibility": "active",
    }


def test_router_search_enforces_visibility_guard(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores

    router.search(
        "query",
        tenant_id="tenant",
        visibility="deleted",
        visibility_override_allowed=False,
    )

    filters = global_store.search_calls[-1]["filters"]
    assert filters["visibility"] == "active"


def test_router_search_allows_authorized_visibility_override(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores

    router.search(
        "query",
        tenant_id="tenant",
        visibility="all",
        visibility_override_allowed=True,
    )

    filters = global_store.search_calls[-1]["filters"]
    assert filters["visibility"] == "all"


def test_router_delegates_to_scope_or_default(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, silo_store = router_and_stores
    router.search("query", tenant_id="tenant", scope="silo")
    assert len(silo_store.search_calls) == 1
    assert len(global_store.search_calls) == 0

    router.search("query", tenant_id="tenant", scope="missing")
    assert len(global_store.search_calls) == 1


def test_router_upsert_delegates_global(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, silo_store = router_and_stores
    chunk = Chunk(content="foo", meta={"tenant_id": "t", "case_id": "case"})
    router.upsert_chunks([chunk])
    assert len(global_store.upsert_calls) == 1
    assert global_store.upsert_calls[0] == [chunk]
    assert len(silo_store.upsert_calls) == 0


def test_router_for_tenant_returns_scoped_client(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    _, global_store, silo_store = router_and_stores
    scoped_router = VectorStoreRouter(
        {"global": global_store, "silo": silo_store},
        tenant_scopes={"tenant-123": "silo"},
    )
    tenant_client = scoped_router.for_tenant("tenant-123")
    tenant_client.search("q", tenant_id="tenant-123", top_k=1)
    assert len(silo_store.search_calls) == 1


def test_router_for_tenant_prefers_schema_scope(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    _, global_store, silo_store = router_and_stores
    router = VectorStoreRouter(
        {"global": global_store, "silo": silo_store},
        schema_scopes={"tenant-123-schema": "silo"},
    )

    tenant_client = router.for_tenant("tenant-123", "tenant-123-schema")
    tenant_client.search("q")

    assert len(silo_store.search_calls) == 1
    assert len(global_store.search_calls) == 0


def test_router_schema_scope_overrides_tenant_scope(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    _, global_store, silo_store = router_and_stores
    scoped_router = VectorStoreRouter(
        {"global": global_store, "silo": silo_store},
        tenant_scopes={"tenant-123": "global"},
        schema_scopes={"tenant-123-schema": "silo"},
    )

    schema_client = scoped_router.for_tenant("tenant-123", "tenant-123-schema")
    schema_client.search("q")
    assert len(silo_store.search_calls) == 1
    assert len(global_store.search_calls) == 0

    tenant_client = scoped_router.for_tenant("tenant-123")
    tenant_client.search("q")
    assert len(global_store.search_calls) == 1


def test_router_for_tenant_with_unknown_schema_falls_back_to_tenant_scope(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    _, global_store, silo_store = router_and_stores
    scoped_router = VectorStoreRouter(
        {"global": global_store, "silo": silo_store},
        tenant_scopes={"tenant-abc": "silo"},
        schema_scopes={"tenant-abc-schema": "global"},
    )

    tenant_client = scoped_router.for_tenant("tenant-abc", "unknown-schema")
    tenant_client.search("q")

    assert len(silo_store.search_calls) == 1
    assert len(global_store.search_calls) == 0


def test_router_upsert_requires_tenant_metadata(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, _, _ = router_and_stores
    chunk = Chunk(content="foo", meta={})
    with pytest.raises(ValueError):
        router.upsert_chunks([chunk])


def test_router_upsert_respects_expected_tenant(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores
    chunk = Chunk(content="foo", meta={"tenant_id": "t", "case_id": "case"})
    router.upsert_chunks([chunk], tenant_id="t")
    assert len(global_store.upsert_calls) == 1
    with pytest.raises(ValueError):
        router.upsert_chunks([chunk], tenant_id="other")


def test_router_health_check_aggregates_scope_results(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, silo_store = router_and_stores
    results = router.health_check()
    assert results == {"global": True, "silo": True}
    assert global_store.health_checks == 1
    assert silo_store.health_checks == 1


def test_tenant_client_rejects_foreign_search_tenant(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, _, _ = router_and_stores
    tenant_client = router.for_tenant("tenant-123")
    with pytest.raises(AssertionError):
        tenant_client.search("query", tenant_id="other-tenant")


def test_tenant_client_search_uses_bound_tenant(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores
    tenant_client = router.for_tenant("tenant-bound")
    tenant_client.search("query")
    assert global_store.search_calls[-1]["tenant_id"] == "tenant-bound"


def test_tenant_client_enforces_tenant_on_upsert(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores
    tenant_client = router.for_tenant("tenant-456")
    chunk = Chunk(content="foo", meta={})
    tenant_client.upsert_chunks([chunk])
    stored = global_store.upsert_calls[-1][0]
    assert stored.meta["tenant_id"] == "tenant-456"


def test_tenant_client_upsert_rejects_foreign_tenant(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, _, _ = router_and_stores
    tenant_client = router.for_tenant("tenant-789")
    chunk = Chunk(content="foo", meta={"tenant_id": "other", "case_id": "case"})
    with pytest.raises(ValueError):
        tenant_client.upsert_chunks([chunk])


def test_router_health_check_records_metrics(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Counter:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def labels(self, **labels: str) -> "_Counter":
            self.calls.append(labels)
            return self

        def inc(self, amount: float = 1.0) -> None:  # noqa: ARG002 - interface compat
            return None

    counter = _Counter()
    monkeypatch.setattr(metrics, "RAG_HEALTH_CHECKS", counter)

    router, _, _ = router_and_stores
    router.health_check()

    assert {"scope": "global", "status": "success"} in counter.calls
    assert {"scope": "silo", "status": "success"} in counter.calls


def test_router_hybrid_search_uses_scoped_store() -> None:
    tenant = "tenant-42"
    global_result = HybridSearchResult(
        chunks=[
            Chunk(
                content="global",
                meta={"tenant_id": tenant, "case_id": "case"},
            )
        ],
        vector_candidates=2,
        lexical_candidates=1,
        fused_candidates=2,
        duration_ms=1.2,
        alpha=0.7,
        min_sim=0.3,
        vec_limit=5,
        lex_limit=5,
    )
    silo_result = HybridSearchResult(
        chunks=[
            Chunk(
                content="silo",
                meta={"tenant_id": tenant, "case_id": "case"},
            )
        ],
        vector_candidates=1,
        lexical_candidates=1,
        fused_candidates=1,
        duration_ms=0.5,
        alpha=0.6,
        min_sim=0.2,
        vec_limit=4,
        lex_limit=4,
    )
    global_store = HybridEnabledStore("global", global_result)
    silo_store = HybridEnabledStore("silo", silo_result)
    router = VectorStoreRouter(
        {"global": global_store, "silo": silo_store},
        tenant_scopes={tenant: "silo"},
    )

    result = router.hybrid_search(
        "frage",
        tenant_id=tenant,
        scope="silo",
        top_k=25,
        filters={"case_id": ""},
        alpha=0.5,
        min_sim=0.1,
        vec_limit=8,
        lex_limit=3,
    )

    assert result is silo_result
    call = silo_store.hybrid_calls[-1]
    assert call["top_k"] == 10  # capped
    assert call["filters"] == {"case_id": None, "visibility": "active"}
    expected_trgm = float(get_limit_setting("RAG_TRGM_LIMIT", 0.30))
    expected_cap = int(get_limit_setting("RAG_MAX_CANDIDATES", 200))
    expected_max_candidates = normalize_max_candidates(10, None, expected_cap)
    assert call["trgm_limit"] == pytest.approx(expected_trgm)
    assert call["max_candidates"] == expected_max_candidates
    assert global_store.hybrid_calls == []

    fallback = router.hybrid_search("frage", tenant_id=tenant, scope="missing")
    assert fallback is global_result
    assert global_store.hybrid_calls[-1]["top_k"] == 5


def test_router_hybrid_search_defaults_to_active_visibility() -> None:
    tenant = "tenant-vis"
    hybrid_result = HybridSearchResult(
        chunks=[
            Chunk(
                content="doc",
                meta={"tenant_id": tenant, "case_id": "case"},
            )
        ],
        vector_candidates=1,
        lexical_candidates=1,
        fused_candidates=1,
        duration_ms=0.2,
        alpha=0.6,
        min_sim=0.3,
        vec_limit=5,
        lex_limit=5,
    )
    store = HybridEnabledStore("global", hybrid_result)
    router = VectorStoreRouter({"global": store})

    result = router.hybrid_search("frage", tenant_id=tenant, visibility="all")

    call = store.hybrid_calls[-1]
    assert call["filters"] == {"visibility": "active"}
    assert call["visibility"] == "active"
    assert call["visibility_override_allowed"] is False
    assert result.visibility == "active"


def test_router_hybrid_search_without_visibility_flag_returns_active() -> None:
    tenant = "tenant-default"
    hybrid_result = HybridSearchResult(
        chunks=[
            Chunk(
                content="doc",
                meta={"tenant_id": tenant, "case_id": "case"},
            )
        ],
        vector_candidates=1,
        lexical_candidates=1,
        fused_candidates=1,
        duration_ms=0.1,
        alpha=0.4,
        min_sim=0.2,
        vec_limit=4,
        lex_limit=4,
    )
    store = HybridEnabledStore("global", hybrid_result)
    router = VectorStoreRouter({"global": store})

    result = router.hybrid_search("frage", tenant_id=tenant)

    call = store.hybrid_calls[-1]
    assert call["filters"] == {"visibility": "active"}
    assert call["visibility"] == "active"
    assert call["visibility_override_allowed"] is False
    assert result.visibility == "active"


def test_router_hybrid_search_allows_authorized_visibility() -> None:
    tenant = "tenant-auth"
    hybrid_result = HybridSearchResult(
        chunks=[
            Chunk(
                content="doc",
                meta={"tenant_id": tenant, "case_id": "case"},
            )
        ],
        vector_candidates=1,
        lexical_candidates=0,
        fused_candidates=1,
        duration_ms=0.1,
        alpha=0.5,
        min_sim=0.2,
        vec_limit=4,
        lex_limit=4,
    )
    store = HybridEnabledStore("global", hybrid_result)
    router = VectorStoreRouter({"global": store})

    result = router.hybrid_search(
        "frage",
        tenant_id=tenant,
        visibility="deleted",
        visibility_override_allowed=True,
    )

    call = store.hybrid_calls[-1]
    assert call["filters"] == {"visibility": "deleted"}
    assert call["visibility"] == "deleted"
    assert call["visibility_override_allowed"] is True
    assert result.visibility == "deleted"


def test_router_hybrid_search_emits_retrieval_span(monkeypatch) -> None:
    tenant = "tenant-span"
    hybrid_result = HybridSearchResult(
        chunks=[
            Chunk(
                content="doc",
                meta={"tenant_id": tenant, "case_id": "case"},
            )
        ],
        vector_candidates=2,
        lexical_candidates=0,
        fused_candidates=2,
        duration_ms=0.3,
        alpha=0.5,
        min_sim=0.2,
        vec_limit=4,
        lex_limit=4,
    )
    hybrid_result.deleted_matches_blocked = 5
    store = HybridEnabledStore("global", hybrid_result)
    router = VectorStoreRouter({"global": store})

    spans: list[dict[str, object]] = []
    monkeypatch.setattr(
        "ai_core.rag.vector_store.tracing.emit_span",
        lambda **kwargs: spans.append(kwargs),
    )

    with log_context(trace_id="trace-span"):
        router.hybrid_search(
            "frage",
            tenant_id=tenant,
            visibility="deleted",
            visibility_override_allowed=True,
        )

    assert spans, "expected retrieval span to be emitted"
    span = spans[-1]
    assert span["trace_id"] == "trace-span"
    assert span["node_name"] == "rag.hybrid.search"
    metadata = span["metadata"]
    assert metadata["visibility_effective"] == "deleted"
    assert metadata["deleted_matches_blocked"] == 5


def test_router_hybrid_search_falls_back_when_not_supported() -> None:
    tenant = "tenant-99"
    fallback_chunks = [
        Chunk(content="fallback", meta={"tenant_id": tenant, "case_id": "case"})
    ]
    store = FakeStore("global", search_result=fallback_chunks)
    router = VectorStoreRouter({"global": store})

    result = router.hybrid_search("fallback", tenant_id=tenant, top_k=3)

    assert isinstance(result, HybridSearchResult)
    assert result.chunks == fallback_chunks
    assert result.vector_candidates == len(fallback_chunks)
    assert result.lexical_candidates == 0
    assert result.fused_candidates == len(fallback_chunks)
    assert result.visibility == "active"
    assert store.search_calls[-1]["top_k"] == 3
    assert store.search_calls[-1]["filters"]["visibility"] == "active"


def test_router_hybrid_search_rejects_small_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = "tenant-10"
    store = FakeStore(
        "global",
        search_result=[
            Chunk(content="x", meta={"tenant_id": tenant, "case_id": "case"})
        ],
    )
    router = VectorStoreRouter({"global": store})

    from ai_core.rag import vector_store as vector_store_module

    emitted: list[RouterInputError] = []

    monkeypatch.setenv("RAG_CANDIDATE_POLICY", "error")
    monkeypatch.setattr(
        vector_store_module,
        "emit_router_validation_failure",
        lambda error: emitted.append(error),
    )

    with pytest.raises(RouterInputError) as excinfo:
        router.hybrid_search(
            "frage",
            tenant_id=tenant,
            top_k=7,
            max_candidates=3,
        )

    assert excinfo.value.code == RouterInputErrorCode.MAX_CANDIDATES_LT_TOP_K
    assert emitted
    assert emitted[0].code == RouterInputErrorCode.MAX_CANDIDATES_LT_TOP_K


def test_router_hybrid_search_normalizes_small_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = "tenant-11"
    hybrid_result = HybridSearchResult(
        chunks=[
            Chunk(
                content="normalized",
                meta={"tenant_id": tenant, "case_id": "case"},
            )
        ],
        vector_candidates=1,
        lexical_candidates=1,
        fused_candidates=1,
        duration_ms=0.2,
        alpha=0.6,
        min_sim=0.3,
        vec_limit=5,
        lex_limit=4,
    )
    store = HybridEnabledStore("global", hybrid_result)
    router = VectorStoreRouter({"global": store})

    monkeypatch.setenv("RAG_CANDIDATE_POLICY", "normalize")

    result = router.hybrid_search(
        "frage",
        tenant_id=tenant,
        top_k=7,
        max_candidates=3,
    )

    assert result is hybrid_result
    call = store.hybrid_calls[-1]
    assert call["max_candidates"] == 7
    assert call["top_k"] == 7


def test_router_logs_warning_when_hybrid_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tenant = "tenant-88"

    class _NullHybridStore(FakeStore):
        def hybrid_search(
            self,
            query: str,
            tenant_id: str,
            *,
            case_id: str | None = None,
            top_k: int = 5,
            filters: Mapping[str, object | None] | None = None,
            alpha: float | None = None,
            min_sim: float | None = None,
            vec_limit: int | None = None,
            lex_limit: int | None = None,
            trgm_limit: float | None = None,
            max_candidates: int | None = None,
            visibility: str | None = None,
            visibility_override_allowed: bool = False,
        ) -> HybridSearchResult | None:
            # Simulate a store that supports hybrid_search but returns no result.
            return None

    store = _NullHybridStore(
        "global",
        search_result=[
            Chunk(content="fallback", meta={"tenant_id": tenant, "case_id": "case"})
        ],
    )
    router = VectorStoreRouter({"global": store})

    with caplog.at_level(logging.WARNING):
        result = router.hybrid_search("frage", tenant_id=tenant, top_k=2)

    assert isinstance(result, HybridSearchResult)
    assert result.chunks
    warning_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "rag.hybrid.router.no_result"
    ]
    assert warning_logs
    assert warning_logs[0].scope == "global"
