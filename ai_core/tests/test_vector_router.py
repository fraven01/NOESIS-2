"""Unit tests for the vector store router abstraction."""

from __future__ import annotations

from typing import Iterable, Mapping

import pytest

from ai_core.rag import metrics
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_store import VectorStore, VectorStoreRouter


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


@pytest.fixture
def router_and_stores() -> tuple[VectorStoreRouter, FakeStore, FakeStore]:
    global_store = FakeStore(
        "global",
        search_result=[Chunk(content="global", meta={"tenant": "t"})],
    )
    silo_store = FakeStore(
        "silo",
        search_result=[Chunk(content="silo", meta={"tenant": "t"})],
    )
    router = VectorStoreRouter({"global": global_store, "silo": silo_store})
    return router, global_store, silo_store


def test_router_requires_tenant_id(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, _, _ = router_and_stores
    with pytest.raises(ValueError, match="tenant_id is required"):
        router.search("query", tenant_id="")


def test_router_caps_top_k_to_10(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores
    router.search("query", tenant_id="tenant", top_k=25)
    assert global_store.search_calls[-1]["top_k"] == 10


def test_router_normalizes_empty_filters(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, _ = router_and_stores
    router.search(
        "query",
        tenant_id="tenant",
        filters={"case": "", "source": "doc", "extra": ""},
    )
    filters = global_store.search_calls[-1]["filters"]
    assert filters == {"case": None, "source": "doc", "extra": None}


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
    chunk = Chunk(content="foo", meta={"tenant": "t"})
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
    assert len(global_store.search_calls) == 0


def test_router_upsert_requires_tenant_metadata(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, _, _ = router_and_stores
    chunk = Chunk(content="foo", meta={})
    with pytest.raises(ValueError):
        router.upsert_chunks([chunk])


def test_router_health_check_aggregates_scope_results(
    router_and_stores: tuple[VectorStoreRouter, FakeStore, FakeStore],
) -> None:
    router, global_store, silo_store = router_and_stores
    results = router.health_check()
    assert results == {"global": True, "silo": True}
    assert global_store.health_checks == 1
    assert silo_store.health_checks == 1


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
