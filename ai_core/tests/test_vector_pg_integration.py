"""Integration test hitting the real pgvector backend via the router."""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("psycopg2", reason="pgvector backing store requires psycopg2")

from ai_core.rag import get_default_router, vector_client
from ai_core.rag.schemas import Chunk


pytestmark = pytest.mark.usefixtures("rag_database")


def _make_chunk(tenant_id: str, ordinal: int, *, hash_id: str) -> Chunk:
    base_value = float(ordinal + 1)
    return Chunk(
        content=f"tenant-{tenant_id}-chunk-{ordinal}",
        meta={
            "tenant": tenant_id,
            "hash": hash_id,
            "source": "integration-test",
            "external_id": f"{hash_id}-external",
        },
        embedding=[base_value] + [0.0] * (vector_client.get_embedding_dim() - 1),
    )


def test_router_roundtrip_with_pgvector_backend(monkeypatch) -> None:
    """Ensure the router can write and read chunks with tenant isolation."""

    dsn = os.environ.get("RAG_DATABASE_URL") or os.environ.get(
        "AI_CORE_TEST_DATABASE_URL"
    )
    if not dsn:
        pytest.skip("RAG test database DSN not configured")

    monkeypatch.setattr(vector_client, "get_embedding_dim", lambda: 1536)

    vector_client.reset_default_client()
    router = get_default_router()

    tenant_id = str(uuid.uuid4())
    other_tenant_id = str(uuid.uuid4())

    tenant_chunks = [
        _make_chunk(tenant_id, idx, hash_id=f"doc-{tenant_id}") for idx in range(3)
    ]
    other_chunks = [
        _make_chunk(other_tenant_id, idx, hash_id=f"doc-{other_tenant_id}")
        for idx in range(2)
    ]

    try:
        router.upsert_chunks([*tenant_chunks, *other_chunks])

        results = router.search(
            "tenant integration query",
            tenant_id=tenant_id,
            top_k=25,
        )
        assert len(results) == len(tenant_chunks)
        assert len(results) <= 10
        assert {chunk.meta.get("tenant") for chunk in results} == {tenant_id}
        assert all("hash" in chunk.meta for chunk in results)
        assert all(0.0 <= chunk.meta.get("score", 0.0) <= 1.0 for chunk in results)

        isolated_results = router.search(
            "tenant integration query",
            tenant_id=other_tenant_id,
            top_k=25,
        )
        assert len(isolated_results) == len(other_chunks)
        assert {chunk.meta.get("tenant") for chunk in isolated_results} == {
            other_tenant_id
        }

        empty_results = router.search(
            "tenant integration query",
            tenant_id=str(uuid.uuid4()),
            top_k=5,
        )
        assert empty_results == []
    finally:
        router.close()
        vector_client.reset_default_client()
