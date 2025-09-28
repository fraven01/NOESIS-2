import uuid

import pytest

from ai_core.rag import metrics, vector_client
from ai_core.rag.schemas import Chunk


@pytest.mark.usefixtures("rag_database")
class TestPgVectorClient:
    def setup_method(self) -> None:
        vector_client.reset_default_client()

    def teardown_method(self) -> None:
        vector_client.reset_default_client()

    def test_missing_tenant_raises(self):
        client = vector_client.get_default_client()
        chunk = Chunk(
            content="text",
            meta={"hash": "h"},
            embedding=[0.0] * vector_client.EMBEDDING_DIM,
        )
        with pytest.raises(ValueError):
            client.upsert_chunks([chunk])

    def test_missing_hash_raises(self):
        client = vector_client.get_default_client()
        chunk = Chunk(
            content="text",
            meta={"tenant": str(uuid.uuid4())},
            embedding=[0.0] * vector_client.EMBEDDING_DIM,
        )
        with pytest.raises(ValueError):
            client.upsert_chunks([chunk])

    def test_upsert_records_metrics_and_handles_legacy_tenant(self, monkeypatch):
        class _Counter:
            def __init__(self) -> None:
                self.value = 0

            def inc(self, amount: float = 1.0) -> None:
                self.value += amount

        class _Histogram:
            def __init__(self) -> None:
                self.samples = []

            def observe(self, amount: float) -> None:
                self.samples.append(amount)

        counter = _Counter()
        histogram = _Histogram()
        monkeypatch.setattr(metrics, "RAG_UPSERT_CHUNKS", counter)
        monkeypatch.setattr(metrics, "RAG_SEARCH_MS", histogram)

        client = vector_client.get_default_client()
        chunk = Chunk(
            content="legacy",
            meta={"tenant": "tenant-1", "hash": "h", "case": "c", "source": "s"},
            embedding=[0.1] * vector_client.EMBEDDING_DIM,
        )
        written = client.upsert_chunks([chunk])
        assert written == 1
        assert counter.value == 1

        results = client.search(
            "legacy",
            tenant_id=chunk.meta["tenant"],
            filters={"case": None},
            top_k=1,
        )
        assert len(results) == 1
        assert histogram.samples
        assert uuid.UUID(results[0].meta["tenant"])  # tenant ids are normalised
        assert 0.0 <= results[0].meta["score"] <= 1.0
        assert results[0].meta.get("hash") == chunk.meta["hash"]

    def test_upsert_replaces_existing_chunks_in_batches(self):
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = "batched"

        def _make_chunk(content: str) -> Chunk:
            return Chunk(
                content=content,
                meta={"tenant": tenant, "hash": doc_hash, "source": "s"},
                embedding=[0.01] * vector_client.EMBEDDING_DIM,
            )

        initial_chunks = [_make_chunk(f"chunk-{idx}") for idx in range(3)]
        written = client.upsert_chunks(initial_chunks)
        assert written == 3

        with client._connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SELECT text, ord, tokens FROM chunks ORDER BY ord")
                rows = cur.fetchall()
                assert [row[0] for row in rows] == [c.content for c in initial_chunks]
                assert [row[1] for row in rows] == [0, 1, 2]
                assert all(row[2] > 0 for row in rows)
                cur.execute("SELECT COUNT(*) FROM embeddings")
                assert cur.fetchone()[0] == 3
                cur.execute("SELECT COUNT(*) FROM documents")
                assert cur.fetchone()[0] == 1

        replacement_chunks = [_make_chunk(f"replacement-{idx}") for idx in range(2)]
        written = client.upsert_chunks(replacement_chunks)
        assert written == 2

        with client._connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SELECT text FROM chunks ORDER BY ord")
                rows = cur.fetchall()
                assert [row[0] for row in rows] == [
                    c.content for c in replacement_chunks
                ]
                cur.execute("SELECT COUNT(*) FROM embeddings")
                assert cur.fetchone()[0] == 2

    def test_search_caps_top_k(self):
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        chunk_hash = "limit"

        chunks = [
            Chunk(
                content="Result",
                meta={"tenant": tenant, "hash": chunk_hash, "source": "src"},
                embedding=[0.02] * vector_client.EMBEDDING_DIM,
            )
            for index in range(12)
        ]

        written = client.upsert_chunks(chunks)
        assert written == len(chunks)

        results = client.search("Result", tenant_id=tenant, top_k=25)
        assert len(results) == 10

    def test_search_applies_metadata_filters(self):
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        chunks = [
            Chunk(
                content="Filtered",
                meta={
                    "tenant": tenant,
                    "hash": "doc-a",
                    "source": "alpha",
                    "doctype": "contract",
                },
                embedding=[0.03] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
            ),
            Chunk(
                content="Filtered",
                meta={
                    "tenant": tenant,
                    "hash": "doc-b",
                    "source": "beta",
                    "doctype": "contract",
                },
                embedding=[0.03] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
            ),
        ]
        written = client.upsert_chunks(chunks)
        assert written == len(chunks)

        results = client.search(
            "Filtered",
            tenant_id=tenant,
            filters={"source": "alpha"},
            top_k=5,
        )
        assert [chunk.meta.get("hash") for chunk in results] == ["doc-a"]
        assert all(chunk.meta.get("source") == "alpha" for chunk in results)

        empty = client.search(
            "Filtered",
            tenant_id=tenant,
            filters={"source": "gamma"},
            top_k=5,
        )
        assert empty == []

    def test_search_supports_boolean_filters(self):
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        chunk = Chunk(
            content="Boolean",
            meta={
                "tenant": tenant,
                "hash": "doc-bool",
                "source": "gamma",
                "published": True,
            },
            embedding=[0.04] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
        )
        client.upsert_chunks([chunk])

        positive = client.search(
            "Boolean",
            tenant_id=tenant,
            filters={"published": True},
            top_k=5,
        )
        assert [c.meta.get("hash") for c in positive] == ["doc-bool"]

        negative = client.search(
            "Boolean",
            tenant_id=tenant,
            filters={"published": False},
            top_k=5,
        )
        assert negative == []


    def test_search_ignores_unknown_filters(self):
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        chunk = Chunk(
            content="Filter tolerant",
            meta={
                "tenant": tenant,
                "hash": "doc-filter-tolerant",
                "source": "alpha",
            },
            embedding=[0.05] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
        )
        client.upsert_chunks([chunk])

        results = client.search(
            "Filter tolerant",
            tenant_id=tenant,
            filters={
                "source": "alpha",
                "project_id": "legacy",  # unbekannter Key, sollte ignoriert werden
                "some_unknown_key": "value",
            },
            top_k=5,
        )

        assert results, "Erwartet mindestens ein Ergebnis trotz unbekannter Filter"
        assert any(r.meta.get("hash") == "doc-filter-tolerant" for r in results)


    def test_retry_metrics_record_attempts(self, monkeypatch):
        client = vector_client.get_default_client()
        client._retries = 2  # type: ignore[attr-defined]

        class _RetryCounter:
            def __init__(self) -> None:
                self.calls: list[dict[str, str]] = []
                self.value = 0.0

            def labels(self, **labels: str) -> "_RetryCounter":
                self.calls.append(labels)
                return self

            def inc(self, amount: float = 1.0) -> None:
                self.value += amount

        counter = _RetryCounter()
        monkeypatch.setattr(metrics, "RAG_RETRY_ATTEMPTS", counter)
        monkeypatch.setattr(vector_client.time, "sleep", lambda _x: None)

        attempts = {"count": 0}

        def _sometimes_fails() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("transient")
            return "ok"

        result = client._run_with_retries(_sometimes_fails, op_name="search")
        assert result == "ok"
        assert counter.value == 1.0
        assert counter.calls == [{"operation": "search"}]

    def test_health_check_runs_simple_query(self):
        client = vector_client.get_default_client()
        assert client.health_check() is True
