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

        results = client.search("legacy", {"tenant": None, "case": None}, top_k=1)
        assert len(results) == 1
        assert histogram.samples
        assert uuid.UUID(results[0].meta["tenant"])  # tenant ids are normalised

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
                cur.execute(
                    "SELECT text, ord, tokens FROM chunks ORDER BY ord"
                )
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
                assert [row[0] for row in rows] == [c.content for c in replacement_chunks]
                cur.execute("SELECT COUNT(*) FROM embeddings")
                assert cur.fetchone()[0] == 2
