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
        chunk = Chunk(content="text", meta={"hash": "h"}, embedding=[0.0] * vector_client.EMBEDDING_DIM)
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
