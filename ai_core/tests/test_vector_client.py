import hashlib
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
            meta={"hash": "h", "external_id": "ext-1"},
            embedding=[0.0] * vector_client.EMBEDDING_DIM,
        )
        with pytest.raises(ValueError):
            client.upsert_chunks([chunk])

    def test_missing_hash_raises(self):
        client = vector_client.get_default_client()
        chunk = Chunk(
            content="text",
            meta={"tenant": str(uuid.uuid4()), "external_id": "ext-1"},
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
        inserted_counter = _Counter()
        replaced_counter = _Counter()
        skipped_counter = _Counter()
        chunks_counter = _Counter()
        monkeypatch.setattr(metrics, "INGESTION_DOCS_INSERTED", inserted_counter)
        monkeypatch.setattr(metrics, "INGESTION_DOCS_REPLACED", replaced_counter)
        monkeypatch.setattr(metrics, "INGESTION_DOCS_SKIPPED", skipped_counter)
        monkeypatch.setattr(metrics, "INGESTION_CHUNKS_WRITTEN", chunks_counter)

        client = vector_client.get_default_client()
        doc_hash = hashlib.sha256(b"legacy").hexdigest()
        chunk = Chunk(
            content="legacy",
            meta={
                "tenant": "tenant-1",
                "hash": doc_hash,
                "case": "c",
                "source": "s",
                "external_id": "legacy-doc",
            },
            embedding=[0.1] * vector_client.EMBEDDING_DIM,
        )
        written = client.upsert_chunks([chunk])
        assert written == 1
        assert counter.value == 1
        assert inserted_counter.value == 1
        assert replaced_counter.value == 0
        assert skipped_counter.value == 0
        assert chunks_counter.value == 1

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
        assert results[0].meta.get("external_id") == "legacy-doc"

    def test_upsert_skips_when_hash_unchanged(self, monkeypatch):
        class _Counter:
            def __init__(self) -> None:
                self.value = 0

            def inc(self, amount: float = 1.0) -> None:
                self.value += amount

        inserted_counter = _Counter()
        replaced_counter = _Counter()
        skipped_counter = _Counter()
        chunks_counter = _Counter()
        monkeypatch.setattr(metrics, "INGESTION_DOCS_INSERTED", inserted_counter)
        monkeypatch.setattr(metrics, "INGESTION_DOCS_REPLACED", replaced_counter)
        monkeypatch.setattr(metrics, "INGESTION_DOCS_SKIPPED", skipped_counter)
        monkeypatch.setattr(metrics, "INGESTION_CHUNKS_WRITTEN", chunks_counter)
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        external_id = "doc-1"
        doc_hash = hashlib.sha256(b"version-1").hexdigest()

        chunk = Chunk(
            content="chunk-1",
            meta={
                "tenant": tenant,
                "hash": doc_hash,
                "source": "s",
                "external_id": external_id,
            },
            embedding=[0.2] * vector_client.EMBEDDING_DIM,
        )
        written = client.upsert_chunks([chunk])
        assert written == 1
        assert inserted_counter.value == 1
        assert skipped_counter.value == 0

        # Re-ingest with identical hash should be skipped
        written_again = client.upsert_chunks([chunk])
        assert written_again == 0
        assert skipped_counter.value == 1
        assert replaced_counter.value == 0
        assert chunks_counter.value == 1

        with client._connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks")
                assert cur.fetchone()[0] == 1

    def test_upsert_replaces_existing_chunks_in_batches(self, monkeypatch):
        class _Counter:
            def __init__(self) -> None:
                self.value = 0

            def inc(self, amount: float = 1.0) -> None:
                self.value += amount

        inserted_counter = _Counter()
        replaced_counter = _Counter()
        skipped_counter = _Counter()
        chunks_counter = _Counter()
        monkeypatch.setattr(metrics, "INGESTION_DOCS_INSERTED", inserted_counter)
        monkeypatch.setattr(metrics, "INGESTION_DOCS_REPLACED", replaced_counter)
        monkeypatch.setattr(metrics, "INGESTION_DOCS_SKIPPED", skipped_counter)
        monkeypatch.setattr(metrics, "INGESTION_CHUNKS_WRITTEN", chunks_counter)
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        external_id = "doc-2"
        doc_hash_v1 = hashlib.sha256(b"version-a").hexdigest()
        doc_hash_v2 = hashlib.sha256(b"version-b").hexdigest()

        def _make_chunk(content: str, doc_hash: str) -> Chunk:
            return Chunk(
                content=content,
                meta={
                    "tenant": tenant,
                    "hash": doc_hash,
                    "source": "s",
                    "external_id": external_id,
                },
                embedding=[0.01] * vector_client.EMBEDDING_DIM,
            )

        initial_chunks = [_make_chunk(f"chunk-{idx}", doc_hash_v1) for idx in range(3)]
        written = client.upsert_chunks(initial_chunks)
        assert written == 3
        assert inserted_counter.value == 1
        assert replaced_counter.value == 0
        assert skipped_counter.value == 0
        assert chunks_counter.value == 3

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

        replacement_chunks = [
            _make_chunk(f"replacement-{idx}", doc_hash_v2) for idx in range(2)
        ]
        written = client.upsert_chunks(replacement_chunks)
        assert written == 2
        assert replaced_counter.value == 1
        assert skipped_counter.value == 0
        assert chunks_counter.value == 5

        with client._connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SELECT text FROM chunks ORDER BY ord")
                rows = cur.fetchall()
                assert [row[0] for row in rows] == [
                    c.content for c in replacement_chunks
                ]
                cur.execute("SELECT COUNT(*) FROM embeddings")
                assert cur.fetchone()[0] == 2

    def test_vector_client_deduplication_skip_and_replace(self):
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        external_id = "doc-dedupe"

        def _make_chunk(content: str) -> Chunk:
            doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            return Chunk(
                content=content,
                meta={
                    "tenant": tenant,
                    "external_id": external_id,
                    "hash": doc_hash,
                    "source": "unit",
                },
                embedding=[0.2] * vector_client.EMBEDDING_DIM,
            )

        initial_chunk = _make_chunk("original content")
        written = client.upsert_chunks([initial_chunk])
        assert written == 1

        initial_results = client.search(
            "original",
            tenant_id=tenant,
            filters={"case": None},
            top_k=3,
        )
        assert len(initial_results) == 1
        assert initial_results[0].content == "original content"

        # Re-ingest with identical content: should be skipped
        duplicate_chunk = _make_chunk("original content")
        written_again = client.upsert_chunks([duplicate_chunk])
        assert written_again == 0

        duplicate_results = client.search(
            "original",
            tenant_id=tenant,
            filters={"case": None},
            top_k=3,
        )
        assert len(duplicate_results) == 1
        assert duplicate_results[0].content == "original content"

        # Change content with the same external_id: should replace previous chunks
        replacement_chunk = _make_chunk("updated content")
        written_replacement = client.upsert_chunks([replacement_chunk])
        assert written_replacement == 1

        updated_results = client.search(
            "updated",
            tenant_id=tenant,
            filters={"case": None},
            top_k=3,
        )
        assert len(updated_results) == 1
        assert updated_results[0].content == "updated content"
        assert (
            updated_results[0].meta.get("hash")
            == hashlib.sha256(b"updated content").hexdigest()
        )

        with client._connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks")
                assert cur.fetchone()[0] == 1
                cur.execute("SELECT COUNT(*) FROM embeddings")
                assert cur.fetchone()[0] == 1

    def test_search_caps_top_k(self):
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        chunk_hash = hashlib.sha256(b"limit").hexdigest()

        chunks = [
            Chunk(
                content="Result",
                meta={
                    "tenant": tenant,
                    "hash": chunk_hash,
                    "source": "src",
                    "external_id": f"doc-{index}",
                },
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
                    "hash": hashlib.sha256(b"doc-a").hexdigest(),
                    "source": "alpha",
                    "doctype": "contract",
                    "external_id": "doc-a",
                },
                embedding=[0.03] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
            ),
            Chunk(
                content="Filtered",
                meta={
                    "tenant": tenant,
                    "hash": hashlib.sha256(b"doc-b").hexdigest(),
                    "source": "beta",
                    "doctype": "contract",
                    "external_id": "doc-b",
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
        expected_hash = hashlib.sha256(b"doc-a").hexdigest()
        assert [chunk.meta.get("hash") for chunk in results] == [expected_hash]
        assert all(chunk.meta.get("source") == "alpha" for chunk in results)
        assert all(chunk.meta.get("external_id") == "doc-a" for chunk in results)

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
                "hash": hashlib.sha256(b"doc-bool").hexdigest(),
                "source": "gamma",
                "published": True,
                "external_id": "doc-bool",
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
        expected_hash = hashlib.sha256(b"doc-bool").hexdigest()
        assert [c.meta.get("hash") for c in positive] == [expected_hash]

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

    def test_hybrid_search_falls_back_on_empty_query_embedding(
        self, monkeypatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"lexical").hexdigest()
        chunk = Chunk(
            content="Lexical fallback example",
            meta={
                "tenant": tenant,
                "hash": doc_hash,
                "source": "lexical",
                "external_id": "doc-lex",
            },
            embedding=[0.3] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
        )
        client.upsert_chunks([chunk])

        class _CounterVec:
            def __init__(self) -> None:
                self.calls: list[dict[str, str]] = []
                self.value = 0.0

            def labels(self, **labels: str) -> "_CounterVec":
                self.calls.append(labels)
                return self

            def inc(self, amount: float = 1.0) -> None:
                self.value += amount

        counter = _CounterVec()
        monkeypatch.setattr(metrics, "RAG_QUERY_EMPTY_VEC_TOTAL", counter)
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.EMBEDDING_DIM,
        )

        from structlog.testing import capture_logs

        with capture_logs() as logs:
            result = client.hybrid_search(
                "Lexical fallback example",
                tenant_id=tenant,
                filters={"case": None},
                top_k=3,
            )

        assert result.vector_candidates == 0
        assert result.lexical_candidates >= 1
        assert result.query_embedding_empty is True
        assert counter.value == 1.0
        assert counter.calls == [{"tenant": tenant}]
        assert result.chunks
        events = [entry["event"] for entry in logs]
        assert "query.embedding.empty_fallback" in events

    def test_hybrid_search_reports_cutoff_statistics(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"cutoff").hexdigest()
        chunk = Chunk(
            content="Cutoff candidate example",
            meta={
                "tenant": tenant,
                "hash": doc_hash,
                "source": "cutoff",
                "external_id": "doc-cutoff",
            },
            embedding=[0.25] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
        )
        client.upsert_chunks([chunk])

        class _CounterVec:
            def __init__(self) -> None:
                self.calls: list[dict[str, str]] = []
                self.value = 0.0

            def labels(self, **labels: str) -> "_CounterVec":
                self.calls.append(labels)
                return self

            def inc(self, amount: float = 1.0) -> None:
                self.value += amount

        cutoff_counter = _CounterVec()
        monkeypatch.setattr(metrics, "RAG_QUERY_BELOW_CUTOFF_TOTAL", cutoff_counter)

        result = client.hybrid_search(
            "candidate cutoff",
            tenant_id=tenant,
            filters={"case": None},
            top_k=3,
            alpha=0.8,
            min_sim=0.95,
        )

        assert result.fused_candidates >= 1
        assert result.below_cutoff >= 1
        assert result.returned_after_cutoff == 0
        assert result.chunks == []
        assert cutoff_counter.calls == [{"tenant": tenant}]
        assert cutoff_counter.value == float(result.below_cutoff)
