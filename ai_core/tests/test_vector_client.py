import hashlib
import threading
import uuid
from collections.abc import Sequence

import pytest
from psycopg2.errors import UndefinedTable

from ai_core.rag import metrics, vector_client
from ai_core.rag.schemas import Chunk
from structlog.testing import capture_logs


@pytest.mark.usefixtures("rag_database")
class TestPgVectorClient:
    def setup_method(self) -> None:
        vector_client.reset_default_client()

    def teardown_method(self) -> None:
        vector_client.reset_default_client()

    def _run_adaptive_fallback(
        self,
        client: vector_client.PgVectorClient,
        monkeypatch: pytest.MonkeyPatch,
        tenant: str,
        doc_hash: str,
        *,
        success_limit: float,
        applied_limit: float,
    ) -> tuple[vector_client.HybridSearchResult, list[dict[str, object]]]:
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.get_embedding_dim(),
        )

        doc_id = uuid.uuid4()
        normalized_tenant = str(client._coerce_tenant_uuid(tenant))

        class _FallbackCursor:
            def __init__(self) -> None:
                self._fetchall_result: list[tuple] = []
                self._fetchone_result: tuple | None = None

            def __enter__(self) -> "_FallbackCursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def execute(self, sql: str, params=None) -> None:
                if "pg_catalog.pg_opclass" in sql:
                    self._fetchone_result = (1,)
                    return
                if "SET LOCAL statement_timeout" in sql:
                    return
                if "SELECT set_limit" in sql:
                    self._fetchone_result = None
                    return
                if "SELECT show_limit()" in sql:
                    self._fetchone_result = (applied_limit,)
                    return
                if "c.text_norm % %s" in sql:
                    self._fetchall_result = []
                    return
                if "similarity(c.text_norm, %s) >= %s" in sql:
                    limit_value = float(params[-2])
                    if limit_value <= success_limit + 1e-9:
                        self._fetchall_result = [
                            (
                                "chunk-adaptive",
                                "Lexical adaptive",
                                {
                                    "tenant_id": normalized_tenant,
                                    "hash": doc_hash,
                                    "external_id": "adaptive-doc",
                                },
                                doc_hash,
                                doc_id,
                                0.7,
                            )
                        ]
                    else:
                        self._fetchall_result = []
                    return
                self._fetchall_result = []

            def fetchall(self) -> list[tuple]:
                result = list(self._fetchall_result)
                self._fetchall_result = []
                return result

            def fetchone(self):
                result = self._fetchone_result
                self._fetchone_result = None
                return result

        class _FakeConnection:
            def cursor(self) -> _FallbackCursor:
                return _FallbackCursor()

            def rollback(self) -> None:
                return None

        def _fake_connection(self):  # type: ignore[no-untyped-def]
            fake = _FakeConnection()

            class _Ctx:
                def __enter__(self_inner):
                    return fake

                def __exit__(self_inner, exc_type, exc, tb):
                    return None

            return _Ctx()

        monkeypatch.setattr(
            client, "_connection", _fake_connection.__get__(client, type(client))
        )

        with capture_logs() as logs:
            result = client.hybrid_search(
                "adaptive fallback",
                tenant_id=tenant,
                filters={},
                top_k=1,
                alpha=0.0,
                trgm_limit=applied_limit,
            )

        return result, logs

    def test_missing_tenant_raises(self):
        client = vector_client.get_default_client()
        chunk = Chunk(
            content="text",
            meta={"hash": "h", "external_id": "ext-1"},
            embedding=[0.0] * vector_client.get_embedding_dim(),
        )
        with pytest.raises(ValueError):
            client.upsert_chunks([chunk])

    def test_missing_hash_raises(self):
        client = vector_client.get_default_client()
        chunk = Chunk(
            content="text",
            meta={
                "tenant_id": str(uuid.uuid4()),
                "case_id": "case-1",
                "external_id": "ext-1",
            },
            embedding=[0.0] * vector_client.get_embedding_dim(),
        )
        with pytest.raises(ValueError):
            client.upsert_chunks([chunk])

    def test_missing_external_id_falls_back_to_hash(self) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"missing-external").hexdigest()
        chunk = Chunk(
            content="missing external id",
            meta={
                "tenant_id": tenant,
                "hash": doc_hash,
                "case_id": "case-fallback",
                "source": "example",
            },
            embedding=[0.25] + [0.0] * (vector_client.get_embedding_dim() - 1),
        )

        written = client.upsert_chunks([chunk])
        assert written == 1

        results = client.search(
            "missing external id",
            tenant_id=tenant,
            filters={"case_id": None},
            top_k=1,
        )

        assert results
        assert results[0].meta["external_id"] == doc_hash

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
                "tenant_id": "tenant-1",
                "hash": doc_hash,
                "case_id": "c",
                "source": "s",
                "external_id": "legacy-doc",
            },
            embedding=[0.1] * vector_client.get_embedding_dim(),
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
            tenant_id=chunk.meta["tenant_id"],
            filters={"case_id": None},
            top_k=1,
        )
        assert len(results) == 1
        assert histogram.samples

    def test_hybrid_search_normalises_textual_tenant_ids(self) -> None:
        client = vector_client.get_default_client()
        tenant_slug = "tenant-case"
        doc_hash = hashlib.sha256(b"tenant-case").hexdigest()
        chunk = Chunk(
            content="tenant case document",
            meta={
                "tenant_id": tenant_slug,
                "hash": doc_hash,
                "case_id": "case-normalised",
                "source": "case-test",
                "external_id": "tenant-case-doc",
            },
            embedding=[0.1] + [0.0] * (vector_client.get_embedding_dim() - 1),
        )

        written = client.upsert_chunks([chunk])
        assert written == 1

        results = client.search(
            "tenant case document",
            tenant_id=tenant_slug.upper(),
            filters={"case_id": None},
            top_k=1,
        )

        assert len(results) == 1
        assert uuid.UUID(results[0].meta["tenant_id"])  # tenant ids are normalised
        assert 0.0 <= results[0].meta["score"] <= 1.0
        assert results[0].meta.get("hash") == chunk.meta["hash"]
        assert results[0].meta.get("external_id") == chunk.meta["external_id"]

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
                "tenant_id": tenant,
                "hash": doc_hash,
                "source": "s",
                "external_id": external_id,
            },
            embedding=[0.2] * vector_client.get_embedding_dim(),
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
                    "tenant_id": tenant,
                    "hash": doc_hash,
                    "source": "s",
                    "external_id": external_id,
                },
                embedding=[0.01] * vector_client.get_embedding_dim(),
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
                    "tenant_id": tenant,
                    "external_id": external_id,
                    "hash": doc_hash,
                    "source": "unit",
                },
                embedding=[0.2] * vector_client.get_embedding_dim(),
            )

        initial_chunk = _make_chunk("original content")
        written = client.upsert_chunks([initial_chunk])
        assert written == 1

        initial_results = client.search(
            "original",
            tenant_id=tenant,
            filters={"case_id": None},
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
            filters={"case_id": None},
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
            filters={"case_id": None},
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
                    "tenant_id": tenant,
                    "hash": chunk_hash,
                    "source": "src",
                    "external_id": f"doc-{index}",
                },
                embedding=[0.02] * vector_client.get_embedding_dim(),
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
                    "tenant_id": tenant,
                    "hash": hashlib.sha256(b"doc-a").hexdigest(),
                    "source": "alpha",
                    "doctype": "contract",
                    "external_id": "doc-a",
                },
                embedding=[0.03] + [0.0] * (vector_client.get_embedding_dim() - 1),
            ),
            Chunk(
                content="Filtered",
                meta={
                    "tenant_id": tenant,
                    "hash": hashlib.sha256(b"doc-b").hexdigest(),
                    "source": "beta",
                    "doctype": "contract",
                    "external_id": "doc-b",
                },
                embedding=[0.03] + [0.0] * (vector_client.get_embedding_dim() - 1),
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
                "tenant_id": tenant,
                "hash": hashlib.sha256(b"doc-bool").hexdigest(),
                "source": "gamma",
                "published": True,
                "external_id": "doc-bool",
            },
            embedding=[0.04] + [0.0] * (vector_client.get_embedding_dim() - 1),
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
                "tenant_id": tenant,
                "hash": "doc-filter-tolerant",
                "source": "alpha",
            },
            embedding=[0.05] + [0.0] * (vector_client.get_embedding_dim() - 1),
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
                "tenant_id": tenant,
                "hash": doc_hash,
                "source": "lexical",
                "external_id": "doc-lex",
            },
            embedding=[0.3] + [0.0] * (vector_client.get_embedding_dim() - 1),
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
            lambda self, _query: [0.0] * vector_client.get_embedding_dim(),
        )

        with capture_logs() as logs:
            result = client.hybrid_search(
                "Lexical fallback example",
                tenant_id=tenant,
                filters={"case_id": None},
                top_k=3,
                alpha=0.0,
            )

        assert result.vector_candidates == 0
        assert result.lexical_candidates >= 1
        assert result.query_embedding_empty is True
        assert counter.value == 1.0
        assert counter.calls == [{"tenant_id": tenant}]
        assert result.chunks
        top_chunk = result.chunks[0]
        assert top_chunk.meta["vscore"] == 0.0
        assert top_chunk.meta["score"] == pytest.approx(top_chunk.meta["lscore"])
        assert [entry["event"] for entry in logs].count(
            "rag.hybrid.null_embedding"
        ) == 1
        logged = [
            entry for entry in logs if entry["event"] == "rag.hybrid.null_embedding"
        ][0]
        assert logged["alpha"] == 0.0
        assert logged["tenant_id"] == tenant
        assert "case_id" in logged
        assert logged["case_id"] is None

    def test_hybrid_search_lexical_matches_database_normalisation(
        self, monkeypatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"zebra-gurke").hexdigest()
        chunk = Chunk(
            content="ZEBRAGURKE",
            meta={
                "tenant_id": tenant,
                "hash": doc_hash,
                "case_id": "lexical-case",
                "source": "lexical",
                "external_id": "lex-1",
            },
            embedding=[0.0] * vector_client.get_embedding_dim(),
        )
        client.upsert_chunks([chunk])
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.get_embedding_dim(),
        )

        result = client.hybrid_search(
            "ZEBRAGURKE",
            tenant_id=tenant,
            filters={"case_id": "lexical-case"},
            alpha=0.0,
            top_k=1,
        )

        assert result.lexical_candidates >= 1
        assert result.chunks
        top_chunk = result.chunks[0]
        assert top_chunk.meta["hash"] == doc_hash
        assert top_chunk.meta["lscore"] == pytest.approx(1.0)

    def test_hybrid_search_reports_cutoff_statistics(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"cutoff").hexdigest()
        chunk = Chunk(
            content="Cutoff candidate example",
            meta={
                "tenant_id": tenant,
                "hash": doc_hash,
                "source": "cutoff",
                "external_id": "doc-cutoff",
            },
            embedding=[0.25] + [0.0] * (vector_client.get_embedding_dim() - 1),
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
            filters={"case_id": None},
            top_k=3,
            alpha=0.8,
            min_sim=0.95,
        )

        assert result.fused_candidates >= 1
        assert result.below_cutoff >= 1
        assert result.returned_after_cutoff == 0
        assert result.chunks == []
        assert cutoff_counter.calls == [{"tenant_id": tenant}]
        assert cutoff_counter.value == float(result.below_cutoff)

    def test_hybrid_search_uses_similarity_fallback_when_trigram_has_no_match(
        self,
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"fallback-trigram").hexdigest()
        chunk = Chunk(
            content="Dies ist ein völlig anderer Inhalt",
            meta={
                "tenant_id": tenant,
                "hash": doc_hash,
                "source": "lexical",
                "external_id": "doc-trigram",
            },
            embedding=[0.12] + [0.0] * (vector_client.get_embedding_dim() - 1),
        )
        client.upsert_chunks([chunk])

        with capture_logs() as logs:
            result = client.hybrid_search(
                "zzzzzzzz",
                tenant_id=tenant,
                filters={"case_id": None},
                top_k=1,
                alpha=0.0,
            )

        assert result.lexical_candidates >= 1
        assert result.chunks
        top_chunk = result.chunks[0]
        assert top_chunk.meta["hash"] == doc_hash
        assert top_chunk.meta["score"] == pytest.approx(top_chunk.meta["lscore"])
        fallback_logs = [
            entry for entry in logs if entry["event"] == "rag.hybrid.trgm_no_match"
        ]
        assert fallback_logs
        assert fallback_logs[0]["fallback"] is True

    def test_fallback_relaxes_trgm_limit_until_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"adaptive-trgm").hexdigest()

        result, logs = self._run_adaptive_fallback(
            client,
            monkeypatch,
            tenant,
            doc_hash,
            success_limit=0.05,
            applied_limit=0.09,
        )

        assert result.lexical_candidates == 1
        assert result.fallback_limit_used == pytest.approx(0.05)
        assert result.applied_trgm_limit == pytest.approx(0.09)
        fallback_logs = [
            entry
            for entry in logs
            if entry["event"] == "rag.hybrid.trgm_fallback_applied"
        ]
        assert fallback_logs
        log_entry = fallback_logs[0]
        assert log_entry["picked_limit"] == pytest.approx(0.05)
        assert log_entry["count"] == 1
        assert log_entry["tried_limits"][0] == pytest.approx(0.09)

    def test_fallback_runs_when_show_limit_empty_and_primary_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"fallback-show-limit").hexdigest()
        doc_id = uuid.uuid4()

        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.get_embedding_dim(),
        )

        class _Cursor:
            def __init__(self) -> None:
                self._fetchone_result: tuple | None = None
                self._next_fetchall: str | None = None
                self._fallback_limit: float | None = None

            def __enter__(self) -> "_Cursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def execute(self, sql: str, params=None) -> None:
                if "pg_catalog.pg_opclass" in sql:
                    self._fetchone_result = (1,)
                    return
                if "SET LOCAL statement_timeout" in sql:
                    return
                if "SELECT set_limit" in sql:
                    self._fetchone_result = None
                    return
                if "SELECT show_limit()" in sql:
                    self._fetchone_result = ()
                    return
                if "FROM embeddings" in sql:
                    self._next_fetchall = "vector"
                    return
                if "c.text_norm % %s" in sql:
                    self._next_fetchall = "primary"
                    return
                if "similarity(c.text_norm, %s) >= %s" in sql:
                    self._fallback_limit = float(params[-2])
                    self._next_fetchall = "fallback"
                    return
                self._next_fetchall = None

            def fetchall(self) -> list[tuple]:
                if self._next_fetchall == "primary":
                    self._next_fetchall = None
                    raise IndexError("simulated primary lexical failure")
                if self._next_fetchall == "fallback":
                    limit = self._fallback_limit or 0.0
                    self._next_fetchall = None
                    if limit <= 0.05:
                        return [
                            (
                                "chunk-fallback",
                                "Fallback lexical",
                                {
                                    "tenant_id": normalized_tenant,
                                    "hash": doc_hash,
                                    "external_id": "fallback-doc",
                                },
                                doc_hash,
                                doc_id,
                                0.42,
                            )
                        ]
                    return []
                self._next_fetchall = None
                return []

            def fetchone(self):
                result = self._fetchone_result
                self._fetchone_result = None
                return result

        class _FakeConnection:
            def cursor(self) -> _Cursor:
                return _Cursor()

            def rollback(self) -> None:
                return None

        def _fake_connection(self):  # type: ignore[no-untyped-def]
            fake = _FakeConnection()

            class _Ctx:
                def __enter__(self_inner):
                    return fake

                def __exit__(self_inner, exc_type, exc, tb):
                    return None

            return _Ctx()

        monkeypatch.setattr(
            client, "_connection", _fake_connection.__get__(client, type(client))
        )

        with capture_logs() as logs:
            result = client.hybrid_search(
                "trigger fallback",
                tenant_id=tenant,
                filters={},
                top_k=1,
                alpha=0.0,
                trgm_limit=0.09,
            )

        assert result.lexical_candidates == 1
        assert result.applied_trgm_limit is None
        assert result.fallback_limit_used == pytest.approx(0.05)
        fallback_logs = [
            entry
            for entry in logs
            if entry["event"] == "rag.hybrid.trgm_fallback_applied"
        ]
        assert fallback_logs
        assert fallback_logs[0]["picked_limit"] == pytest.approx(0.05)

    def test_fallback_reapplies_search_path_after_rollback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"fallback-search-path").hexdigest()
        doc_id = uuid.uuid4()

        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.get_embedding_dim(),
        )

        class _Cursor:
            def __init__(self, owner) -> None:
                self._owner = owner
                self._next_fetchall: str | None = None
                self._fetchone_result: tuple | None = None
                self._fallback_limit: float | None = None

            def __enter__(self) -> "_Cursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def execute(self, sql, params=None) -> None:  # type: ignore[no-untyped-def]
                text = str(sql)
                if "pg_catalog.pg_opclass" in text:
                    self._fetchone_result = (1,)
                    return
                if "SET search_path" in text:
                    self._owner.mark_search_path()
                    return
                if "SET LOCAL statement_timeout" in text:
                    return
                if "SELECT set_limit" in text:
                    self._owner.last_limit = float(params[0]) if params else None
                    return
                if "SELECT show_limit()" in text:
                    if self._owner.last_limit is None:
                        self._fetchone_result = None
                    else:
                        self._fetchone_result = (self._owner.last_limit,)
                    return
                if "FROM embeddings" in text:
                    self._next_fetchall = "vector"
                    return
                if "c.text_norm % %s" in text:
                    self._next_fetchall = "primary"
                    return
                if "similarity(c.text_norm, %s) >= %s" in text:
                    self._next_fetchall = "fallback"
                    self._fallback_limit = float(params[-2]) if params else None
                    return
                self._next_fetchall = None

            def fetchall(self) -> list[tuple]:
                mode = self._next_fetchall
                self._next_fetchall = None
                if mode == "vector":
                    return []
                if mode == "primary":
                    raise IndexError("simulated lexical failure")
                if mode == "fallback":
                    if not self._owner.search_path_set:
                        raise RuntimeError("search path missing")
                    return [
                        (
                            "chunk-fallback",
                            "Lexical fallback",
                            {
                                "tenant_id": normalized_tenant,
                                "hash": doc_hash,
                                "external_id": "fallback-doc",
                            },
                            doc_hash,
                            doc_id,
                            0.38,
                        )
                    ]
                return []

            def fetchone(self):
                result = self._fetchone_result
                self._fetchone_result = None
                return result

        class _FakeConnection:
            def __init__(self) -> None:
                self.search_path_set = True
                self.search_path_calls = 0
                self.last_limit: float | None = None

            def mark_search_path(self) -> None:
                self.search_path_set = True
                self.search_path_calls += 1

            def cursor(self) -> _Cursor:
                return _Cursor(self)

            def rollback(self) -> None:
                self.search_path_set = False

        connections: list[_FakeConnection] = []

        def _fake_connection(self):  # type: ignore[no-untyped-def]
            fake = _FakeConnection()
            connections.append(fake)

            class _Ctx:
                def __enter__(self_inner):
                    return fake

                def __exit__(self_inner, exc_type, exc, tb):
                    return None

            return _Ctx()

        monkeypatch.setattr(
            client, "_connection", _fake_connection.__get__(client, type(client))
        )

        result = client.hybrid_search(
            "trigger fallback",
            tenant_id=tenant,
            filters={},
            top_k=1,
            alpha=0.0,
            trgm_limit=0.09,
        )

        assert result.lexical_candidates == 1
        assert result.chunks
        assert connections
        fake = connections[0]
        assert fake.search_path_calls >= 1
        assert fake.search_path_set is True

    def test_lexical_row_without_negative_index(self, monkeypatch):
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"lexical-noneg").hexdigest()
        doc_id = uuid.uuid4()

        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.get_embedding_dim(),
        )

        class _NoNegativeRow(Sequence):
            def __init__(self, values: tuple[object, ...]) -> None:
                self._values = values

            def __len__(self) -> int:
                return len(self._values)

            def __getitem__(self, index):  # type: ignore[override]
                if isinstance(index, slice):
                    return self._values[index]
                if index < 0:
                    raise IndexError("negative indices not supported")
                return self._values[index]

        class _Cursor:
            def __init__(self) -> None:
                self._next_fetchall: str | None = None
                self._fetchone_result: tuple | None = None
                self._last_limit: float | None = None

            def __enter__(self) -> "_Cursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def execute(self, sql, params=None) -> None:  # type: ignore[no-untyped-def]
                text = str(sql)
                if "pg_catalog.pg_opclass" in text:
                    self._fetchone_result = (1,)
                    return
                if "SET LOCAL statement_timeout" in text:
                    return
                if "SELECT set_limit" in text:
                    self._last_limit = float(params[0]) if params else None
                    return
                if "SELECT show_limit()" in text:
                    if self._last_limit is None:
                        self._fetchone_result = None
                    else:
                        self._fetchone_result = (self._last_limit,)
                    return
                if "FROM embeddings" in text:
                    self._next_fetchall = "vector"
                    return
                if "c.text_norm % %s" in text:
                    self._next_fetchall = "primary"
                    return
                if "similarity(c.text_norm, %s) >= %s" in text:
                    raise AssertionError("fallback should not execute")
                self._next_fetchall = None

            def fetchall(self) -> list[object]:
                mode = self._next_fetchall
                self._next_fetchall = None
                if mode == "vector":
                    return []
                if mode == "primary":
                    return [
                        _NoNegativeRow(
                            (
                                str(uuid.uuid4()),
                                "Lexical row without negative index support",
                                {"tenant_id": normalized_tenant},
                                doc_hash,
                                doc_id,
                                0.42,
                            )
                        )
                    ]
                return []

            def fetchone(self):
                result = self._fetchone_result
                self._fetchone_result = None
                return result

        class _FakeConnection:
            def cursor(self) -> _Cursor:
                return _Cursor()

            def rollback(self) -> None:
                raise AssertionError("rollback should not be required")

        def _fake_connection(self):  # type: ignore[no-untyped-def]
            fake = _FakeConnection()

            class _Ctx:
                def __enter__(self_inner):
                    return fake

                def __exit__(self_inner, exc_type, exc, tb):
                    return None

            return _Ctx()

        monkeypatch.setattr(
            client, "_connection", _fake_connection.__get__(client, type(client))
        )

        with capture_logs() as logs:
            result = client.hybrid_search(
                "primary lexical only",
                tenant_id=tenant,
                filters={},
                top_k=1,
            )

        assert result.lexical_candidates == 1
        assert result.chunks
        failures = [
            entry
            for entry in logs
            if entry.get("event") == "rag.hybrid.lexical_primary_failed"
        ]
        assert failures == []

    def test_fallback_skips_when_limit_low_enough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"trgm-direct").hexdigest()
        doc_id = uuid.uuid4()

        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.get_embedding_dim(),
        )

        class _Cursor:
            def __init__(self, owner) -> None:
                self._owner = owner
                self._fetchall_result: list[tuple] = []
                self._fetchone_result: tuple | None = None

            def __enter__(self) -> "_Cursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def execute(self, sql: str, params=None) -> None:
                if "pg_catalog.pg_opclass" in sql:
                    self._fetchone_result = (1,)
                    return
                if "SET LOCAL statement_timeout" in sql:
                    return
                if "SELECT set_limit" in sql:
                    self._fetchone_result = None
                    return
                if "SELECT show_limit()" in sql:
                    self._fetchone_result = (0.05,)
                    return
                if "c.text_norm % %s" in sql:
                    self._fetchall_result = [
                        (
                            "chunk-direct",
                            "Direct lexical",
                            {
                                "tenant_id": normalized_tenant,
                                "hash": doc_hash,
                                "external_id": "direct-doc",
                            },
                            doc_hash,
                            doc_id,
                            0.9,
                        )
                    ]
                    return
                if "similarity(c.text_norm, %s) >= %s" in sql:
                    self._owner.similarity_limits.append(float(params[-2]))
                    self._fetchall_result = []
                    return
                self._fetchall_result = []

            def fetchall(self) -> list[tuple]:
                result = list(self._fetchall_result)
                self._fetchall_result = []
                return result

            def fetchone(self):
                result = self._fetchone_result
                self._fetchone_result = None
                return result

        class _FakeConnection:
            def __init__(self) -> None:
                self.similarity_limits: list[float] = []

            def cursor(self) -> _Cursor:
                return _Cursor(self)

            def rollback(self) -> None:
                return None

        holder: dict[str, _FakeConnection] = {}

        def _fake_connection(self):  # type: ignore[no-untyped-def]
            fake = _FakeConnection()
            holder["conn"] = fake

            class _Ctx:
                def __enter__(self_inner):
                    return fake

                def __exit__(self_inner, exc_type, exc, tb):
                    return None

            return _Ctx()

        monkeypatch.setattr(
            client, "_connection", _fake_connection.__get__(client, type(client))
        )

        with capture_logs() as logs:
            result = client.hybrid_search(
                "direct lexical",
                tenant_id=tenant,
                filters={},
                top_k=1,
                alpha=0.0,
                trgm_limit=0.05,
            )

        assert result.lexical_candidates == 1
        assert result.fallback_limit_used is None
        assert result.applied_trgm_limit == pytest.approx(0.05)
        assert holder["conn"].similarity_limits == []
        fallback_logs = [
            entry
            for entry in logs
            if entry["event"] == "rag.hybrid.trgm_fallback_applied"
        ]
        assert fallback_logs == []

    def test_meta_includes_fallback_limit_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"fallback-meta").hexdigest()

        result, logs = self._run_adaptive_fallback(
            client,
            monkeypatch,
            tenant,
            doc_hash,
            success_limit=0.04,
            applied_limit=0.09,
        )

        assert result.lexical_candidates == 1
        assert result.applied_trgm_limit == pytest.approx(0.09)
        assert result.fallback_limit_used == pytest.approx(0.04)
        fallback_logs = [
            entry
            for entry in logs
            if entry["event"] == "rag.hybrid.trgm_fallback_applied"
        ]
        assert fallback_logs
        assert fallback_logs[0]["picked_limit"] == pytest.approx(0.04)

    def test_row_shape_mismatch_does_not_crash(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-1",
                        "Vector mismatch",
                        {"tenant_id": tenant},
                        "hash-1",
                        "doc-1",
                    )
                ],
                [],
                2.0,
            )

        monkeypatch.setattr(client, "_run_with_retries", _fake_run)
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_ROW_SHAPE_WARNINGS",
            set(),
        )

        class _Histogram:
            def __init__(self) -> None:
                self.samples: list[float] = []

            def observe(self, amount: float) -> None:
                self.samples.append(amount)

        histogram = _Histogram()
        monkeypatch.setattr(metrics, "RAG_SEARCH_MS", histogram)

        with capture_logs() as logs:
            result = client.hybrid_search(
                "shape mismatch",
                tenant_id=tenant,
                filters={"case_id": None},
                top_k=3,
            )

        warnings = [
            entry for entry in logs if entry["event"] == "rag.hybrid.row_shape_mismatch"
        ]
        assert len(warnings) == 1
        assert warnings[0]["kind"] == "vector"
        assert warnings[0]["row_len"] == 5
        assert result.chunks
        meta = result.chunks[0].meta
        assert meta["vscore"] == 0.0
        assert meta["lscore"] == 0.0

    def test_truncated_vector_row_populates_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-meta",
                        "Vector metadata",
                        {"tenant_id": tenant},
                        "hash-meta",
                        "doc-meta",
                    )
                ],
                [],
                1.0,
            )

        monkeypatch.setattr(client, "_run_with_retries", _fake_run)
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_ROW_SHAPE_WARNINGS",
            set(),
        )

        result = client.hybrid_search(
            "vector meta",
            tenant_id=tenant,
            filters={"case_id": None},
            top_k=1,
        )

        assert result.chunks
        meta = result.chunks[0].meta
        assert meta["hash"] == "hash-meta"
        assert meta["id"] == "doc-meta"

    def test_hybrid_returns_lexical_when_vector_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.1]
            + [0.0] * (vector_client.get_embedding_dim() - 1),
        )

        class _FakeCursor:
            def __init__(self, lexical_rows: list[tuple]) -> None:
                self._lexical_rows = lexical_rows
                self._last_sql = ""
                self._fetchall_result: list[tuple] = []
                self._fetchone_result: tuple | None = None

            def __enter__(self) -> "_FakeCursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def execute(self, sql: str, params=None) -> None:
                self._last_sql = sql
                if "pg_catalog.pg_opclass" in sql:
                    self._fetchone_result = (1,)
                    return
                if "embedding" in sql and "::vector" in sql:
                    raise RuntimeError("vector query failed")
                if "SELECT show_limit()" in sql:
                    self._fetchone_result = (0.3,)
                elif "similarity(c.text_norm" in sql:
                    self._fetchall_result = list(self._lexical_rows)
                else:
                    self._fetchall_result = []

            def fetchall(self) -> list[tuple]:
                result = list(self._fetchall_result)
                self._fetchall_result = []
                return result

            def fetchone(self):
                result = self._fetchone_result
                self._fetchone_result = None
                return result

        class _FakeConnection:
            def __init__(self, lexical_rows: list[tuple]) -> None:
                self._lexical_rows = lexical_rows
                self.rolled_back = False

            def cursor(self):
                return _FakeCursor(self._lexical_rows)

            def rollback(self) -> None:
                self.rolled_back = True

        lexical_rows = [
            (
                "lex-chunk",
                "Lexical fallback",
                {"tenant_id": tenant},
                "lex-hash",
                "lex-doc",
                0.9,
            )
        ]

        def _fake_connection(self):  # type: ignore[no-untyped-def]
            fake = _FakeConnection(lexical_rows)

            class _Ctx:
                def __enter__(self_inner):
                    return fake

                def __exit__(self_inner, exc_type, exc, tb):
                    return None

            return _Ctx()

        monkeypatch.setattr(
            client, "_connection", _fake_connection.__get__(client, type(client))
        )

        result = client.hybrid_search(
            "lexical only",
            tenant_id=tenant,
            filters={"case_id": None},
            top_k=2,
        )

        assert result.lexical_candidates == 1
        assert result.vector_candidates == 0
        assert result.chunks
        top_meta = result.chunks[0].meta
        assert top_meta["hash"] == "lex-hash"
        assert top_meta["lscore"] == pytest.approx(0.9)

    def test_hybrid_uses_fallback_distance_operator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vector_client.reset_default_client()
        client = vector_client.get_default_client()
        client._distance_operator_cache.clear()

        monkeypatch.setattr(
            vector_client,
            "resolve_distance_operator",
            lambda _cur, _kind: "<->",
            raising=True,
        )
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.2]
            + [0.0] * (vector_client.get_embedding_dim() - 1),
        )

        class _FakeCursor:
            def __init__(self, owner) -> None:
                self._owner = owner
                self._fetchall_result: list[tuple] = []
                self._fetchone_result: tuple | None = None

            def __enter__(self) -> "_FakeCursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def execute(self, sql: str, params=None) -> None:
                text = str(sql)
                if "SET LOCAL statement_timeout" in text:
                    return
                if "SET LOCAL hnsw.ef_search" in text:
                    return
                if "SELECT show_limit()" in text:
                    self._fetchone_result = (0.3,)
                    return
                if "SELECT set_limit" in text:
                    self._fetchone_result = None
                    return
                if "embedding" in text and "::vector" in text:
                    self._owner.vector_sql.append(text)
                    tenant_id = str(uuid.uuid4())
                    normalized_candidate = str(
                        client._coerce_tenant_uuid(tenant_id)
                    )
                    doc_hash = hashlib.sha256(b"fallback-operator").hexdigest()
                    doc_id = uuid.uuid4()
                    self._fetchall_result = [
                        (
                            "vector-chunk",
                            "Vector candidate",
                            {"tenant_id": normalized_candidate},
                            doc_hash,
                            doc_id,
                            0.12,
                        )
                    ]
                    return
                if "c.text_norm % %s" in text:
                    self._fetchall_result = []
                    return
                if "similarity(c.text_norm, %s) >= %s" in text:
                    self._fetchall_result = []
                    return
                self._fetchall_result = []

            def fetchall(self) -> list[tuple]:
                result = list(self._fetchall_result)
                self._fetchall_result = []
                return result

            def fetchone(self):
                result = self._fetchone_result
                self._fetchone_result = None
                return result

        class _FakeConnection:
            def __init__(self) -> None:
                self.vector_sql: list[str] = []

            def cursor(self) -> _FakeCursor:
                return _FakeCursor(self)

            def rollback(self) -> None:
                return None

        holder: dict[str, _FakeConnection] = {}

        def _fake_connection(self):  # type: ignore[no-untyped-def]
            fake = _FakeConnection()
            holder["conn"] = fake

            class _Ctx:
                def __enter__(self_inner):
                    return fake

                def __exit__(self_inner, exc_type, exc, tb):
                    return None

            return _Ctx()

        monkeypatch.setattr(
            client, "_connection", _fake_connection.__get__(client, type(client))
        )

        result = client.hybrid_search(
            "fallback distance",
            tenant_id=str(uuid.uuid4()),
            filters={},
            top_k=1,
        )

        fake_conn = holder["conn"]
        assert fake_conn.vector_sql
        assert "<->" in fake_conn.vector_sql[0]
        assert result.vector_candidates == 1

    def test_hybrid_search_counts_candidates_below_min_sim_cutoff(
        self, monkeypatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-cutoff",
                        "Too weak",
                        {"tenant_id": tenant},
                        "hash-cutoff",
                        "doc-cutoff",
                        0.95,
                    )
                ],
                [],
                3.5,
            )

        monkeypatch.setattr(client, "_run_with_retries", _fake_run)

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

        class _Histogram:
            def __init__(self) -> None:
                self.samples: list[float] = []

            def observe(self, amount: float) -> None:
                self.samples.append(amount)

        histogram = _Histogram()
        monkeypatch.setattr(metrics, "RAG_SEARCH_MS", histogram)

        result = client.hybrid_search(
            "min sim cutoff",
            tenant_id=tenant,
            filters={"case_id": None},
            top_k=1,
            min_sim=0.8,
        )

        assert result.below_cutoff == 1
        assert result.chunks == []
        assert cutoff_counter.calls == [{"tenant_id": tenant}]
        assert cutoff_counter.value == 1.0

    def test_hybrid_search_deleted_visibility_count_regression(
        self, monkeypatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        case = "deleted-count"
        embedding_dim = vector_client.get_embedding_dim()
        base_vector = [1.0] + [0.0] * (embedding_dim - 1)

        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: list(base_vector),
        )
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_get_distance_operator",
            lambda self, conn, index_kind: "<=>",
        )

        active_hash = hashlib.sha256(b"deleted-count-active").hexdigest()
        deleted_hash = hashlib.sha256(b"deleted-count-soft").hexdigest()

        active_chunk = Chunk(
            content="Vector candidate still active",
            meta={
                "tenant_id": tenant,
                "hash": active_hash,
                "external_id": "active-doc",
                "case_id": case,
                "source": "example",
            },
            embedding=list(base_vector),
        )
        deleted_chunk = Chunk(
            content="Vector candidate soft deleted",
            meta={
                "tenant_id": tenant,
                "hash": deleted_hash,
                "external_id": "deleted-doc",
                "case_id": case,
                "source": "example",
                "deleted_at": "2024-01-01T00:00:00Z",
            },
            embedding=list(base_vector),
        )

        written = client.upsert_chunks([active_chunk, deleted_chunk])
        assert written == 2

        result = client.hybrid_search(
            "   ",
            tenant_id=tenant,
            filters={"case_id": case},
            top_k=5,
            alpha=1.0,
            min_sim=0.0,
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].meta.get("external_id") == "active-doc"
        assert result.vector_candidates == 1
        assert result.lexical_candidates == 0
        assert result.deleted_matches_blocked == 1

    def test_run_with_retries_logs_exception_context(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        client._retries = 1  # type: ignore[attr-defined]

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

        def _always_fail() -> None:
            raise RuntimeError("boom")

        with capture_logs() as logs:
            with pytest.raises(RuntimeError, match="boom"):
                client._run_with_retries(_always_fail, op_name="search")

        failure_logs = [
            entry
            for entry in logs
            if entry["event"] == "pgvector operation failed, retrying"
        ]
        assert failure_logs
        assert failure_logs[0]["exc_type"] == "RuntimeError"
        assert failure_logs[0]["exc_message"] == "boom"

    def test_hybrid_search_logs_strict_reject_reason(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-strict",
                        "Strict candidate",
                        {"tenant_id": tenant},
                        "hash-strict",
                        "doc-1",
                        0.3,
                    )
                ],
                [],
                1.5,
            )

        class _Histogram:
            def __init__(self) -> None:
                self.samples: list[float] = []

            def observe(self, amount: float) -> None:
                self.samples.append(amount)

        histogram = _Histogram()
        monkeypatch.setattr(metrics, "RAG_SEARCH_MS", histogram)
        monkeypatch.setattr(client, "_run_with_retries", _fake_run)

        with capture_logs() as logs:
            result = client.hybrid_search(
                "Strict candidate",
                tenant_id=tenant,
                case_id="case-required",
                top_k=2,
            )

        rejects = [entry for entry in logs if entry["event"] == "rag.strict.reject"]
        assert rejects
        reject = rejects[0]
        assert reject["reasons"] == ["case_missing"]
        assert reject["candidate_case_id"] is None
        assert reject["candidate_tenant_id"] == tenant
        assert result.chunks == []
        assert result.vector_candidates == 1

    def test_hybrid_search_respects_request_trgm_limit(self) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"trgm-limit").hexdigest()
        chunk = Chunk(
            content="Trigram limit example",
            meta={
                "tenant_id": tenant,
                "hash": doc_hash,
                "case_id": "case-a",
                "source": "example",
            },
            embedding=[0.12] + [0.0] * (vector_client.get_embedding_dim() - 1),
        )
        client.upsert_chunks([chunk])

        with capture_logs() as logs:
            result = client.hybrid_search(
                "Trigram limit example",
                tenant_id=tenant,
                filters={"case_id": None},
                trgm_limit=0.42,
                top_k=3,
            )

        assert result.chunks
        sql_logs = [
            entry for entry in logs if entry["event"] == "rag.hybrid.sql_counts"
        ]
        assert sql_logs
        assert sql_logs[0]["trgm_limit"] == pytest.approx(0.42)
        assert sql_logs[0]["distance_score_mode"] == "inverse"

    def test_hybrid_search_pg_trgm_fallback_records_counts(self) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"fallback").hexdigest()
        chunk = Chunk(
            content="Lexical fallback candidate",
            meta={
                "tenant_id": tenant,
                "hash": doc_hash,
                "case_id": "case-b",
                "source": "example",
            },
            embedding=[0.4] + [0.0] * (vector_client.get_embedding_dim() - 1),
        )
        client.upsert_chunks([chunk])

        with capture_logs() as logs:
            result = client.hybrid_search(
                "Completely different query",
                tenant_id=tenant,
                filters={"case_id": None},
                alpha=0.0,
                trgm_limit=1.0,
                top_k=2,
            )

        assert result.lexical_candidates > 0
        assert result.chunks
        sql_logs = [
            entry for entry in logs if entry["event"] == "rag.hybrid.sql_counts"
        ]
        assert sql_logs
        assert sql_logs[0]["lex_rows"] > 0

    def test_hybrid_search_handles_mapping_rows(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_id = uuid.uuid4()
        doc_hash = hashlib.sha256(b"mapping-row").hexdigest()

        lexical_row = {
            "id": doc_id,
            "text": "Lexical mapping",
            "metadata": {
                "tenant_id": tenant,
                "case_id": "case-mapping",
                "doc_id": str(doc_id),
            },
            "hash": doc_hash,
            "doc_id": doc_id,
            "lscore": 0.62,
        }

        def _fake_run(self, fn, *, op_name: str):
            return ([], [lexical_row], 5.0)

        monkeypatch.setattr(
            client,
            "_run_with_retries",
            _fake_run.__get__(client, type(client)),
        )
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.get_embedding_dim(),
        )

        result = client.hybrid_search(
            "Lexical mapping",
            tenant_id=tenant,
            filters={"case_id": "case-mapping"},
            top_k=1,
            alpha=0.0,
        )

        assert result.lexical_candidates == 1
        assert result.chunks
        chunk_meta = result.chunks[0].meta
        assert chunk_meta["lscore"] == pytest.approx(0.62)

    def test_hybrid_search_score_fusion_alpha_one_cutoff(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-1",
                        "Vector strong",
                        {"tenant_id": tenant, "case_id": "case-1"},
                        "hash-1",
                        "doc-1",
                        0.25,
                    ),
                    (
                        "chunk-2",
                        "Vector weak",
                        {"tenant_id": tenant, "case_id": "case-1"},
                        "hash-2",
                        "doc-2",
                        0.9,
                    ),
                ],
                [
                    (
                        "chunk-1",
                        "Vector strong",
                        {"tenant_id": tenant, "case_id": "case-1"},
                        "hash-1",
                        "doc-1",
                        0.8,
                    ),
                    (
                        "chunk-2",
                        "Vector weak",
                        {"tenant_id": tenant, "case_id": "case-1"},
                        "hash-2",
                        "doc-2",
                        0.3,
                    ),
                ],
                2.5,
            )

        class _Histogram:
            def __init__(self) -> None:
                self.samples: list[float] = []

            def observe(self, amount: float) -> None:
                self.samples.append(amount)

        histogram = _Histogram()

        class _Counter:
            def __init__(self) -> None:
                self.calls: list[dict[str, str]] = []
                self.value = 0.0

            def labels(self, **labels: str) -> "_Counter":
                self.calls.append(labels)
                return self

            def inc(self, amount: float = 1.0) -> None:
                self.value += amount

        cutoff_counter = _Counter()
        monkeypatch.setattr(metrics, "RAG_SEARCH_MS", histogram)
        monkeypatch.setattr(metrics, "RAG_QUERY_BELOW_CUTOFF_TOTAL", cutoff_counter)
        monkeypatch.setattr(client, "_run_with_retries", _fake_run)

        result = client.hybrid_search(
            "Vector strong",
            tenant_id=tenant,
            case_id="case-1",
            alpha=1.0,
            min_sim=0.7,
            top_k=2,
            vec_limit=2,
            lex_limit=2,
        )

        assert len(result.chunks) == 1
        top_meta = result.chunks[0].meta
        assert top_meta["vscore"] == pytest.approx(1.0 / (1.0 + 0.25))
        assert top_meta["score"] == pytest.approx(top_meta["vscore"])
        assert result.below_cutoff == 1
        assert result.returned_after_cutoff == 1
        assert cutoff_counter.calls == [{"tenant_id": tenant}]
        assert cutoff_counter.value == 1.0

    def test_hybrid_search_score_fusion_alpha_zero_returned_after_cutoff(
        self, monkeypatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-1",
                        "Lexical strong",
                        {"tenant_id": tenant, "case_id": "case-2"},
                        "hash-1",
                        "doc-1",
                        0.4,
                    ),
                    (
                        "chunk-2",
                        "Lexical medium",
                        {"tenant_id": tenant, "case_id": "case-2"},
                        "hash-2",
                        "doc-2",
                        0.5,
                    ),
                ],
                [
                    (
                        "chunk-1",
                        "Lexical strong",
                        {"tenant_id": tenant, "case_id": "case-2"},
                        "hash-1",
                        "doc-1",
                        0.9,
                    ),
                    (
                        "chunk-2",
                        "Lexical medium",
                        {"tenant_id": tenant, "case_id": "case-2"},
                        "hash-2",
                        "doc-2",
                        0.5,
                    ),
                ],
                3.2,
            )

        class _Histogram:
            def __init__(self) -> None:
                self.samples: list[float] = []

            def observe(self, amount: float) -> None:
                self.samples.append(amount)

        histogram = _Histogram()
        monkeypatch.setattr(metrics, "RAG_SEARCH_MS", histogram)
        monkeypatch.setattr(client, "_run_with_retries", _fake_run)

        result = client.hybrid_search(
            "Lexical strong",
            tenant_id=tenant,
            case_id="case-2",
            alpha=0.0,
            min_sim=0.4,
            top_k=1,
            vec_limit=2,
            lex_limit=2,
        )

        assert len(result.chunks) == 1
        top_meta = result.chunks[0].meta
        assert top_meta["score"] == pytest.approx(top_meta["lscore"])
        assert top_meta["lscore"] == pytest.approx(0.9)
        assert result.below_cutoff == 0
        assert result.returned_after_cutoff == 2

    def test_hybrid_search_distance_score_mode_linear(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-linear",
                        "Linear score",
                        {"tenant_id": tenant},
                        "hash-linear",
                        "doc-linear",
                        0.25,
                    )
                ],
                [],
                1.8,
            )

        class _Histogram:
            def __init__(self) -> None:
                self.samples: list[float] = []

            def observe(self, amount: float) -> None:
                self.samples.append(amount)

        histogram = _Histogram()
        monkeypatch.setattr(metrics, "RAG_SEARCH_MS", histogram)
        monkeypatch.setattr(client, "_run_with_retries", _fake_run)
        monkeypatch.setenv("RAG_DISTANCE_SCORE_MODE", "linear")

        result = client.hybrid_search(
            "Linear score",
            tenant_id=tenant,
            filters={"case_id": None},
            top_k=1,
            alpha=1.0,
        )

        assert result.chunks
        meta = result.chunks[0].meta
        assert meta["vscore"] == pytest.approx(0.75)
        assert meta["score"] == pytest.approx(0.75)

    def test_format_vector_raises_on_dimension_mismatch(self) -> None:
        client = vector_client.get_default_client()
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            client._format_vector([0.0, 0.1])

    def test_embed_query_returns_non_zero_vector(self) -> None:
        client = vector_client.get_default_client()
        values = client._embed_query("hello world")
        assert len(values) == vector_client.get_embedding_dim()
        assert values[0] > 0.0
        assert all(v == 0.0 for v in values[1:])


def test_hybrid_search_restores_session_after_lexical_error(monkeypatch) -> None:
    embedding_dim = 3
    monkeypatch.setattr(vector_client, "get_embedding_dim", lambda: embedding_dim)

    client = object.__new__(vector_client.PgVectorClient)
    client._schema = "rag"
    client._statement_timeout_ms = 15000
    client._prepare_lock = threading.Lock()
    client._indexes_ready = True
    client._retries = 1
    client._retry_base_delay = 0.0
    client._distance_operator_cache = {}
    monkeypatch.setattr(client, "_get_distance_operator", lambda _conn, _kind: "<=>")

    tenant = str(uuid.uuid4())
    doc_hash = hashlib.sha256(b"lexical-error").hexdigest()
    chunk_id = "vector-chunk"
    doc_id = "doc-123"
    normalized_tenant = str(client._coerce_tenant_uuid(tenant))

    monkeypatch.setattr(
        vector_client.PgVectorClient,
        "_embed_query",
        lambda self, _query: [0.25] + [0.0] * (embedding_dim - 1),
    )

    restore_calls: list[object] = []

    def _record_restore(self, _cur) -> None:  # type: ignore[no-untyped-def]
        restore_calls.append(object())

    monkeypatch.setattr(
        client,
        "_restore_session_after_rollback",
        _record_restore.__get__(client, type(client)),
    )

    class _FakeCursor:
        def __init__(self) -> None:
            self._vector_rows: list[tuple] = []
            self._fetchone_result: tuple | None = None

        def __enter__(self) -> "_FakeCursor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, sql: str, params=None) -> None:
            normalised = " ".join(str(sql).split())
            if "SET LOCAL" in normalised or "SELECT set_limit" in normalised:
                return
            if "SELECT show_limit" in normalised:
                self._fetchone_result = (0.1,)
                return
            if "FROM embeddings e" in normalised and "ORDER BY distance" in normalised:
                self._vector_rows = [
                    (
                        chunk_id,
                        "Vector row",
                        {"tenant_id": normalized_tenant, "hash": doc_hash},
                        doc_hash,
                        doc_id,
                        0.05,
                    )
                ]
                return
            if "c.text_norm %% %s" in normalised and "ORDER BY lscore" in normalised:
                raise UndefinedTable('relation "embeddings" does not exist')
            if "COUNT(DISTINCT id)" in normalised:
                self._fetchone_result = (1,)
                return
            if "similarity(c.text_norm" in normalised and "ORDER BY lscore" in normalised:
                # Lexical fallback counts - no rows returned
                self._vector_rows = []
                return

        def fetchall(self) -> list[tuple]:
            result = list(self._vector_rows)
            self._vector_rows = []
            return result

        def fetchone(self):
            result = self._fetchone_result
            self._fetchone_result = None
            return result

    class _FakeConnection:
        def cursor(self):  # type: ignore[no-untyped-def]
            return _FakeCursor()

        def rollback(self) -> None:
            return None

    def _fake_connection(self):  # type: ignore[no-untyped-def]
        fake = _FakeConnection()

        class _Ctx:
            def __enter__(self_inner):
                return fake

            def __exit__(self_inner, exc_type, exc, tb):
                return None

        return _Ctx()

    monkeypatch.setattr(
        client, "_connection", _fake_connection.__get__(client, type(client))
    )

    result = client.hybrid_search(
        "restore session after lexical error",
        tenant_id=tenant,
        filters={},
        top_k=1,
    )

    assert result.vector_candidates == 1
    assert result.lexical_candidates == 0
    assert restore_calls, "expected session restore after rollback"
