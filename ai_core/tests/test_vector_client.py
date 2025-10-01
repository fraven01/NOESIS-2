import hashlib
import uuid

import pytest

from ai_core.rag import metrics, vector_client
from ai_core.rag.schemas import Chunk
from structlog.testing import capture_logs


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

        with capture_logs() as logs:
            result = client.hybrid_search(
                "Lexical fallback example",
                tenant_id=tenant,
                filters={"case": None},
                top_k=3,
                alpha=0.0,
            )

        assert result.vector_candidates == 0
        assert result.lexical_candidates >= 1
        assert result.query_embedding_empty is True
        assert counter.value == 1.0
        assert counter.calls == [{"tenant": tenant}]
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
        assert logged["tenant"] == tenant
        assert "case" in logged
        assert logged["case"] is None

    def test_hybrid_search_lexical_matches_database_normalisation(
        self, monkeypatch
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"zebra-gurke").hexdigest()
        chunk = Chunk(
            content="ZEBRAGURKE",
            meta={
                "tenant": tenant,
                "hash": doc_hash,
                "case": "lexical-case",
                "source": "lexical",
                "external_id": "lex-1",
            },
            embedding=[0.0] * vector_client.EMBEDDING_DIM,
        )
        client.upsert_chunks([chunk])
        monkeypatch.setattr(
            vector_client.PgVectorClient,
            "_embed_query",
            lambda self, _query: [0.0] * vector_client.EMBEDDING_DIM,
        )

        result = client.hybrid_search(
            "ZEBRAGURKE",
            tenant_id=tenant,
            filters={"case": "lexical-case"},
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

    def test_hybrid_search_uses_similarity_fallback_when_trigram_has_no_match(
        self,
    ) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"fallback-trigram").hexdigest()
        chunk = Chunk(
            content="Dies ist ein völlig anderer Inhalt",
            meta={
                "tenant": tenant,
                "hash": doc_hash,
                "source": "lexical",
                "external_id": "doc-trigram",
            },
            embedding=[0.12] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
        )
        client.upsert_chunks([chunk])

        with capture_logs() as logs:
            result = client.hybrid_search(
                "zzzzzzzz",
                tenant_id=tenant,
                filters={"case": None},
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

    def test_row_shape_mismatch_does_not_crash(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-1",
                        "Vector mismatch",
                        {"tenant": tenant},
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
                filters={"case": None},
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
                        {"tenant": tenant},
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
            filters={"case": None},
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
            lambda self, _query: [0.1] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
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
                if "embedding <=>" in sql:
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
                {"tenant": tenant},
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
            filters={"case": None},
            top_k=2,
        )

        assert result.lexical_candidates == 1
        assert result.vector_candidates == 0
        assert result.chunks
        top_meta = result.chunks[0].meta
        assert top_meta["hash"] == "lex-hash"
        assert top_meta["lscore"] == pytest.approx(0.9)

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
                        {"tenant": tenant},
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
            filters={"case": None},
            top_k=1,
            min_sim=0.8,
        )

        assert result.below_cutoff == 1
        assert result.chunks == []
        assert cutoff_counter.calls == [{"tenant": tenant}]
        assert cutoff_counter.value == 1.0

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
                        {"tenant": tenant},
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
        assert reject["candidate_case"] is None
        assert reject["candidate_tenant"] == tenant
        assert result.chunks == []
        assert result.vector_candidates == 1

    def test_hybrid_search_respects_request_trgm_limit(self) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())
        doc_hash = hashlib.sha256(b"trgm-limit").hexdigest()
        chunk = Chunk(
            content="Trigram limit example",
            meta={
                "tenant": tenant,
                "hash": doc_hash,
                "case": "case-a",
                "source": "example",
            },
            embedding=[0.12] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
        )
        client.upsert_chunks([chunk])

        with capture_logs() as logs:
            result = client.hybrid_search(
                "Trigram limit example",
                tenant_id=tenant,
                filters={"case": None},
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
                "tenant": tenant,
                "hash": doc_hash,
                "case": "case-b",
                "source": "example",
            },
            embedding=[0.4] + [0.0] * (vector_client.EMBEDDING_DIM - 1),
        )
        client.upsert_chunks([chunk])

        with capture_logs() as logs:
            result = client.hybrid_search(
                "Completely different query",
                tenant_id=tenant,
                filters={"case": None},
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

    def test_hybrid_search_score_fusion_alpha_one_cutoff(self, monkeypatch) -> None:
        client = vector_client.get_default_client()
        tenant = str(uuid.uuid4())

        def _fake_run(_fn, *, op_name: str):
            return (
                [
                    (
                        "chunk-1",
                        "Vector strong",
                        {"tenant": tenant, "case": "case-1"},
                        "hash-1",
                        "doc-1",
                        0.25,
                    ),
                    (
                        "chunk-2",
                        "Vector weak",
                        {"tenant": tenant, "case": "case-1"},
                        "hash-2",
                        "doc-2",
                        0.9,
                    ),
                ],
                [
                    (
                        "chunk-1",
                        "Vector strong",
                        {"tenant": tenant, "case": "case-1"},
                        "hash-1",
                        "doc-1",
                        0.8,
                    ),
                    (
                        "chunk-2",
                        "Vector weak",
                        {"tenant": tenant, "case": "case-1"},
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
        assert cutoff_counter.calls == [{"tenant": tenant}]
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
                        {"tenant": tenant, "case": "case-2"},
                        "hash-1",
                        "doc-1",
                        0.4,
                    ),
                    (
                        "chunk-2",
                        "Lexical medium",
                        {"tenant": tenant, "case": "case-2"},
                        "hash-2",
                        "doc-2",
                        0.5,
                    ),
                ],
                [
                    (
                        "chunk-1",
                        "Lexical strong",
                        {"tenant": tenant, "case": "case-2"},
                        "hash-1",
                        "doc-1",
                        0.9,
                    ),
                    (
                        "chunk-2",
                        "Lexical medium",
                        {"tenant": tenant, "case": "case-2"},
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
                        {"tenant": tenant},
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
            filters={"case": None},
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
        assert len(values) == vector_client.EMBEDDING_DIM
        assert values[0] > 0.0
        assert all(v == 0.0 for v in values[1:])
