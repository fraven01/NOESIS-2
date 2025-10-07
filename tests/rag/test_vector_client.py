import math
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone

import pytest
import psycopg2
from psycopg2.extras import Json
from structlog.testing import capture_logs

from ai_core.rag import vector_client
from ai_core.rag.embeddings import EmbeddingBatchResult
from ai_core.rag.schemas import Chunk

pytestmark = pytest.mark.usefixtures("rag_database")


class _FakeCursor:
    def __init__(
        self,
        show_limit_value: float = 0.0,
        vector_rows=None,
        lexical_rows=None,
        *,
        lexical_required_limit: float | None = None,
        respect_set_limit: bool = True,
    ):
        self._limit = float(show_limit_value)
        self._vector_rows = list(vector_rows or [])
        self._lexical_rows = list(lexical_rows or [])
        self._last_sql = ""
        self._fetch_stage = None
        self._required_limit = (
            float(lexical_required_limit)
            if lexical_required_limit is not None
            else None
        )
        self._respect_set_limit = bool(respect_set_limit)
        self._requested_limit: float | None = None
        self._fallback_limit: float | None = None
        self.executed: list[tuple[str, object | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):  # noqa: WPS110 - sql name
        self._last_sql = str(sql)
        self.executed.append((self._last_sql, params))
        text = self._last_sql.lower()
        if "set_limit(" in text:
            try:
                # params may be a tuple like (0.05,)
                self._requested_limit = float((params or (None,))[0])
                if self._respect_set_limit:
                    self._limit = self._requested_limit
            except Exception:
                pass
        elif "show_limit()" in text:
            self._fetch_stage = "show_limit"
        elif "from pg_catalog.pg_opclass" in text:
            # Simulate presence of a compatible pgvector operator class
            # queried by operator_class_exists/resolve_distance_operator.
            self._fetch_stage = "opclass_check"
        elif "from embeddings" in text:
            self._fetch_stage = "vector"
        elif "similarity(" in text:
            # Distinguish initial trigram-based lexical query from fallback
            # which adds a similarity threshold predicate (">= %s").
            if ">= %s" in text:
                self._fetch_stage = "lexical_fallback"
                try:
                    if params and isinstance(params, Sequence):
                        self._fallback_limit = float(params[-2])
                    else:
                        self._fallback_limit = None
                except Exception:
                    self._fallback_limit = None
            else:
                self._fetch_stage = "lexical"
        else:
            self._fetch_stage = None

    def fetchone(self):
        if self._fetch_stage == "show_limit":
            return (self._limit,)
        if self._fetch_stage == "opclass_check":
            # Any non-None row indicates the operator class exists
            return (1,)
        return None

    def fetchall(self):
        if self._fetch_stage == "vector":
            return list(self._vector_rows)
        if self._fetch_stage == "lexical":
            # Simulate trigram operator path. Respect required limit, so when
            # the applied server limit is too high, we return no rows and
            # force the client to attempt the explicit similarity fallback.
            if self._required_limit is None or self._limit <= self._required_limit:
                return list(self._lexical_rows)
            return []
        if self._fetch_stage == "lexical_fallback":
            # Fallback returns rows based on the explicit similarity threshold.
            limit = self._fallback_limit
            rows: list[tuple] = []
            for row in self._lexical_rows:
                score: float | None = None
                try:
                    if isinstance(row, Sequence):
                        score = float(row[-1])
                except Exception:
                    score = None
                if limit is None or (score is not None and score >= limit - 1e-9):
                    rows.append(row)
            return rows
        return []


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _fake_connection_ctx(fake_conn):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        yield fake_conn

    return _ctx


class _FakeLabeledCounter:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def labels(self, **labels):  # noqa: D401 - simple metric stub
        counter = self

        class _Handle:
            def inc(self, value=1):  # noqa: D401 - simple metric stub
                counter.calls.append({"labels": dict(labels), "value": value})

        return _Handle()


class _FakeCounter:
    def __init__(self):
        self.values: list[object] = []

    def inc(self, value=1):  # noqa: D401 - simple metric stub
        self.values.append(value)


def test_replace_chunks_normalises_embeddings(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())
    document_id = uuid.uuid4()
    doc_key = (tenant, "external-1")

    class _RecorderCursor:
        def __init__(self):
            self.executed: list[tuple[str, object | None]] = []
            self.batch_calls: list[tuple[str, list[tuple]]] = []

        def execute(self, sql, params=None):  # noqa: WPS110 - sql name
            self.executed.append((str(sql), params))

        def executemany(self, sql, seq):  # noqa: WPS110 - sql name
            self.batch_calls.append((str(sql), list(seq)))

    recorder = _RecorderCursor()
    monkeypatch.setattr(vector_client, "get_embedding_dim", lambda: 2)

    captured: list[list[float]] = []
    original_format = client._format_vector

    def _capture_format(values: Sequence[float]) -> str:
        captured.append([float(v) for v in values])
        return original_format(values)

    monkeypatch.setattr(client, "_format_vector", _capture_format)

    grouped = {
        doc_key: {
            "tenant_id": tenant,
            "external_id": "external-1",
            "source": "unit-test",
            "metadata": {},
            "chunks": [
                Chunk(
                    content="hello world",
                    meta={"tenant": tenant, "hash": "hash-1", "source": "unit-test"},
                    embedding=[3.0, 4.0],
                )
            ],
        }
    }
    document_ids = {doc_key: document_id}
    doc_actions: dict[tuple[str, str], str] = {}

    client._replace_chunks(recorder, grouped, document_ids, doc_actions)

    assert captured, "expected embedding to be formatted"
    normalised = captured[0]
    assert len(normalised) == 2
    assert math.isclose(math.sqrt(sum(value * value for value in normalised)), 1.0)


def test_query_embedding_is_normalised(monkeypatch):
    client = vector_client.get_default_client()
    monkeypatch.setattr(vector_client, "get_embedding_dim", lambda: 2)

    class _FakeEmbeddingClient:
        def dim(self):  # noqa: D401 - simple interface shim
            return 2

        def embed(self, texts):
            return EmbeddingBatchResult(
                vectors=[[3.0, 4.0]],
                model="fake",
                model_used="primary",
                attempts=1,
                timeout_s=None,
            )

    fake_client = _FakeEmbeddingClient()
    monkeypatch.setattr(vector_client, "get_embedding_client", lambda: fake_client)

    values = client._embed_query("normalise me")

    assert len(values) == 2
    assert math.isclose(math.sqrt(sum(value * value for value in values)), 1.0)


def test_hybrid_search_returns_vector_hits_with_normalised_query(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())
    monkeypatch.setattr(vector_client, "get_embedding_dim", lambda: 2)

    class _FakeEmbeddingClient:
        def dim(self):  # noqa: D401 - simple interface shim
            return 2

        def embed(self, texts):
            return EmbeddingBatchResult(
                vectors=[[3.0, 4.0]],
                model="fake",
                model_used="primary",
                attempts=1,
                timeout_s=None,
            )

    fake_client = _FakeEmbeddingClient()
    monkeypatch.setattr(vector_client, "get_embedding_client", lambda: fake_client)

    vector_row = (
        "chunk-vector",
        "vector candidate",
        {"tenant": tenant},
        "hash-vector",
        "doc-vector",
        0.12,
    )
    cursor = _FakeCursor(vector_rows=[vector_row], lexical_rows=[])
    fake_conn = _FakeConn(cursor)
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))

    captured: list[list[float]] = []
    original_format = client._format_vector

    def _capture_format(values: Sequence[float]) -> str:
        captured.append([float(v) for v in values])
        return original_format(values)

    monkeypatch.setattr(client, "_format_vector", _capture_format)

    result = client.hybrid_search(
        "vector search",
        tenant_id=tenant,
        filters={"case": None},
        alpha=1.0,
        min_sim=0.0,
        top_k=1,
    )

    assert result.vector_candidates >= 1
    assert result.chunks
    assert not result.query_embedding_empty
    assert captured, "expected query embedding to be formatted"
    norm = math.sqrt(sum(value * value for value in captured[0]))
    assert math.isclose(norm, 1.0)


def test_trgm_limit_is_applied_and_yields_lexical_candidates(monkeypatch):
    monkeypatch.setenv("RAG_TRGM_LIMIT", "0.05")
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    lexical_row = (
        "chunk-lex",
        "lexical match",
        {"tenant": tenant},
        "hash-lex",
        "doc-lex",
        0.134,
    )
    cursor = _FakeCursor(
        show_limit_value=0.30,
        lexical_rows=[lexical_row],
        lexical_required_limit=0.05,
    )
    fake_conn = _FakeConn(cursor)
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))

    with capture_logs():
        result = client.hybrid_search(
            "zebragurke",
            tenant_id=tenant,
            filters={"case": None},
            alpha=0.0,
            min_sim=0.01,
            top_k=3,
        )

    assert result.lexical_candidates >= 1
    assert result.vector_candidates == 0
    chunk = result.chunks[0]
    lscore = float(chunk.meta.get("lscore", 0.0))
    assert lscore > 0.0
    assert float(chunk.meta.get("vscore", 0.0)) == pytest.approx(0.0)
    assert float(chunk.meta.get("fused", 0.0)) == pytest.approx(lscore)


def test_lexical_fallback_populates_rows(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    # One lexical row that should only appear via the fallback path
    lexical_row = (
        "chunk-fallback",
        "ZEBRAGURKEN",
        {"tenant": tenant},
        "hash-fallback",
        "doc-fallback",
        0.096,
    )

    # Start with a relatively high pg_trgm limit (0.30), so the first
    # trigram operator path produces zero rows for our FakeCursor when
    # coupled with a stricter required limit (0.10). The client should then
    # execute the explicit similarity fallback which returns the row.
    cursor = _FakeCursor(
        show_limit_value=0.30,
        vector_rows=[],
        lexical_rows=[lexical_row],
        lexical_required_limit=0.10,
    )
    fake_conn = _FakeConn(cursor)
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))

    with capture_logs() as logs:
        result = client.hybrid_search(
            "zebragurke",
            tenant_id=tenant,
            filters={"case": None},
            alpha=0.0,
            min_sim=0.0,
            top_k=3,
        )

    assert result.vector_candidates == 0
    assert result.lexical_candidates == 1
    lscore = float(result.chunks[0].meta.get("lscore", 0.0))
    assert lscore == pytest.approx(0.096, rel=1e-3, abs=1e-6)
    # With alpha=0.0, fusion equals lscore
    assert float(result.chunks[0].meta.get("fused", 0.0)) == pytest.approx(lscore)
    # Ensure our final handoff log recorded the lexical count
    final_logs = [e for e in logs if e.get("event") == "rag.debug.rows.lexical.final"]
    assert final_logs and int(final_logs[-1].get("count", 0)) >= 1

    applied_logs = [
        entry for entry in logs if entry["event"] == "rag.pgtrgm.limit.applied"
    ]
    assert applied_logs, "expected rag.pgtrgm.limit.applied log entry"
    applied = applied_logs[0].get("applied")
    assert float(applied) == pytest.approx(0.05)


def test_explicit_trgm_limit_fallback_uses_requested_threshold(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    lexical_row = (
        "chunk-fallback",  # noqa: S105 - test data
        "lexical match",
        {"tenant": tenant},
        "hash-fallback",
        "doc-fallback",
        0.111,
    )

    cursor = _FakeCursor(
        show_limit_value=0.30,
        lexical_rows=[lexical_row],
        lexical_required_limit=0.05,
        respect_set_limit=False,
    )
    fake_conn = _FakeConn(cursor)
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))

    with capture_logs() as logs:
        result = client.hybrid_search(
            "zebragurke",
            tenant_id=tenant,
            filters={"case": None},
            trgm_limit=0.05,
            alpha=0.0,
            min_sim=0.0,
            top_k=3,
        )

    assert result.vector_candidates == 0
    assert result.lexical_candidates == 1
    meta = result.chunks[0].meta
    assert float(meta.get("lscore", 0.0)) == pytest.approx(0.111)

    # The fallback query should have been executed with the explicitly
    # requested threshold despite the server reporting a higher limit.
    fallback_calls = [
        params for sql, params in cursor.executed if sql and ">= %s" in sql
    ]
    assert fallback_calls, "expected fallback lexical query to run"
    fallback_params = fallback_calls[-1]
    assert float(fallback_params[-2]) == pytest.approx(0.05)

    # Logs should still reflect the server-reported limit value from show_limit().
    applied_logs = [
        entry for entry in logs if entry["event"] == "rag.pgtrgm.limit.applied"
    ]
    assert applied_logs, "expected rag.pgtrgm.limit.applied log entry"
    assert float(applied_logs[0].get("applied")) == pytest.approx(0.30)


def test_applies_set_limit_and_logs_applied_value(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    lexical_row = (
        "chunk-lex",
        "lexical match",
        {"tenant": tenant},
        "hash-lex",
        "doc-lex",
        0.134,
    )

    cursor = _FakeCursor(
        show_limit_value=0.30,
        lexical_rows=[lexical_row],
        lexical_required_limit=0.05,
    )
    fake_conn = _FakeConn(cursor)
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))

    with capture_logs() as logs:
        client.hybrid_search(
            "zebragurke",
            tenant_id=tenant,
            filters={"case": None},
            trgm_limit=0.05,
            alpha=0.0,
            min_sim=0.0,
            top_k=3,
        )

    set_limit_calls = [
        (sql, params) for sql, params in cursor.executed if "set_limit" in sql.lower()
    ]
    assert set_limit_calls, "expected SELECT set_limit call"
    assert float(set_limit_calls[0][1][0]) == pytest.approx(0.05)

    applied_logs = [
        entry for entry in logs if entry["event"] == "rag.pgtrgm.limit.applied"
    ]
    assert applied_logs, "expected rag.pgtrgm.limit.applied log entry"
    assert float(applied_logs[0].get("applied")) == pytest.approx(0.05)


def test_row_shape_mismatch_does_not_crash(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    def _fake_run(_fn, *, op_name: str):
        # Return a vector row with only 5 columns to trigger padding
        return (
            [("chunk-5", "text", {"tenant": tenant}, "hash-5", "doc-5")],
            [],
            1.2,
        )

    monkeypatch.setattr(client, "_run_with_retries", _fake_run)
    monkeypatch.setattr(vector_client.PgVectorClient, "_ROW_SHAPE_WARNINGS", set())

    with capture_logs() as logs:
        result = client.hybrid_search(
            "shape mismatch",
            tenant_id=tenant,
            filters={"case": None},
            top_k=3,
        )

    assert result is not None
    mismatch_logs = [
        entry
        for entry in logs
        if entry["event"] == "rag.hybrid.row_shape_mismatch"
        and entry.get("kind") == "vector"
    ]
    assert mismatch_logs, "expected row_shape_mismatch warning for vector rows"
    assert int(mismatch_logs[0].get("row_len", 0)) == 5


def test_truncated_vector_row_populates_metadata(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    truncated_row = (
        "chunk-short",
        "truncated text",
        {"tenant": tenant},
        "hash-short",
        "doc-short",
    )

    def _fake_run(_fn, *, op_name: str):
        return ([truncated_row], [], 0.7)

    monkeypatch.setattr(client, "_run_with_retries", _fake_run)

    result = client.hybrid_search(
        "truncate me",
        tenant_id=tenant,
        filters={"case": None},
        top_k=1,
    )

    assert len(result.chunks) == 1
    meta = result.chunks[0].meta
    assert meta.get("tenant") == tenant
    assert meta.get("hash") == "hash-short"
    assert meta.get("id") == "doc-short"
    assert meta.get("vscore") == pytest.approx(0.0)
    assert meta.get("lscore") == pytest.approx(0.0)


def test_lexical_only_scoring(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    # Force null/zero embedding so vector path is ignored
    monkeypatch.setattr(
        client, "_embed_query", lambda _q: [0.0] * vector_client.get_embedding_dim()
    )

    lexical_row = (
        "chunk-lex",
        "lexical match",
        {"tenant": tenant, "case": "c1"},
        "hash-lex",
        "doc-lex",
        0.13,
    )

    def _fake_run(_fn, *, op_name: str):
        return ([], [lexical_row], 2.0)

    monkeypatch.setattr(client, "_run_with_retries", _fake_run)

    result = client.hybrid_search(
        "only lexical",
        tenant_id=tenant,
        case_id="c1",
        filters={"case": "c1"},
        top_k=1,
        alpha=0.0,
        min_sim=0.01,
    )

    assert len(result.chunks) == 1
    meta = result.chunks[0].meta
    assert meta.get("vscore") == 0.0
    assert meta.get("lscore", 0.0) > 0.0
    assert meta.get("fused") == meta.get("lscore")


def test_lexical_only_respects_min_sim_with_alpha(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    # Force null/zero embedding so vector path is ignored
    monkeypatch.setattr(
        client, "_embed_query", lambda _q: [0.0] * vector_client.get_embedding_dim()
    )

    lexical_row = (
        "chunk-lex",
        "lexical match",
        {"tenant": tenant},
        "hash-lex",
        "doc-lex",
        0.2,
    )

    def _fake_run(_fn, *, op_name: str):
        return ([], [lexical_row], 1.5)

    monkeypatch.setattr(client, "_run_with_retries", _fake_run)

    result = client.hybrid_search(
        "only lexical",
        tenant_id=tenant,
        filters={"case": None},
        top_k=1,
        alpha=0.7,
        min_sim=0.15,
    )

    assert len(result.chunks) == 1
    meta = result.chunks[0].meta
    assert meta.get("vscore") == 0.0
    assert meta.get("lscore") == pytest.approx(0.2)
    assert meta.get("fused") == pytest.approx(0.2)


def test_hybrid_search_clamps_candidate_limits(monkeypatch):
    vector_client.reset_default_client()
    monkeypatch.setenv("RAG_MAX_CANDIDATES", "100")
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    lexical_row = (
        "chunk-clamped",
        "lexical match",
        {"tenant": tenant},
        "hash-clamped",
        "doc-clamped",
        0.5,
    )
    cursor = _FakeCursor(
        show_limit_value=0.30,
        vector_rows=[],
        lexical_rows=[lexical_row],
    )
    fake_conn = _FakeConn(cursor)
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))

    result = client.hybrid_search(
        "clamp me",
        tenant_id=tenant,
        filters={"case": None},
        alpha=0.0,
        min_sim=0.0,
        top_k=10,
        vec_limit=5000,
        lex_limit=8000,
    )

    assert result.vec_limit == 100
    assert result.lex_limit == 100

    vector_limits = [
        params[-1]
        for sql, params in cursor.executed
        if "from embeddings" in sql.lower()
    ]
    assert vector_limits and all(limit == 100 for limit in vector_limits)

    lexical_limits = [
        params[-1]
        for sql, params in cursor.executed
        if "similarity(c.text_norm" in sql.lower() and "limit %s" in sql.lower()
    ]
    assert lexical_limits and all(limit == 100 for limit in lexical_limits)

    vector_client.reset_default_client()


def test_upsert_retries_operational_error_once(monkeypatch):
    vector_client.reset_default_client()
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())
    chunk = Chunk(
        content="retry me",
        meta={
            "tenant": tenant,
            "hash": "hash-retry",
            "source": "unit-test",
            "external_id": "doc-retry",
        },
    )
    key = (tenant, "doc-retry")
    grouped_doc = {
        key: {
            "id": uuid.uuid4(),
            "tenant_id": tenant,
            "external_id": "doc-retry",
            "hash": "hash-retry",
            "content_hash": "hash-retry",
            "source": "unit-test",
            "metadata": {},
            "chunks": [chunk],
        }
    }

    retry_metric = _FakeLabeledCounter()
    monkeypatch.setattr(vector_client.metrics, "RAG_RETRY_ATTEMPTS", retry_metric)
    monkeypatch.setattr(vector_client.metrics, "RAG_UPSERT_CHUNKS", _FakeCounter())
    monkeypatch.setattr(
        vector_client.metrics, "INGESTION_DOCS_INSERTED", _FakeCounter()
    )
    monkeypatch.setattr(
        vector_client.metrics, "INGESTION_DOCS_REPLACED", _FakeCounter()
    )
    monkeypatch.setattr(vector_client.metrics, "INGESTION_DOCS_SKIPPED", _FakeCounter())
    monkeypatch.setattr(
        vector_client.metrics, "INGESTION_CHUNKS_WRITTEN", _FakeCounter()
    )
    monkeypatch.setattr(vector_client.time, "sleep", lambda _s: None)

    monkeypatch.setattr(client, "_group_by_document", lambda chunks: grouped_doc)

    document_id = uuid.uuid4()

    def _fake_ensure_documents(_cur, grouped):
        assert grouped is grouped_doc
        return {key: document_id}, {key: "inserted"}

    monkeypatch.setattr(client, "_ensure_documents", _fake_ensure_documents)

    def _fake_replace_chunks(_cur, grouped, document_ids, doc_actions):
        assert grouped is grouped_doc
        assert document_ids[key] == document_id
        assert doc_actions[key] == "inserted"
        return 1, {key: {"chunk_count": 1, "duration_ms": 0.5}}

    monkeypatch.setattr(client, "_replace_chunks", _fake_replace_chunks)

    state = {
        "failures": 1,
        "executed": [],
        "rollback_calls": 0,
        "commit_calls": 0,
    }

    class _FlakyCursor:
        def __enter__(self):
            if state["failures"] > 0:
                state["failures"] -= 1
                raise psycopg2.OperationalError("transient failure")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):  # noqa: WPS110 - sql name
            state["executed"].append((str(sql), params))

    class _FlakyConn:
        def cursor(self):
            return _FlakyCursor()

        def rollback(self):
            state["rollback_calls"] += 1

        def commit(self):
            state["commit_calls"] += 1

    fake_conn = _FlakyConn()
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))

    with capture_logs() as logs:
        result = client.upsert_chunks([chunk])

    assert int(result) == 1
    assert state["rollback_calls"] == 1
    assert state["commit_calls"] == 1
    assert any("SET LOCAL statement_timeout" in sql for sql, _ in state["executed"])

    assert retry_metric.calls == [
        {"labels": {"operation": "upsert_chunks"}, "value": 1}
    ]

    retry_logs = [
        entry
        for entry in logs
        if entry.get("event") == "pgvector operation failed, retrying"
    ]
    assert len(retry_logs) == 1
    retry_entry = retry_logs[0]
    assert retry_entry.get("operation") == "upsert_chunks"
    assert int(retry_entry.get("attempt", 0)) == 1


def test_hybrid_search_recovers_when_vector_query_fails(monkeypatch):
    vector_client.reset_default_client()
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())
    lexical_row = (
        "chunk-lex",
        "lexical",
        {"tenant": tenant},
        "hash-lex",
        "doc-lex",
        0.42,
    )

    class _VectorFailCursor(_FakeCursor):
        def __init__(self, *args, fail_state, **kwargs):
            super().__init__(*args, **kwargs)
            self._fail_state = fail_state

        def execute(self, sql, params=None):  # noqa: WPS110 - sql name
            super().execute(sql, params)
            if self._fetch_stage == "vector" and self._fail_state.get("vector", 0) > 0:
                self._fail_state["vector"] -= 1
                raise psycopg2.Error("vector blew up")

    state = {"vector": 1, "rollback": 0}

    class _Conn:
        def cursor(self):
            return _VectorFailCursor(
                show_limit_value=0.30,
                vector_rows=[],
                lexical_rows=[lexical_row],
                fail_state=state,
            )

        def rollback(self):
            state["rollback"] += 1

    fake_conn = _Conn()
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))
    monkeypatch.setattr(client, "_embed_query", lambda _q: [0.1, 0.0])

    with capture_logs() as logs:
        result = client.hybrid_search(
            "vector fails",
            tenant_id=tenant,
            filters={"case": None},
            alpha=0.0,
            min_sim=0.0,
            top_k=3,
        )

    assert result.vector_candidates == 0
    assert result.lexical_candidates == 1
    assert len(result.chunks) == 1
    assert state["rollback"] == 1

    failure_logs = [
        entry
        for entry in logs
        if entry.get("event") == "rag.hybrid.vector_query_failed"
    ]
    assert len(failure_logs) == 1
    assert failure_logs[0].get("tenant") == tenant


def test_hybrid_search_returns_vector_results_when_lexical_fails(monkeypatch):
    vector_client.reset_default_client()
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())
    vector_row = (
        "chunk-vec",
        "vector",
        {"tenant": tenant},
        "hash-vec",
        "doc-vec",
        0.15,
    )

    class _LexicalFailCursor(_FakeCursor):
        def __init__(self, *args, fail_state, **kwargs):
            super().__init__(*args, **kwargs)
            self._fail_state = fail_state

        def execute(self, sql, params=None):  # noqa: WPS110 - sql name
            super().execute(sql, params)
            if (
                self._fetch_stage == "lexical"
                and self._fail_state.get("lexical", 0) > 0
            ):
                self._fail_state["lexical"] -= 1
                raise psycopg2.Error("lexical blew up")

    state = {"lexical": 1, "rollback": 0}

    class _Conn:
        def cursor(self):
            return _LexicalFailCursor(
                show_limit_value=0.30,
                vector_rows=[vector_row],
                lexical_rows=[],
                fail_state=state,
            )

        def rollback(self):
            state["rollback"] += 1

    fake_conn = _Conn()
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))
    monkeypatch.setattr(client, "_embed_query", lambda _q: [0.2, 0.1])

    with capture_logs() as logs:
        result = client.hybrid_search(
            "lexical fails",
            tenant_id=tenant,
            filters={"case": None},
            alpha=0.0,
            min_sim=0.0,
            top_k=3,
        )

    assert result.vector_candidates == 1
    assert result.lexical_candidates == 0
    assert len(result.chunks) == 1
    assert state["rollback"] == 1

    failure_logs = [
        entry
        for entry in logs
        if entry.get("event") == "rag.hybrid.lexical_query_failed"
    ]
    assert len(failure_logs) == 1
    assert failure_logs[0].get("tenant") == tenant


def test_hybrid_search_raises_when_vector_and_lexical_fail(monkeypatch):
    vector_client.reset_default_client()
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    class _DualFailCursor(_FakeCursor):
        def __init__(self, *args, fail_state, **kwargs):
            super().__init__(*args, **kwargs)
            self._fail_state = fail_state

        def execute(self, sql, params=None):  # noqa: WPS110 - sql name
            super().execute(sql, params)
            if self._fetch_stage == "vector" and self._fail_state.get("vector", 0) > 0:
                self._fail_state["vector"] -= 1
                raise psycopg2.Error("vector blew up")
            if (
                self._fetch_stage == "lexical"
                and self._fail_state.get("lexical", 0) > 0
            ):
                self._fail_state["lexical"] -= 1
                raise psycopg2.Error("lexical blew up")

    state = {"vector": 1, "lexical": 1, "rollback": 0}

    class _Conn:
        def cursor(self):
            return _DualFailCursor(
                show_limit_value=0.30,
                vector_rows=[],
                lexical_rows=[],
                fail_state=state,
            )

        def rollback(self):
            state["rollback"] += 1

    fake_conn = _Conn()
    monkeypatch.setattr(client, "_connection", _fake_connection_ctx(fake_conn))
    monkeypatch.setattr(client, "_embed_query", lambda _q: [0.3, 0.2])

    with capture_logs() as logs, pytest.raises(psycopg2.Error):
        client.hybrid_search(
            "both fail",
            tenant_id=tenant,
            filters={"case": None},
            alpha=0.0,
            min_sim=0.0,
            top_k=3,
        )

    assert state["rollback"] >= 2
    vector_logs = [
        entry
        for entry in logs
        if entry.get("event") == "rag.hybrid.vector_query_failed"
    ]
    lexical_logs = [
        entry
        for entry in logs
        if entry.get("event") == "rag.hybrid.lexical_query_failed"
    ]
    assert vector_logs and lexical_logs


def _insert_active_and_soft_deleted_documents(
    client, tenant: str
) -> tuple[uuid.UUID, uuid.UUID, str]:
    active_doc_id = uuid.uuid4()
    deleted_doc_id = uuid.uuid4()
    timestamp = datetime.now(tz=timezone.utc)
    shared_text = "Shared retrieval test"

    with client.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (id, tenant_id, source, hash, metadata, external_id, deleted_at)
            VALUES (%s, %s, %s, %s, %s, %s, NULL)
            """,
            (
                active_doc_id,
                tenant,
                "unit-test",
                "hash-active",
                Json({"hash": "hash-active"}),
                "doc-active",
            ),
        )
        cur.execute(
            """
            INSERT INTO documents (id, tenant_id, source, hash, metadata, external_id, deleted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                deleted_doc_id,
                tenant,
                "unit-test",
                "hash-deleted",
                Json({"hash": "hash-deleted"}),
                "doc-deleted",
                timestamp,
            ),
        )
        cur.execute(
            """
            INSERT INTO chunks (id, document_id, ord, text, tokens, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                uuid.uuid4(),
                active_doc_id,
                0,
                shared_text,
                3,
                Json({"tenant": tenant, "case": "alpha"}),
            ),
        )
        cur.execute(
            """
            INSERT INTO chunks (id, document_id, ord, text, tokens, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                uuid.uuid4(),
                deleted_doc_id,
                0,
                shared_text,
                3,
                Json(
                    {
                        "tenant": tenant,
                        "case": "alpha",
                        "deleted_at": timestamp.isoformat(),
                    }
                ),
            ),
        )
        conn.commit()

    return active_doc_id, deleted_doc_id, shared_text


def test_hybrid_search_filters_soft_deleted_documents():
    vector_client.reset_default_client()
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())
    active_doc_id, _, search_text = _insert_active_and_soft_deleted_documents(
        client, tenant
    )

    result = client.hybrid_search(
        search_text,
        tenant_id=tenant,
        filters={"case": "alpha"},
        alpha=0.0,
        min_sim=0.0,
        top_k=5,
    )

    assert result.vector_candidates == 0
    assert result.lexical_candidates == 1
    assert result.deleted_matches_blocked == 1
    assert [chunk.meta.get("id") for chunk in result.chunks] == [str(active_doc_id)]


def test_hybrid_search_returns_deleted_when_visibility_deleted():
    vector_client.reset_default_client()
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())
    _, deleted_doc_id, search_text = _insert_active_and_soft_deleted_documents(
        client, tenant
    )

    result = client.hybrid_search(
        search_text,
        tenant_id=tenant,
        filters={"case": "alpha"},
        alpha=0.0,
        min_sim=0.0,
        top_k=5,
        visibility="deleted",
    )

    assert result.visibility == "deleted"
    assert result.vector_candidates == 0
    assert result.lexical_candidates == 1
    assert result.deleted_matches_blocked == 0
    assert [chunk.meta.get("id") for chunk in result.chunks] == [str(deleted_doc_id)]


def test_hybrid_search_returns_all_when_visibility_all():
    vector_client.reset_default_client()
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())
    active_doc_id, deleted_doc_id, search_text = _insert_active_and_soft_deleted_documents(
        client, tenant
    )

    result = client.hybrid_search(
        search_text,
        tenant_id=tenant,
        filters={"case": "alpha"},
        alpha=0.0,
        min_sim=0.0,
        top_k=5,
        visibility="all",
    )

    assert result.visibility == "all"
    assert result.vector_candidates == 0
    assert result.lexical_candidates == 2
    assert result.deleted_matches_blocked == 0
    assert set(chunk.meta.get("id") for chunk in result.chunks) == {
        str(active_doc_id),
        str(deleted_doc_id),
    }
