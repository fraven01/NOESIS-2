import uuid

import pytest
from structlog.testing import capture_logs

from ai_core.rag import vector_client

pytestmark = pytest.mark.usefixtures("rag_database")


class _FakeCursor:
    def __init__(
        self,
        show_limit_value: float = 0.0,
        vector_rows=None,
        lexical_rows=None,
        *,
        lexical_required_limit: float | None = None,
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
                self._limit = float((params or (None,))[0])
            except Exception:
                pass
        elif "show_limit()" in text:
            self._fetch_stage = "show_limit"
        elif "from embeddings" in text:
            self._fetch_stage = "vector"
        elif "similarity(" in text:
            # Distinguish initial trigram-based lexical query from fallback
            # which adds a similarity threshold predicate (">= %s").
            if ">= %s" in text:
                self._fetch_stage = "lexical_fallback"
            else:
                self._fetch_stage = "lexical"
        else:
            self._fetch_stage = None

    def fetchone(self):
        if self._fetch_stage == "show_limit":
            return (self._limit,)
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
            # Fallback returns rows irrespective of the server limit threshold.
            return list(self._lexical_rows)
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
        client, "_embed_query", lambda _q: [0.0] * vector_client.EMBEDDING_DIM
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
        client, "_embed_query", lambda _q: [0.0] * vector_client.EMBEDDING_DIM
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
