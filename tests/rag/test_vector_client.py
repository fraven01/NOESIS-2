import uuid

import pytest
from structlog.testing import capture_logs

from ai_core.rag import vector_client

pytestmark = pytest.mark.usefixtures("rag_database")


class _FakeCursor:
    def __init__(
        self, show_limit_value: float = 0.0, vector_rows=None, lexical_rows=None
    ):
        self._limit = float(show_limit_value)
        self._vector_rows = list(vector_rows or [])
        self._lexical_rows = list(lexical_rows or [])
        self._last_sql = ""
        self._fetch_stage = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):  # noqa: WPS110 - sql name
        self._last_sql = str(sql)
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


def test_trgm_limit_is_applied_per_request(monkeypatch):
    client = vector_client.get_default_client()
    tenant = str(uuid.uuid4())

    # Fake DB that reflects set_limit back via show_limit
    cursor = _FakeCursor(show_limit_value=0.30)
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
            top_k=5,
        )

    pg_logs = [entry for entry in logs if entry["event"] == "rag.pgtrgm.limit"]
    assert pg_logs, "expected rag.pgtrgm.limit log entry"
    effective = float(pg_logs[0].get("effective"))
    assert effective == pytest.approx(0.05)


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
        "chunk-lex", "lexical match", {"tenant": tenant}, "hash-lex", "doc-lex", 0.2
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
