from psycopg2 import OperationalError

from ai_core.rag import vector_client
from ai_core.rag.vector_store import NullVectorStore, _build_pgvector_store
from ai_core.rag.visibility import DEFAULT_VISIBILITY


def test_build_pgvector_store_returns_null_when_dsn_unreachable(monkeypatch):
    class DummyPgVectorClient:
        def __init__(self, dsn: str, **kwargs):
            raise OperationalError("unreachable", None, None)

        @classmethod
        def from_env(cls, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("from_env should not be called for explicit DSN")

    monkeypatch.setattr(vector_client, "PgVectorClient", DummyPgVectorClient)

    store = _build_pgvector_store("global", {"dsn": "postgresql://demo"})

    assert isinstance(store, NullVectorStore)


def test_build_pgvector_store_returns_null_when_env_missing(monkeypatch):
    class EnvMissingClient:
        def __init__(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("constructor should not run")

        @classmethod
        def from_env(cls, **kwargs):
            raise RuntimeError("missing env")

    def failing_default():
        raise RuntimeError("missing shared client")

    monkeypatch.setattr(vector_client, "PgVectorClient", EnvMissingClient)
    monkeypatch.setattr(vector_client, "get_default_client", failing_default)

    store = _build_pgvector_store("fallback", {})

    assert isinstance(store, NullVectorStore)


def test_null_vector_store_hybrid_search_returns_empty_result():
    store = NullVectorStore("demo")

    result = store.hybrid_search("query", "tenant")

    assert result.chunks == []
    assert result.vector_candidates == 0
    assert result.lexical_candidates == 0
    assert result.visibility == DEFAULT_VISIBILITY.value
    assert result.vec_limit >= 1
    assert result.lex_limit >= 1
