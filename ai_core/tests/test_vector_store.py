import threading
import types
import uuid

from ai_core.rag.vector_client import PgVectorClient


def _build_stub_client() -> PgVectorClient:
    client = PgVectorClient.__new__(PgVectorClient)
    client._schema = "rag"
    client._indexes_ready = True
    client._prepare_lock = threading.Lock()
    client._pool = None
    client._statement_timeout_ms = 0
    client._retries = 1
    client._retry_base_delay = 0.0
    client._distance_operator_cache = {}
    return client


def test_hybrid_search_skips_malformed_lexical_rows():
    client = _build_stub_client()
    tenant_id = str(uuid.uuid4())

    vector_rows = [
        {
            "id": "vec-1",
            "text": "Vector candidate",
            "metadata": {"tenant_id": tenant_id, "doc_id": "doc-1"},
            "hash": "hash-1",
            "doc_id": "doc-1",
            "distance": 0.1,
        }
    ]
    lexical_rows = [
        ("only_one_column",),
    ]

    def fake_run_with_retries(self, fn, *, op_name):
        return vector_rows, lexical_rows, 1.23

    client._run_with_retries = types.MethodType(fake_run_with_retries, client)

    result = client.hybrid_search(query="test", tenant_id=tenant_id)

    assert result.vector_candidates == 1
    assert result.lexical_candidates == 0
    assert len(result.chunks) == 1
    assert result.chunks[0].content == "Vector candidate"
