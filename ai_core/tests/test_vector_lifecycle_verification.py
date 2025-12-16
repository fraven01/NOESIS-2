import os
import sys
import uuid
import pytest
from ai_core.rag import vector_client
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_client import Visibility

pytestmark = pytest.mark.usefixtures("rag_database")


def _make_chunk(tenant_id, hash_id, doc_uuid):
    return Chunk(
        content="test content verification",
        meta={
            "tenant_id": tenant_id,
            "hash": hash_id,
            "document_id": str(doc_uuid),
            "source": "verification",
        },
        embedding=[0.1] * 1536,
    )


def test_lifecycle_filtering(monkeypatch):
    """Verify that lifecycle states correctly hide/show documents."""
    dsn = os.environ.get("RAG_DATABASE_URL") or os.environ.get(
        "AI_CORE_TEST_DATABASE_URL"
    )
    if not dsn:
        pytest.skip("No DB URL")

    # Mock embedding dim to match our dummy embedding
    monkeypatch.setattr(vector_client, "get_embedding_dim", lambda: 1536)

    vector_client.reset_default_client()
    client = vector_client.get_default_client()

    tenant_id = str(uuid.uuid4())
    doc_uuid = uuid.uuid4()
    doc_hash = "hash-" + str(doc_uuid)

    # 1. Insert Active Document
    print(f"Inserting active doc {doc_uuid}")
    chunk = _make_chunk(tenant_id, doc_hash, doc_uuid)
    client.upsert_chunks([chunk])

    # Verify visible
    res = client.hybrid_search("test", tenant_id=tenant_id)
    assert len(res.chunks) > 0, "Active document should be found"
    assert res.chunks[0].meta["document_id"] == str(doc_uuid)

    # 2. Mark as Retired
    print("Retiring document")
    client.update_lifecycle_state(
        tenant_id=tenant_id, document_ids=[doc_uuid], state="retired"
    )

    # Verify invisible in default search
    res = client.hybrid_search("test", tenant_id=tenant_id)
    assert (
        len(res.chunks) == 0
    ), f"Retired document should NOT be found. Found: {len(res.chunks)}"

    # Verify visible in deleted search
    res_deleted = client.hybrid_search(
        "test", tenant_id=tenant_id, visibility=Visibility.DELETED
    )
    assert (
        len(res_deleted.chunks) > 0
    ), "Retired document SHOULD be found with visibility=DELETED"

    # 3. Restore to Active
    print("Restoring document")
    client.update_lifecycle_state(
        tenant_id=tenant_id, document_ids=[doc_uuid], state="active"
    )

    # Verify visible again
    res = client.hybrid_search("test", tenant_id=tenant_id)
    assert len(res.chunks) > 0, "Restored document should be found"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
