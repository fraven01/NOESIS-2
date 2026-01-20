import pytest
from pydantic import ValidationError

from ai_core.rag.schemas import Chunk


def test_chunk_frozen_config_prevents_field_reassignment():
    """Verify frozen=True blocks assignment to chunk fields."""
    chunk = Chunk(content="Test", meta={"id": "doc-1"})

    with pytest.raises(ValidationError):
        chunk.meta = {"id": "doc-2"}

    with pytest.raises(ValidationError):
        chunk.content = "Modified"


def test_chunk_meta_accepts_chunker_info():
    """Verify ChunkMeta accepts chunker and chunker_mode."""
    from ai_core.rag.ingestion_contracts import ChunkMeta

    meta = ChunkMeta(
        tenant_id="t1",
        case_id="c1",
        source="s1",
        hash="h1",
        external_id="e1",
        content_hash="ch1",
        chunker="hybrid-v1",
        chunker_mode="late",
    )
    assert meta.chunker == "hybrid-v1"
    assert meta.chunker_mode == "late"


def test_chunk_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Chunk(content="Test", meta={"id": "doc-1"}, extra_field="nope")
